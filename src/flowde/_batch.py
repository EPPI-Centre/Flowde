"""Run bounded batches and commit each answer before counting it as complete."""

import signal
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import contextmanager
from functools import partial
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from threading import current_thread, main_thread
from types import FrameType
from typing import Any

from joblib import effective_n_jobs
from joblib.externals.loky import ProcessPoolExecutor
from joblib.externals.loky.backend.context import get_context
from tqdm import tqdm

from flowde._progress import UsageProgress
from flowde._run_settings import referenced_files
from flowde._run_state import ExistingRun, RunState, output_lock, protect_inputs
from flowde.usage import RequestUsage, usage_scope


class _ForcedStop(BaseException):
    """Escape per-image error handling on the second interrupt."""


class _Stop:
    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.interrupted = False

    def interrupt(self, _signum: int, _frame: FrameType | None) -> None:
        if self.interrupted:
            raise _ForcedStop
        self.interrupted = True
        tqdm.write(
            "Stopping after active images finish saving. "
            "Press Ctrl+C again to force stop.",
            file=sys.stderr,
        )

    @property
    def requested(self) -> bool:
        return self.error is not None or self.interrupted


@contextmanager
def _controlled_interrupt() -> Iterator[_Stop]:
    stop = _Stop()
    previous = None
    if current_thread() is main_thread():
        previous = signal.signal(signal.SIGINT, stop.interrupt)
    try:
        yield stop
    except _ForcedStop:
        msg = "Forced stop; only results already saved can be reused."
        raise KeyboardInterrupt(msg) from None
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def _ignore_interrupts() -> None:
    # The parent handles Ctrl+C and decides whether to drain or kill workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _restore_exception(
    error_type: type[BaseException], args: tuple[Any, ...], attributes: dict[str, Any]
) -> BaseException:
    error = error_type.__new__(error_type, *args)
    error.__dict__.update(attributes)
    return error


class _TransportError(Exception):
    """Let loky transfer SDK errors whose constructors cannot unpickle them."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error

    def __reduce__(self) -> Any:
        # For example, OpenAI's errors require response/body constructor arguments
        # that standard exception pickling omits. Restore their saved attributes.
        return _restore_exception, (
            type(self.error),
            self.error.__reduce__()[1],
            vars(self.error),
        )


def _worker(
    fn: Callable[..., Any], kwargs: dict[str, Any], events: Any, key: str
) -> Any:
    try:
        with usage_scope(lambda usage: events.put((key, usage))):
            return fn(**kwargs)
    except (Exception, KeyboardInterrupt) as error:
        raise _TransportError(error) from error


def _parallel(
    fn: Callable[..., Any],
    pending_items: list[dict[str, Any]],
    workers: int,
    stop: _Stop,
    *,
    report: Callable[[str, RequestUsage], None],
    commit: Callable[[str, Any], None],
    fail: Callable[[str, BaseException], None],
) -> None:
    manager = SyncManager(ctx=get_context())
    pool = None
    active: dict[Future[Any], str] = {}
    remaining = iter(pending_items)
    force = True

    def drain() -> None:
        while True:
            try:
                key, usage = events.get_nowait()
            except Empty:
                return
            report(key, usage)

    try:
        manager.start(initializer=_ignore_interrupts)
        events = manager.Queue()
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_ignore_interrupts)
        while True:
            while not stop.requested and len(active) < workers:
                item = next(remaining, None)
                if item is None:
                    break
                future = pool.submit(_worker, fn, item["kwargs"], events, item["key"])
                active[future] = item["key"]
            if not active:
                break
            done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
            drain()
            for future in done:
                key = active.pop(future)
                try:
                    commit(key, future.result())
                except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - Re-raised after active work is saved.
                    fail(key, error)
        drain()
        force = False
    finally:
        try:
            if pool is not None:
                pool.shutdown(wait=True, kill_workers=force)
        finally:
            if hasattr(manager, "shutdown"):
                manager.shutdown()


def run_batch(
    fn: Callable[..., Any],
    inputs: list[dict[str, Any]],
    save_dir: Path,
    *,
    kind: str,
    settings: dict[str, Any],
    encode: Callable[[Any], Any],
    decode: Callable[[Any], Any],
    n_jobs: int,
    show_usage: bool,
    on_existing: ExistingRun,
) -> list[Any]:
    protect_inputs(save_dir, referenced_files(settings))
    workers = effective_n_jobs(n_jobs)
    with output_lock(save_dir), _controlled_interrupt() as stop:
        state = RunState(save_dir, kind, settings, on_existing)
        state.prepare(
            [
                {key: value for key, value in item.items() if key != "kwargs"}
                for item in inputs
            ]
        )
        # Validate saved answers before any new request, and repair missing outputs.
        results = {}
        for item in inputs:
            record = state.items[item["key"]]
            if record["has_result"]:
                results[item["key"]] = decode(record["result"])
                state.publish(item["key"])
        indices = {key: index for index, key in enumerate(state.items)}
        description = (
            "Classifying images..." if kind == "classification" else "Parsing images..."
        )
        display = UsageProgress(
            len(state.items), description, show_usage, totals=state.totals()
        )

        def report(key: str, usage: RequestUsage) -> None:
            state.add_usage(key, usage)
            display.receive(indices[key], usage)

        def commit(key: str, result: Any) -> None:
            encoded = encode(result)
            state.result(key, encoded)
            state.publish(key)
            results[key] = result
            display.receive(indices[key], None)

        def fail(key: str, error: BaseException) -> None:
            if stop.error is None:
                stop.error = error
                error.add_note(
                    f"While processing {state.items[key]['path']}. "
                    f"Saved work is in {save_dir}; resume with on_existing='resume'."
                )
            tqdm.write(
                f"Failed: {state.items[key]['path']}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            try:
                state.failure(key, error)
            except OSError as save_error:
                tqdm.write(
                    f"Could not save the failure record: {save_error}", file=sys.stderr
                )

        pending = [item for item in inputs if item["key"] not in results]
        try:
            if workers > 1 and pending:
                _parallel(
                    fn,
                    pending,
                    min(workers, len(pending)),
                    stop,
                    report=report,
                    commit=commit,
                    fail=fail,
                )
            else:
                for item in pending:
                    if stop.requested:
                        break
                    key = item["key"]
                    try:
                        with usage_scope(partial(report, key)):
                            commit(key, fn(**item["kwargs"]))
                    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - Re-raised after active work is saved.
                        fail(key, error)
        finally:
            display.close()
        if stop.error is not None:
            raise stop.error
        if stop.interrupted:
            msg = "Stopped after saving active images; the run can be resumed."
            raise KeyboardInterrupt(msg)
        return [results[item["key"]] for item in inputs]
