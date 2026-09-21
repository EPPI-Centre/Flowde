"""Run bounded batches and commit each answer before counting it as complete."""

import signal
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from threading import current_thread, main_thread
from types import FrameType
from typing import Any

import cloudpickle
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
    def __init__(self, item_name: str = "images") -> None:
        self.error: BaseException | None = None
        self.interrupted = False
        self.item_name = item_name

    def interrupt(self, _signum: int, _frame: FrameType | None) -> None:
        if self.interrupted:
            raise _ForcedStop
        self.interrupted = True
        tqdm.write(
            f"Stopping after active {self.item_name} finish saving. "
            "Press Ctrl+C again to force stop.",
            file=sys.stderr,
        )

    @property
    def requested(self) -> bool:
        return self.error is not None or self.interrupted


@contextmanager
def _controlled_interrupt(item_name: str = "images") -> Iterator[_Stop]:
    stop = _Stop(item_name)
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


def _thread_pool(
    fn: Callable[..., Any], jobs: Any, answers: Any, events: Any, workers: int
) -> None:
    """Execute whole image jobs in threads inside one disposable process."""
    pool = ThreadPoolExecutor(max_workers=workers)
    active: dict[Future[Any], str] = {}
    try:
        while True:
            try:
                payload = jobs.get(timeout=0.05)
            except Empty:
                pass
            else:
                if payload is None:
                    break
                key, kwargs = cloudpickle.loads(payload)
                active[pool.submit(_worker, fn, kwargs, events, key)] = key
            for future in list(active):
                if not future.done():
                    continue
                key = active.pop(future)
                try:
                    result, error = future.result(), None
                except BaseException as caught:  # noqa: BLE001 - Transfer worker failures to the caller.
                    result, error = None, caught
                # Manager queues use ordinary pickle; preserve notebook-defined
                # result classes and the SDK exception transport used by loky.
                answers.put((key, cloudpickle.dumps((result, error))))
    finally:
        # On a transport failure, let the outer future report it immediately;
        # the parent then kills this process, including any remaining threads.
        pool.shutdown(wait=False)


def _parallel(
    fn: Callable[..., Any],
    pending_items: list[dict[str, Any]],
    workers: int,
    stop: _Stop,
    *,
    report: Callable[[str, RequestUsage], None],
    commit: Callable[[str, Any], None],
    fail: Callable[[str, BaseException], None],
    threaded: bool = False,
) -> None:
    manager = None
    pool: ProcessPoolExecutor | None = None
    active: dict[Future[Any], str] = {}
    thread_futures: dict[str, Future[Any]] = {}
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
        manager = SyncManager(ctx=get_context())
        manager.start(initializer=_ignore_interrupts)
        events = manager.Queue()
        pool = ProcessPoolExecutor(
            max_workers=1 if threaded else workers, initializer=_ignore_interrupts
        )
        if threaded:
            jobs, answers = manager.Queue(), manager.Queue()
            child = pool.submit(_thread_pool, fn, jobs, answers, events, workers)
        while True:
            while not stop.requested and len(active) < workers:
                item = next(remaining, None)
                if item is None:
                    break
                if threaded:
                    future: Future[Any] = Future()
                    thread_futures[item["key"]] = future
                    jobs.put(cloudpickle.dumps((item["key"], item["kwargs"])))
                else:
                    future = pool.submit(
                        _worker, fn, item["kwargs"], events, item["key"]
                    )
                active[future] = item["key"]
            if not active:
                break
            if threaded:
                if child.done():
                    child.result()  # Propagate startup, transport or process failures.
                    msg = "Image worker stopped before returning its active results."
                    raise RuntimeError(msg)
                while True:
                    try:
                        key, payload = answers.get_nowait()
                    except Empty:
                        break
                    result, error = cloudpickle.loads(payload)
                    future = thread_futures.pop(key)
                    if error is None:
                        future.set_result(result)
                    else:
                        future.set_exception(error)
            done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
            drain()
            for future in done:
                key = active.pop(future)
                try:
                    commit(key, future.result())
                except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - Re-raised after active work is saved.
                    fail(key, error)
        drain()
        if threaded:
            jobs.put(None)
            child.result()
        force = False
    finally:
        try:
            if pool is not None:
                pool.shutdown(wait=True, kill_workers=force)
        finally:
            if manager is not None and hasattr(manager, "shutdown"):
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
    threaded: bool = False,
    state_factory: Callable[..., RunState] = RunState,
) -> list[Any]:
    protect_inputs(save_dir, referenced_files(settings))
    if threaded:
        if type(n_jobs) is not int or n_jobs < 1:
            msg = "max_concurrent_jobs must be a positive integer."
            raise ValueError(msg)
        workers = n_jobs
    else:
        workers = effective_n_jobs(n_jobs)
    item_name = "PDFs" if kind == "extraction" else "images"
    with output_lock(save_dir), _controlled_interrupt(item_name) as stop:
        state = state_factory(save_dir, kind, settings, on_existing)
        state.prepare(
            [
                {key: value for key, value in item.items() if key != "kwargs"}
                for item in inputs
            ]
        )
        # Validate saved answers before any new request, and repair missing outputs.
        results = {}
        for item in inputs:
            record = state.input_records[item["key"]]
            if state.can_restore(item["key"]):
                results[item["key"]] = decode(record["result"])
                state.publish(item["key"])
        indices = {key: index for index, key in enumerate(state.input_records)}
        description = {
            "classification": "Classifying images...",
            "parsing": "Parsing images...",
            "rotation": "Rotating images...",
            "extraction": "Extracting images from PDFs",
        }[kind]
        display = UsageProgress(
            len(state.input_records), description, show_usage, totals=state.totals()
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
                    f"While processing {state.input_records[key]['path']}. "
                    f"Saved work is in {save_dir}; resume with on_existing='resume'."
                )
            tqdm.write(
                f"Failed: {state.input_records[key]['path']}: "
                f"{type(error).__name__}: {error}",
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
            if pending and (workers > 1 or threaded):
                _parallel(
                    fn,
                    pending,
                    min(workers, len(pending)),
                    stop,
                    report=report,
                    commit=commit,
                    fail=fail,
                    threaded=threaded,
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
            msg = f"Stopped after saving active {item_name}; the run can be resumed."
            raise KeyboardInterrupt(msg)
        return [results[item["key"]] for item in inputs]
