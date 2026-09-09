"""Deliver worker usage to one progress display in the parent process."""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from html import escape
from importlib import import_module
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from time import monotonic
from typing import Any, Protocol, TypeVar

from joblib.externals.loky.backend.context import get_context
from tqdm import tqdm

from flowde.usage import RequestUsage, UsageTotals, usage_scope

Result = TypeVar("Result")
UsageEvent = tuple[int, RequestUsage | None] | None


class EventQueue(Protocol):
    def put(self, event: UsageEvent) -> None: ...

    def get(self) -> UsageEvent: ...


def run_with_usage(
    events: EventQueue,
    item: int,
    fn: Callable[..., Result],
    *args: Any,
    **kwargs: Any,
) -> Result:
    """Keep all reports inside this callable associated with the same input item."""
    with usage_scope(lambda usage: events.put((item, usage))):
        result = fn(*args, **kwargs)
    events.put((item, None))
    return result


def _in_notebook() -> bool:
    # IPython is optional and is already loaded when running in a notebook.
    ipython = sys.modules.get("IPython")
    return (
        ipython is not None
        and getattr(ipython.get_ipython(), "kernel", None) is not None
    )


class _NotebookProgress:
    def __init__(self, total: int, description: str, initial: int = 0) -> None:
        # Notebook output updates do not require ipywidgets or terminal cursors.
        self.total = total
        self.description = description
        self.started = monotonic()
        self.display = import_module("IPython.display").DisplayHandle()
        self.display.display(self._content(initial, ()), raw=True)

    def _content(self, completed: int, lines: tuple[str, ...]) -> dict[str, str]:
        bar = tqdm.format_meter(
            completed,
            self.total,
            monotonic() - self.started,
            prefix=self.description,
        )
        text = "\n".join((bar, *lines))
        return {
            "text/plain": text,
            "text/html": f'<pre style="white-space: pre-wrap">{escape(text)}</pre>',
        }

    def refresh(self, completed: int, lines: tuple[str, ...]) -> None:
        self.display.update(self._content(completed, lines), raw=True)

    def close(self) -> None:
        # The last update stays in the same output block when the cell finishes.
        pass


class _TerminalProgress:
    def __init__(self, total: int, description: str, initial: int = 0) -> None:
        self.progress = tqdm(
            total=total, desc=description, dynamic_ncols=True, initial=initial
        )
        self.usage_lines: list[tqdm] = []
        self.summary: tuple[str, ...] = ()

    def refresh(self, completed: int, lines: tuple[str, ...]) -> None:
        self.progress.update(completed - self.progress.n)
        self.summary = lines
        if not lines:
            return
        if not self.usage_lines:
            self.usage_lines = [
                tqdm(total=0, bar_format="{desc}", dynamic_ncols=True, position=i)
                for i in (1, 2)
            ]
        for line, text in zip(self.usage_lines, lines, strict=True):
            line.set_description_str(text)

    def close(self) -> None:
        # Clear auxiliary bars before leaving one final summary below the main bar.
        for line in reversed(self.usage_lines):
            line.leave = False
            line.close()
        self.progress.close()
        for text in self.summary:
            tqdm.write(text)


class UsageProgress:
    def __init__(
        self,
        total: int,
        description: str,
        show_usage: bool,
        *,
        totals: UsageTotals | None = None,
    ) -> None:
        self.totals = totals if totals is not None else UsageTotals()
        self.show_usage = show_usage
        renderer = _NotebookProgress if _in_notebook() else _TerminalProgress
        self.display = renderer(
            total, description, initial=len(self.totals.finished_items)
        )
        if totals is not None:
            self._refresh()

    def _refresh(self) -> None:
        lines = (
            self.totals.lines()
            if self.show_usage and self.totals.reported_items
            else ()
        )
        self.display.refresh(len(self.totals.finished_items), lines)

    def receive(self, item: int, usage: RequestUsage | None) -> None:
        if usage is None:
            self.totals.finished_items.add(item)
        else:
            self.totals.add(item, usage)
        self._refresh()

    def close(self) -> None:
        self._refresh()
        self.display.close()


def _consume_usage(events: EventQueue, display: UsageProgress) -> None:
    while (event := events.get()) is not None:
        display.receive(*event)


@contextmanager
def _event_queue(parallel: bool) -> Iterator[EventQueue]:
    if parallel:
        # Match joblib's process context: works in notebooks and unguarded scripts,
        # unlike a default multiprocessing Manager on platforms using spawn.
        with SyncManager(ctx=get_context()) as manager:
            yield manager.Queue()
    else:
        yield Queue()


@contextmanager
def progress_reporting(
    total: int, description: str, *, parallel: bool, show_usage: bool
) -> Iterator[EventQueue]:
    with _event_queue(parallel) as events:
        display = UsageProgress(total, description, show_usage)
        listener = Thread(target=_consume_usage, args=(events, display), daemon=True)
        listener.start()
        try:
            yield events
        finally:
            # Each put is delivered before returning to the worker. Drain all usage
            # already received, including usage preceding an exception, then close.
            events.put(None)
            listener.join()
            display.close()
