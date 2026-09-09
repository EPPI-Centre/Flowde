"""Optional usage reporting, independent of task results and error handling."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class RequestUsage:
    """Usage from one contributing operation; unknown values remain `None`."""

    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


_usage_reporter: ContextVar[Callable[[RequestUsage], None] | None] = ContextVar(
    "flowde_usage_reporter", default=None
)


def usage_reporting_active() -> bool:
    return _usage_reporter.get() is not None


def report_usage(usage: RequestUsage) -> None:
    """
    Report a contributing operation without changing the callable's return value.

    Outside a tracked run this does nothing. Custom functions may call this for
    each model response or separately known charge; reporting is never required.
    Costs are USD and token counts must describe this operation, not a running sum.
    """
    reporter = _usage_reporter.get()
    if reporter is not None:
        reporter(usage)


@contextmanager
def usage_scope(reporter: Callable[[RequestUsage], None]) -> Iterator[None]:
    """Associate reports with the current operation and restore the previous scope."""
    token = _usage_reporter.set(reporter)
    try:
        yield
    finally:
        _usage_reporter.reset(token)


@dataclass
class UsageTotals:
    """Accumulate known figures per item, without inspecting its result."""

    costs: dict[int, float] = field(default_factory=dict)
    tokens: dict[int, int] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)
    missing_token_models: set[str] = field(default_factory=set)
    reported_items: set[int] = field(default_factory=set)
    finished_items: set[int] = field(default_factory=set)

    def add(self, item: int, usage: RequestUsage) -> None:
        self.reported_items.add(item)
        model = f"{usage.provider}/{usage.model}"
        if usage.cost is None:
            self.unpriced_models.add(model)
        else:
            self.costs[item] = self.costs.get(item, 0.0) + usage.cost
        if usage.total_tokens is None:
            self.missing_token_models.add(model)
        else:
            self.tokens[item] = self.tokens.get(item, 0) + usage.total_tokens

    def lines(self) -> tuple[str, str]:
        cost_line = "Est. cost: unavailable"
        if self.costs:
            total = sum(self.costs.values())
            cost_line = (
                f"Est. cost: ${total:.6f} total | "
                f"${total / len(self.costs):.6f} avg/image | "
                f"${max(self.costs.values()):.6f} max/image"
            )
        if self.unpriced_models:
            cost_line += " | unpriced: " + ", ".join(sorted(self.unpriced_models))

        tokens_summary = "Tokens: unavailable"
        if self.tokens:
            total_tokens = sum(self.tokens.values())
            tokens_summary = (
                f"Tokens: {total_tokens:,} total | "
                f"{total_tokens / len(self.tokens):,.1f} avg/image | "
                f"{max(self.tokens.values()):,} max/image"
            )
        if self.missing_token_models:
            tokens_summary += " | usage unavailable: " + ", ".join(
                sorted(self.missing_token_models)
            )

        unreported = len(self.finished_items - self.reported_items)
        if unreported:
            suffix = f" | {unreported} images without usage"
            cost_line += suffix
            tokens_summary += suffix
        if self.unpriced_models or unreported:
            cost_line = cost_line.replace(" total", " known total")
        if self.missing_token_models or unreported:
            tokens_summary = tokens_summary.replace(" total", " known total")
        return cost_line, tokens_summary


def print_request_usage(usage: RequestUsage) -> None:
    """Print an explicitly requested standalone summary, outside a progress display."""
    cost = "unavailable" if usage.cost is None else f"${usage.cost:.6f}"
    print(
        f"Estimated {usage.provider} request cost: {cost}; model={usage.model}; "
        f"input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}, "
        f"total_tokens={usage.total_tokens}"
    )
