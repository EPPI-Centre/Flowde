"""Standard paid-tier USD estimates for image/text input and text output."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TokenPrices:
    """
    Prices per million tokens, optionally with a higher long-context rate.

    `input` applies to uncached tokens; cache reads/writes replace that rate.
    Above `long_context_threshold`, input and cache rates double and output
    increases by 50%, matching the tiered models in the bundled tables.
    """

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None
    long_context_threshold: int | None = None

    def estimate(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        *,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        if cached_tokens and self.cached_input is None:
            return None
        if cache_write_tokens and self.cache_write is None:
            return None
        uncached = input_tokens - cached_tokens - cache_write_tokens
        if uncached < 0:
            return None
        input_cost = (
            uncached * self.input
            + cached_tokens * (self.cached_input or 0)
            + cache_write_tokens * (self.cache_write or 0)
        )
        output_cost = output_tokens * self.output
        if (
            self.long_context_threshold is not None
            and input_tokens > self.long_context_threshold
        ):
            input_cost *= 2
            output_cost *= 1.5
        return (input_cost + output_cost) / 1_000_000


# Verified 2026-09-07. Standard direct-API prices, not Batch/Flex/Fast mode,
# regional uplifts, tool fees, storage fees, or negotiated account rates.
# https://developers.openai.com/api/docs/pricing
# Older models: https://developers.openai.com/api/docs/models/<model-id>
OPENAI_PRICES_PER_1M = {
    "gpt-6-astra": TokenPrices(
        input=10,
        cached_input=1,
        cache_write=12.5,
        output=50,
        long_context_threshold=272_000,
    ),
    "gpt-5.6-sol": TokenPrices(
        input=4,
        cached_input=0.4,
        cache_write=5,
        output=20,
        long_context_threshold=272_000,
    ),
    "gpt-5.6-terra": TokenPrices(
        input=2,
        cached_input=0.2,
        cache_write=2.5,
        output=12,
        long_context_threshold=272_000,
    ),
    "gpt-5.6-luna": TokenPrices(
        input=0.2,
        cached_input=0.02,
        cache_write=0.25,
        output=1.2,
        long_context_threshold=272_000,
    ),
    "gpt-5.5": TokenPrices(
        input=5,
        cached_input=0.5,
        output=30,
        long_context_threshold=272_000,
    ),
    "gpt-5.4": TokenPrices(
        input=2.5,
        cached_input=0.25,
        output=15,
        long_context_threshold=272_000,
    ),
    "gpt-5.4-mini": TokenPrices(input=0.75, cached_input=0.075, output=4.5),
    "gpt-5.4-nano": TokenPrices(input=0.2, cached_input=0.02, output=1.25),
    "gpt-5.2": TokenPrices(input=1.75, cached_input=0.175, output=14),
    "gpt-5.1": TokenPrices(input=1.25, cached_input=0.125, output=10),
    "gpt-5": TokenPrices(input=1.25, cached_input=0.125, output=10),
    "gpt-5-mini": TokenPrices(input=0.25, cached_input=0.025, output=2),
    "gpt-5-nano": TokenPrices(input=0.05, cached_input=0.005, output=0.4),
    "gpt-4.1": TokenPrices(input=2, cached_input=0.5, output=8),
    "gpt-4.1-mini": TokenPrices(input=0.4, cached_input=0.1, output=1.6),
    "gpt-4.1-nano": TokenPrices(input=0.1, cached_input=0.025, output=0.4),
    "gpt-4o": TokenPrices(input=2.5, cached_input=1.25, output=10),
    "gpt-4o-mini": TokenPrices(input=0.15, cached_input=0.075, output=0.6),
}
OPENAI_PRICES_PER_1M["gpt-5.6"] = OPENAI_PRICES_PER_1M["gpt-5.6-sol"]

# https://ai.google.dev/gemini-api/docs/pricing (verified 2026-09-07).
# Input rates here cover text/images, not audio. Output includes thinking.
GEMINI_PRICES_PER_1M = {
    "gemini-3.5-flash": TokenPrices(input=1.5, cached_input=0.15, output=9),
    "gemini-3.5-flash-lite": TokenPrices(input=0.3, cached_input=0.03, output=2.5),
    "gemini-3.1-pro-preview": TokenPrices(
        input=2,
        cached_input=0.2,
        output=12,
        long_context_threshold=200_000,
    ),
    "gemini-3.1-flash-lite": TokenPrices(input=0.25, cached_input=0.025, output=1.5),
    "gemini-2.5-pro": TokenPrices(
        input=1.25,
        cached_input=0.125,
        output=10,
        long_context_threshold=200_000,
    ),
    "gemini-2.5-flash": TokenPrices(input=0.3, cached_input=0.03, output=2.5),
    "gemini-2.5-flash-lite": TokenPrices(input=0.1, cached_input=0.01, output=0.4),
}


def token_count(usage: object, name: str) -> int | None:
    """Read an available count without replacing absent usage with zero."""
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and value >= 0 else None
