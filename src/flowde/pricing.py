"""Standard paid-tier USD estimates for image/text input and text output."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TokenPrices:
    """
    Store token prices for estimating the cost of a model request.

    Parameters
    ----------
    input : float
        Price in USD per million uncached input tokens.
    output : float
        Price in USD per million output tokens.
    cached_input : float | None, optional
        Price in USD per million input tokens read from a cache. This rate
        replaces the ordinary `input` rate for those tokens. Defaults to `None`,
        meaning the cache-read rate is unknown. A value of `0` means cache reads
        are free. An unknown rate prevents a cost estimate only when the
        request includes cache-read tokens.
    cache_write : float | None, optional
        Price in USD per million input tokens written to a cache. This rate
        replaces the ordinary `input` rate for those tokens. Defaults to `None`,
        meaning the cache-write rate is unknown. A value of `0` means cache
        writes are free. An unknown rate prevents a cost estimate only when
        the request includes cache-write tokens.
    long_context_threshold : int | None, optional
        Input-token count above which higher rates apply. Defaults to `None`,
        meaning the same rates apply at every request length. When a request's
        total input-token count is strictly greater than this threshold, all
        input rates, including cache reads and writes, are multiplied by `2`,
        and the output rate is multiplied by `1.5`. The higher rates apply to
        the entire request, not just tokens beyond the threshold. These
        multipliers are fixed; this option is suitable only for prices that
        follow that rule.

    Notes
    -----
    Constructor arguments must be passed by name. The dataclass is frozen, so
    its fields cannot be reassigned after construction. Rates and the threshold
    are stored as supplied, without numeric-range validation or a provider
    price lookup.

    You can pass an instance as `token_prices` to an OpenAI or Gemini model
    factory to override Flowde's bundled prices for cost reporting. The
    override changes estimates, not provider billing or model requests.
    [`estimate()`][flowde.pricing.TokenPrices.estimate] can also calculate a
    cost directly from token counts. Creating a `TokenPrices` instance or
    calculating a cost makes no API request.

    Examples
    --------
    Define illustrative rates of $2 per million uncached input tokens,
    $8 per million output tokens and $0.50 per million cached input tokens:

    ```python
    from flowde.pricing import TokenPrices

    prices = TokenPrices(input=2.0, output=8.0, cached_input=0.5)
    ```

    These are example rates, not prices for a particular model.

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
        """
        Estimate one request's cost from its input, output and cache token counts.

        Parameters
        ----------
        input_tokens : int | None
            Total input tokens, including the tokens counted in `cached_tokens`
            and `cache_write_tokens`. `None` means the count is unknown and
            prevents a cost estimate. This total determines whether the
            long-context threshold is exceeded.
        output_tokens : int | None
            Total output tokens to price at the `output` rate. Include any
            reasoning or thinking tokens charged as output. `None` means the
            count is unknown and prevents a cost estimate.
        cached_tokens : int, optional
            Input tokens read from a cache, priced at `cached_input` instead
            of `input`. Defaults to `0`. These tokens must already be included
            in `input_tokens` and must not also count as cache-write tokens.
        cache_write_tokens : int, optional
            Input tokens written to a cache, priced at `cache_write` instead
            of `input`. Defaults to `0`. These tokens must already be included
            in `input_tokens` and must not also count as cache-read tokens.

        Returns
        -------
        float | None
            Estimated cost in USD. Returns `None` if either total token count
            is unknown, a nonzero cache count has no corresponding price, or
            the two cache counts together exceed `input_tokens`. Known zero
            counts or zero prices can produce a cost of `0.0`.

        Notes
        -----
        Uncached input tokens are calculated as
        `input_tokens - cached_tokens - cache_write_tokens`. Each input group
        is multiplied by its own per-million-token rate; output tokens are
        multiplied by the output rate. The sum is divided by `1_000_000`.

        When `input_tokens` exceeds a configured `long_context_threshold`, the combined
        input cost is doubled and the output cost is multiplied by `1.5`
        before summing. At exactly the threshold, the original rates apply.
        Output tokens do not contribute to the threshold comparison.

        Counts are expected to be nonnegative integers. Apart from the checks
        described under Returns, counts and prices are used as supplied.
        The method does not validate every numeric range, contact a provider,
        round the estimate or include charges unrelated to these token counts.

        Examples
        --------
        Calculate a cost using illustrative rates and 1,000 input tokens,
        of which 400 were read from a cache, plus 200 output tokens:

        ```python
        from flowde.pricing import TokenPrices

        prices = TokenPrices(input=2.0, output=8.0, cached_input=0.5)
        cost = prices.estimate(
            input_tokens=1_000,
            output_tokens=200,
            cached_tokens=400,
        )
        ```

        The estimated cost is `$0.003`: `$0.0012` for the 600 uncached input
        tokens, `$0.0002` for the 400 cached input tokens and `$0.0016` for the
        200 output tokens.

        """
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
