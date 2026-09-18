# Tokens and costs

[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs),
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs) and
[`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)
collect token usage and estimated API costs reported by their model helpers and
record them in the `.flowde/run.state` file inside `save_dir`. The
[OpenAI, Azure OpenAI and Gemini helpers](setup-default-funcs.md) automatically
report usage from model responses.

## Display usage

By default, [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs),
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs) and
[`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs) display
token usage and API costs as they process images.

For example, suppose Flowde records 100 tokens and a cost of $0.0001
for one image, and 300 tokens and $0.0003 for a second image. The display is:

```text
Est. cost: $0.000400 total | $0.000200 avg/image | $0.000300 max/image
Tokens: 400 total | 200.0 avg/image | 300 max/image
```

Set `show_usage=False` to hide the usage display.

## Custom token prices

If the current package version does not include prices for a model, you can
supply custom prices to an existing classifier, rotation or parsing helper:

```python
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.pricing import TokenPrices

prices = TokenPrices(input=1.0, output=4.0, cached_input=0.1)

classify_fn = make_openai_classify_fn(
    input_text="Return 1 if the image is a flowchart, otherwise return 0.",
    model="gpt-7",
    effort="medium",
    token_prices=prices,
)
```

See [`TokenPrices`](../reference/data-types.md#flowde.pricing.TokenPrices) for
all available pricing parameters.

## Report usage from a custom function

Usage reporting is optional for custom functions. Inside your custom classifier
or parser, you can use
[`report_usage()`](../reference/data-types.md#flowde.usage.report_usage)
after receiving a model response. For example, a response with 120 input tokens,
30 output tokens and a known cost of $0.00024 would need to be reported like this:

```python
from flowde.usage import RequestUsage, report_usage

report_usage(
    RequestUsage(
        provider="my-provider",
        model="my-model",
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cost=0.00024,
    )
)
```

Flowde does not collect usage reports when the custom function runs outside
`extract_imgs()`, `classify_imgs()`, `rotate_imgs()` or `parse_imgs()`. Calling
[`report_usage()`](../reference/data-types.md#flowde.usage.report_usage) outside
one of these runs has no effect.

See the
[`RequestUsage`](../reference/data-types.md#flowde.usage.RequestUsage) and
[`report_usage()`](../reference/data-types.md#flowde.usage.report_usage)
API references for details.
