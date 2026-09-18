# Data types

## Classification schemas

Import these types from `flowde.classify_fns.classify_types`.

| Type                                                                                 | Output                                                                                                           |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `Classification[LabelType]`{#flowde.classify_fns.classify_types.Classification}      | Pydantic model with a single `label` field. Specialise the generic with a type or a `Literal` of allowed values. |
| `BinaryClassification`{#flowde.classify_fns.classify_types.BinaryClassification}     | `label` is integer `0` or `1`.                                                                                   |
| `ConsortClassification`                                                              | Binary schema for the supplied CONSORT classification prompt: `0` for other images, `1` for CONSORT diagrams.    |
| `RotationClassification`{#flowde.classify_fns.classify_types.RotationClassification} | `label` is `0`, `90`, `180` or `270`: the clockwise correction to apply.                                         |

Classification schemas use strict validation. For example, the JSON string
`"1"` is not the same as the integer `1` required by `BinaryClassification`.
Model factories return the validated label from the response.

```python
from typing import Literal

from flowde.classify_fns.classify_types import Classification

ImageType = Classification[Literal["flowchart", "table", "other"]]
answer = ImageType.model_validate({"label": "flowchart"})
print(answer.label)
```

## Parsing schemas

Import `ParseType`, `Node`, `Flowchart` and the schema-building helpers from
`flowde.parsing_fns.parsing_types`.

| `ParseType` value    | JSON fields requested                                      |
| -------------------- | ---------------------------------------------------------- |
| `"node_text"`        | `nodes`, with `node_number` and `text` for every node      |
| `"labels"`           | `nodes`, with `node_number` and `labels` for every node    |
| `"flow"`             | `nodes`, with `node_number` and `points_to` for every node |
| `"additional_texts"` | Top-level `additional_texts` list                          |

`Node` has `node_number: int`, `text: str`, `labels: list[str]` and
`points_to: list[int]`. `Flowchart` has `nodes: list[Node]` and
`additional_texts: list[str]`. The default parsing factories create a schema for
the selected parts using
[`build_partial_flowchart_schema()`](helpers.md#flowde.parsing_fns.parsing_types.build_partial_flowchart_schema)
.

Inference writes one JSON object per image. Ground truth uses an `options`
wrapper and separates the four parts into different directories. See the
[parsing benchmark format](../benchmarking/parsing.md#prepare-the-input-directories).

Pydantic results support
[`model_dump()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump)
for a Python dictionary,
[`model_dump_json(indent=2)`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump_json)
for JSON text, and
[`model_json_schema()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_json_schema)
for the
schema. See [parsing](../pipeline/parsing.md) for complete examples.

## Few-shot example

<!-- prettier-ignore-start -->

::: flowde.utils.VisionFewShotExample
    options:
      members:
        - expected_output_text

<!-- prettier-ignore-end -->

See [few-shot examples](../pipeline/few-shot.md) for classification and parsing
examples using model factories.

## Token prices

::: flowde.pricing.TokenPrices

::: flowde.pricing.TokenPrices.estimate

See [tokens and costs](../pipeline/usage.md) for usage reporting and custom-price
examples.

## Usage reports

::: flowde.usage.RequestUsage

Create a report with `model` and `provider`, plus any known `input_tokens`,
`output_tokens`, `total_tokens` and `cost`. Unknown values default to `None`.
For a custom function, supply `total_tokens` explicitly if you want the total to
appear in Flowde's summary; the report object does not calculate that field.

::: flowde.usage.report_usage

Records a `RequestUsage` report for the currently running item. Custom
classifiers and parsers can report more than one request per image. Calling
[`report_usage()`](data-types.md#flowde.usage.report_usage) outside a managed usage
context has no effect. See
[custom usage reporting](../pipeline/usage.md#report-usage-from-a-custom-function).

## Function contracts

| Protocol                                                                            | Import                               | Required callable behaviour                                                                                                   |
| ----------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `ExtractImgsFunction`                                                               | `flowde.extract_fns.extract_types`   | Accept `pdf_path: Path` and `save_dir: Path`; write PNGs into the supplied directory.                                         |
| `ClassificationFunction`                                                            | `flowde.classify_fns.classify_types` | Accept `img_path: Path`; return one label.                                                                                    |
| [`ParsingFunction`](data-types.md#flowde.parsing_fns.parsing_types.ParsingFunction) | `flowde.parsing_fns.parsing_types`   | Accept `img_path: Path` and optional `partial_flowchart`; return a Pydantic model and expose its class as `result_structure`. |

The [custom-function guide](../pipeline/custom-functions.md) shows how to declare
settings as well as satisfy these callable contracts.

<!-- prettier-ignore-start -->

::: flowde.parsing_fns.parsing_types.ParsingFunction
    options:
      members:
        - __call__

<!-- prettier-ignore-end -->
