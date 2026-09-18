# Function factories and helpers

Factories create the callable passed into a pipeline stage. Creating a callable
does not run the dataset. Factory settings are recorded so a resumed run can
reject incompatible changes.

## Paddle extraction

::: flowde.extract_fns.paddle_layout_detect_extraction.make_paddle_layout_extract_fn

See the [extraction guide](../pipeline/image-extraction.md) for a directory-based
example.

## OpenAI classification and rotation

::: flowde.classify_fns.openai_classify_fn.make_openai_classify_fn

The returned callable takes an image path and returns the validated `label`,
not a complete classification model. Use
[`RotationClassification`](data-types.md#flowde.classify_fns.classify_types.RotationClassification)
as the
`result_structure` when the callable will be passed to
[`rotate_imgs()`](pipeline.md#flowde.rotate_imgs.rotate_imgs) .

## Gemini classification and rotation

::: flowde.classify_fns.gemini_classify_fn.make_gemini_classify_fn

This factory has the same classification contract as the OpenAI factory.
`effort` is passed to Gemini as the thinking level. There is no `from_azure`
parameter.

## OpenAI parsing

::: flowde.parsing_fns.openai_parse.make_openai_parse_fn

Returns a callable accepting an image path and optional partial-flowchart model.
The callable returns a validated Pydantic model.

## Gemini parsing

::: flowde.parsing_fns.gemini_parse.make_gemini_parse_fn

This factory has the same parsing contract as the OpenAI factory. `effort` is
passed as the thinking level; `from_azure` is not supported.

### Shared model-factory parameters

| Parameter           | Meaning                                                                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_text`        | Instructions sent to the model. Use a supplied CONSORT prompt or your own prompt.                                                                                                        |
| `model`             | Provider model ID. For Azure, use the deployment name.                                                                                                                                   |
| `effort`            | Reasoning effort or thinking level. Defaults to `"high"`; choose a value supported by the model.                                                                                         |
| `result_structure`  | Classification schema or custom parsing Pydantic schema. Classification defaults to `BinaryClassification`.                                                                              |
| `parts_to_parse`    | Parsing only: a set containing any of `"node_text"`, `"labels"`, `"flow"`, `"additional_texts"`. When neither this parameter nor a custom schema is supplied, all four parts are parsed. |
| `few_shot_examples` | Optional list of `VisionFewShotExample` instances, containing example images and expected responses.                                                                                     |
| `from_azure`        | OpenAI factories only: use `AZURE_API_KEY` and `AZURE_API_BASE`. Defaults to `False`.                                                                                                    |
| `token_prices`      | Optional `TokenPrices` used for cost estimates. Does not change provider billing.                                                                                                        |

Do not supply both `parts_to_parse` and `result_structure`. Use a custom schema
for an output structure different from Flowde's built-in parts.

See [provider setup](../pipeline/setup-default-funcs.md),
[few-shot examples](../pipeline/few-shot.md) and
[usage reporting](../pipeline/usage.md).

## Connection check

::: flowde.api_utils.openai_utils.check_openai_connection

## Parsing helpers

::: flowde.combine_parsed_parts.combine_parsed_parts

::: flowde.parsing_fns.parsing_types.build_partial_flowchart_schema

Returns a Pydantic class with exactly the requested fields. Generated schemas
reject extra fields. `None` selects all four parsing parts. See the field map
under [parsing schemas](data-types.md#parsing-schemas).

::: flowde.parsing_fns.parsing_types.build_a_partial_flowchart

::: flowde.utils.pretty_json

Returns indented JSON text for a Pydantic model, a list of Pydantic models, or
ordinary JSON-compatible data. Use
[`print()`](https://docs.python.org/3/library/functions.html#print) to display the text
or
[`Path.write_text()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.write_text)
to save the text.

## Custom function settings

::: flowde.model_function

Use
[`model_function(fn, result_structure=..., **run_settings)`](helpers.md#flowde.model_function)
to declare the
settings of a custom callable. `result_structure` is required for parsers.
Settings describe the callable; they do not configure or change the callable.
Include a version setting when changing the callable's algorithm should make
old runs incompatible. Declare input files as `Path` values so their contents
are checked on resume. See [custom functions](../pipeline/custom-functions.md).

## Supplied CONSORT prompts

| Import                                                    | Intended task                            |
| --------------------------------------------------------- | ---------------------------------------- |
| `flowde.prompts.consort.consort_nodes_prompt`             | Node text and node numbers               |
| `flowde.prompts.consort_labels.consort_labels_prompt`     | Node labels                              |
| `flowde.prompts.consort_flow.consort_flow_prompt`         | Directed flow between nodes              |
| `flowde.prompts.consort_add_text.consort_add_text_prompt` | Additional text outside nodes and labels |

The [CONSORT recipe](../recipes/consort.md) demonstrates these parsing prompts
and combines their outputs. Classification and rotation use short prompts
written directly in the example; those prompts are not separate package imports.
