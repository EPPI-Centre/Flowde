# API reference

The reference describes the functions used to build a Flowde workflow. Function
signatures below are generated from the package source. For an introduction with
runnable examples, begin with the [pipeline guide](../pipeline/index.md).

| Page                                         | Contents                                                                                                 |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Pipeline functions](pipeline.md)            | Directory-based extraction, classification, rotation and parsing                                         |
| [Function factories and helpers](helpers.md) | Paddle, OpenAI and Gemini factories; connection checks; partial-result joining; custom function settings |
| [Data types](data-types.md)                  | Classification and parsing schemas, few-shot examples, usage reports and token prices                    |
| [Benchmarks](benchmarks.md)                  | Benchmark constructors, metric methods and detailed matching results                                     |

Pass filesystem paths as `pathlib.Path` objects. Directory functions inspect
top-level files, sort the input paths, and save results in the supplied
`save_dir`. The [run-management guide](../pipeline/resuming.md) explains the
shared `on_existing` options.

The reference focuses on the workflow entry points and their public helpers.
Modules whose names begin with `_` implement the saved-state and worker
machinery; applications should use the pipeline functions to manage those
operations.
