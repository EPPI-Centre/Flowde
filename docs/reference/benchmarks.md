# Benchmarks

Benchmarks read saved JSONs and run locally. Constructing or evaluating a
benchmark does not call an LLM. See the stage-specific benchmarking guides for
input formats.

## Classification and rotation

::: flowde.benchmarks.classification.classification_benchmark.ClassificationBenchmark

::: flowde.benchmarks.classification.classification_benchmark.BinaryClassificationBenchmark

::: flowde.benchmarks.classification.classification_benchmark.MulticlassClassificationBenchmark

### RotationBenchmark {#flowde.benchmarks.rotation.rotation_benchmark.RotationBenchmark}

```python
RotationBenchmark(*, true_path: Path, pred_path: Path)
```

<!-- Griffe includes the inherited labels argument despite init=False here. -->
<!-- Keep the actual signature above and render the source docstring below. -->
<!-- prettier-ignore -->
::: flowde.benchmarks.rotation.rotation_benchmark.RotationBenchmark
    options:
      show_root_heading: false
      show_root_toc_entry: false

### Individual classification results

::: flowde.benchmarks.classification.classification_bench_types.SingleClassificationResult

## Parsing

::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark

### Methods

Select a method below for its parameters, return values and error conditions.
Select a result type to inspect the fields and properties on each returned
object. Lists follow sorted flowchart filename-stem order.

| Method | Required predicted parts | Return value |
| --- | --- | --- |
| [`node_matches(range_indices=None)`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches) | Node text | List of [`NodeMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches) |
| [`node_matches_with_flow()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches_with_flow) | Node text and flow | List of [`NodeMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches) |
| [`node_and_label_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches) | Node text and labels | List of [`NodeMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches) |
| [`additional_text_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.additional_text_matches) | Node text and additional text | List of [`TextListMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.TextListMatches) |
| [`diagram_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches) | All four parts | List of [`DiagramMatch`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.DiagramMatch) |
| [`total_node_text_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_node_text_cost) | Node text | `float` |
| [`total_label_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_label_cost) | Node text and labels | `float` |
| [`total_additional_text_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_additional_text_cost) | Node text and additional text | `float` |
| [`all_flow_scores()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_scores) | Node text and flow | List of [`FlowScores`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.FlowScores) |
| [`all_flow_jaccard_scores()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_jaccard_scores) | Node text and flow | `list[float]` |
| [`avg_flow_jaccard()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.avg_flow_jaccard) | Node text and flow | `float` |

All methods use the available predicted parts to select Node Matches and a
Ground-Truth Option; see [how diagrams are matched](../benchmarking/parsing.md#how-diagrams-are-matched).
Each method calculates its results from the loaded data when called. Results
are not cached between calls, and scoring makes no model requests.

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches_with_flow
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.additional_text_matches
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_node_text_cost
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_label_cost
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_additional_text_cost
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_scores
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_jaccard_scores
    options:
      heading_level: 4
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.avg_flow_jaccard
    options:
      heading_level: 4
      signature_crossrefs: true

### Matching result objects

These Pydantic models hold the comparisons returned by the benchmark.
A match can pair a prediction with ground truth or record an unmatched node
or string. The following entries describe the stored fields and computed
properties.

| Type | Contains |
| --- | --- |
| [`NodeMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches) | Node Matches for one flowchart and the selected Ground-Truth Option. |
| [`NodeMatch`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatch) | One paired or unmatched node, with text cost and optional label matches. |
| [`TextListMatches`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.TextListMatches) | Comparisons for one node's labels or one flowchart's additional text. |
| [`TextListMatch`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.TextListMatch) | One paired or unmatched string, with source-list indices and cost. |
| [`DiagramMatch`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.DiagramMatch) | A complete prediction, its selected ground-truth diagram and all part comparisons. |
| [`FlowScores`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.FlowScores) | Directed-connection counts, metrics, and missing and extra connections. |
| [`Node`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.Node) | An individual predicted or ground-truth node held inside a match. |
| [`Diagram`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.Diagram) | The prediction or selected ground-truth diagram held inside a complete comparison. |

You can inspect stored fields with
[`model_dump()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump)
or serialise stored fields with
[`model_dump_json()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump_json).
Computed properties such as `total_text_cost` and `flow_score` are accessed
directly and are not included in those dumps.

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.NodeMatches
    options:
      heading_level: 4
      merge_init_into_class: false
      members:
        - get_matched_true_node
        - get_matched_pred_node
      signature_crossrefs: true

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.NodeMatch
    options:
      heading_level: 4
      merge_init_into_class: false

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.TextListMatches
    options:
      heading_level: 4
      merge_init_into_class: false

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.TextListMatch
    options:
      heading_level: 4
      merge_init_into_class: false

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.DiagramMatch
    options:
      heading_level: 4
      merge_init_into_class: false

### Flow scores

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.FlowScores
    options:
      heading_level: 4
      merge_init_into_class: false

### Nodes and diagrams

These benchmark models preserve the source text and node numbers alongside
the metadata identifying the flowchart and Ground-Truth Option.

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.Node
    options:
      heading_level: 4
      merge_init_into_class: false

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.parsing_bench_types.Diagram
    options:
      heading_level: 4
      merge_init_into_class: false

## Text distances

Import the following functions from
`flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn`.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_with_text_normalisation

[`ParsingBenchmark`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark)
uses this function when you omit `distance_fn`. You can still pass
[`levenshtein_fn()`](benchmarks.md#flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_fn)
for raw character comparisons, or another distance function.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_fn

Counts the minimum number of single-character insertions, deletions and
substitutions. Treats a missing string as empty; two missing strings are invalid.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.number_only_levenshtein

Compares the extracted numbers in the two strings. Use this for a separate
number-focused check, not as a measure of whether the surrounding words agree.
`NodeMatches.total_numbers_only_node_text_cost` provides this check for the
already selected node matches.

<!-- prettier-ignore -->
::: flowde.benchmarks.parsing.text_distance_fns.distance_fn_protocol.DistanceFnProtocol
    options:
      members:
        - __call__
