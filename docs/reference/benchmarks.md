# Benchmarks

Benchmarks read saved JSONs and run locally. Constructing or evaluating a
benchmark does not call an LLM. See the stage-specific benchmarking guides for
input formats.

## Classification and rotation

::: flowde.benchmarks.classification.classification_benchmark.ClassificationBenchmark

::: flowde.benchmarks.classification.classification_benchmark.BinaryClassificationBenchmark

::: flowde.benchmarks.classification.classification_benchmark.MulticlassClassificationBenchmark

<!-- Griffe includes the inherited labels argument despite init=False here. -->
<!-- prettier-ignore -->
::: flowde.benchmarks.rotation.rotation_benchmark.RotationBenchmark
    options:
      show_signature: false

```text
RotationBenchmark(*, true_path: Path, pred_path: Path)
```

Pass `true_path` and `pred_path` as keyword arguments pointing to JSON lists of
`{"img_path": "...", "label": ...}` records. The two lists must identify the same
images. `BinaryClassificationBenchmark` requires integer labels `0` and `1`.
`MulticlassClassificationBenchmark` also requires a `labels` tuple containing the
allowed classes. `RotationBenchmark` uses `(0, 90, 180, 270)` automatically.

### Individual classification results

::: flowde.benchmarks.classification.classification_bench_types.SingleClassificationResult

### Shared properties

| Property or operation                                                    | Return value                                                    |
| ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| [`len(benchmark)`](https://docs.python.org/3/library/functions.html#len) | Number of paired results                                        |
| `result_list`                                                            | Tuple of `SingleClassificationResult` records                   |
| `img_paths`, `trues`, `preds`                                            | Aligned tuples of image paths, true labels and predicted labels |
| `correct_predictions`, `incorrect_predictions`                           | Tuples of records in each group                                 |
| `accuracy`                                                               | Fraction with exactly equal true and predicted labels           |

Each `SingleClassificationResult` contains `img_path`, `true` and `pred`.
The two lists must put images in the same order. At each position, the filenames
(including their extensions) must match; the parent directories can differ.

### Binary properties

| Property                               | Meaning                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `tp`, `fp`, `fn`, `tn`                 | Tuples of true positives, false positives, false negatives and true negatives |
| `num_tp`, `num_fp`, `num_fn`, `num_tn` | Counts of those records                                                       |
| `precision`                            | TP / (TP + FP)                                                                |
| `recall`, `tpr`                        | TP / (TP + FN)                                                                |
| `f1_score`                             | 2 × TP / (2 × TP + FP + FN)                                                   |
| `fpr`                                  | FP / (FP + TN)                                                                |
| `specificity`, `tnr`                   | TN / (TN + FP)                                                                |

Ratios with zero denominators return `0`.

### Multiclass and rotation properties

| Property             | Return value                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `labels`             | Tuple of allowed labels                                                                           |
| `confusion_matrix`   | Nested dictionary indexed first by true label, then predicted label                               |
| `num_per_true_class` | Number of ground-truth records for each label                                                     |
| `num_per_pred_class` | Number of predicted records for each label                                                        |
| `per_class_accuracy` | Correct predictions for a label divided by the number of true records for that label; `0` if none |

Rotation uses exact angle equality; a 90-degree mistake does not receive a
better score than a 180-degree mistake. See the
[rotation example](../benchmarking/rotation.md).

## Parsing

::: flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark

| Parameter                     | Meaning                                                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pred_diagrams_dir`           | One directory of combined prediction JSONs, or a sequence of directories containing non-overlapping parsing parts. Separate parts are joined by stem and node number.    |
| `distance_fn`                 | Callable taking a true string and a predicted string, either possibly `None`, and returning a non-negative finite distance. Smaller values indicate better text matches. |
| `allow_missing_pred_diagrams` | Defaults to `False`. If `True`, evaluate only ground-truth diagrams with predictions. Missing diagrams receive no penalty and are not included in averages.              |
| `true_nodes_dir`              | Ground-truth node-text JSON directory.                                                                                                                                   |
| `true_labels_dir`             | Ground-truth label JSON directory.                                                                                                                                       |
| `true_additional_texts_dir`   | Ground-truth additional-text JSON directory.                                                                                                                             |
| `true_flow_dir`               | Ground-truth directed-flow JSON directory.                                                                                                                               |
| `expected_num_diagrams`       | Number of diagrams in the complete ground-truth dataset, before filtering missing predictions.                                                                           |

Supply all four ground-truth directories and `expected_num_diagrams` explicitly.
The constructor's default ground-truth paths and default count of `346` refer
to the original research dataset, which is not included in a package install.

Ground truth must contain matching, same-stem files with aligned `options`
lists. Predictions need node text for the supported benchmark methods. The
constructor checks the prediction structure and exposes supported methods in
`benchmark.capabilities.available_methods`.

### Methods

Methods below return results in sorted diagram-stem order. Calling a method
without the required predicted parts raises `ValueError`.

| Method                                                                                                                                                                                                                    | Required predicted parts      | Return value                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------- |
| [`node_matches(range_indices=None)`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches}                         | Node text                     | List of `NodeMatches`; optional `(start, stop)` slice           |
| [`node_matches_with_flow()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches_with_flow){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches_with_flow}             | Node text and flow            | List of `NodeMatches`, using the shared matching policy         |
| [`node_and_label_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches}             | Node text and labels          | List of `NodeMatches` including label matches                   |
| [`additional_text_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.additional_text_matches){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.additional_text_matches}          | Node text and additional text | List of `TextListMatches` for the selected ground-truth options |
| [`diagram_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches}                                  | All four parts                | List of `DiagramMatch`                                          |
| [`total_node_text_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_node_text_cost){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_node_text_cost}                   | Node text                     | Sum of node-text distances                                      |
| [`total_label_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_label_cost){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_label_cost}                               | Node text and labels          | Sum of label distances                                          |
| [`total_additional_text_cost()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_additional_text_cost){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.total_additional_text_cost} | Node text and additional text | Sum of additional-text distances                                |
| [`all_flow_scores()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_scores){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_scores}                                  | Node text and flow            | List of `FlowScores`                                            |
| [`all_flow_jaccard_scores()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_jaccard_scores){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.all_flow_jaccard_scores}          | Node text and flow            | List of diagram Jaccard scores                                  |
| [`avg_flow_jaccard()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.avg_flow_jaccard){#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.avg_flow_jaccard}                               | Node text and flow            | Arithmetic mean of diagram Jaccard scores                       |

The matching policy uses all available predicted parts, including when calling
[`node_matches()`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches)
. See [how diagrams are matched](../benchmarking/matching.md).

### Matching result objects

These Pydantic result types are defined in
`flowde.benchmarks.parsing.parsing_bench_types` . Use
[`model_dump()`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump)
to inspect
their stored fields or
[`model_dump_json(indent=2)`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump_json)
to serialize those fields.
Computed properties such as costs and `flow_score` can be accessed directly.

| Object            | Useful attributes                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NodeMatch`       | `true_node`, `pred_node`, `node_text_cost`, `label_matches`, `match_type`, `parent_img_code`, `numbers_only_node_text_cost`                                                                |
| `NodeMatches`     | `matches`, `true_diagram_option_idx`, `parent_img_code`, `true_nodes`, `pred_nodes`, `total_node_text_cost`, `total_numbers_only_node_text_cost`, `total_label_error_cost`, `flow_score`   |
| `TextListMatch`   | `true_index`, `pred_index`, `true_text`, `pred_text`, `cost`, `match_type`, `parent_img_code`, `true_option_idx`                                                                           |
| `TextListMatches` | `matches`, `total_cost`                                                                                                                                                                    |
| `DiagramMatch`    | `node_matches`, `additional_text_matches`, `true_diagram`, `pred_diagram`, `total_node_text_cost`, `total_label_error_cost`, `total_additional_text_cost`, `total_text_cost`, `flow_score` |

`NodeMatch.true_node` or `pred_node` can be `None` for an unmatched node. Text
matches use `None` for the absent string and index. `true_diagram_option_idx`
and `true_option_idx` are zero-based option indices. `parent_img_code` is the
diagram's filename stem.

[`NodeMatches.get_matched_true_node(pred_node_number)`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches.get_matched_true_node){#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches.get_matched_true_node}
returns the true node
matched to a predicted node, or `None`. The reverse lookup is
[`get_matched_pred_node(true_node_number)`](benchmarks.md#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches.get_matched_pred_node){#flowde.benchmarks.parsing.parsing_bench_types.NodeMatches.get_matched_pred_node}
.

### Flow scores

| `FlowScores` attribute | Meaning                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `tp`                   | Directed edges present in both diagrams after matching nodes     |
| `fp`                   | Predicted directed edges absent from the selected ground truth   |
| `fn`                   | Ground-truth directed edges missing from the prediction          |
| `precision`            | TP / (TP + FP), or `0` for a zero denominator                    |
| `recall`               | TP / (TP + FN), or `0` for a zero denominator                    |
| `f1`                   | Harmonic mean of precision and recall, or `0` when both are zero |
| `jaccard`              | TP / (TP + FP + FN); `1` when both diagrams have no edges        |
| `missing_edges`        | Set of missing `(source, destination)` node-number pairs         |
| `extra_edges`          | Set of extra directed node-number pairs                          |

Flow comparisons express matched nodes using ground-truth node numbers.
Unmatched predicted nodes receive synthetic identifiers starting at `10000`
when constructing edge comparisons. The [worked example](../benchmarking/parsing.md)
shows how to interpret every score.

## Text distances

Import the following functions from
`flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn`.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_fn

Counts the minimum number of single-character insertions, deletions and
substitutions. Treats a missing string as empty; two missing strings are invalid.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_with_nfc_and_space_normalisation

Normalises Unicode and whitespace/punctuation conventions before counting edits.
The output remains an edit count, not a score scaled between zero and one.

::: flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.number_only_levenshtein

Compares the extracted numbers in the two strings. Use this for a separate
number-focused check, not as a measure of whether the surrounding words agree.
`NodeMatches.total_numbers_only_node_text_cost` provides this check for the
already selected node matches.
