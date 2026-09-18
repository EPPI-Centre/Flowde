# Parsing benchmarking

The parsing benchmark compares saved diagram predictions with annotated
ground truth. It evaluates node text, labels, connections and additional text,
depending on which parts you predicted.

Predicted and ground-truth node numbers need not identify the same visual
nodes. The benchmark first finds [Node Matches](matching.md), then uses those
matches to score text and connections.

## Prepare the input directories

For one complete prediction, the files might be:

```text
data/ground-truth/
├── nodes/
│   └── example_0.json
├── labels/
│   └── example_0.json
├── flow/
│   └── example_0.json
└── additional_texts/
    └── example_0.json
results/predictions/
└── example_0.json
```

The same filename stem identifies the diagram in every directory. Ground truth
always supplies all four components, even when you evaluate a partial
prediction. Use empty lists for genuinely absent labels or additional text.

### Ground-truth node text

Save this as `data/ground-truth/nodes/example_0.json`:

```json
{
  "options": [
    {
      "nodes": [
        { "node_number": 1, "text": "Screened (n = 10)" },
        { "node_number": 2, "text": "Included (n = 8)" },
        { "node_number": 3, "text": "Analysed (n = 7)" }
      ]
    }
  ]
}
```

### Ground-truth labels

Save this as `data/ground-truth/labels/example_0.json`:

```json
{
  "options": [
    {
      "nodes": [
        { "node_number": 1, "labels": ["Enrollment"] },
        { "node_number": 2, "labels": [] },
        { "node_number": 3, "labels": [] }
      ]
    }
  ]
}
```

### Ground-truth flow

Save this as `data/ground-truth/flow/example_0.json`:

```json
{
  "options": [
    {
      "nodes": [
        { "node_number": 1, "points_to": [2] },
        { "node_number": 2, "points_to": [3] },
        { "node_number": 3, "points_to": [] }
      ]
    }
  ]
}
```

### Ground-truth additional text

Save this as `data/ground-truth/additional_texts/example_0.json`:

```json
{ "options": [{ "additional_texts": ["Figure 1"] }] }
```

### Ground-Truth Options

Each `options` entry is one accepted interpretation. All four files for a
diagram must have the same number of options. Entries at the same index belong
together:

| Option index | Node-text file           | Label file             | Flow file                   | Additional-text file               |
| ------------ | ------------------------ | ---------------------- | --------------------------- | ---------------------------------- |
| `0`          | Interpretation 0's nodes | Labels for those nodes | Connections for those nodes | Interpretation 0's additional text |
| `1`          | Interpretation 1's nodes | Labels for those nodes | Connections for those nodes | Interpretation 1's additional text |

Within each option, the node numbers in `nodes`, `labels` and `flow` must match.
Use consecutive node numbers starting at `1`. Every connection must refer to
an existing node. The benchmark also validates the diagram structure; for
example, a multi-node diagram's flow cannot leave a node completely isolated.

The [parsing examples](../pipeline/parsing.md#understand-the-flowchart-format)
illustrate why splitting a shared visual box or joining labels may produce
different accepted interpretations. Annotate each option as a complete,
internally consistent interpretation. Do not independently reorder the options
in the four files.

## Prediction format

Predictions have the ordinary inference format, without an `options` wrapper.
For this worked example, save the following prediction as
`results/predictions/example_0.json`:

```json
{
  "nodes": [
    {
      "node_number": 1,
      "text": "Screened (n = 11)",
      "labels": ["Enrolment"],
      "points_to": [2, 3]
    },
    {
      "node_number": 2,
      "text": "Included (n = 8)",
      "labels": [],
      "points_to": []
    },
    {
      "node_number": 3,
      "text": "Analysed (n = 7)",
      "labels": [],
      "points_to": []
    }
  ],
  "additional_texts": ["Figure 2"]
}
```

Prediction files in one directory must have consistent component structures.
For example, do not mix full parses with node-text-only parses in the same
directory. Node text is required for all parsing benchmark operations.

## Create the benchmark

```python
from pathlib import Path

from flowde.benchmarks.parsing.parsing_bench import ParsingBenchmark
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import levenshtein_fn

truth_dir = Path("data/ground-truth")

benchmark = ParsingBenchmark(
    pred_diagrams_dir=Path("results/predictions"),
    distance_fn=levenshtein_fn,
    true_nodes_dir=truth_dir / "nodes",
    true_labels_dir=truth_dir / "labels",
    true_flow_dir=truth_dir / "flow",
    true_additional_texts_dir=truth_dir / "additional_texts",
    expected_num_diagrams=1,
)
```

Always provide the ground-truth directories and the expected number of
diagrams for your dataset. The constructor's historical defaults refer to a
particular research dataset and expect 346 diagrams.

Creating the benchmark validates the files. Scoring methods then perform the
matching and calculations locally.

## Read text costs

```python
print(benchmark.total_node_text_cost())       # 1
print(benchmark.total_label_cost())           # 1
print(benchmark.total_additional_text_cost()) # 1
```

This example uses raw Levenshtein distance: the number of character insertions,
deletions or substitutions required to make two strings equal.

- Node text: `10` became `11`, so one substitution is needed.
- Labels: `Enrollment` became `Enrolment`, so one character differs by deletion.
- Additional text: `Figure 1` became `Figure 2`, so one substitution is needed.

Each total sums the matched and unmatched text costs across the evaluated
diagrams. Lower is better; zero means no differences under the selected
distance function. These totals are not percentages and are not divided by the
number of characters or diagrams.

### Choose a distance function

| Function                                                                                                                                                                             | Comparison                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| [`levenshtein_fn`](../reference/benchmarks.md#flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_fn)                                                             | Raw character edit distance.                                                           |
| [`levenshtein_with_nfc_and_space_normalisation`](../reference/benchmarks.md#flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_with_nfc_and_space_normalisation) | Edit distance after normalising Unicode, whitespace and selected punctuation variants. |
| [`number_only_levenshtein`](../reference/benchmarks.md#flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.number_only_levenshtein)                                           | Edit distance between the extracted sequences of numbers.                              |

The normalised-text function still returns an edit count; it does not divide
by text length. For example, `n=10` and `n = 10` have raw distance `2` and
normalised-text distance `0`.

Numbers-only comparison turns `Screened (n = 10)` and `Included (n = 10)` into
the same number sequence, giving cost `0`. Use that distance only when this is
the distinction you intend to measure. Changing the distance can also change
which nodes and Ground-Truth Options are selected.

An unmatched string is compared with an absent string. The supplied distance
functions treat absence as an empty string, so an unmatched `ABC` has raw
cost `3`.

## Read flow scores

```python
score = benchmark.all_flow_scores()[0]
print(score.tp, score.fp, score.fn)  # 1 1 1
print(score.precision)              # 0.5
print(score.recall)                 # 0.5
print(score.f1)                     # 0.5
print(score.jaccard)                # 0.3333333333333333
print(score.missing_edges)          # {(2, 3)}
print(score.extra_edges)            # {(1, 3)}
```

After matching the nodes, the ground truth has edges `1 → 2` and `2 → 3`.
The prediction has edges `1 → 2` and `1 → 3`. There is one correct edge, one
extra edge and one missing edge.

| Metric      | Calculation in this example                         |
| ----------- | --------------------------------------------------- |
| `tp`        | Number of edges present in both diagrams: `1`.      |
| `fp`        | Predicted edges absent from the ground truth: `1`.  |
| `fn`        | Ground-truth edges absent from the prediction: `1`. |
| `precision` | `TP / (TP + FP) = 1 / 2`.                           |
| `recall`    | `TP / (TP + FN) = 1 / 2`.                           |
| `f1`        | Harmonic mean of precision and recall: `0.5`.       |
| `jaccard`   | `TP / (TP + FP + FN) = 1 / 3`.                      |

Edges are directed: `1 → 2` differs from `2 → 1`. Flow scores use sets of
edges, so repeating the same connection does not add another correct edge.
Edges attached to unmatched predicted nodes count as extra edges. Reported
edge numbers use the ground-truth numbering after matching; unmatched
endpoints can be represented by synthetic numbers.

When both diagrams have no edges, Jaccard is `1.0`. Precision, recall and F1
return `0.0` for their zero-denominator cases.

### Average across diagrams

```python
print(benchmark.all_flow_jaccard_scores())
print(benchmark.avg_flow_jaccard())
```

[`avg_flow_jaccard()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.avg_flow_jaccard)
gives each evaluated diagram equal weight. If two diagrams
have Jaccard scores `1.0` and `0.5`, the average is `0.75`, regardless of their
different numbers of edges. It is not calculated by pooling all edges first.

## Inspect individual matches

```python
for matches in benchmark.node_matches():
    print(matches.parent_img_code)
    print("Selected option:", matches.true_diagram_option_idx)
    print("Text cost:", matches.total_node_text_cost)
    print("Numbers-only cost:", matches.total_numbers_only_node_text_cost)
    for match in matches.matches:
        print(match.true_node, match.pred_node, match.node_text_cost)
```

`true_diagram_option_idx` is zero-based. `true_node` or `pred_node` is `None`
for an unmatched node. The numbers-only cost here evaluates the already chosen
Node Matches; it does not rematch nodes using a different distance.

Use
[`node_and_label_matches()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches)
to include each node's `label_matches` .
[`additional_text_matches()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.additional_text_matches)
returns the matched and unmatched additional-text
strings and their costs. For complete predictions,
[`diagram_matches()`](../reference/benchmarks.md#flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.diagram_matches)
returns
both kinds of matches and the selected true and predicted diagrams. Its
`total_text_cost` sums node, label and additional-text costs; flow is reported
separately.

## Benchmark separately parsed parts

Instead of combining JSON files first, pass a list of prediction directories:

```python
benchmark = ParsingBenchmark(
    pred_diagrams_dir=[
        Path("results/parsing/node_text"),
        Path("results/parsing/labels"),
        Path("results/parsing/flow"),
        Path("results/parsing/additional_texts"),
    ],
    distance_fn=levenshtein_fn,
    true_nodes_dir=truth_dir / "nodes",
    true_labels_dir=truth_dir / "labels",
    true_flow_dir=truth_dir / "flow",
    true_additional_texts_dir=truth_dir / "additional_texts",
    expected_num_diagrams=1,
)
```

Every supplied prediction directory must contain the same diagram stems.
The component fields must not overlap across directories. For example, do not
supply both a complete parse directory and a node-text directory for the same
diagrams. Node-based components are joined using the predicted node numbers.

You can supply fewer prediction components. Check available operations with:

```python
print(benchmark.capabilities.available_methods)
```

| Predicted components        | Available evaluation                            |
| --------------------------- | ----------------------------------------------- |
| Node text                   | Node Matches and node-text costs                |
| Node text + labels          | Also label matching and label costs             |
| Node text + flow            | Also flow scores                                |
| Node text + additional text | Also additional-text matching and costs         |
| All four                    | All methods, including complete diagram matches |

Calling an operation that needs a missing component raises an explanatory
error. Every operation uses the same option-selection rules and all available
predicted evidence; the method name does not request a different matching
policy.

## Missing predictions

By default, predictions must cover the expected ground-truth diagrams. With
`allow_missing_pred_diagrams=True`, Flowde evaluates only the diagrams that have
predictions. `expected_num_diagrams` still describes the complete ground-truth
dataset.

Missing predictions are excluded, not counted as wrong. Report the number of
evaluated diagrams and the missing stems alongside scores. Predictions with
no corresponding ground truth still need to be separated from the benchmark
inputs, as the [CONSORT recipe](../recipes/consort.md) demonstrates.

See [matching rules](matching.md) for how ties are resolved and the
[benchmark reference](../reference/benchmarks.md#parsing) for all public methods.
