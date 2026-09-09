# Classification benchmarking

Classification benchmarking evaluates how well a classification function has
performed against a set of ground-truth labels.

In a typical workflow, you will first run
[classification](../pipeline/classification.md#image-classification) and save
the predictions to JSON. You can then compare those predictions against a
separate JSON file containing the true labels.

## Input format

Classification benchmarks compare two JSON files:

- a ground-truth file containing the true labels;
- a prediction file containing the predicted labels.

The prediction file can be created automatically by passing `json_path` to
`classify_imgs` or `classify_imgs_from_paths`.

Both files should contain a list of objects with the following keys:

```json
[
  {
    "img_path": "data/extracted-images/paper-1_0.png",
    "label": 1
  },
  {
    "img_path": "data/extracted-images/paper-1_1.png",
    "label": 0
  }
]
```

The benchmark expects the true and predicted files to contain the same images in
the same order. The parent directories do not have to match, but the image
filenames must match.

For example, this is valid:

```text
true: data/ground-truth/paper-1_0.png
pred: data/predictions/paper-1_0.png
```

This is not valid:

```text
true: data/ground-truth/paper-1_0.png
pred: data/predictions/paper-2_0.png
```

## Binary classification benchmark

You will most likely want to benchmark a binary classification task; meaning
your classification function has outputted the following labels:

```text
0 = negative
1 = positive
```

For example, you most likely want to evaluate whether your classification
function is correctly classifying images as flowcharts.

Use `BinaryClassificationBenchmark` for this case:

```python
from pathlib import Path

from flowde.benchmarks.classification.classification_benchmark import (
    BinaryClassificationBenchmark,
)

benchmark = BinaryClassificationBenchmark(
    true_path=Path("data/true-classifications.json"),
    pred_path=Path("data/pred-classifications.json"),
)
```

The benchmark loads both JSON files, checks that they are valid, and compares
the predicted labels with the true labels.

### Binary benchmark results

After creating the benchmark object, you can inspect the results through its
properties.

| Property                          | Description                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `len(benchmark)`                  | Number of classified images in the benchmark.                                     |
| `benchmark.result_list`           | All classification results, one per image.                                        |
| `benchmark.correct_predictions`   | Results where `pred` matches `true`.                                              |
| `benchmark.incorrect_predictions` | Results where `pred` does not match `true`.                                       |
| `benchmark.accuracy`              | Proportion of images where the predicted label matches the true label.            |
| `benchmark.img_paths`             | Image paths used in the benchmark.                                                |
| `benchmark.trues`                 | True labels, in benchmark order.                                                  |
| `benchmark.preds`                 | Predicted labels, in benchmark order.                                             |
| `benchmark.tp`                    | True positives: predicted `1`, true label `1`.                                    |
| `benchmark.fp`                    | False positives: predicted `1`, true label `0`.                                   |
| `benchmark.fn`                    | False negatives: predicted `0`, true label `1`.                                   |
| `benchmark.tn`                    | True negatives: predicted `0`, true label `0`.                                    |
| `benchmark.num_tp`                | Number of true positives.                                                         |
| `benchmark.num_fp`                | Number of false positives.                                                        |
| `benchmark.num_fn`                | Number of false negatives.                                                        |
| `benchmark.num_tn`                | Number of true negatives.                                                         |
| `benchmark.precision`             | Of the images predicted positive, the proportion that were truly positive.        |
| `benchmark.recall`                | Of the truly positive images, the proportion predicted positive.                  |
| `benchmark.f1_score`              | Harmonic mean of precision and recall.                                            |
| `benchmark.tpr`                   | True positive rate. This is the same as `recall`.                                 |
| `benchmark.fpr`                   | False positive rate: proportion of true negatives incorrectly predicted positive. |
| `benchmark.specificity`           | True negative rate: proportion of true negatives correctly predicted negative.    |
| `benchmark.tnr`                   | True negative rate. This is the same as `specificity`.                            |

For example, you can print the main benchmark metrics:

```python
print(f"Accuracy: {benchmark.accuracy:.3f}")
print(f"Precision: {benchmark.precision:.3f}")
print(f"Recall: {benchmark.recall:.3f}")
print(f"F1 score: {benchmark.f1_score:.3f}")
```

You can also inspect the individual incorrect predictions:

```python
for result in benchmark.incorrect_predictions:
    print(result.img_path)
    print(f"true: {result.true}")
    print(f"pred: {result.pred}")
```

Each item in `result_list`, `correct_predictions`, `incorrect_predictions`,
`tp`, `fp`, `fn`, and `tn` is a `SingleClassificationResult` with:

| Attribute  | Description                   |
| ---------- | ----------------------------- |
| `img_path` | Path to the classified image. |
| `true`     | Ground-truth label.           |
| `pred`     | Predicted label.              |

## Multiclass classification benchmark

Use `MulticlassClassificationBenchmark` when each image belongs to exactly one
of more than two possible classes.

For example, you might classify extracted images as:

```text
consort_flowchart
other_flowchart
table
graph
other
```

In this case, you need to pass the full set of possible labels to the benchmark.

```python
from pathlib import Path

from flowde.benchmarks.classification.classification_benchmark import (
    MulticlassClassificationBenchmark,
)

benchmark = MulticlassClassificationBenchmark(
    true_path=Path("data/true-classifications.json"),
    pred_path=Path("data/pred-classifications.json"),
    labels=(
        "consort_flowchart",
        "other_flowchart",
        "table",
        "graph",
        "other",
    ),
)
```

Passing `labels` ensures that all possible classes are included in the
benchmark, even if one of the classes does not appear in a particular evaluation
set.

### Multiclass benchmark results

After creating the benchmark object, you can inspect the results through its
properties.

| Property                          | Description                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------ |
| `len(benchmark)`                  | Number of classified images in the benchmark.                                  |
| `benchmark.result_list`           | All classification results, one per image.                                     |
| `benchmark.correct_predictions`   | Results where `pred` matches `true`.                                           |
| `benchmark.incorrect_predictions` | Results where `pred` does not match `true`.                                    |
| `benchmark.accuracy`              | Proportion of images where the predicted label matches the true label.         |
| `benchmark.img_paths`             | Image paths used in the benchmark.                                             |
| `benchmark.trues`                 | True labels, in benchmark order.                                               |
| `benchmark.preds`                 | Predicted labels, in benchmark order.                                          |
| `benchmark.labels`                | Full set of possible labels passed to the benchmark.                           |
| `benchmark.confusion_matrix`      | Nested dictionary where rows are true labels and columns are predicted labels. |
| `benchmark.num_per_true_class`    | Number of examples for each true class.                                        |
| `benchmark.num_per_pred_class`    | Number of predictions made for each class.                                     |
| `benchmark.per_class_accuracy`    | For each true class, the proportion of examples classified correctly.          |

For example, you can print the overall accuracy and class-level summaries:

```python
print(f"Accuracy: {benchmark.accuracy:.3f}")
print(benchmark.num_per_true_class)
print(benchmark.num_per_pred_class)
print(benchmark.per_class_accuracy)
```

The confusion matrix shows which classes are being confused with each other:

```python
print(benchmark.confusion_matrix)
```

This returns a nested dictionary where the outer keys are true labels and the
inner keys are predicted labels:

```python
{
    "consort_flowchart": {
        "consort_flowchart": 10,
        "other_flowchart": 2,
        "table": 0,
        "graph": 0,
        "other": 1,
    },
    "other_flowchart": {
        "consort_flowchart": 1,
        "other_flowchart": 8,
        "table": 0,
        "graph": 0,
        "other": 0,
    },
}
```

You can also inspect the individual incorrect predictions:

```python
for result in benchmark.incorrect_predictions:
    print(result.img_path)
    print(f"true: {result.true}")
    print(f"pred: {result.pred}")
```

Each item in `result_list`, `correct_predictions`, and `incorrect_predictions`
is a `SingleClassificationResult` with:

| Attribute  | Description                   |
| ---------- | ----------------------------- |
| `img_path` | Path to the classified image. |
| `true`     | Ground-truth label.           |
| `pred`     | Predicted label.              |
