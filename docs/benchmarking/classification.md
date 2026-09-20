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

Classification saves predictions automatically to
`save_dir / "classifications.json"`. Pass that file to the benchmark as
`pred_path`. The benchmark reads saved JSON and makes no model requests.

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

You can use
[`BinaryClassificationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_benchmark.BinaryClassificationBenchmark)
to benchmark a binary classification task:

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

See the
[`BinaryClassificationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_benchmark.BinaryClassificationBenchmark)
API reference for full details of the parameters and results.

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
`tp`, `fp`, `fn`, and `tn` is a
[`SingleClassificationResult`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_bench_types.SingleClassificationResult)
with:

| Attribute  | Description                   |
| ---------- | ----------------------------- |
| `img_path` | Path to the classified image. |
| `true`     | Ground-truth label.           |
| `pred`     | Predicted label.              |

## Multiclass classification benchmark

For classifications where each image belongs to one of more than two possible
classes, you can use
[`MulticlassClassificationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_benchmark.MulticlassClassificationBenchmark):

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

The benchmark loads both JSON files, checks that they are valid, and compares
the predicted labels with the true labels.

The `labels` tuple declares the possible classes and must include every label
in the ground-truth and prediction files. The benchmark includes every declared
class in its counts and confusion matrix, even when no image has that label.

See the
[`MulticlassClassificationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_benchmark.MulticlassClassificationBenchmark)
API reference for full details of the parameters and results.

### Multiclass benchmark results

After creating the benchmark object, you can inspect the results through its
properties.

| Property                          | Description                                                                 |
| --------------------------------- | --------------------------------------------------------------------------- |
| `len(benchmark)`                  | Number of classified images in the benchmark.                               |
| `benchmark.result_list`           | All classification results, one per image.                                  |
| `benchmark.correct_predictions`   | Results where `pred` matches `true`.                                        |
| `benchmark.incorrect_predictions` | Results where `pred` does not match `true`.                                 |
| `benchmark.accuracy`              | Proportion of images where the predicted label matches the true label.      |
| `benchmark.img_paths`             | Image paths used in the benchmark.                                          |
| `benchmark.trues`                 | True labels, in benchmark order.                                            |
| `benchmark.preds`                 | Predicted labels, in benchmark order.                                       |
| `benchmark.labels`                | Class labels supplied through `labels`.                                     |
| `benchmark.confusion_matrix`      | Number of images for each true-label and predicted-label pair.              |
| `benchmark.num_per_true_class`    | Number of images with each ground-truth label.                              |
| `benchmark.num_per_pred_class`    | Number of images assigned each predicted label.                             |
| `benchmark.per_class_accuracy`    | For each ground-truth class, the proportion of images classified correctly. |

For example, you can print the overall accuracy and class-level summaries:

```python
print(f"Accuracy: {benchmark.accuracy:.3f}")
print(benchmark.confusion_matrix)
print(benchmark.num_per_true_class)
print(benchmark.num_per_pred_class)
print(benchmark.per_class_accuracy)
```

The confusion matrix is a nested dictionary indexed by true label, then
predicted label. For example,
`benchmark.confusion_matrix["consort_flowchart"]["table"]` counts CONSORT
flowcharts incorrectly classified as tables.

You can also inspect the individual incorrect predictions:

```python
for result in benchmark.incorrect_predictions:
    print(result.img_path)
    print(f"true: {result.true}")
    print(f"pred: {result.pred}")
```

Each item in `result_list`, `correct_predictions`, and `incorrect_predictions`
is a
[`SingleClassificationResult`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_bench_types.SingleClassificationResult)
with:

| Attribute  | Description                   |
| ---------- | ----------------------------- |
| `img_path` | Path to the classified image. |
| `true`     | Ground-truth label.           |
| `pred`     | Predicted label.              |

See the
[`MulticlassClassificationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.classification.classification_benchmark.MulticlassClassificationBenchmark)
API reference for full details of the parameters and results.
