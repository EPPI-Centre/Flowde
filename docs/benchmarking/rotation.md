# Rotation benchmarking

Rotation benchmarking evaluates how well a rotation classification function has
performed against a set of ground-truth rotation labels.

In a typical workflow, you will first run
[image rotation](../pipeline/rotation.md#image-rotation) and save the predicted
rotations to JSON. You can then compare those predictions against a separate
JSON file containing the true rotations.

## Input format

Rotation benchmarks compare two JSON files:

- a ground-truth file containing the true rotation labels;
- a prediction file containing the predicted rotation labels.

Rotation saves predictions automatically to `save_dir / "rotations.json"`.
Pass that file to the benchmark as `pred_path`. The benchmark reads saved JSON
and makes no model requests.

Both files should contain a list of objects with the following keys:

```json
[
  {
    "img_path": "data/flowchart-images/paper-1_0.png",
    "label": 0
  },
  {
    "img_path": "data/flowchart-images/paper-1_1.png",
    "label": 90
  }
]
```

The `label` is the clockwise rotation, in degrees, needed to correct the image
orientation: `0`, `90`, `180` or `270`. Ground-truth labels describe the
corrections needed by the original input images.

The benchmark expects the true and predicted files to contain the same images in
the same order. The parent directories do not have to match, but the image
filenames must match.

For example, this is valid:

```text
true: data/ground-truth-rotations/paper-1_0.png
pred: data/predicted-rotations/paper-1_0.png
```

This is not valid:

```text
true: data/ground-truth-rotations/paper-1_0.png
pred: data/predicted-rotations/paper-2_0.png
```

## Rotation benchmark

You can use
[`RotationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.rotation.rotation_benchmark.RotationBenchmark)
to evaluate predicted rotation labels:

```python
from pathlib import Path

from flowde.benchmarks.rotation.rotation_benchmark import RotationBenchmark

benchmark = RotationBenchmark(
    true_path=Path("data/true-rotations.json"),
    pred_path=Path("data/pred-rotations.json"),
)
```

The benchmark loads both JSON files, checks that they are valid, and compares
the predicted rotation labels with the true rotation labels.

The benchmark uses the labels `(0, 90, 180, 270)` automatically and includes
every angle in its counts and confusion matrix, even when no image has that
label. A prediction counts as correct only when the predicted angle exactly
matches the ground-truth angle.

### Rotation benchmark results

After creating the benchmark object, you can inspect the results through its
properties.

| Property                          | Description                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `len(benchmark)`                  | Number of images in the benchmark.                                                                                        |
| `benchmark.result_list`           | All rotation results, one per image.                                                                                      |
| `benchmark.correct_predictions`   | Results where `pred` matches `true`.                                                                                      |
| `benchmark.incorrect_predictions` | Results where `pred` does not match `true`.                                                                               |
| `benchmark.accuracy`              | Proportion of images where the predicted angle matches the ground-truth angle.                                            |
| `benchmark.img_paths`             | Image paths used in the benchmark.                                                                                        |
| `benchmark.trues`                 | Ground-truth angles, in benchmark order.                                                                                  |
| `benchmark.preds`                 | Predicted angles, in benchmark order.                                                                                     |
| `benchmark.labels`                | The supported rotation labels: `(0, 90, 180, 270)`.                                                                       |
| `benchmark.confusion_matrix`      | Number of images for each ground-truth angle and predicted angle pair.                                                    |
| `benchmark.num_per_true_class`    | Number of images requiring each ground-truth angle.                                                                       |
| `benchmark.num_per_pred_class`    | Number of images assigned each predicted angle.                                                                           |
| `benchmark.per_class_accuracy`    | For each ground-truth angle, the proportion of images with a correct prediction; `0.0` when no image requires that angle. |

For example, you can print the overall accuracy and class-level summaries:

```python
print(f"Accuracy: {benchmark.accuracy:.3f}")
print(benchmark.confusion_matrix)
print(benchmark.num_per_true_class)
print(benchmark.num_per_pred_class)
print(benchmark.per_class_accuracy)
```

The confusion matrix is a nested dictionary indexed by ground-truth angle,
then predicted angle. For example, `benchmark.confusion_matrix[90][0]` counts
images that needed a 90-degree clockwise correction but received a prediction
of `0`.

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

| Attribute  | Description                                    |
| ---------- | ---------------------------------------------- |
| `img_path` | Image path from the ground-truth file.         |
| `true`     | Ground-truth clockwise correction, in degrees. |
| `pred`     | Predicted clockwise correction, in degrees.    |

See the
[`RotationBenchmark`](../reference/benchmarks.md#flowde.benchmarks.rotation.rotation_benchmark.RotationBenchmark)
API reference for full details of the parameters and results.
