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

The prediction file can be created automatically by passing `json_path` to
`rotate_imgs` or `rotate_imgs_from_paths`.

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

The `label` should be the clockwise rotation needed to correct the image
orientation. Rotation labels must be one of:

```text
0
90
180
270
```

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

## Run the rotation benchmark

Use `RotationBenchmark` to evaluate predicted rotation labels:

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

## Results

After creating the benchmark object, you can inspect the results through its
properties.

| Property                          | Description                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| `len(benchmark)`                  | Number of images in the benchmark.                                                   |
| `benchmark.result_list`           | All rotation results, one per image.                                                 |
| `benchmark.correct_predictions`   | Results where `pred` matches `true`.                                                 |
| `benchmark.incorrect_predictions` | Results where `pred` does not match `true`.                                          |
| `benchmark.accuracy`              | Proportion of images where the predicted rotation matches the true rotation.         |
| `benchmark.img_paths`             | Image paths used in the benchmark.                                                   |
| `benchmark.trues`                 | True rotation labels, in benchmark order.                                            |
| `benchmark.preds`                 | Predicted rotation labels, in benchmark order.                                       |
| `benchmark.labels`                | The supported rotation labels: `0`, `90`, `180`, and `270`.                          |
| `benchmark.confusion_matrix`      | Nested dictionary where rows are true rotations and columns are predicted rotations. |
| `benchmark.num_per_true_class`    | Number of examples for each true rotation.                                           |
| `benchmark.num_per_pred_class`    | Number of predictions made for each rotation.                                        |
| `benchmark.per_class_accuracy`    | For each true rotation, the proportion of examples classified correctly.             |

For example, you can print the overall accuracy and class-level summaries:

```python
print(f"Accuracy: {benchmark.accuracy:.3f}")
print(benchmark.num_per_true_class)
print(benchmark.num_per_pred_class)
print(benchmark.per_class_accuracy)
```

The confusion matrix shows which rotations are being confused with each other:

```python
print(benchmark.confusion_matrix)
```

This returns a nested dictionary where the outer keys are true rotation labels
and the inner keys are predicted rotation labels:

```python
{
    0: {
        0: 20,
        90: 1,
        180: 0,
        270: 0,
    },
    90: {
        0: 2,
        90: 18,
        180: 0,
        270: 1,
    },
    180: {
        0: 0,
        90: 0,
        180: 15,
        270: 2,
    },
    270: {
        0: 1,
        90: 0,
        180: 3,
        270: 17,
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

| Attribute  | Description                  |
| ---------- | ---------------------------- |
| `img_path` | Path to the rotated image.   |
| `true`     | Ground-truth rotation label. |
| `pred`     | Predicted rotation label.    |
