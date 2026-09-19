from dataclasses import dataclass, field

from flowde.benchmarks.classification.classification_benchmark import (
    MulticlassClassificationBenchmark,
)
from flowde.classify_fns.classify_types import RotationLabel


@dataclass(frozen=True, kw_only=True)
class RotationBenchmark(MulticlassClassificationBenchmark[RotationLabel]):
    """
    Compare predicted image rotations with ground-truth rotation labels.

    Labels describe the clockwise correction, in degrees, needed by each
    original input image: `0`, `90`, `180` or `270`. A prediction is correct
    only when the predicted angle equals the ground-truth angle. The benchmark
    reads saved labels without opening or rotating images or making model
    requests.

    Parameters
    ----------
    true_path : Path
        Ground-truth JSON file containing a list of objects with exactly
        `img_path` and `label` keys. Each `img_path` must be a string; each
        `label` specifies the clockwise correction for the original image.
    pred_path : Path
        Prediction JSON file in the same format as `true_path`, such as
        `rotations.json` saved by
        [`rotate_imgs()`][flowde.rotate_imgs.rotate_imgs]. Both lists must
        contain the same number of entries in the same image order. At each
        position, filenames including extensions must match; parent
        directories may differ.

    Attributes
    ----------
    labels : tuple[RotationLabel, ...]
        Fixed class labels `(0, 90, 180, 270)`. The constructor does not accept
        a `labels` argument.
    confusion_matrix : dict[RotationLabel, dict[RotationLabel, int]]
        Counts indexed first by ground-truth angle, then by predicted angle.
        For example, `confusion_matrix[90][0]` counts images requiring a
        90-degree correction that received a prediction of zero degrees.
        Includes all four angles, even when no image has a particular angle.
    num_per_true_class : dict[RotationLabel, int]
        Number of images requiring each ground-truth correction.
    num_per_pred_class : dict[RotationLabel, int]
        Number of images assigned each predicted correction.
    per_class_accuracy : dict[RotationLabel, float]
        For each ground-truth angle, correct predictions divided by images
        requiring that angle. Returns `0.0` when no image requires the angle.

    Raises
    ------
    OSError
        If either JSON file cannot be opened.
    TypeError
        If a JSON document is not a list, an entry is not an object, an
        `img_path` is not a string, or a saved label has an unsupported type.
    ValueError
        If either file contains invalid JSON, the lists have different
        lengths, an entry has missing or extra keys, paired filenames differ,
        or a saved label is outside the supported rotation classes.

    Notes
    -----
    Pass `true_path` and `pred_path` by keyword. Result tuples retain the JSON
    list order. Neither input file is changed. Two empty JSON lists produce
    zero counts and accuracies. Incorrect angles receive no partial credit
    based on how close the predicted angle is to the ground-truth angle.

    The benchmark also exposes `accuracy`, `result_list`, `img_paths`,
    `trues`, `preds`, `correct_predictions` and `incorrect_predictions`;
    `len(benchmark)` counts paired images. See the shared properties on
    [`ClassificationBenchmark`][flowde.benchmarks.classification.classification_benchmark.ClassificationBenchmark].

    Examples
    --------
    Using existing ground-truth and prediction JSON files:

    >>> from pathlib import Path
    >>> from flowde.benchmarks.rotation.rotation_benchmark import RotationBenchmark
    >>> benchmark = RotationBenchmark(
    ...     true_path=Path("data/true-rotations.json"),
    ...     pred_path=Path("results/rotation/rotations.json"),
    ... )
    >>> print(benchmark.accuracy)
    >>> print(benchmark.confusion_matrix)

    """

    labels: tuple[RotationLabel, ...] = field(
        default=(0, 90, 180, 270),
        init=False,
    )
