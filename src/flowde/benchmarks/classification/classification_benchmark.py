from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic

from flowde.benchmarks.classification.classification_bench_types import (
    SingleClassificationResult,
)
from flowde.classify_fns.classify_types import ClassificationLabel, LabelType
from flowde.utils import load_json


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationBenchmark(Generic[LabelType]):
    """
    Compare saved image labels with ground truth and report accuracy.

    The benchmark reads JSON files without opening images or making model
    requests. Binary, multiclass and rotation benchmarks inherit the shared
    result properties documented here.

    Parameters
    ----------
    true_path : Path
        Ground-truth JSON file containing a list of objects with exactly
        `img_path` and `label` keys. Image paths must be strings; labels must
        be `str`, `int` or `bool` values.
    pred_path : Path
        Prediction JSON file with the same format and number of entries as
        `true_path`. Entries are paired by list position. Paired image
        filenames, including extensions, must match; parent directories may
        differ. The benchmark does not sort or reorder either list.

    Attributes
    ----------
    result_list : tuple[SingleClassificationResult[LabelType], ...]
        One record per paired image, in the JSON list order. Each record
        contains `img_path` from ground truth, `true` and `pred`.
    correct_predictions : tuple[SingleClassificationResult[LabelType], ...]
        Records whose predicted labels equal their ground-truth labels.
    incorrect_predictions : tuple[SingleClassificationResult[LabelType], ...]
        Records whose predicted labels differ from their ground-truth labels.
    accuracy : float
        Number of correct predictions divided by the number of paired images.
        Returns `0.0` for two empty JSON lists.
    img_paths : tuple[str, ...]
        Image paths from the ground-truth JSON, in the JSON list order.
    trues : tuple[LabelType, ...]
        Ground-truth labels, in the JSON list order.
    preds : tuple[LabelType, ...]
        Predicted labels, in the JSON list order.

    Raises
    ------
    OSError
        If either JSON file cannot be opened.
    TypeError
        If a JSON document is not a list, an entry is not an object, an
        `img_path` is not a string, or a label has an unsupported type.
    ValueError
        If either file contains invalid JSON, the lists have different
        lengths, an entry has missing or extra keys, or paired filenames
        differ.

    Notes
    -----
    Pass constructor arguments by keyword. `len(benchmark)` gives the number
    of paired images. Construction loads the saved labels into memory and
    leaves both source files unchanged. Subsequent edits to the source files
    do not update an existing benchmark object.

    """

    true_path: Path
    pred_path: Path
    result_list: tuple[SingleClassificationResult[LabelType], ...] = field(
        init=False,
    )

    def __post_init__(self) -> None:
        result_list = load_classification_benchmark_results(
            true_path=self.true_path,
            pred_path=self.pred_path,
        )
        object.__setattr__(self, "result_list", result_list)

    def __len__(self) -> int:
        return len(self.result_list)

    @property
    def correct_predictions(
        self,
    ) -> tuple[SingleClassificationResult[LabelType], ...]:
        return tuple(r for r in self.result_list if r.pred == r.true)

    @property
    def incorrect_predictions(
        self,
    ) -> tuple[SingleClassificationResult[LabelType], ...]:
        return tuple(r for r in self.result_list if r.pred != r.true)

    @property
    def accuracy(self) -> float:
        if len(self.result_list) == 0:
            return 0.0
        return len(self.correct_predictions) / len(self.result_list)

    @property
    def img_paths(self) -> tuple[str, ...]:
        return tuple(r.img_path for r in self.result_list)

    @property
    def preds(self) -> tuple[LabelType, ...]:
        return tuple(r.pred for r in self.result_list)

    @property
    def trues(self) -> tuple[LabelType, ...]:
        return tuple(r.true for r in self.result_list)


@dataclass(frozen=True, kw_only=True)
class BinaryClassificationBenchmark(ClassificationBenchmark[int]):
    """
    Evaluate binary image classification using saved labels `0` and `1`.

    The benchmark treats `1` as positive and `0` as negative. Construction
    loads and validates the saved labels; metric properties compare those
    labels without opening images or making model requests.

    Parameters
    ----------
    true_path : Path
        Ground-truth JSON file containing a list of objects with exactly
        `img_path` and `label` keys. Each `img_path` must be a string and each
        `label` must be the integer `0` or `1`. Boolean labels are rejected.
    pred_path : Path
        Prediction JSON file in the same format as `true_path`, such as
        `classifications.json` saved by
        [`classify_imgs()`][flowde.classify_imgs.classify_imgs]. Both lists must
        contain the same number of entries in the same image order. At each
        position, filenames including extensions must match; parent
        directories may differ.

    Attributes
    ----------
    tp : tuple[SingleClassificationResult[int], ...]
        True-positive records: ground-truth label `1`, predicted label `1`.
    fp : tuple[SingleClassificationResult[int], ...]
        False-positive records: ground-truth label `0`, predicted label `1`.
    fn : tuple[SingleClassificationResult[int], ...]
        False-negative records: ground-truth label `1`, predicted label `0`.
    tn : tuple[SingleClassificationResult[int], ...]
        True-negative records: ground-truth label `0`, predicted label `0`.
    num_tp : int
        Number of true-positive records.
    num_fp : int
        Number of false-positive records.
    num_fn : int
        Number of false-negative records.
    num_tn : int
        Number of true-negative records.
    precision : float
        Proportion of predicted positives that are correct:
        `num_tp / (num_tp + num_fp)`.
    recall : float
        Proportion of ground-truth positives found:
        `num_tp / (num_tp + num_fn)`.
    f1_score : float
        Harmonic mean of precision and recall:
        `2 * precision * recall / (precision + recall)`.
    tpr : float
        True-positive rate; equal to `recall`.
    fpr : float
        Proportion of ground-truth negatives incorrectly predicted positive:
        `num_fp / (num_fp + num_tn)`.
    specificity : float
        Proportion of ground-truth negatives correctly predicted negative:
        `num_tn / (num_tn + num_fp)`.
    tnr : float
        True-negative rate; equal to `specificity`.

    Raises
    ------
    OSError
        If either JSON file cannot be opened.
    TypeError
        If a JSON document is not a list, an entry is not an object, an
        `img_path` is not a string, or a label is not an integer.
    ValueError
        If either file contains invalid JSON, the lists have different
        lengths, an entry has missing or extra keys, paired filenames differ,
        or a label is an integer other than `0` or `1`.

    Notes
    -----
    Pass constructor arguments by keyword. All metric properties return
    `0.0` when their denominator is zero, including for two empty JSON lists.
    Result tuples retain the JSON list order. Neither input file is changed.

    The benchmark also exposes `accuracy`, `result_list`, `img_paths`,
    `trues`, `preds`, `correct_predictions` and `incorrect_predictions`;
    `len(benchmark)` counts paired images. See the shared properties on
    [`ClassificationBenchmark`][flowde.benchmarks.classification.classification_benchmark.ClassificationBenchmark].

    Examples
    --------
    Using existing ground-truth and prediction JSON files:

    >>> from pathlib import Path
    >>> from flowde.benchmarks.classification.classification_benchmark import (
    ...     BinaryClassificationBenchmark,
    ... )
    >>> benchmark = BinaryClassificationBenchmark(
    ...     true_path=Path("data/true-classifications.json"),
    ...     pred_path=Path("results/classification/classifications.json"),
    ... )
    >>> print(benchmark.accuracy, benchmark.precision, benchmark.recall)

    """

    _num_tp: int = field(init=False, repr=False, compare=False)
    _num_fp: int = field(init=False, repr=False, compare=False)
    _num_fn: int = field(init=False, repr=False, compare=False)
    _num_tn: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        num_tp = 0
        num_fp = 0
        num_fn = 0
        num_tn = 0

        for result in self.result_list:
            if type(result.pred) is not int or type(result.true) is not int:
                msg = (
                    "BinaryClassificationBenchmark expects integer labels. "
                    f"Found pred={result.pred!r}, true={result.true!r} "
                    f"for image {result.img_path!r}."
                )
                raise TypeError(msg)

            if result.pred not in {0, 1} or result.true not in {0, 1}:
                msg = (
                    "BinaryClassificationBenchmark expects only 0 and 1 labels. "
                    f"Found pred={result.pred!r}, true={result.true!r} "
                    f"for image {result.img_path!r}."
                )
                raise ValueError(msg)

            if result.pred == 1 and result.true == 1:
                num_tp += 1
            elif result.pred == 1 and result.true == 0:
                num_fp += 1
            elif result.pred == 0 and result.true == 1:
                num_fn += 1
            elif result.pred == 0 and result.true == 0:
                num_tn += 1

        object.__setattr__(self, "_num_tp", num_tp)
        object.__setattr__(self, "_num_fp", num_fp)
        object.__setattr__(self, "_num_fn", num_fn)
        object.__setattr__(self, "_num_tn", num_tn)

    @property
    def tp(self) -> tuple[SingleClassificationResult[int], ...]:
        return tuple(r for r in self.result_list if r.pred == 1 and r.true == 1)

    @property
    def fp(self) -> tuple[SingleClassificationResult[int], ...]:
        return tuple(r for r in self.result_list if r.pred == 1 and r.true == 0)

    @property
    def fn(self) -> tuple[SingleClassificationResult[int], ...]:
        return tuple(r for r in self.result_list if r.pred == 0 and r.true == 1)

    @property
    def tn(self) -> tuple[SingleClassificationResult[int], ...]:
        return tuple(r for r in self.result_list if r.pred == 0 and r.true == 0)

    @property
    def num_tp(self) -> int:
        return self._num_tp

    @property
    def num_fp(self) -> int:
        return self._num_fp

    @property
    def num_fn(self) -> int:
        return self._num_fn

    @property
    def num_tn(self) -> int:
        return self._num_tn

    @property
    def precision(self) -> float:
        if self.num_tp + self.num_fp == 0:
            return 0.0
        return self.num_tp / (self.num_tp + self.num_fp)

    @property
    def recall(self) -> float:
        if self.num_tp + self.num_fn == 0:
            return 0.0
        return self.num_tp / (self.num_tp + self.num_fn)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def tpr(self) -> float:
        return self.recall

    @property
    def fpr(self) -> float:
        if self.num_fp + self.num_tn == 0:
            return 0.0
        return self.num_fp / (self.num_fp + self.num_tn)

    @property
    def specificity(self) -> float:
        if self.num_tn + self.num_fp == 0:
            return 0.0
        return self.num_tn / (self.num_tn + self.num_fp)

    @property
    def tnr(self) -> float:
        return self.specificity


@dataclass(frozen=True, kw_only=True)
class MulticlassClassificationBenchmark(ClassificationBenchmark[LabelType]):
    """
    Evaluate image classification across a declared set of class labels.

    The benchmark compares saved predicted labels with ground-truth labels
    and provides overall accuracy, a confusion matrix and per-class counts.
    Construction reads JSON files without opening images or making model
    requests.

    Parameters
    ----------
    true_path : Path
        Ground-truth JSON file containing a list of objects with exactly
        `img_path` and `label` keys. Image paths must be strings; labels must
        be `str`, `int` or `bool` values included in `labels`.
    pred_path : Path
        Prediction JSON file in the same format as `true_path`, such as
        `classifications.json` saved by
        [`classify_imgs()`][flowde.classify_imgs.classify_imgs]. Both lists must
        contain the same number of entries in the same image order. At each
        position, filenames including extensions must match; parent
        directories may differ.
    labels : tuple[LabelType, ...]
        Non-empty tuple of unique class labels. Must include every label in
        both JSON files; may also include classes absent from both files.
        Tuple order determines the order of dictionary keys in the confusion
        matrix and class summaries. Uniqueness follows Python equality, so
        `1` and `True`, for example, cannot be separate classes.

    Attributes
    ----------
    confusion_matrix : dict[LabelType, dict[LabelType, int]]
        Counts indexed first by ground-truth label, then by predicted label.
        `confusion_matrix["table"]["flowchart"]` counts true tables predicted
        as flowcharts. Includes every pair of declared labels, with zero for
        pairs absent from the saved results.
    num_per_true_class : dict[LabelType, int]
        Number of images with each ground-truth label, including zero counts.
    num_per_pred_class : dict[LabelType, int]
        Number of images with each predicted label, including zero counts.
    per_class_accuracy : dict[LabelType, float]
        For each ground-truth class, the number of correct predictions divided
        by the number of images in that class. Returns `0.0` for a declared
        class with no ground-truth images.

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
        `labels` is empty or contains duplicates, or a saved label is absent
        from `labels`.

    Notes
    -----
    Pass constructor arguments by keyword. Result tuples retain the JSON
    list order. Neither input file is changed. Two empty JSON lists are
    accepted with a non-empty `labels` tuple and produce zero counts and
    accuracies.

    The benchmark also exposes `accuracy`, `result_list`, `img_paths`,
    `trues`, `preds`, `correct_predictions` and `incorrect_predictions`;
    `len(benchmark)` counts paired images. See the shared properties on
    [`ClassificationBenchmark`][flowde.benchmarks.classification.classification_benchmark.ClassificationBenchmark].

    Examples
    --------
    Using existing ground-truth and prediction JSON files:

    >>> from pathlib import Path
    >>> from flowde.benchmarks.classification.classification_benchmark import (
    ...     MulticlassClassificationBenchmark,
    ... )
    >>> benchmark = MulticlassClassificationBenchmark(
    ...     true_path=Path("data/true-classifications.json"),
    ...     pred_path=Path("results/classification/classifications.json"),
    ...     labels=("flowchart", "table", "other"),
    ... )
    >>> print(benchmark.confusion_matrix)
    >>> print(benchmark.per_class_accuracy)

    """

    labels: tuple[LabelType, ...]

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.labels) == 0:
            msg = "labels cannot be empty."
            raise ValueError(msg)

        if len(set(self.labels)) != len(self.labels):
            msg = f"labels must be unique. Found labels: {self.labels!r}."
            raise ValueError(msg)

        for result in self.result_list:
            if result.true not in self.labels:
                msg = (
                    f"True label {result.true!r} for image {result.img_path!r} "
                    f"is not in labels: {self.labels!r}."
                )
                raise ValueError(msg)

            if result.pred not in self.labels:
                msg = (
                    f"Predicted label {result.pred!r} for image {result.img_path!r} "
                    f"is not in labels: {self.labels!r}."
                )
                raise ValueError(msg)

    @property
    def confusion_matrix(self) -> dict[LabelType, dict[LabelType, int]]:
        matrix = {
            true_label: dict.fromkeys(self.labels, 0) for true_label in self.labels
        }

        for result in self.result_list:
            matrix[result.true][result.pred] += 1

        return matrix

    @property
    def num_per_true_class(self) -> dict[LabelType, int]:
        counts = Counter(self.trues)
        return {label: counts[label] for label in self.labels}

    @property
    def num_per_pred_class(self) -> dict[LabelType, int]:
        counts = Counter(self.preds)
        return {label: counts[label] for label in self.labels}

    @property
    def per_class_accuracy(self) -> dict[LabelType, float]:
        scores = {}

        for label in self.labels:
            true_class_results = tuple(r for r in self.result_list if r.true == label)

            if len(true_class_results) == 0:
                scores[label] = 0.0
            else:
                scores[label] = len(
                    tuple(r for r in true_class_results if r.pred == r.true)
                ) / len(true_class_results)

        return scores


def load_classification_benchmark_results(
    true_path: Path,
    pred_path: Path,
) -> tuple[SingleClassificationResult[ClassificationLabel], ...]:
    true_classifications = load_json(true_path)
    pred_classifications = load_json(pred_path)

    check_generic_classification_benchmark_data(
        true_path=true_path,
        pred_path=pred_path,
    )

    return tuple(
        SingleClassificationResult(
            img_path=true_cls["img_path"],
            true=true_cls["label"],
            pred=pred_cls["label"],
        )
        for true_cls, pred_cls in zip(
            true_classifications,
            pred_classifications,
            strict=True,
        )
    )


def check_generic_classification_benchmark_data(
    true_path: Path,
    pred_path: Path,
) -> None:
    true_classifications = load_json(true_path)
    pred_classifications = load_json(pred_path)

    if not isinstance(true_classifications, list):
        msg = "True classifications should be a list."
        raise TypeError(msg)

    if not isinstance(pred_classifications, list):
        msg = "Pred classifications should be a list."
        raise TypeError(msg)

    if len(true_classifications) != len(pred_classifications):
        msg = (
            f"True and pred classifications should have the same length. "
            f"Found {len(true_classifications)} and {len(pred_classifications)}."
        )
        raise ValueError(msg)

    required_keys = {"img_path", "label"}

    for classification in true_classifications:
        if not isinstance(classification, dict):
            msg = f"Items in {true_path} should be dictionaries."
            raise TypeError(msg)

        if set(classification.keys()) != required_keys:
            msg = f"Found wrong keys in {true_path}."
            raise ValueError(msg)

    for classification in pred_classifications:
        if not isinstance(classification, dict):
            msg = f"Items in {pred_path} should be dictionaries."
            raise TypeError(msg)

        if set(classification.keys()) != required_keys:
            msg = f"Found wrong keys in {pred_path}."
            raise ValueError(msg)

    for pred_cls, true_cls in zip(
        pred_classifications,
        true_classifications,
        strict=True,
    ):
        if not isinstance(pred_cls["img_path"], str):
            msg = f"Image path should be a string. Found {type(pred_cls['img_path'])}."
            raise TypeError(msg)

        if not isinstance(true_cls["img_path"], str):
            msg = f"Image path should be a string. Found {type(true_cls['img_path'])}."
            raise TypeError(msg)

        if type(pred_cls["label"]) not in {str, int, bool}:
            msg = (
                "Predicted label should be a string, integer, or boolean. "
                f"Found {type(pred_cls['label'])}."
            )
            raise TypeError(msg)

        if type(true_cls["label"]) not in {str, int, bool}:
            msg = (
                "True label should be a string, integer, or boolean. "
                f"Found {type(true_cls['label'])}."
            )
            raise TypeError(msg)

        pred_img_path = Path(pred_cls["img_path"])
        true_img_path = Path(true_cls["img_path"])

        if pred_img_path.name != true_img_path.name:
            msg = (
                "Image paths should match between true and pred classifications. "
                f"Found {pred_img_path} and {true_img_path}."
            )
            raise ValueError(msg)
