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
