import json
from pathlib import Path

import pytest

import flowde.benchmarks.rotation.rotation_benchmark as rotation_benchmark_module
from flowde.benchmarks.classification.classification_benchmark import (
    MulticlassClassificationBenchmark,
)


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def make_classification(img_path: str, label: int) -> dict:
    return {
        "img_path": img_path,
        "label": label,
    }


def test_rotation_benchmark_uses_rotation_labels_by_default(tmp_path: Path) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification("image_1.png", 0),
            make_classification("image_2.png", 90),
            make_classification("image_3.png", 180),
            make_classification("image_4.png", 270),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification("image_1.png", 0),
            make_classification("image_2.png", 180),
            make_classification("image_3.png", 180),
            make_classification("image_4.png", 90),
        ],
    )

    benchmark = rotation_benchmark_module.RotationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert benchmark.labels == (0, 90, 180, 270)


def test_rotation_benchmark_calculates_multiclass_metrics(tmp_path: Path) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification("image_1.png", 0),
            make_classification("image_2.png", 90),
            make_classification("image_3.png", 90),
            make_classification("image_4.png", 270),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification("image_1.png", 0),
            make_classification("image_2.png", 90),
            make_classification("image_3.png", 180),
            make_classification("image_4.png", 180),
        ],
    )

    benchmark = rotation_benchmark_module.RotationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert benchmark.accuracy == pytest.approx(2 / 4)

    assert benchmark.confusion_matrix == {
        0: {0: 1, 90: 0, 180: 0, 270: 0},
        90: {0: 0, 90: 1, 180: 1, 270: 0},
        180: {0: 0, 90: 0, 180: 0, 270: 0},
        270: {0: 0, 90: 0, 180: 1, 270: 0},
    }

    assert benchmark.num_per_true_class == {
        0: 1,
        90: 2,
        180: 0,
        270: 1,
    }

    assert benchmark.num_per_pred_class == {
        0: 1,
        90: 1,
        180: 2,
        270: 0,
    }

    assert benchmark.per_class_accuracy == {
        0: 1.0,
        90: 0.5,
        180: 0.0,
        270: 0.0,
    }


@pytest.mark.parametrize(
    ("true_label", "pred_label", "expected_msg"),
    [
        pytest.param(45, 0, "True label 45", id="invalid-true-rotation"),
        pytest.param(0, 45, "Predicted label 45", id="invalid-pred-rotation"),
    ],
)
def test_rotation_benchmark_raises_for_invalid_rotation_label(
    tmp_path: Path,
    true_label: int,
    pred_label: int,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification("image_1.png", true_label)])
    write_json(pred_path, [make_classification("image_1.png", pred_label)])

    with pytest.raises(ValueError) as exc_info:
        rotation_benchmark_module.RotationBenchmark(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert expected_msg in str(exc_info.value)


def test_rotation_benchmark_is_multiclass_benchmark(tmp_path: Path) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [])
    write_json(pred_path, [])

    benchmark = rotation_benchmark_module.RotationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert isinstance(benchmark, MulticlassClassificationBenchmark)
