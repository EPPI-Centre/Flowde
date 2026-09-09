import json
from pathlib import Path

import pytest

import flowde.benchmarks.classification.classification_benchmark as classification_benchmark_module


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def make_img_path(dataset: str, paper: str, filename: str) -> str:
    return str(
        Path("experiment_data") / "extraction" / dataset / paper / "method_1" / filename
    )


def make_classification(img_path: str, label) -> dict:
    return {
        "img_path": img_path,
        "label": label,
    }


def test_classification_benchmark_loads_results_and_common_properties(
    tmp_path: Path,
) -> None:
    img_path_1 = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    img_path_2 = make_img_path("test_dataset", "paper_2", "paper_2_0.png")
    img_path_3 = make_img_path("test_dataset", "paper_3", "paper_3_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path_1, 1),
            make_classification(img_path_2, 0),
            make_classification(img_path_3, 1),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path_1, 1),
            make_classification(img_path_2, 1),
            make_classification(img_path_3, 1),
        ],
    )

    benchmark = classification_benchmark_module.ClassificationBenchmark[int](
        true_path=true_path,
        pred_path=pred_path,
    )

    assert len(benchmark) == 3
    assert benchmark.result_list == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=1,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=0,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=1,
            pred=1,
        ),
    )
    assert benchmark.correct_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=1,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=1,
            pred=1,
        ),
    )
    assert benchmark.incorrect_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=0,
            pred=1,
        ),
    )
    assert benchmark.accuracy == 2 / 3
    assert benchmark.img_paths == (img_path_1, img_path_2, img_path_3)
    assert benchmark.trues == (1, 0, 1)
    assert benchmark.preds == (1, 1, 1)


def test_classification_benchmark_accuracy_is_zero_for_empty_results(
    tmp_path: Path,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [])
    write_json(pred_path, [])

    benchmark = classification_benchmark_module.ClassificationBenchmark[int](
        true_path=true_path,
        pred_path=pred_path,
    )

    assert len(benchmark) == 0
    assert benchmark.result_list == ()
    assert benchmark.correct_predictions == ()
    assert benchmark.incorrect_predictions == ()
    assert benchmark.accuracy == 0.0


@pytest.mark.parametrize(
    (
        "true_labels",
        "pred_labels",
        "expected_num_tp",
        "expected_num_tn",
        "expected_num_fp",
        "expected_num_fn",
        "expected_accuracy",
        "expected_precision",
        "expected_recall",
        "expected_f1_score",
        "expected_fpr",
        "expected_specificity",
    ),
    [
        pytest.param(
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            2,
            3,
            1,
            1,
            5 / 7,
            2 / 3,
            2 / 3,
            2 / 3,
            1 / 4,
            3 / 4,
            id="mixed-unbalanced",
        ),
        pytest.param(
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            3,
            3,
            2,
            0,
            6 / 8,
            3 / 5,
            1.0,
            3 / 4,
            2 / 5,
            3 / 5,
            id="high-recall-lower-precision",
        ),
        pytest.param(
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            1,
            3,
            1,
            3,
            4 / 8,
            1 / 2,
            1 / 4,
            1 / 3,
            1 / 4,
            3 / 4,
            id="low-recall",
        ),
        pytest.param(
            [1, 0, 0],
            [0, 0, 0],
            0,
            2,
            0,
            1,
            2 / 3,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            id="no-predicted-positives",
        ),
        pytest.param(
            [0, 0, 0],
            [1, 0, 0],
            0,
            2,
            1,
            0,
            2 / 3,
            0.0,
            0.0,
            0.0,
            1 / 3,
            2 / 3,
            id="no-true-positives",
        ),
    ],
)
def test_binary_classification_benchmark_calculates_binary_metrics(
    tmp_path: Path,
    true_labels: list[int],
    pred_labels: list[int],
    expected_num_tp: int,
    expected_num_tn: int,
    expected_num_fp: int,
    expected_num_fn: int,
    expected_accuracy: float,
    expected_precision: float,
    expected_recall: float,
    expected_f1_score: float,
    expected_fpr: float,
    expected_specificity: float,
) -> None:
    img_paths = [
        make_img_path(
            "test_dataset",
            f"paper_{idx}",
            f"paper_{idx}_0.png",
        )
        for idx in range(len(true_labels))
    ]

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path, true_label)
            for img_path, true_label in zip(img_paths, true_labels, strict=True)
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path, pred_label)
            for img_path, pred_label in zip(img_paths, pred_labels, strict=True)
        ],
    )

    benchmark = classification_benchmark_module.BinaryClassificationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert benchmark.num_tp == expected_num_tp
    assert benchmark.num_tn == expected_num_tn
    assert benchmark.num_fp == expected_num_fp
    assert benchmark.num_fn == expected_num_fn

    assert len(benchmark.tp) == expected_num_tp
    assert len(benchmark.tn) == expected_num_tn
    assert len(benchmark.fp) == expected_num_fp
    assert len(benchmark.fn) == expected_num_fn

    assert all(r.pred == 1 and r.true == 1 for r in benchmark.tp)
    assert all(r.pred == 0 and r.true == 0 for r in benchmark.tn)
    assert all(r.pred == 1 and r.true == 0 for r in benchmark.fp)
    assert all(r.pred == 0 and r.true == 1 for r in benchmark.fn)

    assert benchmark.accuracy == pytest.approx(expected_accuracy)
    assert benchmark.precision == pytest.approx(expected_precision)
    assert benchmark.recall == pytest.approx(expected_recall)
    assert benchmark.f1_score == pytest.approx(expected_f1_score)
    assert benchmark.tpr == pytest.approx(expected_recall)
    assert benchmark.fpr == pytest.approx(expected_fpr)
    assert benchmark.specificity == pytest.approx(expected_specificity)
    assert benchmark.tnr == pytest.approx(expected_specificity)


@pytest.mark.parametrize(
    ("true_label", "pred_label", "expected_error", "expected_msg"),
    [
        pytest.param(
            1,
            "1",
            TypeError,
            "BinaryClassificationBenchmark expects integer labels.",
            id="pred-label-not-int",
        ),
        pytest.param(
            "1",
            1,
            TypeError,
            "BinaryClassificationBenchmark expects integer labels.",
            id="true-label-not-int",
        ),
        pytest.param(
            1,
            2,
            ValueError,
            "BinaryClassificationBenchmark expects only 0 and 1 labels.",
            id="pred-label-not-binary",
        ),
        pytest.param(
            2,
            1,
            ValueError,
            "BinaryClassificationBenchmark expects only 0 and 1 labels.",
            id="true-label-not-binary",
        ),
    ],
)
def test_binary_classification_benchmark_raises_for_non_binary_labels(
    tmp_path: Path,
    true_label,
    pred_label,
    expected_error,
    expected_msg: str,
) -> None:
    img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(img_path, true_label)])
    write_json(pred_path, [make_classification(img_path, pred_label)])

    with pytest.raises(expected_error) as exc_info:
        classification_benchmark_module.BinaryClassificationBenchmark(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    (
        "labels",
        "true_labels",
        "pred_labels",
        "expected_accuracy",
        "expected_confusion_matrix",
        "expected_num_per_true_class",
        "expected_num_per_pred_class",
        "expected_per_class_accuracy",
    ),
    [
        pytest.param(
            (0, 90, 180, 270),
            [0, 90, 90, 180],
            [0, 90, 180, 270],
            2 / 4,
            {
                0: {0: 1, 90: 0, 180: 0, 270: 0},
                90: {0: 0, 90: 1, 180: 1, 270: 0},
                180: {0: 0, 90: 0, 180: 0, 270: 1},
                270: {0: 0, 90: 0, 180: 0, 270: 0},
            },
            {
                0: 1,
                90: 2,
                180: 1,
                270: 0,
            },
            {
                0: 1,
                90: 1,
                180: 1,
                270: 1,
            },
            {
                0: 1.0,
                90: 1 / 2,
                180: 0.0,
                270: 0.0,
            },
            id="rotation-with-missing-true-class",
        ),
        pytest.param(
            (0, 90, 180, 270),
            [0, 0, 90, 270, 270],
            [0, 90, 90, 270, 180],
            3 / 5,
            {
                0: {0: 1, 90: 1, 180: 0, 270: 0},
                90: {0: 0, 90: 1, 180: 0, 270: 0},
                180: {0: 0, 90: 0, 180: 0, 270: 0},
                270: {0: 0, 90: 0, 180: 1, 270: 1},
            },
            {
                0: 2,
                90: 1,
                180: 0,
                270: 2,
            },
            {
                0: 1,
                90: 2,
                180: 1,
                270: 1,
            },
            {
                0: 1 / 2,
                90: 1.0,
                180: 0.0,
                270: 1 / 2,
            },
            id="rotation-unbalanced",
        ),
        pytest.param(
            ("consort", "other_flowchart", "table"),
            [
                "consort",
                "consort",
                "other_flowchart",
                "table",
                "table",
                "table",
            ],
            [
                "consort",
                "table",
                "other_flowchart",
                "table",
                "consort",
                "other_flowchart",
            ],
            3 / 6,
            {
                "consort": {
                    "consort": 1,
                    "other_flowchart": 0,
                    "table": 1,
                },
                "other_flowchart": {
                    "consort": 0,
                    "other_flowchart": 1,
                    "table": 0,
                },
                "table": {
                    "consort": 1,
                    "other_flowchart": 1,
                    "table": 1,
                },
            },
            {
                "consort": 2,
                "other_flowchart": 1,
                "table": 3,
            },
            {
                "consort": 2,
                "other_flowchart": 2,
                "table": 2,
            },
            {
                "consort": 1 / 2,
                "other_flowchart": 1.0,
                "table": 1 / 3,
            },
            id="string-labels",
        ),
    ],
)
def test_multiclass_classification_benchmark_calculates_multiclass_metrics(
    tmp_path: Path,
    labels,
    true_labels,
    pred_labels,
    expected_accuracy: float,
    expected_confusion_matrix,
    expected_num_per_true_class,
    expected_num_per_pred_class,
    expected_per_class_accuracy,
) -> None:
    img_paths = [
        make_img_path(
            "test_dataset",
            f"paper_{idx}",
            f"paper_{idx}_0.png",
        )
        for idx in range(len(true_labels))
    ]

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path, true_label)
            for img_path, true_label in zip(img_paths, true_labels, strict=True)
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path, pred_label)
            for img_path, pred_label in zip(img_paths, pred_labels, strict=True)
        ],
    )

    benchmark = classification_benchmark_module.MulticlassClassificationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
        labels=labels,
    )

    assert benchmark.accuracy == pytest.approx(expected_accuracy)
    assert benchmark.confusion_matrix == expected_confusion_matrix
    assert benchmark.num_per_true_class == expected_num_per_true_class
    assert benchmark.num_per_pred_class == expected_num_per_pred_class

    assert benchmark.per_class_accuracy.keys() == expected_per_class_accuracy.keys()
    for label, expected_score in expected_per_class_accuracy.items():
        assert benchmark.per_class_accuracy[label] == pytest.approx(expected_score)


@pytest.mark.parametrize(
    ("labels", "expected_msg"),
    [
        pytest.param((), "labels cannot be empty.", id="empty-labels"),
        pytest.param((0, 90, 90), "labels must be unique.", id="duplicate-labels"),
    ],
)
def test_multiclass_classification_benchmark_raises_for_invalid_labels(
    tmp_path: Path,
    labels,
    expected_msg: str,
) -> None:
    img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(img_path, 0)])
    write_json(pred_path, [make_classification(img_path, 0)])

    with pytest.raises(ValueError) as exc_info:
        classification_benchmark_module.MulticlassClassificationBenchmark[int](
            true_path=true_path,
            pred_path=pred_path,
            labels=labels,
        )

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("true_label", "pred_label", "expected_msg"),
    [
        pytest.param(
            270,
            0,
            "True label 270",
            id="true-label-not-in-labels",
        ),
        pytest.param(
            0,
            270,
            "Predicted label 270",
            id="pred-label-not-in-labels",
        ),
    ],
)
def test_multiclass_classification_benchmark_raises_if_result_label_not_in_labels(
    tmp_path: Path,
    true_label: int,
    pred_label: int,
    expected_msg: str,
) -> None:
    img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(img_path, true_label)])
    write_json(pred_path, [make_classification(img_path, pred_label)])

    with pytest.raises(ValueError) as exc_info:
        classification_benchmark_module.MulticlassClassificationBenchmark[int](
            true_path=true_path,
            pred_path=pred_path,
            labels=(0, 90, 180),
        )

    assert expected_msg in str(exc_info.value)


def test_check_generic_classification_benchmark_data_passes_for_valid_data(
    tmp_path: Path,
) -> None:
    img_path_1 = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    img_path_2 = make_img_path("test_dataset", "paper_2", "paper_2_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path_1, 1),
            make_classification(img_path_2, "other"),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path_1, 0),
            make_classification(img_path_2, "flowchart"),
        ],
    )

    result = (
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )
    )

    assert result is None


@pytest.mark.parametrize(
    ("true_content", "pred_content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {"img_path": "not a list", "label": 1},
            [],
            TypeError,
            "True classifications should be a list.",
            id="true-classifications-not-list",
        ),
        pytest.param(
            [],
            {"img_path": "not a list", "label": 1},
            TypeError,
            "Pred classifications should be a list.",
            id="pred-classifications-not-list",
        ),
        pytest.param(
            [
                make_classification(
                    make_img_path("test_dataset", "paper_1", "paper_1_0.png"),
                    1,
                ),
            ],
            [
                make_classification(
                    make_img_path("test_dataset", "paper_1", "paper_1_0.png"),
                    1,
                ),
                make_classification(
                    make_img_path("test_dataset", "paper_2", "paper_2_0.png"),
                    0,
                ),
            ],
            ValueError,
            "True and pred classifications should have the same length. Found 1 and 2.",
            id="different-lengths",
        ),
    ],
)
def test_check_generic_classification_benchmark_data_raises_for_top_level_invalid_data(
    tmp_path: Path,
    true_content,
    pred_content,
    expected_error,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, true_content)
    write_json(pred_path, pred_content)

    with pytest.raises(expected_error) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert str(exc_info.value) == expected_msg


@pytest.mark.parametrize(
    ("true_content", "pred_content", "expected_error", "expected_msg"),
    [
        pytest.param(
            ["not a dict"],
            [make_classification("paper_1_0.png", 1)],
            TypeError,
            "Items in",
            id="true-item-not-dict",
        ),
        pytest.param(
            [make_classification("paper_1_0.png", 1)],
            ["not a dict"],
            TypeError,
            "Items in",
            id="pred-item-not-dict",
        ),
    ],
)
def test_check_generic_classification_benchmark_data_raises_for_non_dict_items(
    tmp_path: Path,
    true_content,
    pred_content,
    expected_error,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, true_content)
    write_json(pred_path, pred_content)

    with pytest.raises(expected_error) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("bad_true_classification", "expected_msg"),
    [
        pytest.param(
            {"img_path": make_img_path("test_dataset", "paper_1", "paper_1_0.png")},
            "Found wrong keys in",
            id="missing-label",
        ),
        pytest.param(
            {"label": 1},
            "Found wrong keys in",
            id="missing-img-path",
        ),
        pytest.param(
            {
                "img_path": make_img_path("test_dataset", "paper_1", "paper_1_0.png"),
                "label": 1,
                "extra": "not allowed",
            },
            "Found wrong keys in",
            id="extra-key",
        ),
    ],
)
def test_check_generic_classification_benchmark_data_raises_for_wrong_true_keys(
    tmp_path: Path,
    bad_true_classification,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")

    write_json(true_path, [bad_true_classification])
    write_json(pred_path, [make_classification(img_path, 1)])

    with pytest.raises(ValueError) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert expected_msg in str(exc_info.value)
    assert str(true_path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("bad_pred_classification", "expected_msg"),
    [
        pytest.param(
            {"img_path": make_img_path("test_dataset", "paper_1", "paper_1_0.png")},
            "Found wrong keys in",
            id="missing-label",
        ),
        pytest.param(
            {"label": 1},
            "Found wrong keys in",
            id="missing-img-path",
        ),
        pytest.param(
            {
                "img_path": make_img_path("test_dataset", "paper_1", "paper_1_0.png"),
                "label": 1,
                "extra": "not allowed",
            },
            "Found wrong keys in",
            id="extra-key",
        ),
    ],
)
def test_check_generic_classification_benchmark_data_raises_for_wrong_pred_keys(
    tmp_path: Path,
    bad_pred_classification,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")

    write_json(true_path, [make_classification(img_path, 1)])
    write_json(pred_path, [bad_pred_classification])

    with pytest.raises(ValueError) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert expected_msg in str(exc_info.value)
    assert str(pred_path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("true_classification", "pred_classification", "expected_error", "expected_msg"),
    [
        pytest.param(
            make_classification(123, 1),
            make_classification("paper_1_0.png", 1),
            TypeError,
            "Image path should be a string. Found <class 'int'>.",
            id="true-img-path-not-string",
        ),
        pytest.param(
            make_classification("paper_1_0.png", 1),
            make_classification(123, 1),
            TypeError,
            "Image path should be a string. Found <class 'int'>.",
            id="pred-img-path-not-string",
        ),
        pytest.param(
            make_classification("paper_1_0.png", 1.5),
            make_classification("paper_1_0.png", 1),
            TypeError,
            "True label should be a string, integer, or boolean. Found <class 'float'>.",
            id="true-label-invalid-type",
        ),
        pytest.param(
            make_classification("paper_1_0.png", 1),
            make_classification("paper_1_0.png", 1.5),
            TypeError,
            "Predicted label should be a string, integer, or boolean. Found <class 'float'>.",
            id="pred-label-invalid-type",
        ),
    ],
)
def test_check_generic_classification_benchmark_data_raises_for_invalid_value_types(
    tmp_path: Path,
    true_classification,
    pred_classification,
    expected_error,
    expected_msg: str,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [true_classification])
    write_json(pred_path, [pred_classification])

    with pytest.raises(expected_error) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert str(exc_info.value) == expected_msg


def test_check_generic_classification_benchmark_data_raises_if_image_filenames_do_not_match(
    tmp_path: Path,
) -> None:
    true_img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    pred_img_path = make_img_path("test_dataset", "paper_2", "paper_2_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(true_img_path, 1)])
    write_json(pred_path, [make_classification(pred_img_path, 1)])

    with pytest.raises(ValueError) as exc_info:
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )

    assert str(exc_info.value) == (
        "Image paths should match between true and pred classifications. "
        f"Found {Path(pred_img_path)} and {Path(true_img_path)}."
    )


def test_check_generic_classification_benchmark_data_allows_different_parent_dirs_if_filenames_match(
    tmp_path: Path,
) -> None:
    true_img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    pred_img_path = str(
        Path("some") / "other" / "directory" / "structure" / "paper_1_0.png"
    )

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(true_img_path, 1)])
    write_json(pred_path, [make_classification(pred_img_path, 0)])

    result = (
        classification_benchmark_module.check_generic_classification_benchmark_data(
            true_path=true_path,
            pred_path=pred_path,
        )
    )

    assert result is None


def test_load_classification_benchmark_results_uses_true_image_path_when_parent_dirs_differ(
    tmp_path: Path,
) -> None:
    true_img_path = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    pred_img_path = str(
        Path("some") / "other" / "directory" / "structure" / "paper_1_0.png"
    )

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(true_path, [make_classification(true_img_path, 1)])
    write_json(pred_path, [make_classification(pred_img_path, 0)])

    results = classification_benchmark_module.load_classification_benchmark_results(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert results == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=true_img_path,
            true=1,
            pred=0,
        ),
    )


def test_single_classification_result_dataclass_stores_values() -> None:
    result = classification_benchmark_module.SingleClassificationResult(
        img_path="some/path.png",
        true=1,
        pred=0,
    )

    assert result.img_path == "some/path.png"
    assert result.true == 1
    assert result.pred == 0


def test_binary_classification_benchmark_inherits_common_properties(
    tmp_path: Path,
) -> None:
    img_path_1 = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    img_path_2 = make_img_path("test_dataset", "paper_2", "paper_2_0.png")
    img_path_3 = make_img_path("test_dataset", "paper_3", "paper_3_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path_1, 1),
            make_classification(img_path_2, 0),
            make_classification(img_path_3, 1),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path_1, 1),
            make_classification(img_path_2, 1),
            make_classification(img_path_3, 1),
        ],
    )

    benchmark = classification_benchmark_module.BinaryClassificationBenchmark(
        true_path=true_path,
        pred_path=pred_path,
    )

    assert isinstance(
        benchmark,
        classification_benchmark_module.ClassificationBenchmark,
    )
    assert len(benchmark) == 3
    assert benchmark.result_list == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=1,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=0,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=1,
            pred=1,
        ),
    )
    assert benchmark.correct_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=1,
            pred=1,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=1,
            pred=1,
        ),
    )
    assert benchmark.incorrect_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=0,
            pred=1,
        ),
    )
    assert benchmark.accuracy == pytest.approx(2 / 3)
    assert benchmark.img_paths == (img_path_1, img_path_2, img_path_3)
    assert benchmark.trues == (1, 0, 1)
    assert benchmark.preds == (1, 1, 1)


def test_multiclass_classification_benchmark_inherits_common_properties(
    tmp_path: Path,
) -> None:
    img_path_1 = make_img_path("test_dataset", "paper_1", "paper_1_0.png")
    img_path_2 = make_img_path("test_dataset", "paper_2", "paper_2_0.png")
    img_path_3 = make_img_path("test_dataset", "paper_3", "paper_3_0.png")

    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    write_json(
        true_path,
        [
            make_classification(img_path_1, 0),
            make_classification(img_path_2, 90),
            make_classification(img_path_3, 180),
        ],
    )
    write_json(
        pred_path,
        [
            make_classification(img_path_1, 0),
            make_classification(img_path_2, 180),
            make_classification(img_path_3, 180),
        ],
    )

    benchmark = classification_benchmark_module.MulticlassClassificationBenchmark[int](
        true_path=true_path,
        pred_path=pred_path,
        labels=(0, 90, 180, 270),
    )

    assert isinstance(
        benchmark,
        classification_benchmark_module.ClassificationBenchmark,
    )
    assert len(benchmark) == 3
    assert benchmark.result_list == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=0,
            pred=0,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=90,
            pred=180,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=180,
            pred=180,
        ),
    )
    assert benchmark.correct_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_1,
            true=0,
            pred=0,
        ),
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_3,
            true=180,
            pred=180,
        ),
    )
    assert benchmark.incorrect_predictions == (
        classification_benchmark_module.SingleClassificationResult(
            img_path=img_path_2,
            true=90,
            pred=180,
        ),
    )
    assert benchmark.accuracy == pytest.approx(2 / 3)
    assert benchmark.img_paths == (img_path_1, img_path_2, img_path_3)
    assert benchmark.trues == (0, 90, 180)
    assert benchmark.preds == (0, 180, 180)


@pytest.mark.parametrize(
    "benchmark_cls, extra_kwargs",
    [
        pytest.param(
            classification_benchmark_module.BinaryClassificationBenchmark,
            {},
            id="binary",
        ),
        pytest.param(
            classification_benchmark_module.MulticlassClassificationBenchmark,
            {"labels": (0, 1, 90, 180, 270)},
            id="multiclass",
        ),
    ],
)
def test_specialised_benchmarks_call_shared_result_loader(
    monkeypatch,
    tmp_path: Path,
    benchmark_cls,
    extra_kwargs,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    calls = {}

    fake_result_list = (
        classification_benchmark_module.SingleClassificationResult(
            img_path="paper_1_0.png",
            true=1,
            pred=1,
        ),
    )

    def fake_load_classification_benchmark_results(true_path, pred_path):
        calls["true_path"] = true_path
        calls["pred_path"] = pred_path
        return fake_result_list

    monkeypatch.setattr(
        classification_benchmark_module,
        "load_classification_benchmark_results",
        fake_load_classification_benchmark_results,
    )

    benchmark = benchmark_cls(
        true_path=true_path,
        pred_path=pred_path,
        **extra_kwargs,
    )

    assert calls == {
        "true_path": true_path,
        "pred_path": pred_path,
    }
    assert benchmark.result_list == fake_result_list


@pytest.mark.parametrize(
    ("benchmark_cls", "extra_kwargs"),
    [
        pytest.param(
            classification_benchmark_module.BinaryClassificationBenchmark,
            {},
            id="binary",
        ),
        pytest.param(
            classification_benchmark_module.MulticlassClassificationBenchmark,
            {"labels": (0, 1)},
            id="multiclass",
        ),
    ],
)
def test_specialised_benchmarks_call_super_post_init(
    monkeypatch,
    tmp_path: Path,
    benchmark_cls,
    extra_kwargs,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"

    calls = {"base_post_init_called": False}

    fake_result_list = (
        classification_benchmark_module.SingleClassificationResult(
            img_path="paper_1_0.png",
            true=1,
            pred=1,
        ),
    )

    def fake_base_post_init(self) -> None:
        calls["base_post_init_called"] = True
        object.__setattr__(self, "result_list", fake_result_list)

    monkeypatch.setattr(
        classification_benchmark_module.ClassificationBenchmark,
        "__post_init__",
        fake_base_post_init,
    )

    benchmark = benchmark_cls(
        true_path=true_path,
        pred_path=pred_path,
        **extra_kwargs,
    )

    assert calls["base_post_init_called"] is True
    assert benchmark.result_list == fake_result_list
