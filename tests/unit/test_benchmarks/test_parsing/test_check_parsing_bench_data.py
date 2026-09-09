import json
from pathlib import Path

import pytest

import flowde.benchmarks.parsing.check_parsing_bench_data as check_module


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def path_for(parent: str, stem: str, suffix: str = ".json") -> Path:
    return Path(parent) / f"{stem}{suffix}"


def make_node_texts() -> dict:
    return {
        "options": [
            {
                "nodes": [
                    {"node_number": 1, "text": "Assessed for eligibility"},
                    {"node_number": 2, "text": "Randomised"},
                ]
            }
        ]
    }


def make_labels() -> dict:
    return {
        "options": [
            {
                "nodes": [
                    {"node_number": 1, "labels": ["screening", "eligibility"]},
                    {"node_number": 2, "labels": ["randomisation", "allocation"]},
                ]
            }
        ]
    }


def make_additional_texts() -> dict:
    return {
        "options": [
            {
                "additional_texts": [
                    "Figure 1",
                    "Participant flow diagram",
                ]
            }
        ]
    }


def make_flow() -> dict:
    return {
        "options": [
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2]},
                    {"node_number": 2, "points_to": [1]},
                ]
            }
        ]
    }


def write_valid_true_files(
    tmp_path: Path,
    stem: str = "diagram_1",
) -> tuple[Path, Path, Path, Path]:
    nodes_path = tmp_path / "nodes" / f"{stem}.json"
    labels_path = tmp_path / "labels" / f"{stem}.json"
    additional_texts_path = tmp_path / "additional_texts" / f"{stem}.json"
    flow_path = tmp_path / "flow" / f"{stem}.json"

    write_json(nodes_path, make_node_texts())
    write_json(labels_path, make_labels())
    write_json(additional_texts_path, make_additional_texts())
    write_json(flow_path, make_flow())

    return nodes_path, labels_path, additional_texts_path, flow_path


def write_valid_pred_file(
    tmp_path: Path,
    stem: str = "diagram_1",
    content: dict | None = None,
) -> Path:
    pred_path = tmp_path / "pred" / f"{stem}.json"
    write_json(
        pred_path,
        content
        or {
            "nodes": [
                {
                    "node_number": 1,
                    "text": "Assessed for eligibility",
                    "labels": ["screening", "eligibility"],
                    "points_to": [2],
                },
                {
                    "node_number": 2,
                    "text": "Randomised",
                    "labels": ["randomisation", "allocation"],
                    "points_to": [1],
                },
            ],
            "additional_texts": ["Figure 1"],
        },
    )
    return pred_path


def test_check_benchmark_files_match_passes_when_all_stems_match(
    tmp_path,
):
    pred_paths = [
        tmp_path / "pred" / "diagram_1.json",
        tmp_path / "pred" / "diagram_2.json",
    ]
    nodes_paths = [
        tmp_path / "nodes" / "diagram_1.json",
        tmp_path / "nodes" / "diagram_2.json",
    ]
    labels_paths = [
        tmp_path / "labels" / "diagram_1.json",
        tmp_path / "labels" / "diagram_2.json",
    ]
    additional_texts_paths = [
        tmp_path / "additional_texts" / "diagram_1.json",
        tmp_path / "additional_texts" / "diagram_2.json",
    ]
    flow_paths = [
        tmp_path / "flow" / "diagram_1.json",
        tmp_path / "flow" / "diagram_2.json",
    ]

    result = check_module.check_benchmark_files_match(
        pred_paths=pred_paths,
        nodes_paths=nodes_paths,
        labels_paths=labels_paths,
        additional_texts_paths=additional_texts_paths,
        flow_paths=flow_paths,
        expected_num_diagrams=2,
    )

    assert result is None


def test_check_benchmark_files_match_allows_missing_pred_diagrams(
    tmp_path,
):
    pred_paths = [
        tmp_path / "pred" / "diagram_1.json",
    ]
    nodes_paths = [
        tmp_path / "nodes" / "diagram_1.json",
        tmp_path / "nodes" / "diagram_2.json",
    ]
    labels_paths = [
        tmp_path / "labels" / "diagram_1.json",
        tmp_path / "labels" / "diagram_2.json",
    ]
    additional_texts_paths = [
        tmp_path / "additional_texts" / "diagram_1.json",
        tmp_path / "additional_texts" / "diagram_2.json",
    ]
    flow_paths = [
        tmp_path / "flow" / "diagram_1.json",
        tmp_path / "flow" / "diagram_2.json",
    ]

    result = check_module.check_benchmark_files_match(
        pred_paths=pred_paths,
        nodes_paths=nodes_paths,
        labels_paths=labels_paths,
        additional_texts_paths=additional_texts_paths,
        flow_paths=flow_paths,
        allow_missing_pred_diagrams=True,
        expected_num_diagrams=2,
    )

    assert result is None


@pytest.mark.parametrize(
    (
        "pred_paths",
        "nodes_paths",
        "labels_paths",
        "additional_texts_paths",
        "flow_paths",
        "allow_missing_pred_diagrams",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            False,
            ["Expected 2 diagrams", "Number of predicted diagrams: 1"],
            id="pred-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1"), path_for("pred", "diagram_2")],
            [path_for("nodes", "diagram_1")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            False,
            ["Expected 2 diagrams", "Number of true nodes files: 1"],
            id="nodes-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1"), path_for("pred", "diagram_2")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            False,
            ["Expected 2 diagrams", "Number of true labels files: 1"],
            id="labels-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1"), path_for("pred", "diagram_2")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [path_for("additional_texts", "diagram_1")],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            False,
            ["Expected 2 diagrams", "Number of true additional texts files: 1"],
            id="additional-texts-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1"), path_for("pred", "diagram_2")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1")],
            False,
            ["Expected 2 diagrams", "Number of true flow files: 1"],
            id="flow-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1")],
            [path_for("labels", "diagram_1")],
            [path_for("additional_texts", "diagram_1")],
            [path_for("flow", "diagram_1")],
            False,
            [
                "Expected 2 diagrams",
                "Number of predicted diagrams: 1",
                "Number of true nodes files: 1",
                "Number of true labels files: 1",
                "Number of true additional texts files: 1",
                "Number of true flow files: 1",
            ],
            id="all-counts-wrong",
        ),
        pytest.param(
            [],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            True,
            [],
            id="allow-missing-preds-ignores-pred-count",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            True,
            ["Expected 2 diagrams", "Number of true nodes files: 1"],
            id="allow-missing-preds-nodes-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            True,
            ["Expected 2 diagrams", "Number of true labels files: 1"],
            id="allow-missing-preds-labels-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [path_for("additional_texts", "diagram_1")],
            [path_for("flow", "diagram_1"), path_for("flow", "diagram_2")],
            True,
            ["Expected 2 diagrams", "Number of true additional texts files: 1"],
            id="allow-missing-preds-additional-texts-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1"), path_for("nodes", "diagram_2")],
            [path_for("labels", "diagram_1"), path_for("labels", "diagram_2")],
            [
                path_for("additional_texts", "diagram_1"),
                path_for("additional_texts", "diagram_2"),
            ],
            [path_for("flow", "diagram_1")],
            True,
            ["Expected 2 diagrams", "Number of true flow files: 1"],
            id="allow-missing-preds-flow-count-wrong",
        ),
        pytest.param(
            [path_for("pred", "diagram_1")],
            [path_for("nodes", "diagram_1")],
            [path_for("labels", "diagram_1")],
            [path_for("additional_texts", "diagram_1")],
            [path_for("flow", "diagram_1")],
            True,
            [
                "Expected 2 diagrams",
                "Number of true nodes files: 1",
                "Number of true labels files: 1",
                "Number of true additional texts files: 1",
                "Number of true flow files: 1",
            ],
            id="allow-missing-preds-all-true-counts-wrong",
        ),
    ],
)
def test_check_benchmark_files_match_raises_for_wrong_counts(
    pred_paths,
    nodes_paths,
    labels_paths,
    additional_texts_paths,
    flow_paths,
    allow_missing_pred_diagrams,
    expected_msg_parts,
):
    if expected_msg_parts == []:
        result = check_module.check_benchmark_files_match(
            pred_paths=pred_paths,
            nodes_paths=nodes_paths,
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
            allow_missing_pred_diagrams=allow_missing_pred_diagrams,
            expected_num_diagrams=2,
        )

        assert result is None
        return

    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_files_match(
            pred_paths=pred_paths,
            nodes_paths=nodes_paths,
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
            allow_missing_pred_diagrams=allow_missing_pred_diagrams,
            expected_num_diagrams=2,
        )

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg


@pytest.mark.parametrize(
    (
        "nodes_path",
        "labels_path",
        "additional_texts_path",
        "flow_path",
    ),
    [
        pytest.param(
            path_for("nodes", "diagram_1"),
            path_for("labels", "different"),
            path_for("additional_texts", "diagram_1"),
            path_for("flow", "diagram_1"),
            id="labels-stem-different",
        ),
        pytest.param(
            path_for("nodes", "diagram_1"),
            path_for("labels", "diagram_1"),
            path_for("additional_texts", "different"),
            path_for("flow", "diagram_1"),
            id="additional-texts-stem-different",
        ),
        pytest.param(
            path_for("nodes", "diagram_1"),
            path_for("labels", "diagram_1"),
            path_for("additional_texts", "diagram_1"),
            path_for("flow", "different"),
            id="flow-stem-different",
        ),
        pytest.param(
            path_for("nodes", "different"),
            path_for("labels", "diagram_1"),
            path_for("additional_texts", "diagram_1"),
            path_for("flow", "diagram_1"),
            id="nodes-stem-different",
        ),
        pytest.param(
            path_for("nodes", "diagram_1"),
            path_for("labels", "different_labels"),
            path_for("additional_texts", "different_additional_texts"),
            path_for("flow", "different_flow"),
            id="multiple-stems-different",
        ),
    ],
)
def test_check_benchmark_files_match_raises_if_true_component_stems_do_not_match(
    nodes_path,
    labels_path,
    additional_texts_path,
    flow_path,
):
    pred_path = path_for("pred", nodes_path.stem)

    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_files_match(
            pred_paths=[pred_path],
            nodes_paths=[nodes_path],
            labels_paths=[labels_path],
            additional_texts_paths=[additional_texts_path],
            flow_paths=[flow_path],
            expected_num_diagrams=1,
        )

    msg = str(exc_info.value)

    assert "should have the same stem" in msg
    assert str(nodes_path) in msg
    assert str(labels_path) in msg
    assert str(additional_texts_path) in msg
    assert str(flow_path) in msg


def test_check_benchmark_files_match_raises_if_pred_stem_does_not_match_true_nodes(
    tmp_path,
):
    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_files_match(
            pred_paths=[tmp_path / "pred" / "diagram_2.json"],
            nodes_paths=[tmp_path / "nodes" / "diagram_1.json"],
            labels_paths=[tmp_path / "labels" / "diagram_1.json"],
            additional_texts_paths=[tmp_path / "additional_texts" / "diagram_1.json"],
            flow_paths=[tmp_path / "flow" / "diagram_1.json"],
            expected_num_diagrams=1,
        )

    assert (
        "should have the same stem and parent directory as the true nodes file"
        in str(exc_info.value)
    )


def test_check_benchmark_files_match_raises_if_missing_pred_does_not_match_any_true_node(
    tmp_path,
):
    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_files_match(
            pred_paths=[tmp_path / "pred" / "diagram_2.json"],
            nodes_paths=[tmp_path / "nodes" / "diagram_1.json"],
            labels_paths=[tmp_path / "labels" / "diagram_1.json"],
            additional_texts_paths=[tmp_path / "additional_texts" / "diagram_1.json"],
            flow_paths=[tmp_path / "flow" / "diagram_1.json"],
            allow_missing_pred_diagrams=True,
            expected_num_diagrams=1,
        )

    assert (
        "should have the same stem and parent directory as one of the true nodes files"
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    ("pred_paths", "nodes_paths", "expected_msg"),
    [
        pytest.param(
            [
                path_for("pred", "diagram_1"),
                path_for("other_pred", "diagram_1"),
            ],
            [
                path_for("nodes", "diagram_1"),
                path_for("nodes", "diagram_2"),
            ],
            "Duplicate predicted files found.",
            id="duplicate-pred-stems",
        ),
        pytest.param(
            [
                path_for("pred", "diagram_1"),
            ],
            [
                path_for("nodes", "diagram_1"),
                path_for("other_nodes", "diagram_1"),
            ],
            "Duplicate true nodes files found.",
            id="duplicate-true-node-stems",
        ),
    ],
)
def test_check_benchmark_files_match_raises_for_duplicate_stems(
    pred_paths,
    nodes_paths,
    expected_msg,
):
    labels_paths = nodes_paths.copy()
    additional_texts_paths = nodes_paths.copy()
    flow_paths = nodes_paths.copy()

    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_files_match(
            pred_paths=pred_paths,
            nodes_paths=nodes_paths,
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
            allow_missing_pred_diagrams=True,
            expected_num_diagrams=2,
        )

    assert expected_msg in str(exc_info.value)


def test_check_benchmark_options_match_passes_for_matching_options_and_node_numbers(
    tmp_path,
):
    nodes_path, labels_path, additional_texts_path, flow_path = write_valid_true_files(
        tmp_path
    )

    result = check_module.check_benchmark_options_match(
        nodes_paths=[nodes_path],
        labels_paths=[labels_path],
        additional_texts_paths=[additional_texts_path],
        flow_paths=[flow_path],
    )

    assert result is None


@pytest.mark.parametrize(
    (
        "nodes_n_options",
        "labels_n_options",
        "additional_texts_n_options",
        "flow_n_options",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            1,
            2,
            1,
            1,
            [
                "The number of options",
                "Found 1 options in",
                "Found 2 options in",
            ],
            id="labels-option-count-different",
        ),
        pytest.param(
            1,
            1,
            2,
            1,
            [
                "The number of options",
                "Found 1 options in",
                "Found 2 options in",
            ],
            id="additional-texts-option-count-different",
        ),
        pytest.param(
            1,
            1,
            1,
            2,
            [
                "The number of options",
                "Found 1 options in",
                "Found 2 options in",
            ],
            id="flow-option-count-different",
        ),
        pytest.param(
            2,
            1,
            1,
            1,
            [
                "The number of options",
                "Found 2 options in",
                "Found 1 options in",
            ],
            id="nodes-option-count-different",
        ),
        pytest.param(
            1,
            2,
            3,
            4,
            [
                "The number of options",
                "Found 1 options in",
                "Found 2 options in",
                "Found 3 options in",
                "Found 4 options in",
            ],
            id="all-option-counts-different",
        ),
        pytest.param(
            0,
            1,
            1,
            1,
            [
                "The number of options",
                "Found 0 options in",
                "Found 1 options in",
            ],
            id="nodes-has-zero-options",
        ),
        pytest.param(
            1,
            0,
            1,
            1,
            [
                "The number of options",
                "Found 1 options in",
                "Found 0 options in",
            ],
            id="labels-has-zero-options",
        ),
        pytest.param(
            1,
            1,
            0,
            1,
            [
                "The number of options",
                "Found 1 options in",
                "Found 0 options in",
            ],
            id="additional-texts-has-zero-options",
        ),
        pytest.param(
            1,
            1,
            1,
            0,
            [
                "The number of options",
                "Found 1 options in",
                "Found 0 options in",
            ],
            id="flow-has-zero-options",
        ),
    ],
)
def test_check_benchmark_options_match_raises_if_option_counts_do_not_match(
    tmp_path,
    nodes_n_options,
    labels_n_options,
    additional_texts_n_options,
    flow_n_options,
    expected_msg_parts,
):
    nodes_path, labels_path, additional_texts_path, flow_path = write_valid_true_files(
        tmp_path
    )

    def make_nodes_options(n_options):
        return [
            {
                "nodes": [
                    {"node_number": 1, "text": f"Node 1 option {i}"},
                    {"node_number": 2, "text": f"Node 2 option {i}"},
                ]
            }
            for i in range(n_options)
        ]

    def make_labels_options(n_options):
        return [
            {
                "nodes": [
                    {"node_number": 1, "labels": [f"label 1 option {i}"]},
                    {"node_number": 2, "labels": [f"label 2 option {i}"]},
                ]
            }
            for i in range(n_options)
        ]

    def make_additional_texts_options(n_options):
        return [
            {
                "additional_texts": [
                    f"Figure option {i}",
                    f"Participant flow option {i}",
                ]
            }
            for i in range(n_options)
        ]

    def make_flow_options(n_options):
        return [
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2]},
                    {"node_number": 2, "points_to": [1]},
                ]
            }
            for i in range(n_options)
        ]

    write_json(nodes_path, {"options": make_nodes_options(nodes_n_options)})
    write_json(labels_path, {"options": make_labels_options(labels_n_options)})
    write_json(
        additional_texts_path,
        {"options": make_additional_texts_options(additional_texts_n_options)},
    )
    write_json(flow_path, {"options": make_flow_options(flow_n_options)})

    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_options_match(
            nodes_paths=[nodes_path],
            labels_paths=[labels_path],
            additional_texts_paths=[additional_texts_path],
            flow_paths=[flow_path],
        )

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg


@pytest.mark.parametrize(
    ("bad_labels", "bad_flow", "expected_msg_part"),
    [
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "labels": ["screening"]},
                            {"node_number": 3, "labels": ["wrong"]},
                        ]
                    }
                ]
            },
            make_flow(),
            "The node numbers in nodes, flows, and labels should match",
            id="labels-node-numbers-do-not-match",
        ),
        pytest.param(
            make_labels(),
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "points_to": [3]},
                            {"node_number": 3, "points_to": [1]},
                        ]
                    }
                ]
            },
            "The node numbers in nodes, flows, and labels should match",
            id="flow-node-numbers-do-not-match",
        ),
    ],
)
def test_check_benchmark_options_match_raises_if_node_numbers_do_not_match(
    tmp_path,
    bad_labels,
    bad_flow,
    expected_msg_part,
):
    nodes_path, labels_path, additional_texts_path, flow_path = write_valid_true_files(
        tmp_path
    )

    write_json(labels_path, bad_labels)
    write_json(flow_path, bad_flow)

    with pytest.raises(ValueError) as exc_info:
        check_module.check_benchmark_options_match(
            nodes_paths=[nodes_path],
            labels_paths=[labels_path],
            additional_texts_paths=[additional_texts_path],
            flow_paths=[flow_path],
        )

    assert expected_msg_part in str(exc_info.value)


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            [],
            TypeError,
            "The JSON file should be a dictionary.",
            id="json-not-dict",
        ),
        pytest.param(
            {"not_options": []},
            ValueError,
            "The JSON file should have 'options' as its only top level key.",
            id="wrong-top-level-key",
        ),
        pytest.param(
            {"options": []},
            ValueError,
            "The JSON file should have at least one option.",
            id="empty-options",
        ),
        pytest.param(
            {"options": ["not a dict"]},
            TypeError,
            "Each option should be a dictionary.",
            id="option-not-dict",
        ),
        pytest.param(
            {"options": "not a list"},
            TypeError,
            "The 'options' key should be a list.",
            id="options-not-list",
        ),
    ],
)
def test_check_true_diagram_component_generic_structure_raises_for_invalid_content(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "diagram.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_diagram_component_generic_structure(path)

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {"options": [{"not_nodes": []}]},
            ValueError,
            "Each option should have 'nodes' as its only key.",
            id="option-missing-nodes-key",
        ),
        pytest.param(
            {"options": [{"nodes": [{"text": "Node without number"}]}]},
            ValueError,
            "Each node should have a 'node_number' key.",
            id="node-missing-node-number",
        ),
        pytest.param(
            {"options": [{"nodes": [{"node_number": "1", "text": "Node 1"}]}]},
            TypeError,
            "The 'node_number' should be an integer.",
            id="node-number-not-int",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "text": "Node 1"},
                            {"node_number": 1, "text": "Node 1 duplicate"},
                        ]
                    }
                ]
            },
            ValueError,
            "The 'node_number' values should be unique.",
            id="duplicate-node-numbers",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "text": "Node 1"},
                            {"node_number": 3, "text": "Node 3"},
                        ]
                    }
                ]
            },
            ValueError,
            "The 'node_number' values should be sequential starting from 1.",
            id="non-sequential-node-numbers",
        ),
        pytest.param(
            {"options": [{"nodes": []}]},
            ValueError,
            "There should be at least one node.",
            id="empty-nodes",
        ),
    ],
)
def test_check_true_diagram_generic_nodes_list_structure_raises_for_invalid_content(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "diagram.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_diagram_generic_nodes_list_structure(path)

    assert expected_msg in str(exc_info.value)


def test_check_true_nodes_structure_passes_for_valid_nodes(tmp_path):
    path = tmp_path / "nodes" / "diagram_1.json"
    write_json(path, make_node_texts())

    result = check_module.check_true_nodes_structure([path])

    assert result is None


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {
                                "node_number": 1,
                                "text": "Node 1",
                                "labels": ["extra"],
                            }
                        ]
                    }
                ]
            },
            ValueError,
            "Each node should only have 'node_number' and 'text' as keys.",
            id="wrong-node-keys",
        ),
        pytest.param(
            {"options": [{"nodes": [{"node_number": 1, "text": 123}]}]},
            TypeError,
            "The 'text' for each node should be a string.",
            id="text-not-string",
        ),
    ],
)
def test_check_true_nodes_structure_raises_for_invalid_nodes(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "nodes" / "diagram_1.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_nodes_structure([path])

    assert expected_msg in str(exc_info.value)


def test_check_true_labels_structure_passes_for_valid_labels(tmp_path):
    path = tmp_path / "labels" / "diagram_1.json"
    write_json(path, make_labels())

    result = check_module.check_true_labels_structure([path])

    assert result is None


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {
                                "node_number": 1,
                                "labels": ["screening"],
                                "text": "extra",
                            }
                        ]
                    }
                ]
            },
            ValueError,
            "Each node should only have 'node_number' and 'labels' as keys.",
            id="wrong-node-keys",
        ),
        pytest.param(
            {"options": [{"nodes": [{"node_number": 1, "labels": "screening"}]}]},
            TypeError,
            "'labels' should be a list.",
            id="labels-not-list",
        ),
        pytest.param(
            {"options": [{"nodes": [{"node_number": 1, "labels": ["a", 2]}]}]},
            TypeError,
            "Each label should be a string.",
            id="label-not-string",
        ),
    ],
)
def test_check_true_labels_structure_raises_for_invalid_labels(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "labels" / "diagram_1.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_labels_structure([path])

    assert expected_msg in str(exc_info.value)


def test_check_true_additional_texts_structure_passes_for_valid_additional_texts(
    tmp_path,
):
    path = tmp_path / "additional_texts" / "diagram_1.json"
    write_json(path, make_additional_texts())

    result = check_module.check_true_additional_texts_structure([path])

    assert result is None


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {"options": [{"additional_texts": ["Figure 1"], "nodes": []}]},
            ValueError,
            "Each 'additional texts' JSON file should have 'additional_texts' as its only key.",
            id="wrong-option-keys",
        ),
        pytest.param(
            {"options": [{"additional_texts": "Figure 1"}]},
            TypeError,
            "The 'additional_texts' key should be a list.",
            id="additional-texts-not-list",
        ),
        pytest.param(
            {"options": [{"additional_texts": ["Figure 1", 2]}]},
            TypeError,
            "Each item in 'additional_texts' should be a string.",
            id="additional-text-not-string",
        ),
    ],
)
def test_check_true_additional_texts_structure_raises_for_invalid_additional_texts(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "additional_texts" / "diagram_1.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_additional_texts_structure([path])

    assert expected_msg in str(exc_info.value)


def test_check_true_flow_structure_passes_for_valid_flow(tmp_path):
    path = tmp_path / "flow" / "diagram_1.json"
    write_json(path, make_flow())

    result = check_module.check_true_flow_structure([path])

    assert result is None


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {
                                "node_number": 1,
                                "points_to": [2],
                                "text": "extra",
                            },
                            {"node_number": 2, "points_to": [1]},
                        ]
                    }
                ]
            },
            ValueError,
            "Each node should only have 'node_number' and 'points_to' as keys.",
            id="wrong-node-keys",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "points_to": 2},
                            {"node_number": 2, "points_to": [1]},
                        ]
                    }
                ]
            },
            TypeError,
            "The 'points_to' key should be a list.",
            id="points-to-not-list",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "points_to": [2, "3"]},
                            {"node_number": 2, "points_to": [1]},
                            {"node_number": 3, "points_to": [1]},
                        ]
                    }
                ]
            },
            TypeError,
            "Each item in 'points_to' should be an integer.",
            id="points-to-item-not-int",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "points_to": [2, 2]},
                            {"node_number": 2, "points_to": [1]},
                        ]
                    }
                ]
            },
            ValueError,
            "There should be no duplicate edges in 'points_to'.",
            id="duplicate-edges",
        ),
        pytest.param(
            {
                "options": [
                    {
                        "nodes": [
                            {"node_number": 1, "points_to": []},
                            {"node_number": 2, "points_to": []},
                        ]
                    }
                ]
            },
            ValueError,
            "Every node should point to or be pointed to by another node.",
            id="isolated-nodes",
        ),
    ],
)
def test_check_true_flow_structure_raises_for_invalid_flow(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    path = tmp_path / "flow" / "diagram_1.json"
    write_json(path, content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_true_flow_structure([path])

    assert expected_msg in str(exc_info.value)


def make_valid_pred_content(
    benchmark_nodes: bool,
    benchmark_labels: bool,
    benchmark_additional_texts: bool,
    benchmark_flow: bool,
) -> dict:
    content = {}

    if benchmark_nodes or benchmark_labels or benchmark_flow:
        nodes = []
        for node_number in [1, 2]:
            node = {"node_number": node_number}

            if benchmark_nodes:
                node["text"] = f"Node {node_number}"

            if benchmark_labels:
                node["labels"] = [f"label {node_number}", "shared label"]

            if benchmark_flow:
                node["points_to"] = [2] if node_number == 1 else []

            nodes.append(node)

        content["nodes"] = nodes

    if benchmark_additional_texts:
        content["additional_texts"] = ["Figure 1", "Trial flow"]

    return content


@pytest.mark.parametrize(
    (
        "benchmark_nodes",
        "benchmark_labels",
        "benchmark_additional_texts",
        "benchmark_flow",
    ),
    [
        pytest.param(True, False, False, False, id="nodes-only"),
        pytest.param(False, True, False, False, id="labels-only"),
        pytest.param(False, False, True, False, id="additional-texts-only"),
        pytest.param(False, False, False, True, id="flow-only"),
        pytest.param(True, True, False, False, id="nodes-and-labels"),
        pytest.param(True, False, True, False, id="nodes-and-additional-texts"),
        pytest.param(True, False, False, True, id="nodes-and-flow"),
        pytest.param(False, True, True, False, id="labels-and-additional-texts"),
        pytest.param(False, True, False, True, id="labels-and-flow"),
        pytest.param(False, False, True, True, id="additional-texts-and-flow"),
        pytest.param(True, True, True, False, id="nodes-labels-additional-texts"),
        pytest.param(True, True, False, True, id="nodes-labels-flow"),
        pytest.param(True, False, True, True, id="nodes-additional-texts-flow"),
        pytest.param(False, True, True, True, id="labels-additional-texts-flow"),
        pytest.param(True, True, True, True, id="all-benchmarks"),
    ],
)
def test_check_pred_structure_passes_for_valid_pred_structures(
    tmp_path,
    benchmark_nodes,
    benchmark_labels,
    benchmark_additional_texts,
    benchmark_flow,
):
    pred_diagrams_dir = tmp_path / "pred"
    content = make_valid_pred_content(
        benchmark_nodes=benchmark_nodes,
        benchmark_labels=benchmark_labels,
        benchmark_additional_texts=benchmark_additional_texts,
        benchmark_flow=benchmark_flow,
    )

    write_json(pred_diagrams_dir / "diagram_1.json", content)

    result = check_module.check_pred_structure(
        pred_diagrams_dir=pred_diagrams_dir,
    )

    assert result == check_module.PredDiagramStructure(
        node_text=benchmark_nodes,
        labels=benchmark_labels,
        additional_texts=benchmark_additional_texts,
        flow=benchmark_flow,
    )


def test_check_pred_structure_ignores_nested_json_files(tmp_path):
    pred_diagrams_dir = tmp_path / "pred"

    write_json(
        pred_diagrams_dir / "diagram_1.json",
        make_valid_pred_content(
            benchmark_nodes=True,
            benchmark_labels=False,
            benchmark_additional_texts=False,
            benchmark_flow=False,
        ),
    )
    write_json(
        pred_diagrams_dir / "nested" / "bad_nested.json",
        {"wrong_top_level_key": []},
    )

    result = check_module.check_pred_structure(
        pred_diagrams_dir=pred_diagrams_dir,
    )

    assert result == check_module.PredDiagramStructure(
        node_text=True,
        labels=False,
        additional_texts=False,
        flow=False,
    )


@pytest.mark.parametrize(
    ("content", "expected_error", "expected_msg"),
    [
        pytest.param(
            {"wrong_top_level_key": []},
            ValueError,
            "The JSON file has invalid top-level keys.",
            id="wrong-top-level-key",
        ),
        pytest.param(
            {"nodes": [], "additional_texts": [], "extra": []},
            ValueError,
            "The JSON file has invalid top-level keys.",
            id="extra-top-level-key",
        ),
        pytest.param(
            {"nodes": "not a list"},
            TypeError,
            "The 'nodes' key should be a list.",
            id="nodes-not-list",
        ),
        pytest.param(
            {"nodes": [{"node_number": "1", "text": "Node 1"}]},
            TypeError,
            "The 'node_number' should be an integer.",
            id="node-number-not-int",
        ),
        pytest.param(
            {"nodes": [{"node_number": 1, "text": 123}]},
            TypeError,
            "The 'text' should be a string.",
            id="text-not-string",
        ),
        pytest.param(
            {"nodes": [{"node_number": 1, "labels": "screening"}]},
            TypeError,
            "The 'labels' key should be a list.",
            id="labels-not-list",
        ),
        pytest.param(
            {"nodes": [{"node_number": 1, "labels": [123]}]},
            TypeError,
            "Each label should be a string.",
            id="label-not-string",
        ),
        pytest.param(
            {"nodes": [{"node_number": 1, "points_to": "2"}]},
            TypeError,
            "The 'points_to' key should be a list.",
            id="flow-points-to-not-list",
        ),
        pytest.param(
            {"nodes": [{"node_number": 1, "points_to": ["2"]}]},
            TypeError,
            "Each item in 'points_to' should be an integer.",
            id="flow-points-to-item-not-int",
        ),
        pytest.param(
            {"additional_texts": "Figure 1"},
            TypeError,
            "The 'additional_texts' key should be a list.",
            id="additional-texts-not-list",
        ),
        pytest.param(
            {"additional_texts": ["Figure 1", 123]},
            TypeError,
            "Each item in 'additional_texts' should be a string.",
            id="additional-text-not-string",
        ),
        pytest.param(
            {
                "nodes": [
                    {"node_number": 1, "points_to": [3]},
                    {"node_number": 2, "points_to": []},
                ]
            },
            ValueError,
            "For pred diagrams, each item in 'points_to' should be a valid node_number.",
            id="flow-invalid-pointed-to-node",
        ),
        pytest.param(
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                    {"node_number": 1, "text": "Duplicate"},
                ]
            },
            ValueError,
            "The 'node_number' values should be unique.",
            id="duplicate-node-numbers-nodes-only",
        ),
        pytest.param(
            {
                "nodes": [
                    {"node_number": 1, "labels": ["screening"]},
                    {"node_number": 1, "labels": ["duplicate"]},
                ]
            },
            ValueError,
            "The 'node_number' values should be unique.",
            id="duplicate-node-numbers-labels-only",
        ),
        pytest.param(
            {
                "nodes": [
                    {"node_number": 1, "points_to": []},
                    {"node_number": 1, "points_to": []},
                ]
            },
            ValueError,
            "The 'node_number' values should be unique.",
            id="duplicate-node-numbers-flow-only",
        ),
        pytest.param(
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["screening"],
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Node 2",
                        "labels": ["randomisation"],
                        "points_to": [],
                    },
                ],
                "additional_texts": "Figure 1",
            },
            TypeError,
            "The 'additional_texts' key should be a list.",
            id="all-benchmarks-additional-texts-not-list",
        ),
        pytest.param(
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["screening"],
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Node 2",
                        "labels": ["randomisation"],
                        "points_to": [],
                    },
                ],
                "additional_texts": ["Figure 1", 123],
            },
            TypeError,
            "Each item in 'additional_texts' should be a string.",
            id="all-benchmarks-additional-text-item-not-string",
        ),
        pytest.param(
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["screening"],
                        "points_to": [3],
                    },
                    {
                        "node_number": 2,
                        "text": "Node 2",
                        "labels": ["randomisation"],
                        "points_to": [],
                    },
                ],
                "additional_texts": ["Figure 1"],
            },
            ValueError,
            "For pred diagrams, each item in 'points_to' should be a valid node_number.",
            id="all-benchmarks-invalid-pointed-to-node",
        ),
    ],
)
def test_check_pred_structure_raises_for_invalid_pred_structures(
    tmp_path,
    content,
    expected_error,
    expected_msg,
):
    pred_diagrams_dir = tmp_path / "pred"
    write_json(pred_diagrams_dir / "diagram_1.json", content)

    with pytest.raises(expected_error) as exc_info:
        check_module.check_pred_structure(
            pred_diagrams_dir=pred_diagrams_dir,
        )

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("node_fields", "missing_key"),
    [
        pytest.param({"text": "Node"}, "text", id="node-text"),
        pytest.param({"labels": ["screening"]}, "labels", id="labels"),
        pytest.param({"points_to": []}, "points_to", id="flow"),
        pytest.param(
            {"text": "Node", "labels": ["screening"]},
            "text",
            id="node-text-and-labels-missing-text",
        ),
        pytest.param(
            {"text": "Node", "labels": ["screening"]},
            "labels",
            id="node-text-and-labels-missing-labels",
        ),
        pytest.param(
            {"text": "Node", "points_to": []},
            "text",
            id="node-text-and-flow-missing-text",
        ),
        pytest.param(
            {"text": "Node", "points_to": []},
            "points_to",
            id="node-text-and-flow-missing-flow",
        ),
        pytest.param(
            {"labels": ["screening"], "points_to": []},
            "labels",
            id="labels-and-flow-missing-labels",
        ),
        pytest.param(
            {"labels": ["screening"], "points_to": []},
            "points_to",
            id="labels-and-flow-missing-flow",
        ),
        pytest.param(
            {"text": "Node", "labels": ["screening"], "points_to": []},
            "text",
            id="node-text-labels-and-flow-missing-text",
        ),
        pytest.param(
            {"text": "Node", "labels": ["screening"], "points_to": []},
            "labels",
            id="node-text-labels-and-flow-missing-labels",
        ),
        pytest.param(
            {"text": "Node", "labels": ["screening"], "points_to": []},
            "points_to",
            id="node-text-labels-and-flow-missing-flow",
        ),
    ],
)
def test_check_pred_structure_raises_if_later_node_has_different_keys(
    tmp_path,
    node_fields,
    missing_key,
):
    pred_diagrams_dir = tmp_path / "pred"
    first_node = {"node_number": 1, **node_fields}
    later_node = {"node_number": 2, **node_fields}
    later_node.pop(missing_key)
    write_json(
        pred_diagrams_dir / "diagram_1.json",
        {"nodes": [first_node, later_node]},
    )

    with pytest.raises(ValueError) as exc_info:
        check_module.check_pred_structure(pred_diagrams_dir=pred_diagrams_dir)

    assert "Node has invalid keys." in str(exc_info.value)


@pytest.mark.parametrize("missing_key", ["nodes", "additional_texts"])
def test_check_pred_structure_raises_if_later_file_has_different_top_level_keys(
    tmp_path,
    missing_key,
):
    pred_diagrams_dir = tmp_path / "pred"
    complete_content = make_valid_pred_content(
        benchmark_nodes=True,
        benchmark_labels=False,
        benchmark_additional_texts=True,
        benchmark_flow=False,
    )
    incomplete_content = {
        key: value for key, value in complete_content.items() if key != missing_key
    }
    write_json(pred_diagrams_dir / "diagram_1.json", complete_content)
    write_json(pred_diagrams_dir / "diagram_2.json", complete_content)
    write_json(pred_diagrams_dir / "diagram_3.json", incomplete_content)

    with pytest.raises(ValueError) as exc_info:
        check_module.check_pred_structure(pred_diagrams_dir=pred_diagrams_dir)

    assert "All predicted diagram files should have the same top-level keys." in str(
        exc_info.value
    )


def test_count_functions_in_this_module_counts_functions():
    result = check_module.count_functions_in_this_module()

    assert result == 12


def test_run_all_benchmark_true_data_checks_calls_all_base_checks(
    tmp_path,
    monkeypatch,
):
    pred_diagrams_dir = tmp_path / "pred"
    true_nodes_dir = tmp_path / "nodes"
    true_labels_dir = tmp_path / "labels"
    true_additional_texts_dir = tmp_path / "additional_texts"
    true_flow_dir = tmp_path / "flow"

    pred_path = write_valid_pred_file(tmp_path)
    pred_sources = check_module.validate_pred_diagram_sources(pred_diagrams_dir)
    nodes_path, labels_path, additional_texts_path, flow_path = write_valid_true_files(
        tmp_path
    )

    write_json(pred_diagrams_dir / "nested" / "ignored_pred.json", {})
    write_json(true_nodes_dir / "nested" / "ignored_nodes.json", {})
    write_json(true_labels_dir / "nested" / "ignored_labels.json", {})
    write_json(
        true_additional_texts_dir / "nested" / "ignored_additional_texts.json", {}
    )
    write_json(true_flow_dir / "nested" / "ignored_flow.json", {})

    for directory in (
        true_nodes_dir,
        true_labels_dir,
        true_additional_texts_dir,
        true_flow_dir,
    ):
        write_json(directory / "z_diagram.json", {})
        write_json(directory / "a_diagram.json", {})

    calls = []

    def fake_check_benchmark_files_match(**kwargs):
        calls.append(("check_benchmark_files_match", kwargs))

    def fake_check_benchmark_options_match(**kwargs):
        calls.append(("check_benchmark_options_match", kwargs))

    def fake_check_true_nodes_structure(**kwargs):
        calls.append(("check_true_nodes_structure", kwargs))

    def fake_check_true_labels_structure(**kwargs):
        calls.append(("check_true_labels_structure", kwargs))

    def fake_check_true_additional_texts_structure(**kwargs):
        calls.append(("check_true_additional_texts_structure", kwargs))

    def fake_check_true_flow_structure(**kwargs):
        calls.append(("check_true_flow_structure", kwargs))

    monkeypatch.setattr(
        check_module,
        "check_benchmark_files_match",
        fake_check_benchmark_files_match,
    )
    monkeypatch.setattr(
        check_module,
        "check_benchmark_options_match",
        fake_check_benchmark_options_match,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_nodes_structure",
        fake_check_true_nodes_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_labels_structure",
        fake_check_true_labels_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_additional_texts_structure",
        fake_check_true_additional_texts_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_flow_structure",
        fake_check_true_flow_structure,
    )

    monkeypatch.setattr(check_module, "count_functions_in_this_module", lambda: 12)

    result = check_module.run_all_benchmark_true_data_checks(
        pred_sources=pred_sources,
        true_nodes_dir=true_nodes_dir,
        true_labels_dir=true_labels_dir,
        true_additional_texts_dir=true_additional_texts_dir,
        true_flow_dir=true_flow_dir,
        allow_missing_pred_diagrams=False,
        expected_num_diagrams=1,
    )

    assert result is None

    assert [call_name for call_name, _ in calls] == [
        "check_benchmark_files_match",
        "check_benchmark_options_match",
        "check_true_nodes_structure",
        "check_true_labels_structure",
        "check_true_additional_texts_structure",
        "check_true_flow_structure",
    ]

    assert calls[0][1]["pred_paths"] == [pred_path]
    assert calls[0][1]["nodes_paths"] == sorted(
        [
            true_nodes_dir / "a_diagram.json",
            nodes_path,
            true_nodes_dir / "z_diagram.json",
        ]
    )
    assert calls[0][1]["labels_paths"] == sorted(
        [
            true_labels_dir / "a_diagram.json",
            labels_path,
            true_labels_dir / "z_diagram.json",
        ]
    )
    assert calls[0][1]["additional_texts_paths"] == sorted(
        [
            true_additional_texts_dir / "a_diagram.json",
            additional_texts_path,
            true_additional_texts_dir / "z_diagram.json",
        ]
    )
    assert calls[0][1]["flow_paths"] == sorted(
        [
            true_flow_dir / "a_diagram.json",
            flow_path,
            true_flow_dir / "z_diagram.json",
        ]
    )
    assert calls[0][1]["allow_missing_pred_diagrams"] is False
    assert calls[0][1]["expected_num_diagrams"] == 1


def test_run_all_benchmark_true_data_checks_raises_if_not_all_checks_are_run(
    tmp_path,
    monkeypatch,
):
    pred_diagrams_dir = tmp_path / "pred"
    true_nodes_dir = tmp_path / "nodes"
    true_labels_dir = tmp_path / "labels"
    true_additional_texts_dir = tmp_path / "additional_texts"
    true_flow_dir = tmp_path / "flow"

    write_valid_pred_file(tmp_path)
    pred_sources = check_module.validate_pred_diagram_sources(pred_diagrams_dir)
    write_valid_true_files(tmp_path)

    monkeypatch.setattr(
        check_module, "check_benchmark_files_match", lambda **kwargs: None
    )
    monkeypatch.setattr(
        check_module, "check_benchmark_options_match", lambda **kwargs: None
    )
    monkeypatch.setattr(
        check_module, "check_true_nodes_structure", lambda **kwargs: None
    )
    monkeypatch.setattr(
        check_module, "check_true_labels_structure", lambda **kwargs: None
    )
    monkeypatch.setattr(
        check_module,
        "check_true_additional_texts_structure",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        check_module, "check_true_flow_structure", lambda **kwargs: None
    )
    monkeypatch.setattr(check_module, "count_functions_in_this_module", lambda: 13)

    with pytest.raises(ValueError) as exc_info:
        check_module.run_all_benchmark_true_data_checks(
            pred_sources=pred_sources,
            true_nodes_dir=true_nodes_dir,
            true_labels_dir=true_labels_dir,
            true_additional_texts_dir=true_additional_texts_dir,
            true_flow_dir=true_flow_dir,
            allow_missing_pred_diagrams=False,
            expected_num_diagrams=1,
        )

    assert "Not all benchmark check functions are being run." in str(exc_info.value)


def test_check_true_nodes_structure_calls_generic_structure_checks_for_each_file(
    tmp_path,
    monkeypatch,
):
    paths = [
        tmp_path / "nodes" / "diagram_1.json",
        tmp_path / "nodes" / "diagram_2.json",
    ]

    for path in paths:
        write_json(path, make_node_texts())

    calls = {
        "component": [],
        "nodes_list": [],
    }

    def fake_check_true_diagram_component_generic_structure(true_diagram_path):
        calls["component"].append(true_diagram_path)

    def fake_check_true_diagram_generic_nodes_list_structure(true_diagram_path):
        calls["nodes_list"].append(true_diagram_path)

    monkeypatch.setattr(
        check_module,
        "check_true_diagram_component_generic_structure",
        fake_check_true_diagram_component_generic_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_diagram_generic_nodes_list_structure",
        fake_check_true_diagram_generic_nodes_list_structure,
    )

    result = check_module.check_true_nodes_structure(paths)

    assert result is None
    assert calls["component"] == paths
    assert calls["nodes_list"] == paths


def test_check_true_labels_structure_calls_generic_structure_checks_for_each_file(
    tmp_path,
    monkeypatch,
):
    paths = [
        tmp_path / "labels" / "diagram_1.json",
        tmp_path / "labels" / "diagram_2.json",
    ]

    for path in paths:
        write_json(path, make_labels())

    calls = {
        "component": [],
        "nodes_list": [],
    }

    def fake_check_true_diagram_component_generic_structure(true_diagram_path):
        calls["component"].append(true_diagram_path)

    def fake_check_true_diagram_generic_nodes_list_structure(true_diagram_path):
        calls["nodes_list"].append(true_diagram_path)

    monkeypatch.setattr(
        check_module,
        "check_true_diagram_component_generic_structure",
        fake_check_true_diagram_component_generic_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_diagram_generic_nodes_list_structure",
        fake_check_true_diagram_generic_nodes_list_structure,
    )

    result = check_module.check_true_labels_structure(paths)

    assert result is None
    assert calls["component"] == paths
    assert calls["nodes_list"] == paths


def test_check_true_additional_texts_structure_calls_component_generic_check_for_each_file(
    tmp_path,
    monkeypatch,
):
    paths = [
        tmp_path / "additional_texts" / "diagram_1.json",
        tmp_path / "additional_texts" / "diagram_2.json",
    ]

    for path in paths:
        write_json(path, make_additional_texts())

    calls = {
        "component": [],
        "nodes_list": [],
    }

    def fake_check_true_diagram_component_generic_structure(true_diagram_path):
        calls["component"].append(true_diagram_path)

    def fake_check_true_diagram_generic_nodes_list_structure(true_diagram_path):
        calls["nodes_list"].append(true_diagram_path)

    monkeypatch.setattr(
        check_module,
        "check_true_diagram_component_generic_structure",
        fake_check_true_diagram_component_generic_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_diagram_generic_nodes_list_structure",
        fake_check_true_diagram_generic_nodes_list_structure,
    )

    result = check_module.check_true_additional_texts_structure(paths)

    assert result is None
    assert calls["component"] == paths
    assert calls["nodes_list"] == []


def test_check_true_flow_structure_calls_generic_structure_checks_for_each_file(
    tmp_path,
    monkeypatch,
):
    paths = [
        tmp_path / "flow" / "diagram_1.json",
        tmp_path / "flow" / "diagram_2.json",
    ]

    for path in paths:
        write_json(path, make_flow())

    calls = {
        "component": [],
        "nodes_list": [],
    }

    def fake_check_true_diagram_component_generic_structure(true_diagram_path):
        calls["component"].append(true_diagram_path)

    def fake_check_true_diagram_generic_nodes_list_structure(true_diagram_path):
        calls["nodes_list"].append(true_diagram_path)

    monkeypatch.setattr(
        check_module,
        "check_true_diagram_component_generic_structure",
        fake_check_true_diagram_component_generic_structure,
    )
    monkeypatch.setattr(
        check_module,
        "check_true_diagram_generic_nodes_list_structure",
        fake_check_true_diagram_generic_nodes_list_structure,
    )

    result = check_module.check_true_flow_structure(paths)

    assert result is None
    assert calls["component"] == paths
    assert calls["nodes_list"] == paths
