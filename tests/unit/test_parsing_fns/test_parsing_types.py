import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import flowde.parsing_fns.parsing_types as parsing_types_module


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def write_nodes_json(path: Path, nodes: list[dict]) -> None:
    write_json(path, {"nodes": nodes})


def write_additional_texts_json(path: Path, additional_texts: list[str]) -> None:
    write_json(path, {"additional_texts": additional_texts})


def make_node_texts() -> list[dict]:
    return [
        {"node_number": 1, "text": "Assessed for eligibility"},
        {"node_number": 2, "text": "Randomised"},
    ]


def make_labels() -> list[dict]:
    return [
        {"node_number": 1, "labels": ["screening", "eligibility"]},
        {"node_number": 2, "labels": ["randomisation", "allocation"]},
    ]


def make_flow() -> list[dict]:
    return [
        {"node_number": 1, "points_to": [2]},
        {"node_number": 2, "points_to": []},
    ]


def test_build_partial_flowchart_schema_defaults_to_full_schema():
    Schema = parsing_types_module.build_partial_flowchart_schema()

    flowchart = Schema.model_validate(
        {
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
                    "points_to": [],
                },
            ],
            "additional_texts": ["Figure 1", "Trial flow"],
        }
    )

    assert flowchart.model_dump() == {
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
                "points_to": [],
            },
        ],
        "additional_texts": ["Figure 1", "Trial flow"],
    }


@pytest.mark.parametrize(
    ("parts_to_parse", "valid_content", "invalid_content"),
    [
        pytest.param(
            {"node_text"},
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["extra"],
                    },
                ],
            },
            id="node-text-extra-labels",
        ),
        pytest.param(
            {"node_text"},
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "points_to": [2],
                    },
                ],
            },
            id="node-text-extra-flow",
        ),
        pytest.param(
            {"node_text"},
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                ],
            },
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                ],
                "additional_texts": ["extra"],
            },
            id="node-text-extra-additional-texts",
        ),
        pytest.param(
            {"labels"},
            {
                "nodes": [
                    {"node_number": 1, "labels": ["included", "screened"]},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["included", "screened"],
                        "text": "extra",
                    },
                ],
            },
            id="labels-extra-text",
        ),
        pytest.param(
            {"labels"},
            {
                "nodes": [
                    {"node_number": 1, "labels": ["included", "screened"]},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
            },
            id="labels-extra-flow",
        ),
        pytest.param(
            {"flow"},
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2, 3]},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "points_to": [2, 3],
                        "text": "extra",
                    },
                ],
            },
            id="flow-extra-text",
        ),
        pytest.param(
            {"flow"},
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2, 3]},
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "points_to": [2, 3],
                        "labels": ["extra"],
                    },
                ],
            },
            id="flow-extra-labels",
        ),
        pytest.param(
            {"additional_texts"},
            {
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "additional_texts": ["Some note", "Another note"],
                "nodes": [{"node_number": 1, "text": "extra"}],
            },
            id="additional-text-extra-nodes",
        ),
        pytest.param(
            {"node_text", "labels"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                    },
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
            },
            id="node-text-labels-extra-flow",
        ),
        pytest.param(
            {"node_text", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "points_to": [2],
                    },
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "points_to": [2],
                        "labels": ["extra"],
                    },
                ],
            },
            id="node-text-flow-extra-labels",
        ),
        pytest.param(
            {"labels", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["included", "screened"],
                        "points_to": [2],
                        "text": "extra",
                    },
                ],
            },
            id="labels-flow-extra-text",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
                "additional_texts": ["extra"],
            },
            id="node-text-labels-flow-extra-additional-texts",
        ),
        pytest.param(
            {"node_text", "additional_texts"},
            {
                "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["extra"],
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            id="node-text-additional-text-extra-labels",
        ),
        pytest.param(
            {"labels", "additional_texts"},
            {
                "nodes": [
                    {"node_number": 1, "labels": ["included", "screened"]},
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            id="labels-additional-text-extra-flow",
        ),
        pytest.param(
            {"flow", "additional_texts"},
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2]},
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "points_to": [2],
                        "text": "extra",
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            id="flow-additional-text-extra-text",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                        "unexpected": "extra",
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            id="full-schema-extra-node-field",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
            },
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    },
                ],
                "additional_texts": ["Some note", "Another note"],
                "unexpected": "extra",
            },
            id="full-schema-extra-flowchart-field",
        ),
    ],
)
def test_build_partial_flowchart_schema_forbids_extra_fields(
    parts_to_parse,
    valid_content,
    invalid_content,
):
    Schema = parsing_types_module.build_partial_flowchart_schema(parts_to_parse)

    assert Schema.model_validate(valid_content).model_dump() == valid_content

    with pytest.raises(ValidationError):
        Schema.model_validate(invalid_content)


@pytest.mark.parametrize(
    (
        "include_labels",
        "include_flow",
        "include_additional_texts",
        "expected",
    ),
    [
        pytest.param(
            False,
            False,
            False,
            {
                "nodes": [
                    {"node_number": 1, "text": "Assessed for eligibility"},
                    {"node_number": 2, "text": "Randomised"},
                ],
            },
            id="node-text-only",
        ),
        pytest.param(
            True,
            False,
            False,
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "labels": ["screening", "eligibility"],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "labels": ["randomisation", "allocation"],
                    },
                ],
            },
            id="node-text-and-labels",
        ),
        pytest.param(
            False,
            True,
            False,
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "points_to": [],
                    },
                ],
            },
            id="node-text-and-flow",
        ),
        pytest.param(
            False,
            False,
            True,
            {
                "nodes": [
                    {"node_number": 1, "text": "Assessed for eligibility"},
                    {"node_number": 2, "text": "Randomised"},
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="node-text-and-additional-texts",
        ),
        pytest.param(
            True,
            True,
            False,
            {
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
                        "points_to": [],
                    },
                ],
            },
            id="node-text-labels-and-flow",
        ),
        pytest.param(
            True,
            False,
            True,
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "labels": ["screening", "eligibility"],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "labels": ["randomisation", "allocation"],
                    },
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="node-text-labels-and-additional-texts",
        ),
        pytest.param(
            False,
            True,
            True,
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "points_to": [],
                    },
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="node-text-flow-and-additional-texts",
        ),
        pytest.param(
            True,
            True,
            True,
            {
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
                        "points_to": [],
                    },
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="all-parts",
        ),
    ],
)
def test_build_a_partial_flowchart_with_different_part_combinations(
    tmp_path,
    include_labels,
    include_flow,
    include_additional_texts,
    expected,
):
    nodes_path = tmp_path / "nodes" / "diagram_1.json"
    labels_path = tmp_path / "labels" / "diagram_1.json"
    flow_path = tmp_path / "flow" / "diagram_1.json"
    additional_texts_path = tmp_path / "additional_texts" / "diagram_1.json"

    write_nodes_json(nodes_path, make_node_texts())

    if include_labels:
        write_nodes_json(labels_path, make_labels())
    else:
        labels_path = None

    if include_flow:
        write_nodes_json(flow_path, make_flow())
    else:
        flow_path = None

    if include_additional_texts:
        write_additional_texts_json(
            additional_texts_path,
            ["Figure 1", "Trial flow"],
        )
    else:
        additional_texts_path = None

    flowchart = parsing_types_module.build_a_partial_flowchart(
        nodes_path=nodes_path,
        labels_path=labels_path,
        additional_texts_path=additional_texts_path,
        flow_path=flow_path,
    )

    assert flowchart.model_dump() == expected


def test_build_a_partial_flowchart_sorts_node_parts_before_joining(tmp_path):
    nodes_path = tmp_path / "nodes" / "diagram_1.json"
    labels_path = tmp_path / "labels" / "diagram_1.json"
    flow_path = tmp_path / "flow" / "diagram_1.json"
    additional_texts_path = tmp_path / "additional_texts" / "diagram_1.json"

    write_nodes_json(
        nodes_path,
        [
            {"node_number": 2, "text": "Randomised"},
            {"node_number": 1, "text": "Assessed for eligibility"},
        ],
    )
    write_nodes_json(
        labels_path,
        [
            {"node_number": 2, "labels": ["randomisation", "allocation"]},
            {"node_number": 1, "labels": ["screening", "eligibility"]},
        ],
    )
    write_nodes_json(
        flow_path,
        [
            {"node_number": 2, "points_to": []},
            {"node_number": 1, "points_to": [2]},
        ],
    )
    write_additional_texts_json(
        additional_texts_path,
        ["Figure 1", "Trial flow"],
    )

    flowchart = parsing_types_module.build_a_partial_flowchart(
        nodes_path=nodes_path,
        labels_path=labels_path,
        flow_path=flow_path,
        additional_texts_path=additional_texts_path,
    )

    assert flowchart.model_dump() == {
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
                "points_to": [],
            },
        ],
        "additional_texts": ["Figure 1", "Trial flow"],
    }


def test_build_a_partial_flowchart_raises_if_nodes_path_is_none():
    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_a_partial_flowchart(nodes_path=None)

    assert str(exc_info.value) == (
        "nodes_path must be provided to build a partial flowchart, as node numbers "
        "are needed to join the different parts together."
    )


def test_build_a_partial_flowchart_raises_if_path_stems_do_not_match(tmp_path):
    nodes_path = tmp_path / "nodes" / "diagram_1.json"
    labels_path = tmp_path / "labels" / "diagram_2.json"

    write_nodes_json(nodes_path, make_node_texts())
    write_nodes_json(labels_path, make_labels())

    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_a_partial_flowchart(
            nodes_path=nodes_path,
            labels_path=labels_path,
        )

    msg = str(exc_info.value)

    assert "Mismatched stems at index 0" in msg
    assert "diagram_1" in msg
    assert "diagram_2" in msg


def test_build_a_partial_flowchart_raises_if_node_numbers_are_not_consecutive(
    tmp_path,
):
    nodes_path = tmp_path / "diagram_1.json"

    write_nodes_json(
        nodes_path,
        [
            {"node_number": 1, "text": "Node 1"},
            {"node_number": 3, "text": "Node 3"},
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_a_partial_flowchart(nodes_path=nodes_path)

    assert str(exc_info.value) == (
        "Node numbers must be consecutive integers starting from 1. "
        "Found node numbers [1, 3]."
    )


def test_build_a_partial_flowchart_raises_if_no_nodes_are_provided(tmp_path):
    nodes_path = tmp_path / "diagram_1.json"

    write_nodes_json(nodes_path, [])

    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_a_partial_flowchart(nodes_path=nodes_path)

    assert str(exc_info.value) == (
        "At least one node must be provided to build a flowchart."
    )


@pytest.mark.parametrize(
    (
        "labels_nodes",
        "flow_nodes",
        "expected_msg",
    ),
    [
        pytest.param(
            [
                {"node_number": 1, "labels": ["label 1", "included"]},
                {"node_number": 3, "labels": ["label 3", "excluded"]},
            ],
            None,
            (
                "All provided node-based parts must have the same node numbers in the same "
                "order. Expected node numbers [1, 2] but got [1, 3]."
            ),
            id="labels-wrong-node-number",
        ),
        pytest.param(
            None,
            [
                {"node_number": 1, "points_to": [2]},
                {"node_number": 3, "points_to": []},
            ],
            (
                "All provided node-based parts must have the same node numbers in the same "
                "order. Expected node numbers [1, 2] but got [1, 3]."
            ),
            id="flow-wrong-node-number",
        ),
        pytest.param(
            [
                {"node_number": 1, "labels": ["label 1", "included"]},
                {"node_number": 2, "labels": ["label 2", "randomised"]},
            ],
            [
                {"node_number": 1, "points_to": [2]},
                {"node_number": 3, "points_to": []},
            ],
            (
                "All provided node-based parts must have the same node numbers in the same "
                "order. Expected node numbers [1, 2] but got [1, 3]."
            ),
            id="labels-correct-flow-wrong-node-number",
        ),
        pytest.param(
            [
                {"node_number": 1, "labels": ["label 1", "included"]},
                {"node_number": 3, "labels": ["label 3", "excluded"]},
            ],
            [
                {"node_number": 1, "points_to": [2]},
                {"node_number": 2, "points_to": []},
            ],
            (
                "All provided node-based parts must have the same node numbers in the same "
                "order. Expected node numbers [1, 2] but got [1, 3]."
            ),
            id="labels-wrong-flow-correct-node-number",
        ),
    ],
)
def test_build_a_partial_flowchart_raises_if_node_parts_have_different_node_numbers(
    tmp_path,
    labels_nodes,
    flow_nodes,
    expected_msg,
):
    nodes_path = tmp_path / "nodes" / "diagram_1.json"
    labels_path = tmp_path / "labels" / "diagram_1.json"
    flow_path = tmp_path / "flow" / "diagram_1.json"

    write_nodes_json(
        nodes_path,
        [
            {"node_number": 1, "text": "Node 1"},
            {"node_number": 2, "text": "Node 2"},
        ],
    )

    if labels_nodes is not None:
        write_nodes_json(labels_path, labels_nodes)
    else:
        labels_path = None

    if flow_nodes is not None:
        write_nodes_json(flow_path, flow_nodes)
    else:
        flow_path = None

    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_a_partial_flowchart(
            nodes_path=nodes_path,
            labels_path=labels_path,
            flow_path=flow_path,
        )

    assert str(exc_info.value) == expected_msg


@pytest.mark.parametrize(
    (
        "include_labels",
        "include_flow",
        "include_additional_texts",
        "expected",
    ),
    [
        pytest.param(
            False,
            False,
            False,
            [
                {
                    "nodes": [
                        {"node_number": 1, "text": "Assessed for eligibility"},
                        {"node_number": 2, "text": "Randomised"},
                    ],
                },
                {
                    "nodes": [
                        {"node_number": 1, "text": "Invited"},
                        {"node_number": 2, "text": "Completed"},
                    ],
                },
            ],
            id="node-text-only",
        ),
        pytest.param(
            True,
            False,
            False,
            [
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Assessed for eligibility",
                            "labels": ["screening", "eligibility"],
                        },
                        {
                            "node_number": 2,
                            "text": "Randomised",
                            "labels": ["randomisation", "allocation"],
                        },
                    ],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "labels": ["invitation", "approached"],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "labels": ["completion", "follow-up"],
                        },
                    ],
                },
            ],
            id="node-text-and-labels",
        ),
        pytest.param(
            False,
            True,
            False,
            [
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Assessed for eligibility",
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Randomised",
                            "points_to": [],
                        },
                    ],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "points_to": [],
                        },
                    ],
                },
            ],
            id="node-text-and-flow",
        ),
        pytest.param(
            False,
            False,
            True,
            [
                {
                    "nodes": [
                        {"node_number": 1, "text": "Assessed for eligibility"},
                        {"node_number": 2, "text": "Randomised"},
                    ],
                    "additional_texts": ["Figure 1", "Trial flow"],
                },
                {
                    "nodes": [
                        {"node_number": 1, "text": "Invited"},
                        {"node_number": 2, "text": "Completed"},
                    ],
                    "additional_texts": ["Figure 2", "Participant flow"],
                },
            ],
            id="node-text-and-additional-texts",
        ),
        pytest.param(
            True,
            True,
            False,
            [
                {
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
                            "points_to": [],
                        },
                    ],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "labels": ["invitation", "approached"],
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "labels": ["completion", "follow-up"],
                            "points_to": [],
                        },
                    ],
                },
            ],
            id="node-text-labels-and-flow",
        ),
        pytest.param(
            True,
            False,
            True,
            [
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Assessed for eligibility",
                            "labels": ["screening", "eligibility"],
                        },
                        {
                            "node_number": 2,
                            "text": "Randomised",
                            "labels": ["randomisation", "allocation"],
                        },
                    ],
                    "additional_texts": ["Figure 1", "Trial flow"],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "labels": ["invitation", "approached"],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "labels": ["completion", "follow-up"],
                        },
                    ],
                    "additional_texts": ["Figure 2", "Participant flow"],
                },
            ],
            id="node-text-labels-and-additional-texts",
        ),
        pytest.param(
            False,
            True,
            True,
            [
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Assessed for eligibility",
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Randomised",
                            "points_to": [],
                        },
                    ],
                    "additional_texts": ["Figure 1", "Trial flow"],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "points_to": [],
                        },
                    ],
                    "additional_texts": ["Figure 2", "Participant flow"],
                },
            ],
            id="node-text-flow-and-additional-texts",
        ),
        pytest.param(
            True,
            True,
            True,
            [
                {
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
                            "points_to": [],
                        },
                    ],
                    "additional_texts": ["Figure 1", "Trial flow"],
                },
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Invited",
                            "labels": ["invitation", "approached"],
                            "points_to": [2],
                        },
                        {
                            "node_number": 2,
                            "text": "Completed",
                            "labels": ["completion", "follow-up"],
                            "points_to": [],
                        },
                    ],
                    "additional_texts": ["Figure 2", "Participant flow"],
                },
            ],
            id="all-parts",
        ),
    ],
)
def test_build_partial_flowcharts_builds_multiple_partial_flowcharts_with_different_part_combinations(
    tmp_path,
    include_labels,
    include_flow,
    include_additional_texts,
    expected,
):
    nodes_path_1 = tmp_path / "nodes" / "diagram_1.json"
    nodes_path_2 = tmp_path / "nodes" / "diagram_2.json"

    labels_path_1 = tmp_path / "labels" / "diagram_1.json"
    labels_path_2 = tmp_path / "labels" / "diagram_2.json"

    flow_path_1 = tmp_path / "flow" / "diagram_1.json"
    flow_path_2 = tmp_path / "flow" / "diagram_2.json"

    additional_texts_path_1 = tmp_path / "additional_texts" / "diagram_1.json"
    additional_texts_path_2 = tmp_path / "additional_texts" / "diagram_2.json"

    write_nodes_json(nodes_path_1, make_node_texts())
    write_nodes_json(
        nodes_path_2,
        [
            {"node_number": 1, "text": "Invited"},
            {"node_number": 2, "text": "Completed"},
        ],
    )

    if include_labels:
        write_nodes_json(labels_path_1, make_labels())
        write_nodes_json(
            labels_path_2,
            [
                {"node_number": 1, "labels": ["invitation", "approached"]},
                {"node_number": 2, "labels": ["completion", "follow-up"]},
            ],
        )
        labels_paths = [labels_path_1, labels_path_2]
    else:
        labels_paths = None

    if include_flow:
        write_nodes_json(flow_path_1, make_flow())
        write_nodes_json(
            flow_path_2,
            [
                {"node_number": 1, "points_to": [2]},
                {"node_number": 2, "points_to": []},
            ],
        )
        flow_paths = [flow_path_1, flow_path_2]
    else:
        flow_paths = None

    if include_additional_texts:
        write_additional_texts_json(
            additional_texts_path_1,
            ["Figure 1", "Trial flow"],
        )
        write_additional_texts_json(
            additional_texts_path_2,
            ["Figure 2", "Participant flow"],
        )
        additional_texts_paths = [additional_texts_path_1, additional_texts_path_2]
    else:
        additional_texts_paths = None

    flowcharts = parsing_types_module.build_partial_flowcharts(
        nodes_paths=[nodes_path_1, nodes_path_2],
        labels_paths=labels_paths,
        additional_texts_paths=additional_texts_paths,
        flow_paths=flow_paths,
    )

    assert [flowchart.model_dump() for flowchart in flowcharts] == expected


@pytest.mark.parametrize(
    ("labels_paths", "additional_texts_paths", "flow_paths", "expected_match"),
    [
        pytest.param(
            "short",
            None,
            None,
            r"zip\(\) argument 2 is shorter than argument 1",
            id="labels-short",
        ),
        pytest.param(
            None,
            "short",
            None,
            r"zip\(\) argument 3 is shorter than arguments 1-2",
            id="additional-texts-short",
        ),
        pytest.param(
            None,
            None,
            "short",
            r"zip\(\) argument 4 is shorter than arguments 1-3",
            id="flow-short",
        ),
        pytest.param(
            "long",
            None,
            None,
            r"zip\(\) argument 2 is longer than argument 1",
            id="labels-long",
        ),
        pytest.param(
            None,
            "long",
            None,
            r"zip\(\) argument 3 is longer than arguments 1-2",
            id="additional-texts-long",
        ),
        pytest.param(
            None,
            None,
            "long",
            r"zip\(\) argument 4 is longer than arguments 1-3",
            id="flow-long",
        ),
    ],
)
def test_build_partial_flowcharts_raises_if_optional_path_lists_have_wrong_length(
    tmp_path,
    labels_paths,
    additional_texts_paths,
    flow_paths,
    expected_match,
):
    nodes_path_1 = tmp_path / "nodes" / "diagram_1.json"
    nodes_path_2 = tmp_path / "nodes" / "diagram_2.json"

    labels_path_1 = tmp_path / "labels" / "diagram_1.json"
    labels_path_2 = tmp_path / "labels" / "diagram_2.json"
    labels_path_3 = tmp_path / "labels" / "diagram_3.json"

    additional_texts_path_1 = tmp_path / "additional_texts" / "diagram_1.json"
    additional_texts_path_2 = tmp_path / "additional_texts" / "diagram_2.json"
    additional_texts_path_3 = tmp_path / "additional_texts" / "diagram_3.json"

    flow_path_1 = tmp_path / "flow" / "diagram_1.json"
    flow_path_2 = tmp_path / "flow" / "diagram_2.json"
    flow_path_3 = tmp_path / "flow" / "diagram_3.json"

    write_nodes_json(nodes_path_1, make_node_texts())
    write_nodes_json(nodes_path_2, make_node_texts())

    write_nodes_json(labels_path_1, make_labels())
    write_nodes_json(labels_path_2, make_labels())
    write_nodes_json(labels_path_3, make_labels())

    write_additional_texts_json(additional_texts_path_1, ["Figure 1", "Trial flow"])
    write_additional_texts_json(additional_texts_path_2, ["Figure 2", "Trial flow"])
    write_additional_texts_json(additional_texts_path_3, ["Figure 3", "Trial flow"])

    write_nodes_json(flow_path_1, make_flow())
    write_nodes_json(flow_path_2, make_flow())
    write_nodes_json(flow_path_3, make_flow())

    if labels_paths == "short":
        labels_paths = [labels_path_1]
    elif labels_paths == "long":
        labels_paths = [labels_path_1, labels_path_2, labels_path_3]

    if additional_texts_paths == "short":
        additional_texts_paths = [additional_texts_path_1]
    elif additional_texts_paths == "long":
        additional_texts_paths = [
            additional_texts_path_1,
            additional_texts_path_2,
            additional_texts_path_3,
        ]

    if flow_paths == "short":
        flow_paths = [flow_path_1]
    elif flow_paths == "long":
        flow_paths = [flow_path_1, flow_path_2, flow_path_3]

    with pytest.raises(ValueError, match=expected_match):
        parsing_types_module.build_partial_flowcharts(
            nodes_paths=[nodes_path_1, nodes_path_2],
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
        )


@pytest.mark.parametrize(
    "nodes_paths",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty-list"),
    ],
)
def test_build_partial_flowcharts_raises_if_nodes_paths_missing(nodes_paths):
    with pytest.raises(ValueError) as exc_info:
        parsing_types_module.build_partial_flowcharts(nodes_paths=nodes_paths)

    assert str(exc_info.value) == (
        "nodes_paths must be provided to build partial flowcharts, as node numbers "
        "are needed to join the different parts together."
    )
