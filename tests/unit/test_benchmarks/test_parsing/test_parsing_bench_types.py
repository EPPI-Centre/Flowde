from pathlib import Path

import pytest
from pydantic import ValidationError

import flowde.benchmarks.parsing.parsing_bench_types as types_module

# TODO: Really need to refactor this. Ran out of time.


def write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_full_node_dict(
    node_number: int = 1,
    text: str = "Node 1",
    labels: list[str] | None = None,
    points_to: list[int] | None = None,
    diagram_type: str = "pred",
    true_option_idx: int | None = None,
    parent_img_code: str = "diagram_1",
) -> dict:
    return {
        "node_number": node_number,
        "text": text,
        "labels": labels if labels is not None else ["label 1"],
        "points_to": points_to if points_to is not None else [],
        "diagram_type": diagram_type,
        "true_option_idx": true_option_idx,
        "parent_img_code": parent_img_code,
    }


def make_node_dict(
    node_number: int = 1,
    text: str | None = "Node 1",
    labels: list[str] | None = None,
    points_to: list[int] | None = None,
    diagram_type: str = "pred",
    true_option_idx: int | None = None,
    parent_img_code: str = "diagram_1",
) -> dict:
    return {
        "node_number": node_number,
        "text": text,
        "labels": labels,
        "points_to": points_to,
        "diagram_type": diagram_type,
        "true_option_idx": true_option_idx,
        "parent_img_code": parent_img_code,
    }


def make_pred_node(
    node_number: int = 1,
    text: str | None = "Node 1",
    labels: list[str] | None = None,
    points_to: list[int] | None = None,
    parent_img_code: str = "diagram_1",
) -> types_module.Node:
    return types_module.Node.model_validate(
        make_node_dict(
            node_number=node_number,
            text=text,
            labels=labels,
            points_to=points_to,
            diagram_type="pred",
            true_option_idx=None,
            parent_img_code=parent_img_code,
        )
    )


def make_true_node(
    node_number: int = 1,
    text: str | None = "Node 1",
    labels: list[str] | None = None,
    points_to: list[int] | None = None,
    true_option_idx: int = 0,
    parent_img_code: str = "diagram_1",
) -> types_module.Node:
    return types_module.Node.model_validate(
        make_node_dict(
            node_number=node_number,
            text=text,
            labels=labels,
            points_to=points_to,
            diagram_type="true",
            true_option_idx=true_option_idx,
            parent_img_code=parent_img_code,
        )
    )


def make_full_pred_diagram(
    parent_img_code: str = "diagram_1",
    nodes: list[types_module.Node] | None = None,
    additional_texts: list[str] | None = None,
) -> types_module.Diagram:
    return types_module.Diagram(
        nodes=nodes
        if nodes is not None
        else [
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code=parent_img_code,
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code=parent_img_code,
            ),
        ],
        additional_texts=additional_texts
        if additional_texts is not None
        else ["Pred Figure 1"],
        parent_img_code=parent_img_code,
        diagram_type="pred",
        true_option_idx=None,
    )


def make_pred_diagram(
    parent_img_code: str = "diagram_1",
    nodes: list[types_module.Node] | None = None,
    additional_texts: list[str] | None = None,
) -> types_module.Diagram:
    return types_module.Diagram(
        nodes=nodes,
        additional_texts=additional_texts,
        parent_img_code=parent_img_code,
        diagram_type="pred",
        true_option_idx=None,
    )


def make_true_diagram(
    parent_img_code: str = "diagram_1",
    true_option_idx: int = 0,
    nodes: list[types_module.Node] | None = None,
    additional_texts: list[str] | None = None,
) -> types_module.Diagram:
    return types_module.Diagram(
        nodes=nodes
        if nodes is not None
        else [
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=["true label 1"],
                points_to=[2],
                true_option_idx=true_option_idx,
                parent_img_code=parent_img_code,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=["true label 2"],
                points_to=[1],
                true_option_idx=true_option_idx,
                parent_img_code=parent_img_code,
            ),
        ],
        additional_texts=additional_texts
        if additional_texts is not None
        else ["True Figure 1"],
        parent_img_code=parent_img_code,
        diagram_type="true",
        true_option_idx=true_option_idx,
    )


def make_text_list_match(
    true_index: int | None = 0,
    pred_index: int | None = 0,
    true_text: str | None = "true text",
    pred_text: str | None = "pred text",
    cost: float = 0,
) -> types_module.TextListMatch:
    return types_module.TextListMatch(
        true_index=true_index,
        pred_index=pred_index,
        true_text=true_text,
        pred_text=pred_text,
        cost=cost,
    )


def make_node_match(
    true_node: types_module.Node | None,
    pred_node: types_module.Node | None,
    node_text_cost: float = 0,
    label_cost: float = 0,
) -> types_module.NodeMatch:
    return types_module.NodeMatch(
        true_node=true_node,
        pred_node=pred_node,
        node_text_cost=node_text_cost,
        label_matches=types_module.TextListMatches(
            matches=[
                make_text_list_match(
                    true_text="true label",
                    pred_text="pred label",
                    cost=label_cost,
                )
            ]
        ),
    )


def make_node_matches(
    true_diagram: types_module.Diagram,
    pred_diagram: types_module.Diagram,
) -> types_module.NodeMatches:
    return types_module.NodeMatches(
        matches=[
            make_node_match(
                true_node=true_diagram.nodes[0],
                pred_node=pred_diagram.nodes[0],
                node_text_cost=1,
                label_cost=2,
            ),
            make_node_match(
                true_node=true_diagram.nodes[1],
                pred_node=pred_diagram.nodes[1],
                node_text_cost=3,
                label_cost=4,
            ),
        ],
        true_diagram_option_idx=true_diagram.true_option_idx,
    )


def make_diagram_match() -> types_module.DiagramMatch:
    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=["true label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=["true label 2"],
                points_to=[1],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
        ],
        additional_texts=["True Figure 1"],
    )
    pred_diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code="diagram_1",
            ),
        ],
        additional_texts=["Pred Figure 1"],
    )
    node_matches = make_node_matches(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
    )

    return types_module.DiagramMatch(
        node_matches=node_matches,
        additional_text_matches=types_module.TextListMatches(
            matches=[
                make_text_list_match(
                    true_text="True Figure 1",
                    pred_text="Pred Figure 1",
                    cost=5,
                )
            ]
        ),
        pred_diagram=pred_diagram,
        true_diagram=true_diagram,
    )


def make_true_component_jsons(
    tmp_path: Path,
    parent: str = "paper_1",
    stem: str = "diagram_1",
) -> tuple[Path, Path, Path, Path]:
    nodes_path = tmp_path / "nodes" / parent / f"{stem}.json"
    labels_path = tmp_path / "labels" / parent / f"{stem}.json"
    additional_texts_path = tmp_path / "additional_texts" / parent / f"{stem}.json"
    flow_path = tmp_path / "flow" / parent / f"{stem}.json"

    write_json(
        nodes_path,
        """
        {
          "options": [
            {
              "nodes": [
                {"node_number": 2, "text": "Node 2"},
                {"node_number": 1, "text": "Node 1"}
              ]
            }
          ]
        }
        """,
    )
    write_json(
        labels_path,
        """
        {
          "options": [
            {
              "nodes": [
                {"node_number": 2, "labels": ["label 2"]},
                {"node_number": 1, "labels": ["label 1"]}
              ]
            }
          ]
        }
        """,
    )
    write_json(
        additional_texts_path,
        """
        {
          "options": [
            {
              "additional_texts": ["Figure 1", "Trial flow"]
            }
          ]
        }
        """,
    )
    write_json(
        flow_path,
        """
        {
          "options": [
            {
              "nodes": [
                {"node_number": 2, "points_to": [1]},
                {"node_number": 1, "points_to": [2]}
              ]
            }
          ]
        }
        """,
    )

    return nodes_path, labels_path, additional_texts_path, flow_path


def test_node_accepts_valid_pred_node_and_properties_work():
    node = make_pred_node(
        node_number=1,
        text="Node 1",
        labels=["screening", "eligibility"],
        points_to=[2],
    )

    assert node.node_number == 1
    assert node.labels_done is True
    assert node.text_done is True
    assert node.flow_done is True
    assert node.is_complete is True


def test_node_accepts_valid_true_node():
    node = make_true_node(
        node_number=1,
        text="Node 1",
        labels=["screening"],
        points_to=[2],
        true_option_idx=0,
    )

    assert node.diagram_type == "true"
    assert node.true_option_idx == 0


@pytest.mark.parametrize(
    (
        "labels",
        "points_to",
        "expected_labels_done",
        "expected_text_done",
        "expected_flow_done",
        "expected_is_complete",
    ),
    [
        pytest.param(
            ["label 1"],
            [2],
            True,
            True,
            True,
            True,
            id="all-done",
        ),
        pytest.param(
            None,
            [2],
            False,
            True,
            True,
            False,
            id="labels-not-done",
        ),
        pytest.param(
            ["label 1"],
            None,
            True,
            True,
            False,
            False,
            id="flow-not-done",
        ),
        pytest.param(
            None,
            None,
            False,
            True,
            False,
            False,
            id="labels-and-flow-not-done",
        ),
        pytest.param(
            [],
            [],
            True,
            True,
            True,
            True,
            id="empty-lists-count-as-done",
        ),
    ],
)
def test_node_done_and_complete_properties(
    labels,
    points_to,
    expected_labels_done,
    expected_text_done,
    expected_flow_done,
    expected_is_complete,
):
    node = make_pred_node(
        labels=labels,
        points_to=points_to,
    )

    assert node.labels_done is expected_labels_done
    assert node.text_done is expected_text_done
    assert node.flow_done is expected_flow_done
    assert node.is_complete is expected_is_complete


@pytest.mark.parametrize(
    ("node_data", "expected_msg"),
    [
        pytest.param(
            make_full_node_dict(
                diagram_type="true",
                true_option_idx=None,
            ),
            "True nodes should have a true option index.",
            id="true-node-missing-option-idx",
        ),
        pytest.param(
            make_full_node_dict(
                diagram_type="pred",
                true_option_idx=0,
            ),
            "Pred nodes should not have a true option index.",
            id="pred-node-has-option-idx",
        ),
        pytest.param(
            make_full_node_dict(
                node_number=1000,
                diagram_type="pred",
                true_option_idx=None,
            ),
            "we assume all node numbers are less than 1000",
            id="node-number-too-large",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "extra": "not allowed",
            },
            "Extra inputs are not permitted",
            id="extra-field-forbidden",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "node_number": "1",
            },
            "Input should be a valid integer",
            id="strict-node-number-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "node_number": 1.0,
            },
            "Input should be a valid integer",
            id="strict-node-number-float",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "node_number": True,
            },
            "Input should be a valid integer",
            id="strict-node-number-bool",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "node_number": None,
            },
            "Input should be a valid integer",
            id="node-number-none",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "text": "",
            },
            "String should have at least 1 character",
            id="empty-text",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "text": 123,
            },
            "Input should be a valid string",
            id="text-not-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "text": None,
            },
            "Input should be a valid string",
            id="text-none",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "labels": "screening",
            },
            "Input should be a valid list",
            id="labels-not-list",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "labels": ["screening", 123],
            },
            "Input should be a valid string",
            id="label-item-not-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "labels": [""],
            },
            "String should have at least 1 character",
            id="empty-label-item",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "points_to": 2,
            },
            "Input should be a valid list",
            id="points-to-not-list",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "points_to": ["2"],
            },
            "Input should be a valid integer",
            id="points-to-item-not-int-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "points_to": [2.0],
            },
            "Input should be a valid integer",
            id="points-to-item-not-int-float",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "points_to": [True],
            },
            "Input should be a valid integer",
            id="points-to-item-not-int-bool",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "diagram_type": "unknown",
            },
            "Input should be",
            id="invalid-diagram-type",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "diagram_type": None,
            },
            "Input should be",
            id="diagram-type-none",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "true_option_idx": "0",
            },
            "Input should be a valid integer",
            id="true-option-idx-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "true_option_idx": 0.0,
            },
            "Input should be a valid integer",
            id="true-option-idx-float",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "true_option_idx": True,
            },
            "Input should be a valid integer",
            id="true-option-idx-bool",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "parent_img_code": 123,
            },
            "Input should be a valid string",
            id="parent-img-code-not-string",
        ),
        pytest.param(
            {
                **make_full_node_dict(),
                "parent_img_code": None,
            },
            "Input should be a valid string",
            id="parent-img-code-none",
        ),
        pytest.param(
            {
                key: value
                for key, value in make_full_node_dict().items()
                if key != "node_number"
            },
            "Field required",
            id="missing-node-number",
        ),
        pytest.param(
            {
                key: value
                for key, value in make_full_node_dict().items()
                if key != "text"
            },
            "Field required",
            id="missing-text",
        ),
        pytest.param(
            {
                key: value
                for key, value in make_full_node_dict().items()
                if key != "diagram_type"
            },
            "Field required",
            id="missing-diagram-type",
        ),
        pytest.param(
            {
                key: value
                for key, value in make_full_node_dict().items()
                if key != "parent_img_code"
            },
            "Field required",
            id="missing-parent-img-code",
        ),
    ],
)
def test_node_raises_for_invalid_data(node_data, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        types_module.Node.model_validate(node_data)

    assert expected_msg in str(exc_info.value)


def test_diagram_accepts_valid_pred_diagram_and_properties_work():
    diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code="diagram_1",
            ),
        ],
        additional_texts=["Pred Figure 1"],
    )

    assert diagram.labels_done is True
    assert diagram.text_done is True
    assert diagram.flow_done is True
    assert diagram.additional_texts_done is True
    assert diagram.is_complete is True


@pytest.mark.parametrize(
    (
        "nodes",
        "additional_texts",
        "expected_labels_done",
        "expected_text_done",
        "expected_flow_done",
        "expected_additional_texts_done",
        "expected_is_complete",
    ),
    [
        pytest.param(
            [
                make_pred_node(node_number=1, labels=["label 1"], points_to=[2]),
                make_pred_node(node_number=2, labels=["label 2"], points_to=[]),
            ],
            ["Figure 1"],
            True,
            True,
            True,
            True,
            True,
            id="all-done",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=["label 1"], points_to=[2]),
                make_pred_node(node_number=2, labels=["label 2"], points_to=[]),
            ],
            None,
            True,
            True,
            True,
            False,
            False,
            id="additional-texts-not-done",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=[], points_to=[]),
            ],
            [],
            True,
            True,
            True,
            True,
            True,
            id="empty-lists-count-as-done",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=None, points_to=None),
                make_pred_node(node_number=2, labels=None, points_to=None),
            ],
            ["Figure 1"],
            False,
            True,
            False,
            True,
            False,
            id="all-node-parts-not-done",
        ),
        pytest.param(
            None,
            ["Figure 1"],
            False,
            False,
            False,
            True,
            False,
            id="nodes-not-done-additional-texts-done",
        ),
    ],
)
def test_diagram_done_and_complete_properties(
    nodes,
    additional_texts,
    expected_labels_done,
    expected_text_done,
    expected_flow_done,
    expected_additional_texts_done,
    expected_is_complete,
):
    diagram = make_pred_diagram(
        nodes=nodes,
        additional_texts=additional_texts,
    )

    assert diagram.labels_done is expected_labels_done
    assert diagram.text_done is expected_text_done
    assert diagram.flow_done is expected_flow_done
    assert diagram.additional_texts_done is expected_additional_texts_done
    assert diagram.is_complete is expected_is_complete


@pytest.mark.parametrize(
    ("nodes", "additional_texts", "expected_msg_parts"),
    [
        pytest.param(
            [
                make_pred_node(node_number=1, labels=None, points_to=[2]),
                make_pred_node(node_number=2, labels=["label 2"], points_to=[]),
            ],
            ["Figure 1"],
            [
                "If labels are done for one node in a diagram, labels should be done for all nodes in that diagram.",
                "Nodes with labels done: [2]",
                "Nodes without labels done: [1]",
            ],
            id="one-node-labels-not-done",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=["label 1"], points_to=None),
                make_pred_node(node_number=2, labels=["label 2"], points_to=[]),
            ],
            ["Figure 1"],
            [
                "If flow is done for one node in a diagram, flow should be done for all nodes in that diagram.",
                "Nodes with flow done: [2]",
                "Nodes without flow done: [1]",
            ],
            id="one-node-flow-not-done",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=["label 1"], points_to=[2]),
                make_pred_node(node_number=2, labels=None, points_to=None),
            ],
            ["Figure 1"],
            [
                "If labels are done for one node in a diagram, labels should be done for all nodes in that diagram.",
                "Nodes with labels done: [1]",
                "Nodes without labels done: [2]",
            ],
            id="one-node-complete-other-node-incomplete-labels-first",
        ),
        pytest.param(
            [
                make_pred_node(node_number=1, labels=["label 1"], points_to=[2]),
                make_pred_node(node_number=2, labels=["label 2"], points_to=None),
            ],
            ["Figure 1"],
            [
                "If flow is done for one node in a diagram, flow should be done for all nodes in that diagram.",
                "Nodes with flow done: [1]",
                "Nodes without flow done: [2]",
            ],
            id="one-node-complete-other-node-incomplete-flow",
        ),
    ],
)
def test_diagram_raises_if_node_done_state_is_partially_complete(
    nodes,
    additional_texts,
    expected_msg_parts,
):
    with pytest.raises(ValidationError) as exc_info:
        make_pred_diagram(
            nodes=nodes,
            additional_texts=additional_texts,
        )

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg


def test_diagram_accepts_valid_true_diagram():
    diagram = make_true_diagram()

    assert diagram.diagram_type == "true"
    assert diagram.true_option_idx == 0
    assert diagram.is_complete is True


@pytest.mark.parametrize(
    ("diagram", "expected_msg"),
    [
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(node_number=1, parent_img_code="diagram_1"),
                    make_pred_node(node_number=1, parent_img_code="diagram_1"),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "Node numbers should be unique within a diagram.",
            id="duplicate-node-numbers",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        text="Node 1",
                        points_to=[3],
                        labels=["label 1"],
                        parent_img_code="diagram_1",
                    ),
                    make_true_node(
                        node_number=3,
                        text="Node 3",
                        points_to=[1],
                        labels=["label 3"],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "For true diagrams, node numbers should be sequential starting from 1",
            id="true-node-numbers-not-sequential",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        text="node 1",
                        labels=["node 1"],
                        parent_img_code="diagram_1",
                        points_to=[],
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=None,
            ),
            "True diagrams should have a true option index.",
            id="true-diagram-missing-option-idx",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(node_number=1, parent_img_code="diagram_1"),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=0,
            ),
            "Pred diagrams should not have a true option index.",
            id="pred-diagram-has-option-idx",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        parent_img_code="diagram_1",
                        text="node 1",
                        labels=["node 1"],
                        points_to=[],
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "All nodes in a diagram should have the same diagram type as the diagram.",
            id="node-diagram-type-mismatch",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        parent_img_code="other_diagram",
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "All nodes in a diagram should have the same parent img code as the diagram.",
            id="node-parent-img-code-mismatch",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        parent_img_code="diagram_1",
                        text="node 1",
                        labels=["node 1"],
                        points_to=[],
                        true_option_idx=1,
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "All nodes in a diagram should have the same true option index as the diagram.",
            id="node-option-idx-mismatch",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=[3],
                        parent_img_code="diagram_1",
                    ),
                    make_pred_node(
                        node_number=2,
                        labels=["label 2"],
                        points_to=[],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "Nodes should not point to nodes that do not exist.",
            id="fake-node-in-flow",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=[999],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "Nodes should not point to nodes that do not exist.",
            id="single-pred-node-points-to-fake-node",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        labels=None,
                        points_to=[2],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        labels=None,
                        points_to=[1],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "True diagrams should be complete.",
            id="true-diagram-labels-incomplete",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=None,
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        labels=["label 2"],
                        points_to=None,
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "True diagrams should be complete.",
            id="true-diagram-flow-incomplete",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=[2],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        labels=["label 2"],
                        points_to=[1],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=None,
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "True diagrams should be complete.",
            id="true-diagram-additional-texts-incomplete",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        labels=None,
                        points_to=None,
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        labels=None,
                        points_to=None,
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=None,
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "True diagrams should be complete.",
            id="true-diagram-all-parts-incomplete",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        points_to=[],
                        text="node 1",
                        labels=["label 1"],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        points_to=[],
                        text="node 2",
                        labels=["label 2"],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "all nodes should point to or be pointed to by another node",
            id="true-diagram-isolated-nodes",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_true_node(
                        node_number=1,
                        points_to=[2],
                        text="node 1",
                        labels=["label 1"],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=2,
                        points_to=[],
                        text="node 2",
                        labels=["label 2"],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                    make_true_node(
                        node_number=3,
                        points_to=[],
                        text="node 3",
                        labels=["label 3"],
                        parent_img_code="diagram_1",
                        true_option_idx=0,
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="true",
                true_option_idx=0,
            ),
            "all nodes should point to or be pointed to by another node",
            id="true-diagram-one-isolated-node",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=None,
                additional_texts=None,
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "Either text or additional texts should be done for a diagram.",
            id="pred-diagram-no-nodes-and-no-additional-texts",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "List should have at least 1 item",
            id="empty-nodes-list",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=[],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=[""],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "String should have at least 1 character",
            id="empty-additional-text-item",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        labels=["label 1"],
                        points_to=[],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=["Figure 1", ""],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "String should have at least 1 character",
            id="one-empty-additional-text-item",
        ),
        pytest.param(
            lambda: types_module.Diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        labels=[""],
                        points_to=[],
                        parent_img_code="diagram_1",
                    ),
                ],
                additional_texts=["Figure 1"],
                parent_img_code="diagram_1",
                diagram_type="pred",
                true_option_idx=None,
            ),
            "String should have at least 1 character",
            id="empty-label-item-in-nested-node",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": "1",
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                        }
                    ],
                    "additional_texts": ["Figure 1"],
                    "parent_img_code": "diagram_1",
                    "diagram_type": "pred",
                    "true_option_idx": None,
                }
            ),
            "Input should be a valid integer",
            id="nested-node-strict-node-number",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                            "extra": "not allowed",
                        }
                    ],
                    "additional_texts": ["Figure 1"],
                    "parent_img_code": "diagram_1",
                    "diagram_type": "pred",
                    "true_option_idx": None,
                }
            ),
            "Extra inputs are not permitted",
            id="nested-node-extra-field",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                        }
                    ],
                    "additional_texts": ["Figure 1"],
                    "parent_img_code": "diagram_1",
                    "diagram_type": "pred",
                    "true_option_idx": None,
                    "extra": "not allowed",
                }
            ),
            "Extra inputs are not permitted",
            id="diagram-extra-field",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                        }
                    ],
                    "additional_texts": "Figure 1",
                    "parent_img_code": "diagram_1",
                    "diagram_type": "pred",
                    "true_option_idx": None,
                }
            ),
            "Input should be a valid list",
            id="additional-texts-not-list",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                        }
                    ],
                    "additional_texts": ["Figure 1"],
                    "parent_img_code": 123,
                    "diagram_type": "pred",
                    "true_option_idx": None,
                }
            ),
            "Input should be a valid string",
            id="parent-img-code-not-string",
        ),
        pytest.param(
            lambda: types_module.Diagram.model_validate(
                {
                    "nodes": [
                        {
                            "node_number": 1,
                            "text": "Node 1",
                            "labels": ["label 1"],
                            "points_to": [],
                            "diagram_type": "pred",
                            "true_option_idx": None,
                            "parent_img_code": "diagram_1",
                        }
                    ],
                    "additional_texts": ["Figure 1"],
                    "parent_img_code": "diagram_1",
                    "diagram_type": "unknown",
                    "true_option_idx": None,
                }
            ),
            "Input should be",
            id="invalid-diagram-type",
        ),
    ],
)
def test_diagram_raises_for_invalid_data(diagram, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        diagram()

    assert expected_msg in str(exc_info.value)


def test_diagram_allows_non_sequential_pred_node_numbers():
    diagram = types_module.Diagram(
        nodes=[
            make_pred_node(node_number=1, parent_img_code="diagram_1"),
            make_pred_node(node_number=3, parent_img_code="diagram_1"),
        ],
        additional_texts=["Figure 1"],
        parent_img_code="diagram_1",
        diagram_type="pred",
        true_option_idx=None,
    )

    assert [node.node_number for node in diagram.nodes] == [1, 3]


def test_diagram_inherits_node_validation_errors_from_nested_node_dicts():
    with pytest.raises(ValidationError) as exc_info:
        types_module.Diagram.model_validate(
            {
                "nodes": [
                    make_node_dict(
                        node_number=1000,
                        diagram_type="pred",
                        true_option_idx=None,
                    )
                ],
                "additional_texts": ["Figure 1"],
                "parent_img_code": "diagram_1",
                "diagram_type": "pred",
                "true_option_idx": None,
            }
        )

    assert "we assume all node numbers are less than 1000" in str(exc_info.value)


def test_diagram_from_path_adds_parent_and_type_metadata(tmp_path):
    path = tmp_path / "paper_1" / "diagram_1.json"
    write_json(
        path,
        """
        {
          "nodes": [
            {
              "node_number": 1,
              "text": "Node 1",
              "labels": ["label 1"],
              "points_to": [2]
            },
            {
              "node_number": 2,
              "text": "Node 2",
              "labels": ["label 2"],
              "points_to": []
            }
          ],
          "additional_texts": ["Figure 1"]
        }
        """,
    )

    diagram = types_module.Diagram.from_path(
        path=path,
        diagram_type="pred",
        true_option_idx=None,
    )

    assert diagram.parent_img_code == "diagram_1"
    assert diagram.diagram_type == "pred"
    assert diagram.true_option_idx is None
    assert [node.parent_img_code for node in diagram.nodes] == [
        "diagram_1",
        "diagram_1",
    ]
    assert [node.diagram_type for node in diagram.nodes] == ["pred", "pred"]


def test_diagram_from_path_wraps_errors_in_runtime_error(tmp_path):
    path = tmp_path / "paper_1" / "diagram_1.json"
    write_json(
        path,
        """
        {
          "nodes": [
            {
              "node_number": 1000,
              "text": "Node 1",
              "labels": ["label 1"],
              "points_to": []
            }
          ],
          "additional_texts": ["Figure 1"]
        }
        """,
    )

    with pytest.raises(RuntimeError) as exc_info:
        types_module.Diagram.from_path(
            path=path,
            diagram_type="pred",
            true_option_idx=None,
        )

    msg = str(exc_info.value)
    assert "Failed to create Diagram" in msg
    assert "Original error: ValidationError" in msg
    assert "we assume all node numbers are less than 1000" in msg


def test_diagram_options_from_true_paths_builds_true_diagram_options(tmp_path):
    nodes_path, labels_path, additional_texts_path, flow_path = (
        make_true_component_jsons(tmp_path)
    )

    diagram_options = types_module.DiagramOptions.from_true_paths(
        true_nodes_path=nodes_path,
        true_labels_path=labels_path,
        true_additional_texts_path=additional_texts_path,
        true_flow_path=flow_path,
    )

    assert diagram_options.parent_img_code == "diagram_1"
    assert diagram_options.parent_nodes_path == nodes_path
    assert diagram_options.parent_labels_path == labels_path
    assert diagram_options.parent_additional_texts_path == additional_texts_path
    assert diagram_options.parent_flow_path == flow_path

    assert len(diagram_options.options) == 1

    diagram = diagram_options.options[0]
    assert diagram.diagram_type == "true"
    assert diagram.true_option_idx == 0
    assert diagram.parent_img_code == "diagram_1"
    assert diagram.additional_texts == ["Figure 1", "Trial flow"]

    assert [node.node_number for node in diagram.nodes] == [1, 2]
    assert [node.text for node in diagram.nodes] == ["Node 1", "Node 2"]
    assert [node.labels for node in diagram.nodes] == [["label 1"], ["label 2"]]
    assert [node.points_to for node in diagram.nodes] == [[2], [1]]


def test_diagram_options_from_true_paths_wraps_nested_diagram_errors(tmp_path):
    nodes_path, labels_path, additional_texts_path, flow_path = (
        make_true_component_jsons(tmp_path)
    )

    write_json(
        nodes_path,
        """
        {
          "options": [
            {
              "nodes": [
                {"node_number": 1000, "text": "Node 1000"},
                {"node_number": 2, "text": "Node 2"}
              ]
            }
          ]
        }
        """,
    )

    with pytest.raises(RuntimeError) as exc_info:
        types_module.DiagramOptions.from_true_paths(
            true_nodes_path=nodes_path,
            true_labels_path=labels_path,
            true_additional_texts_path=additional_texts_path,
            true_flow_path=flow_path,
        )

    msg = str(exc_info.value)
    assert "Failed to create DiagramOptions" in msg
    assert "Original error: ValidationError" in msg
    assert "we assume all node numbers are less than 1000" in msg


@pytest.mark.parametrize(
    (
        "wrong_path_name",
        "wrong_path",
    ),
    [
        pytest.param(
            "parent_nodes_path",
            lambda tmp_path: tmp_path / "nodes" / "paper_1" / "different.json",
            id="nodes-stem-different",
        ),
        pytest.param(
            "parent_labels_path",
            lambda tmp_path: tmp_path / "labels" / "paper_1" / "different.json",
            id="labels-stem-different",
        ),
        pytest.param(
            "parent_additional_texts_path",
            lambda tmp_path: (
                tmp_path / "additional_texts" / "paper_1" / "different.json"
            ),
            id="additional-texts-stem-different",
        ),
        pytest.param(
            "parent_flow_path",
            lambda tmp_path: tmp_path / "flow" / "paper_1" / "different.json",
            id="flow-stem-different",
        ),
    ],
)
def test_diagram_options_raises_if_true_paths_are_not_from_same_img(
    tmp_path,
    wrong_path_name,
    wrong_path,
):
    nodes_path, labels_path, additional_texts_path, flow_path = (
        make_true_component_jsons(tmp_path)
    )

    wrong_path = wrong_path(tmp_path)

    source_paths = {
        "parent_nodes_path": nodes_path,
        "parent_labels_path": labels_path,
        "parent_additional_texts_path": additional_texts_path,
        "parent_flow_path": flow_path,
    }

    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_text(
        source_paths[wrong_path_name].read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    kwargs = {
        "options": [make_true_diagram(parent_img_code="diagram_1")],
        "parent_img_code": "diagram_1",
        "parent_nodes_path": nodes_path,
        "parent_labels_path": labels_path,
        "parent_additional_texts_path": additional_texts_path,
        "parent_flow_path": flow_path,
    }

    kwargs[wrong_path_name] = wrong_path

    with pytest.raises(ValidationError) as exc_info:
        types_module.DiagramOptions(**kwargs)

    msg = str(exc_info.value)

    assert "should have the same stem." in msg
    assert str(kwargs["parent_nodes_path"]) in msg
    assert str(kwargs["parent_labels_path"]) in msg
    assert str(kwargs["parent_additional_texts_path"]) in msg
    assert str(kwargs["parent_flow_path"]) in msg


@pytest.mark.parametrize(
    ("file_to_change", "replacement_json", "expected_msg_parts"),
    [
        pytest.param(
            "nodes",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 1, "text": "Node 1"},
                    {"node_number": 3, "text": "Node 3"}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [1, 3]",
                "Labels had: [1, 2]",
                "Flow had: [1, 2]",
            ],
            id="nodes-node-numbers-do-not-match",
        ),
        pytest.param(
            "labels",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 1, "labels": ["label 1"]},
                    {"node_number": 3, "labels": ["label 3"]}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [1, 2]",
                "Labels had: [1, 3]",
                "Flow had: [1, 2]",
            ],
            id="labels-node-numbers-do-not-match",
        ),
        pytest.param(
            "flow",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 1, "points_to": [3]},
                    {"node_number": 3, "points_to": [1]}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [1, 2]",
                "Labels had: [1, 2]",
                "Flow had: [1, 3]",
            ],
            id="flow-node-numbers-do-not-match",
        ),
        pytest.param(
            "nodes",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 4, "text": "Node 4"},
                    {"node_number": 5, "text": "Node 5"}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [4, 5]",
                "Labels had: [1, 2]",
                "Flow had: [1, 2]",
            ],
            id="nodes-completely-different-node-numbers",
        ),
        pytest.param(
            "labels",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 4, "labels": ["label 4"]},
                    {"node_number": 5, "labels": ["label 5"]}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [1, 2]",
                "Labels had: [4, 5]",
                "Flow had: [1, 2]",
            ],
            id="labels-completely-different-node-numbers",
        ),
        pytest.param(
            "flow",
            """
            {
              "options": [
                {
                  "nodes": [
                    {"node_number": 4, "points_to": [5]},
                    {"node_number": 5, "points_to": [4]}
                  ]
                }
              ]
            }
            """,
            [
                "Nodes had: [1, 2]",
                "Labels had: [1, 2]",
                "Flow had: [4, 5]",
            ],
            id="flow-completely-different-node-numbers",
        ),
    ],
)
def test_diagram_options_raises_if_json_node_numbers_do_not_match(
    tmp_path,
    file_to_change,
    replacement_json,
    expected_msg_parts,
):
    nodes_path, labels_path, additional_texts_path, flow_path = (
        make_true_component_jsons(tmp_path)
    )

    paths_by_name = {
        "nodes": nodes_path,
        "labels": labels_path,
        "flow": flow_path,
    }

    write_json(paths_by_name[file_to_change], replacement_json)

    with pytest.raises(ValidationError) as exc_info:
        types_module.DiagramOptions(
            options=[make_true_diagram(parent_img_code="diagram_1")],
            parent_img_code="diagram_1",
            parent_nodes_path=nodes_path,
            parent_labels_path=labels_path,
            parent_additional_texts_path=additional_texts_path,
            parent_flow_path=flow_path,
        )

    msg = str(exc_info.value)

    assert "the node numbers in nodes, flows,and labels files should match" in msg

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg


@pytest.mark.parametrize(
    (
        "true_index",
        "pred_index",
        "true_text",
        "pred_text",
        "expected_match_type",
    ),
    [
        pytest.param(
            0,
            0,
            "true text",
            "pred text",
            "match",
            id="match",
        ),
        pytest.param(
            0,
            None,
            "true text",
            None,
            "unmatched_true",
            id="unmatched-true",
        ),
        pytest.param(
            None,
            0,
            None,
            "pred text",
            "unmatched_pred",
            id="unmatched-pred",
        ),
    ],
)
def test_text_list_match_accepts_valid_data_and_match_type_property_works(
    true_index,
    pred_index,
    true_text,
    pred_text,
    expected_match_type,
):
    match = make_text_list_match(
        true_index=true_index,
        pred_index=pred_index,
        true_text=true_text,
        pred_text=pred_text,
        cost=4,
    )

    assert match.true_index == true_index
    assert match.pred_index == pred_index
    assert match.true_text == true_text
    assert match.pred_text == pred_text
    assert match.cost == 4
    assert match.match_type == expected_match_type


@pytest.mark.parametrize(
    ("data", "expected_msg"),
    [
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": "1",
            },
            "Input should be a valid number",
            id="strict-cost-string",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": True,
            },
            "Input should be a valid number",
            id="strict-cost-bool",
        ),
        pytest.param(
            {
                "true_index": "0",
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-true-index-string",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": "0",
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-pred-index-string",
        ),
        pytest.param(
            {
                "true_index": 0.0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-true-index-float",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0.0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-pred-index-float",
        ),
        pytest.param(
            {
                "true_index": True,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-true-index-bool",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": True,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid integer",
            id="strict-pred-index-bool",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": 123,
                "pred_text": "pred",
                "cost": 1,
            },
            "Input should be a valid string",
            id="true-text-not-string",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": 123,
                "cost": 1,
            },
            "Input should be a valid string",
            id="pred-text-not-string",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "",
                "pred_text": "pred",
                "cost": 1,
            },
            "String should have at least 1 character",
            id="true-text-empty",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "",
                "cost": 1,
            },
            "String should have at least 1 character",
            id="pred-text-empty",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
                "extra": "not allowed",
            },
            "Extra inputs are not permitted",
            id="extra-field",
        ),
        pytest.param(
            {
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Field required",
            id="missing-true-index",
        ),
        pytest.param(
            {
                "true_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Field required",
            id="missing-pred-index",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "pred_text": "pred",
                "cost": 1,
            },
            "Field required",
            id="missing-true-text",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "cost": 1,
            },
            "Field required",
            id="missing-pred-text",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
            },
            "Field required",
            id="missing-cost",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": None,
                "pred_text": "pred",
                "cost": 1,
            },
            "True index and true text should both be None or both be set.",
            id="true-index-set-true-text-none",
        ),
        pytest.param(
            {
                "true_index": None,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "True index and true text should both be None or both be set.",
            id="true-index-none-true-text-set",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": 0,
                "true_text": "true",
                "pred_text": None,
                "cost": 1,
            },
            "Pred index and pred text should both be None or both be set.",
            id="pred-index-set-pred-text-none",
        ),
        pytest.param(
            {
                "true_index": 0,
                "pred_index": None,
                "true_text": "true",
                "pred_text": "pred",
                "cost": 1,
            },
            "Pred index and pred text should both be None or both be set.",
            id="pred-index-none-pred-text-set",
        ),
        pytest.param(
            {
                "true_index": None,
                "pred_index": None,
                "true_text": None,
                "pred_text": None,
                "cost": 1,
            },
            "At least one of true index or pred index should be set.",
            id="both-true-and-pred-unmatched",
        ),
    ],
)
def test_text_list_match_raises_for_invalid_data(data, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        types_module.TextListMatch.model_validate(data)

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("costs", "expected_total_cost"),
    [
        pytest.param([], 0, id="empty"),
        pytest.param([2], 2, id="single-match"),
        pytest.param([1.5, 2.5, 3], 7, id="multiple-matches"),
    ],
)
def test_text_list_matches_provides_collection_access_and_total_cost(
    costs,
    expected_total_cost,
):
    individual_matches = [
        make_text_list_match(
            true_index=i,
            pred_index=i,
            true_text=f"true text {i}",
            pred_text=f"pred text {i}",
            cost=cost,
        )
        for i, cost in enumerate(costs)
    ]

    matches = types_module.TextListMatches(matches=individual_matches)

    assert matches.matches == individual_matches
    assert matches.total_cost == expected_total_cost


def test_node_match_accepts_match_and_parent_img_code_property_uses_true_node():
    true_node = make_true_node(parent_img_code="diagram_1")
    pred_node = make_pred_node(parent_img_code="diagram_1")

    match = make_node_match(
        true_node=true_node,
        pred_node=pred_node,
    )

    assert match.parent_img_code == "diagram_1"
    assert match.match_type == "match"


def test_node_match_parent_img_code_property_uses_pred_node_if_true_node_is_none():
    pred_node = make_pred_node(parent_img_code="diagram_1")

    match = make_node_match(
        true_node=None,
        pred_node=pred_node,
    )

    assert match.parent_img_code == "diagram_1"
    assert match.match_type == "unmatched_pred"


def test_node_match_parent_img_code_property_uses_true_node_if_pred_node_is_none():
    true_node = make_true_node(parent_img_code="diagram_1")

    match = make_node_match(
        true_node=true_node,
        pred_node=None,
    )

    assert match.parent_img_code == "diagram_1"
    assert match.match_type == "unmatched_true"


@pytest.mark.parametrize(
    (
        "true_node",
        "pred_node",
        "node_text_cost",
        "label_matches",
        "expected_msg",
    ),
    [
        pytest.param(
            make_true_node(parent_img_code="true_diagram"),
            make_pred_node(parent_img_code="pred_diagram"),
            0,
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "should have the same parent img code",
            id="parent-img-code-mismatch",
        ),
        pytest.param(
            None,
            None,
            0,
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "At least one of true node or pred node should be not None.",
            id="both-nodes-none",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            make_pred_node(parent_img_code="diagram_1"),
            "0",
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "Input should be a valid number",
            id="node-text-cost-string",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            make_pred_node(parent_img_code="diagram_1"),
            True,
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "Input should be a valid number",
            id="node-text-cost-bool",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            make_pred_node(parent_img_code="diagram_1"),
            0,
            "not a list",
            "Input should be a valid dictionary or instance of TextListMatches",
            id="label-matches-not-text-list-matches",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            make_pred_node(parent_img_code="diagram_1"),
            0,
            {
                "matches": [
                    {
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "true",
                        "pred_text": "pred",
                        "cost": "1",
                    }
                ]
            },
            "Input should be a valid number",
            id="nested-label-match-invalid-cost",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            make_pred_node(parent_img_code="diagram_1"),
            0,
            {
                "matches": [
                    {
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "",
                        "pred_text": "pred",
                        "cost": 1,
                    }
                ]
            },
            "String should have at least 1 character",
            id="nested-label-match-empty-true-text",
        ),
        pytest.param(
            "not a node",
            make_pred_node(parent_img_code="diagram_1"),
            0,
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "Input should be a valid dictionary or instance of Node",
            id="true-node-not-node",
        ),
        pytest.param(
            make_true_node(parent_img_code="diagram_1"),
            "not a node",
            0,
            types_module.TextListMatches(matches=[make_text_list_match()]),
            "Input should be a valid dictionary or instance of Node",
            id="pred-node-not-node",
        ),
    ],
)
def test_node_match_raises_for_invalid_nodes(
    true_node,
    pred_node,
    node_text_cost,
    label_matches,
    expected_msg,
):
    with pytest.raises(ValidationError) as exc_info:
        types_module.NodeMatch(
            true_node=true_node,
            pred_node=pred_node,
            node_text_cost=node_text_cost,
            label_matches=label_matches,
        )

    assert expected_msg in str(exc_info.value)


def test_node_matches_properties_add_and_lookup_methods_work():
    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=["true label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=["true label 2"],
                points_to=[1],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
        ],
        additional_texts=["True Figure 1"],
    )
    pred_diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code="diagram_1",
            ),
        ],
        additional_texts=["Pred Figure 1"],
    )

    node_matches = make_node_matches(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
    )

    assert node_matches.parent_img_code == "diagram_1"
    assert node_matches.total_node_text_cost == 4
    assert node_matches.total_label_error_cost == 6
    assert node_matches.true_nodes == true_diagram.nodes
    assert node_matches.pred_nodes == pred_diagram.nodes

    assert node_matches.get_matched_true_node(1) == true_diagram.nodes[0]
    assert node_matches.get_matched_pred_node(2) == pred_diagram.nodes[1]

    node_matches.add(
        true_node=None,
        pred_node=make_pred_node(
            node_number=3,
            text="Extra pred",
            labels=["extra"],
            points_to=[],
            parent_img_code="diagram_1",
        ),
        node_text_cost=7,
        label_matches=types_module.TextListMatches(matches=[]),
    )

    assert node_matches.total_node_text_cost == 11
    assert node_matches.pred_nodes[-1].node_number == 3
    assert isinstance(node_matches.matches[-1], types_module.NodeMatch)


def test_node_matches_add_passes_parameters_to_node_match(monkeypatch):
    true_node = make_true_node(
        node_number=1,
        text="True node",
        labels=["true label"],
        points_to=[2],
        parent_img_code="diagram_1",
    )
    pred_node = make_pred_node(
        node_number=1,
        text="Pred node",
        labels=["pred label"],
        points_to=[2],
        parent_img_code="diagram_1",
    )
    label_matches = types_module.TextListMatches(
        matches=[
            make_text_list_match(
                true_index=0,
                pred_index=0,
                true_text="true label",
                pred_text="pred label",
                cost=3,
            )
        ]
    )

    calls = []

    class FakeNodeMatch:
        def __init__(
            self,
            true_node,
            pred_node,
            node_text_cost,
            label_matches=None,
        ):
            calls.append(
                {
                    "true_node": true_node,
                    "pred_node": pred_node,
                    "node_text_cost": node_text_cost,
                    "label_matches": label_matches,
                }
            )

            self.true_node = true_node
            self.pred_node = pred_node
            self.node_text_cost = node_text_cost
            self.label_matches = label_matches

    monkeypatch.setattr(types_module, "NodeMatch", FakeNodeMatch)

    node_matches = types_module.NodeMatches(
        matches=[],
        true_diagram_option_idx=0,
    )

    node_matches.add(
        true_node=true_node,
        pred_node=pred_node,
        node_text_cost=7,
        label_matches=label_matches,
    )

    assert calls == [
        {
            "true_node": true_node,
            "pred_node": pred_node,
            "node_text_cost": 7,
            "label_matches": label_matches,
        }
    ]

    assert len(node_matches.matches) == 1
    assert node_matches.matches[0].true_node == true_node
    assert node_matches.matches[0].pred_node == pred_node
    assert node_matches.matches[0].node_text_cost == 7
    assert node_matches.matches[0].label_matches == label_matches


@pytest.mark.parametrize(
    ("matches", "expected_msg"),
    [
        pytest.param(
            [],
            "There should be at least one node match to get parent img code.",
            id="empty-matches-parent-img-code",
        ),
    ],
)
def test_node_matches_parent_img_code_raises_for_empty_matches(matches, expected_msg):
    node_matches = types_module.NodeMatches(
        matches=matches,
        true_diagram_option_idx=0,
    )

    with pytest.raises(ValueError) as exc_info:
        _ = node_matches.parent_img_code

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    ("lookup_method", "node_number", "expected_msg"),
    [
        pytest.param(
            "get_matched_true_node",
            999,
            "Pred node 999 not found in node matches",
            id="missing-pred-node",
        ),
        pytest.param(
            "get_matched_pred_node",
            999,
            "True node 999 not found in node matches",
            id="missing-true-node",
        ),
    ],
)
def test_node_matches_lookup_methods_raise_if_node_not_found(
    lookup_method,
    node_number,
    expected_msg,
):
    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=["true label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=["true label 2"],
                points_to=[1],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
        ],
        additional_texts=["True Figure 1"],
    )
    pred_diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code="diagram_1",
            ),
        ],
        additional_texts=["Pred Figure 1"],
    )
    node_matches = make_node_matches(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
    )

    with pytest.raises(ValueError) as exc_info:
        getattr(node_matches, lookup_method)(node_number)

    assert expected_msg in str(exc_info.value)


def test_node_matches_get_matched_true_node_handles_fake_pred_node_only_if_allowed():
    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=["true label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=["true label 2"],
                points_to=[1],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
        ],
        additional_texts=["True Figure 1"],
    )
    pred_diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text="Pred node 1",
                labels=["pred label 1"],
                points_to=[2],
                parent_img_code="diagram_1",
            ),
            make_pred_node(
                node_number=2,
                text="Pred node 2",
                labels=["pred label 2"],
                points_to=[],
                parent_img_code="diagram_1",
            ),
        ],
        additional_texts=["Pred Figure 1"],
    )

    node_matches = make_node_matches(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
    )

    fake_pred_node_number = 999

    with pytest.raises(ValueError) as exc_info:
        node_matches.get_matched_true_node(fake_pred_node_number)

    assert f"Pred node {fake_pred_node_number} not found in node matches" in str(
        exc_info.value
    )
    assert "for img diagram_1" in str(exc_info.value)

    assert (
        node_matches.get_matched_true_node(
            fake_pred_node_number,
            allow_fake_pred_node=True,
        )
        is None
    )


def test_node_matches_properties_return_expected_values():
    true_node_1 = make_true_node(
        node_number=1,
        text="True node 1",
        labels=["true label 1"],
        points_to=[2],
        parent_img_code="diagram_1",
    )
    true_node_2 = make_true_node(
        node_number=2,
        text="True node 2",
        labels=["true label 2"],
        points_to=[1],
        parent_img_code="diagram_1",
    )
    true_node_3 = make_true_node(
        node_number=3,
        text="Unmatched true node",
        labels=["true label 3"],
        points_to=[],
        parent_img_code="diagram_1",
    )

    pred_node_1 = make_pred_node(
        node_number=1,
        text="Pred node 1",
        labels=["pred label 1"],
        points_to=[2],
        parent_img_code="diagram_1",
    )
    pred_node_2 = make_pred_node(
        node_number=2,
        text="Pred node 2",
        labels=["pred label 2"],
        points_to=[],
        parent_img_code="diagram_1",
    )
    pred_node_3 = make_pred_node(
        node_number=3,
        text="Unmatched pred node",
        labels=["pred label 3"],
        points_to=[],
        parent_img_code="diagram_1",
    )

    label_match_1 = make_text_list_match(
        true_text="true label 1",
        pred_text="pred label 1",
        cost=2,
    )
    label_match_2 = make_text_list_match(
        true_text="true label 2",
        pred_text="pred label 2",
        cost=3,
    )
    label_match_3 = make_text_list_match(
        true_text="true label 3",
        pred_text=None,
        pred_index=None,
        cost=4,
    )
    label_match_4 = make_text_list_match(
        true_text=None,
        true_index=None,
        pred_text="pred label 3",
        cost=5.5,
    )

    node_match_1 = types_module.NodeMatch(
        true_node=true_node_1,
        pred_node=pred_node_1,
        node_text_cost=10.5,
        label_matches=types_module.TextListMatches(
            matches=[label_match_1, label_match_2]
        ),
    )
    node_match_2 = types_module.NodeMatch(
        true_node=true_node_2,
        pred_node=pred_node_2,
        node_text_cost=20,
        label_matches=types_module.TextListMatches(matches=[]),
    )
    node_match_3 = types_module.NodeMatch(
        true_node=true_node_3,
        pred_node=None,
        node_text_cost=30,
        label_matches=types_module.TextListMatches(matches=[label_match_3]),
    )
    node_match_4 = types_module.NodeMatch(
        true_node=None,
        pred_node=pred_node_3,
        node_text_cost=40,
        label_matches=types_module.TextListMatches(matches=[label_match_4]),
    )

    node_matches = types_module.NodeMatches(
        matches=[
            node_match_1,
            node_match_2,
            node_match_3,
            node_match_4,
        ],
        true_diagram_option_idx=0,
    )

    assert node_matches.parent_img_code == "diagram_1"
    assert node_matches.total_node_text_cost == 100.5
    assert node_matches.total_label_error_cost == 14.5
    assert node_matches.true_nodes == [
        true_node_1,
        true_node_2,
        true_node_3,
    ]
    assert node_matches.pred_nodes == [
        pred_node_1,
        pred_node_2,
        pred_node_3,
    ]


def test_node_matches_total_label_cost_raises_before_labels_are_matched():
    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=make_true_node(),
                pred_node=make_pred_node(),
                node_text_cost=0,
                label_matches=None,
            )
        ],
        true_diagram_option_idx=0,
    )

    with pytest.raises(
        ValueError,
        match="All node matches must have label matches before calculating",
    ):
        _ = node_matches.total_label_error_cost


def test_node_matches_parent_img_code_raises_if_matches_empty():
    node_matches = types_module.NodeMatches(
        matches=[],
        true_diagram_option_idx=0,
    )

    with pytest.raises(ValueError) as exc_info:
        _ = node_matches.parent_img_code

    assert "There should be at least one node match to get parent img code." in str(
        exc_info.value
    )


@pytest.mark.parametrize(
    ("matches", "expected_msg"),
    [
        pytest.param(
            lambda: [
                make_node_match(
                    true_node=make_true_node(node_number=1),
                    pred_node=make_pred_node(node_number=1),
                ),
                make_node_match(
                    true_node=make_true_node(node_number=1),
                    pred_node=make_pred_node(node_number=2),
                ),
            ],
            "True node 1 is matched more than once.",
            id="true-node-matched-twice",
        ),
        pytest.param(
            lambda: [
                make_node_match(
                    true_node=make_true_node(node_number=1),
                    pred_node=make_pred_node(node_number=1),
                ),
                make_node_match(
                    true_node=make_true_node(node_number=2),
                    pred_node=make_pred_node(node_number=1),
                ),
            ],
            "Pred node 1 is matched more than once.",
            id="pred-node-matched-twice",
        ),
    ],
)
def test_node_matches_raises_if_node_matched_twice(matches, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        types_module.NodeMatches(
            matches=matches(),
            true_diagram_option_idx=0,
        )

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    (
        "true_points_to_by_node",
        "pred_points_to_by_node",
        "expected_flow_score",
    ),
    [
        pytest.param(
            {
                1: [2],
                2: [],
            },
            {
                1: [2],
                2: [],
            },
            types_module.FlowScores(
                tp=1,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                jaccard=1.0,
                missing_edges=set(),
                extra_edges=set(),
            ),
            id="perfect-single-edge",
        ),
        pytest.param(
            {
                1: [2],
                2: [],
            },
            {
                1: [],
                2: [],
            },
            types_module.FlowScores(
                tp=0,
                fp=0,
                fn=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                jaccard=0.0,
                missing_edges={(1, 2)},
                extra_edges=set(),
            ),
            id="missing-single-edge",
        ),
        pytest.param(
            {
                1: [],
                2: [],
            },
            {
                1: [2],
                2: [],
            },
            types_module.FlowScores(
                tp=0,
                fp=1,
                fn=0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                jaccard=0.0,
                missing_edges=set(),
                extra_edges={(1, 2)},
            ),
            id="extra-single-edge",
        ),
        pytest.param(
            {
                1: [2],
                2: [],
            },
            {
                1: [],
                2: [1],
            },
            types_module.FlowScores(
                tp=0,
                fp=1,
                fn=1,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                jaccard=0.0,
                missing_edges={(1, 2)},
                extra_edges={(2, 1)},
            ),
            id="wrong-direction",
        ),
        pytest.param(
            {
                1: [2, 3],
                2: [],
                3: [],
            },
            {
                1: [2],
                2: [],
                3: [],
            },
            types_module.FlowScores(
                tp=1,
                fp=0,
                fn=1,
                precision=1.0,
                recall=0.5,
                f1=2 / 3,
                jaccard=0.5,
                missing_edges={(1, 3)},
                extra_edges=set(),
            ),
            id="one-of-two-true-edges-found",
        ),
        pytest.param(
            {
                1: [2],
                2: [],
                3: [],
            },
            {
                1: [2, 3],
                2: [],
                3: [],
            },
            types_module.FlowScores(
                tp=1,
                fp=1,
                fn=0,
                precision=0.5,
                recall=1.0,
                f1=2 / 3,
                jaccard=0.5,
                missing_edges=set(),
                extra_edges={(1, 3)},
            ),
            id="one-correct-one-extra-edge",
        ),
        pytest.param(
            {
                1: [2, 3],
                2: [3],
                3: [],
            },
            {
                1: [2, 3],
                2: [],
                3: [2],
            },
            types_module.FlowScores(
                tp=2,
                fp=1,
                fn=1,
                precision=2 / 3,
                recall=2 / 3,
                f1=2 / 3,
                jaccard=0.5,
                missing_edges={(2, 3)},
                extra_edges={(3, 2)},
            ),
            id="mixed-tp-fp-fn",
        ),
        pytest.param(
            {
                1: [],
                2: [],
            },
            {
                1: [],
                2: [],
            },
            types_module.FlowScores(
                tp=0,
                fp=0,
                fn=0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                jaccard=1.0,
                missing_edges=set(),
                extra_edges=set(),
            ),
            id="no-true-or-pred-edges",
        ),
        pytest.param(
            {
                1: [2],
                2: [3],
                3: [],
            },
            {
                1: [2],
                2: [3],
                3: [],
            },
            types_module.FlowScores(
                tp=2,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                jaccard=1.0,
                missing_edges=set(),
                extra_edges=set(),
            ),
            id="perfect-chain",
        ),
        pytest.param(
            {
                1: [2],
                2: [3],
                3: [],
            },
            {
                1: [3],
                2: [],
                3: [],
            },
            types_module.FlowScores(
                tp=0,
                fp=1,
                fn=2,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                jaccard=0.0,
                missing_edges={(1, 2), (2, 3)},
                extra_edges={(1, 3)},
            ),
            id="shortcut-edge-instead-of-chain",
        ),
    ],
)
def test_node_matches_flow_score_for_matched_nodes(
    true_points_to_by_node,
    pred_points_to_by_node,
    expected_flow_score,
):
    true_nodes = [
        make_true_node(
            node_number=node_number,
            text=f"True node {node_number}",
            labels=[f"true label {node_number}"],
            points_to=points_to,
            parent_img_code="diagram_1",
            true_option_idx=0,
        )
        for node_number, points_to in true_points_to_by_node.items()
    ]
    pred_nodes = [
        make_pred_node(
            node_number=node_number,
            text=f"Pred node {node_number}",
            labels=[f"pred label {node_number}"],
            points_to=points_to,
            parent_img_code="diagram_1",
        )
        for node_number, points_to in pred_points_to_by_node.items()
    ]

    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_node,
                pred_node=pred_node,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            )
            for true_node, pred_node in zip(true_nodes, pred_nodes, strict=True)
        ],
        true_diagram_option_idx=0,
    )

    assert node_matches.flow_score == expected_flow_score


def test_node_matches_flow_score_counts_edge_from_unmatched_pred_node_as_false_positive():
    true_node_1 = make_true_node(
        node_number=1,
        points_to=[2],
        parent_img_code="diagram_1",
    )
    true_node_2 = make_true_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )

    pred_node_1 = make_pred_node(
        node_number=1,
        points_to=[2],
        parent_img_code="diagram_1",
    )
    pred_node_2 = make_pred_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )
    unmatched_pred_node_3 = make_pred_node(
        node_number=3,
        points_to=[1],
        parent_img_code="diagram_1",
    )

    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_node_1,
                pred_node=pred_node_1,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=true_node_2,
                pred_node=pred_node_2,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=None,
                pred_node=unmatched_pred_node_3,
                node_text_cost=10,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
        ],
        true_diagram_option_idx=0,
    )

    assert node_matches.flow_score == types_module.FlowScores(
        tp=1,
        fp=1,
        fn=0,
        precision=0.5,
        recall=1.0,
        f1=2 / 3,
        jaccard=0.5,
        missing_edges=set(),
        extra_edges={(10000, 1)},
    )


def test_node_matches_flow_score_counts_edge_to_unmatched_pred_node_as_false_positive():
    true_node_1 = make_true_node(
        node_number=1,
        points_to=[2],
        parent_img_code="diagram_1",
    )
    true_node_2 = make_true_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )

    pred_node_1 = make_pred_node(
        node_number=1,
        points_to=[3],
        parent_img_code="diagram_1",
    )
    pred_node_2 = make_pred_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )
    unmatched_pred_node_3 = make_pred_node(
        node_number=3,
        points_to=[],
        parent_img_code="diagram_1",
    )

    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_node_1,
                pred_node=pred_node_1,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=true_node_2,
                pred_node=pred_node_2,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=None,
                pred_node=unmatched_pred_node_3,
                node_text_cost=10,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
        ],
        true_diagram_option_idx=0,
    )

    assert node_matches.flow_score == types_module.FlowScores(
        tp=0,
        fp=1,
        fn=1,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        jaccard=0.0,
        missing_edges={(1, 2)},
        extra_edges={(1, 10000)},
    )


# TODO: When back, add a pred edge to one that doesn't exist within pred
def test_node_matches_flow_score_counts_edge_to_fake_node_as_false_positive():
    true_node_1 = make_true_node(
        node_number=1,
        points_to=[],
        parent_img_code="diagram_1",
    )
    true_node_2 = make_true_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )

    pred_node_1 = make_pred_node(
        node_number=1,
        points_to=[],
        parent_img_code="diagram_1",
    )
    pred_node_2 = make_pred_node(
        node_number=2,
        points_to=[],
        parent_img_code="diagram_1",
    )
    unmatched_pred_node_3 = make_pred_node(
        node_number=3,
        points_to=[4],
        parent_img_code="diagram_1",
    )
    unmatched_pred_node_4 = make_pred_node(
        node_number=4,
        points_to=[5],
        parent_img_code="diagram_1",
    )

    node_matches = types_module.NodeMatches.model_construct(
        matches=[
            types_module.NodeMatch(
                true_node=true_node_1,
                pred_node=pred_node_1,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=true_node_2,
                pred_node=pred_node_2,
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=None,
                pred_node=unmatched_pred_node_3,
                node_text_cost=10,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
            types_module.NodeMatch(
                true_node=None,
                pred_node=unmatched_pred_node_4,
                node_text_cost=10,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
        ],
        true_diagram_option_idx=0,
    )

    assert node_matches.flow_score == types_module.FlowScores(
        tp=0,
        fp=2,
        fn=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        jaccard=0.0,
        missing_edges=set(),
        extra_edges={(10000, 10001), (10002, 10003)},
    )


def test_diagram_match_accepts_valid_data_and_cost_properties_work():
    diagram_match = make_diagram_match()

    assert diagram_match.total_node_text_cost == 4
    assert diagram_match.total_label_error_cost == 6
    assert diagram_match.total_additional_text_cost == 5
    assert diagram_match.total_text_cost == 15
    assert isinstance(diagram_match.flow_score, types_module.FlowScores)


@pytest.mark.parametrize(
    ("diagram_match", "expected_msg"),
    [
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(),
                    pred_diagram=make_full_pred_diagram(
                        nodes=[
                            make_pred_node(
                                node_number=1,
                                labels=None,
                                points_to=[2],
                            ),
                            make_pred_node(
                                node_number=2,
                                labels=None,
                                points_to=[],
                            ),
                        ]
                    ),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(
                    nodes=[
                        make_pred_node(node_number=1, labels=None, points_to=[2]),
                        make_pred_node(node_number=2, labels=None, points_to=[]),
                    ]
                ),
                true_diagram=make_true_diagram(),
            ),
            "Pred diagram is not complete.",
            id="pred-diagram-incomplete",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(
                        nodes=[
                            make_true_node(
                                node_number=1,
                                labels=None,
                                points_to=[2],
                            ),
                            make_true_node(
                                node_number=2,
                                labels=None,
                                points_to=[1],
                            ),
                        ]
                    ),
                    pred_diagram=make_full_pred_diagram(),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(),
                true_diagram=make_true_diagram(
                    nodes=[
                        make_true_node(node_number=1, labels=None, points_to=[2]),
                        make_true_node(node_number=2, labels=None, points_to=[1]),
                    ]
                ),
            ),
            "True diagrams should be complete.",
            id="true-diagram-incomplete-inherited",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(parent_img_code="true_diagram"),
                    pred_diagram=make_full_pred_diagram(parent_img_code="pred_diagram"),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(parent_img_code="pred_diagram"),
                true_diagram=make_true_diagram(parent_img_code="true_diagram"),
            ),
            "should have the same parent img code",
            id="diagram-origin-mismatch",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(),
                    pred_diagram=make_full_pred_diagram(),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Wrong pred text",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(),
                true_diagram=make_true_diagram(),
            ),
            "Pred additional texts do not match additional texts from matches list.",
            id="pred-additional-texts-not-in-matches",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(),
                    pred_diagram=make_full_pred_diagram(),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="Wrong true text",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(),
                true_diagram=make_true_diagram(),
            ),
            "True additional texts do not match additional texts from matches list.",
            id="true-additional-texts-not-in-matches",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=types_module.NodeMatches(
                    matches=[
                        make_node_match(
                            true_node=make_true_diagram().nodes[0],
                            pred_node=make_full_pred_diagram().nodes[0],
                        )
                    ],
                    true_diagram_option_idx=0,
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(),
                true_diagram=make_true_diagram(),
            ),
            "True nodes do not match nodes from matches list.",
            id="not-every-node-in-matches",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=types_module.NodeMatches(
                    matches=[
                        make_node_match(
                            true_node=make_true_diagram().nodes[0],
                            pred_node=make_full_pred_diagram().nodes[0],
                        ),
                        make_node_match(
                            true_node=make_true_diagram().nodes[1],
                            pred_node=None,
                        ),
                    ],
                    true_diagram_option_idx=0,
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_full_pred_diagram(),
                true_diagram=make_true_diagram(),
            ),
            "Pred nodes do not match nodes from matches list.",
            id="not-every-pred-node-in-matches",
        ),
        pytest.param(
            lambda: types_module.DiagramMatch(
                node_matches=make_node_matches(
                    true_diagram=make_true_diagram(
                        additional_texts=["True Figure 1"],
                    ),
                    pred_diagram=make_true_diagram(
                        additional_texts=["Pred Figure 1"],
                    ),
                ),
                additional_text_matches=types_module.TextListMatches(
                    matches=[
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        )
                    ]
                ),
                pred_diagram=make_true_diagram(
                    additional_texts=["Pred Figure 1"],
                ),
                true_diagram=make_true_diagram(
                    additional_texts=["True Figure 1"],
                ),
            ),
            "The pred diagram in a diagram match should have diagram type 'pred'.",
            id="pred-diagram-has-true-type",
        ),
        pytest.param(
            lambda: (
                lambda true_diagram, pred_diagram: types_module.DiagramMatch(
                    node_matches=types_module.NodeMatches(
                        matches=[
                            make_node_match(
                                true_node=true_diagram.nodes[0],
                                pred_node=pred_diagram.nodes[0],
                            ),
                            make_node_match(
                                true_node=true_diagram.nodes[1],
                                pred_node=pred_diagram.nodes[1],
                            ),
                        ],
                        true_diagram_option_idx=0,
                    ),
                    additional_text_matches=types_module.TextListMatches(
                        matches=[
                            make_text_list_match(
                                true_text="True Figure 1",
                                pred_text="Pred Figure 1",
                            )
                        ]
                    ),
                    pred_diagram=pred_diagram,
                    true_diagram=true_diagram,
                )
            )(
                make_full_pred_diagram(additional_texts=["True Figure 1"]),
                make_full_pred_diagram(additional_texts=["Pred Figure 1"]),
            ),
            "The true diagram in a diagram match should have diagram type 'true'.",
            id="true-diagram-has-pred-type",
        ),
    ],
)
def test_diagram_match_raises_for_invalid_data(diagram_match, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        diagram_match()

    assert expected_msg in str(exc_info.value)


def test_diagram_match_inherits_diagram_errors_from_nested_diagram_dicts():
    pred_diagram = make_full_pred_diagram().model_dump()
    pred_diagram["nodes"][1]["node_number"] = 1

    true_diagram = make_true_diagram().model_dump()

    with pytest.raises(ValidationError) as exc_info:
        types_module.DiagramMatch.model_validate(
            {
                "node_matches": make_node_matches(
                    true_diagram=make_true_diagram(),
                    pred_diagram=make_full_pred_diagram(),
                ).model_dump(),
                "additional_text_matches": {
                    "matches": [
                        make_text_list_match(
                            true_text="True Figure 1",
                            pred_text="Pred Figure 1",
                        ).model_dump()
                    ]
                },
                "pred_diagram": pred_diagram,
                "true_diagram": true_diagram,
            }
        )

    assert "Node numbers should be unique within a diagram." in str(exc_info.value)


def test_diagram_match_raises_if_pred_diagram_type_is_not_pred():
    true_diagram = make_true_diagram()
    pred_diagram = types_module.Diagram(
        nodes=[
            make_true_node(
                node_number=1, points_to=[2], labels=["label 1"], true_option_idx=0
            ),
            make_true_node(
                node_number=2, points_to=[1], labels=["label 2"], true_option_idx=0
            ),
        ],
        additional_texts=["Pred Figure 1"],
        parent_img_code="diagram_1",
        diagram_type="true",
        true_option_idx=0,
    )

    node_matches = types_module.NodeMatches(
        matches=[
            make_node_match(
                true_node=true_diagram.nodes[0],
                pred_node=pred_diagram.nodes[0],
            ),
            make_node_match(
                true_node=true_diagram.nodes[1],
                pred_node=pred_diagram.nodes[1],
            ),
        ],
        true_diagram_option_idx=0,
    )

    with pytest.raises(ValidationError) as exc_info:
        types_module.DiagramMatch(
            node_matches=node_matches,
            additional_text_matches=types_module.TextListMatches(
                matches=[
                    make_text_list_match(
                        true_text="True Figure 1",
                        pred_text="Pred Figure 1",
                    )
                ]
            ),
            pred_diagram=pred_diagram,
            true_diagram=true_diagram,
        )

    assert (
        "The pred diagram in a diagram match should have diagram type 'pred'."
        in str(exc_info.value)
    )


def test_diagram_match_raises_if_pred_and_true_diagrams_have_different_parent_img_codes() -> (
    None
):
    true_diagram = make_true_diagram(parent_img_code="true_diagram")
    pred_diagram = make_full_pred_diagram(parent_img_code="pred_diagram")

    node_matches = types_module.NodeMatches(
        matches=[
            make_node_match(
                true_node=true_diagram.nodes[0],
                pred_node=None,
            ),
            make_node_match(
                true_node=true_diagram.nodes[1],
                pred_node=None,
            ),
            make_node_match(
                true_node=None,
                pred_node=pred_diagram.nodes[0],
            ),
            make_node_match(
                true_node=None,
                pred_node=pred_diagram.nodes[1],
            ),
        ],
        true_diagram_option_idx=0,
    )

    with pytest.raises(ValidationError) as exc_info:
        types_module.DiagramMatch(
            node_matches=node_matches,
            additional_text_matches=types_module.TextListMatches(
                matches=[
                    make_text_list_match(
                        true_text="True Figure 1",
                        pred_text=None,
                        pred_index=None,
                    ),
                    make_text_list_match(
                        true_text=None,
                        true_index=None,
                        pred_text="Pred Figure 1",
                    ),
                ]
            ),
            pred_diagram=pred_diagram,
            true_diagram=true_diagram,
        )

    msg = str(exc_info.value)

    assert (
        "The pred and true diagrams in a diagram match should have the "
        "same parent img code."
    ) in msg
    assert "Found pred diagram parent img code: pred_diagram" in msg
    assert "Found true diagram parent img code: true_diagram" in msg


def test_diagram_match_validate_diagrams_complete_raises_if_true_diagram_incomplete() -> (
    None
):
    pred_diagram = make_full_pred_diagram()

    incomplete_true_diagram = types_module.Diagram.model_construct(
        nodes=[
            make_true_node(
                node_number=1,
                text="True node 1",
                labels=None,
                points_to=[2],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="True node 2",
                labels=None,
                points_to=[1],
                parent_img_code="diagram_1",
                true_option_idx=0,
            ),
        ],
        additional_texts=["True Figure 1"],
        parent_img_code="diagram_1",
        diagram_type="true",
        true_option_idx=0,
    )

    diagram_match = types_module.DiagramMatch.model_construct(
        node_matches=types_module.NodeMatches.model_construct(
            matches=[],
            true_diagram_option_idx=0,
        ),
        additional_text_matches=types_module.TextListMatches(matches=[]),
        pred_diagram=pred_diagram,
        true_diagram=incomplete_true_diagram,
    )

    with pytest.raises(ValueError) as exc_info:
        diagram_match.validate_diagrams_complete()

    msg = str(exc_info.value)

    assert "Diagrams should be complete for a diagram match." in msg
    assert "True diagram is not complete." in msg
    assert "labels_done: False" in msg
