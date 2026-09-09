from pathlib import Path
from types import SimpleNamespace

import pytest

import flowde.benchmarks.parsing.match_diagrams as match_module
import flowde.benchmarks.parsing.parsing_bench_types as types_module
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_fn,
)


def custom_distance_fn(
    *,
    true_text: str | None,
    pred_text: str | None,
) -> int:
    custom_costs = {
        ("abc", "abc"): 7,
        ("abc", None): 13,
        (None, "abcd"): 19,
        ("cat", "cat"): 5,
        ("dog", "dig"): 8,
        ("dog", "dog"): 6,
        ("mouse", None): 17,
        (None, "mouse"): 23,
        ("allocated", "allocated"): 100,
        ("randomised", "randomised"): 100,
        ("allocated", "randomised"): 1,
        ("randomised", "allocated"): 1,
        ("extreme", "extreme"): 1000,
    }

    if (true_text, pred_text) in custom_costs:
        return custom_costs[(true_text, pred_text)]

    if true_text is None:
        return 97

    if pred_text is None:
        return 89

    return 83


def fractional_levenshtein_fn(
    *,
    true_text: str | None,
    pred_text: str | None,
) -> float:
    return levenshtein_fn(true_text=true_text, pred_text=pred_text) / 10


def make_true_node(
    node_number: int,
    text: str,
    labels: list[str],
    points_to: list[int],
    true_option_idx: int = 0,
    parent_img_code: str = "diagram_1",
) -> types_module.Node:
    return types_module.Node(
        node_number=node_number,
        text=text,
        labels=labels,
        points_to=points_to,
        diagram_type="true",
        true_option_idx=true_option_idx,
        parent_img_code=parent_img_code,
    )


def make_pred_node(
    node_number: int,
    text: str,
    labels: list[str] | None = None,
    points_to: list[int] | None = None,
    parent_img_code: str = "diagram_1",
) -> types_module.Node:
    return types_module.Node(
        node_number=node_number,
        text=text,
        labels=labels,
        points_to=points_to,
        diagram_type="pred",
        true_option_idx=None,
        parent_img_code=parent_img_code,
    )


def make_true_diagram(
    true_option_idx: int = 0,
    parent_img_code: str = "diagram_1",
    nodes: list[types_module.Node] | None = None,
    additional_texts: list[str] | None = None,
) -> types_module.Diagram:
    return types_module.Diagram(
        nodes=nodes
        if nodes is not None
        else [
            make_true_node(
                node_number=1,
                text="Node 1",
                labels=["label 1"],
                points_to=[2],
                true_option_idx=true_option_idx,
                parent_img_code=parent_img_code,
            ),
            make_true_node(
                node_number=2,
                text="Node 2",
                labels=["label 2"],
                points_to=[1],
                true_option_idx=true_option_idx,
                parent_img_code=parent_img_code,
            ),
        ],
        additional_texts=additional_texts
        if additional_texts is not None
        else ["Figure 1"],
        parent_img_code=parent_img_code,
        diagram_type="true",
        true_option_idx=true_option_idx,
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


def make_full_pred_diagram(
    parent_img_code: str = "diagram_1",
    nodes: list[types_module.Node] | None = None,
    additional_texts: list[str] | None = None,
) -> types_module.Diagram:
    return make_pred_diagram(
        nodes=nodes
        if nodes is not None
        else [
            make_pred_node(
                node_number=1,
                text="Node 1",
                labels=["label 1"],
                points_to=[2],
                parent_img_code=parent_img_code,
            ),
            make_pred_node(
                node_number=2,
                text="Node 2",
                labels=["label 2"],
                points_to=[],
                parent_img_code=parent_img_code,
            ),
        ],
        additional_texts=additional_texts
        if additional_texts is not None
        else ["Figure 1"],
        parent_img_code=parent_img_code,
    )


def make_diagram_options(
    options: list[types_module.Diagram],
    parent_img_code: str = "diagram_1",
) -> types_module.DiagramOptions:
    return types_module.DiagramOptions.model_construct(
        options=options,
        parent_img_code=parent_img_code,
        parent_nodes_path=Path("nodes") / "paper_1" / f"{parent_img_code}.json",
        parent_labels_path=Path("labels") / "paper_1" / f"{parent_img_code}.json",
        parent_additional_texts_path=Path("additional_texts")
        / "paper_1"
        / f"{parent_img_code}.json",
        parent_flow_path=Path("flow") / "paper_1" / f"{parent_img_code}.json",
    )


def make_option_selection_true_diagram(
    true_option_idx: int,
    node_texts: tuple[str, str] = ("Node 1", "Node 2"),
    node_labels: tuple[str, str] = ("label 1", "label 2"),
    node_points_to: tuple[tuple[int, ...], tuple[int, ...]] = ((2,), ()),
    additional_text: str = "Participant flow diagram",
) -> types_module.Diagram:
    return make_true_diagram(
        true_option_idx=true_option_idx,
        nodes=[
            make_true_node(
                node_number=1,
                text=node_texts[0],
                labels=[node_labels[0]],
                points_to=list(node_points_to[0]),
                true_option_idx=true_option_idx,
            ),
            make_true_node(
                node_number=2,
                text=node_texts[1],
                labels=[node_labels[1]],
                points_to=list(node_points_to[1]),
                true_option_idx=true_option_idx,
            ),
        ],
        additional_texts=[additional_text],
    )


def make_option_selection_pred_diagram(
    node_texts: tuple[str, str] = ("Node 1", "Node 2"),
    node_labels: tuple[str, str] | None = ("label 1", "label 2"),
    node_points_to: tuple[tuple[int, ...], tuple[int, ...]] | None = ((2,), ()),
    additional_text: str | None = "Participant flow diagram",
) -> types_module.Diagram:
    return make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=1,
                text=node_texts[0],
                labels=[node_labels[0]] if node_labels is not None else None,
                points_to=(
                    list(node_points_to[0]) if node_points_to is not None else None
                ),
            ),
            make_pred_node(
                node_number=2,
                text=node_texts[1],
                labels=[node_labels[1]] if node_labels is not None else None,
                points_to=(
                    list(node_points_to[1]) if node_points_to is not None else None
                ),
            ),
        ],
        additional_texts=[additional_text] if additional_text is not None else None,
    )


@pytest.mark.parametrize(
    "distance_fn",
    [
        pytest.param(levenshtein_fn, id="levenshtein"),
        pytest.param(custom_distance_fn, id="custom"),
    ],
)
def test_match_texts_lists_returns_empty_matches_for_two_empty_lists(distance_fn):
    matches = match_module.match_texts_lists(
        true_texts=[],
        pred_texts=[],
        distance_fn=distance_fn,
    )

    assert isinstance(matches, types_module.TextListMatches)
    assert matches.matches == []


@pytest.mark.parametrize(
    ("distance_fn", "true_texts", "pred_texts", "expected"),
    [
        pytest.param(
            levenshtein_fn,
            ["abc"],
            ["abc"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "abc",
                    "pred_text": "abc",
                    "cost": 0,
                }
            ],
            id="levenshtein-single-exact-match",
        ),
        pytest.param(
            custom_distance_fn,
            ["abc"],
            ["abc"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "abc",
                    "pred_text": "abc",
                    "cost": 7,
                }
            ],
            id="custom-single-exact-match",
        ),
        pytest.param(
            levenshtein_fn,
            ["abc"],
            [],
            [
                {
                    "match_type": "unmatched_true",
                    "true_index": 0,
                    "pred_index": None,
                    "true_text": "abc",
                    "pred_text": None,
                    "cost": 3,
                }
            ],
            id="levenshtein-single-unmatched-true",
        ),
        pytest.param(
            custom_distance_fn,
            ["abc"],
            [],
            [
                {
                    "match_type": "unmatched_true",
                    "true_index": 0,
                    "pred_index": None,
                    "true_text": "abc",
                    "pred_text": None,
                    "cost": 13,
                }
            ],
            id="custom-single-unmatched-true",
        ),
        pytest.param(
            levenshtein_fn,
            [],
            ["abcd"],
            [
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 0,
                    "true_text": None,
                    "pred_text": "abcd",
                    "cost": 4,
                }
            ],
            id="levenshtein-single-unmatched-pred",
        ),
        pytest.param(
            custom_distance_fn,
            [],
            ["abcd"],
            [
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 0,
                    "true_text": None,
                    "pred_text": "abcd",
                    "cost": 19,
                }
            ],
            id="custom-single-unmatched-pred",
        ),
        pytest.param(
            levenshtein_fn,
            ["cat", "dog"],
            ["cat", "dig"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 0,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dig",
                    "cost": 1,
                },
            ],
            id="levenshtein-two-matches-with-one-cost",
        ),
        pytest.param(
            custom_distance_fn,
            ["cat", "dog"],
            ["cat", "dig"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 5,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dig",
                    "cost": 8,
                },
            ],
            id="custom-two-matches-with-one-cost",
        ),
        pytest.param(
            levenshtein_fn,
            ["cat", "dog", "mouse"],
            ["cat", "dog"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 0,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dog",
                    "cost": 0,
                },
                {
                    "match_type": "unmatched_true",
                    "true_index": 2,
                    "pred_index": None,
                    "true_text": "mouse",
                    "pred_text": None,
                    "cost": 5,
                },
            ],
            id="levenshtein-more-true-than-pred",
        ),
        pytest.param(
            custom_distance_fn,
            ["cat", "dog", "mouse"],
            ["cat", "dog"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 5,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dog",
                    "cost": 6,
                },
                {
                    "match_type": "unmatched_true",
                    "true_index": 2,
                    "pred_index": None,
                    "true_text": "mouse",
                    "pred_text": None,
                    "cost": 17,
                },
            ],
            id="custom-more-true-than-pred",
        ),
        pytest.param(
            levenshtein_fn,
            ["cat", "dog"],
            ["cat", "dog", "mouse"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 0,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dog",
                    "cost": 0,
                },
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 2,
                    "true_text": None,
                    "pred_text": "mouse",
                    "cost": 5,
                },
            ],
            id="levenshtein-more-pred-than-true",
        ),
        pytest.param(
            custom_distance_fn,
            ["cat", "dog"],
            ["cat", "dog", "mouse"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "cat",
                    "pred_text": "cat",
                    "cost": 5,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 1,
                    "true_text": "dog",
                    "pred_text": "dog",
                    "cost": 6,
                },
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 2,
                    "true_text": None,
                    "pred_text": "mouse",
                    "cost": 23,
                },
            ],
            id="custom-more-pred-than-true",
        ),
        pytest.param(
            custom_distance_fn,
            ["allocated", "randomised"],
            ["allocated", "randomised"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 1,
                    "true_text": "allocated",
                    "pred_text": "randomised",
                    "cost": 1,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 0,
                    "true_text": "randomised",
                    "pred_text": "allocated",
                    "cost": 1,
                },
            ],
            id="custom-distance-changes-assignment",
        ),
        pytest.param(
            levenshtein_fn,
            ["randomised", "allocated"],
            ["allocated", "randomised"],
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 1,
                    "true_text": "randomised",
                    "pred_text": "randomised",
                    "cost": 0,
                },
                {
                    "match_type": "match",
                    "true_index": 1,
                    "pred_index": 0,
                    "true_text": "allocated",
                    "pred_text": "allocated",
                    "cost": 0,
                },
            ],
            id="levensthein-distance-does-not-match-same-indices",
        ),
    ],
)
def test_match_texts_lists_returns_expected_matches(
    distance_fn,
    true_texts,
    pred_texts,
    expected,
):
    matches = match_module.match_texts_lists(
        true_texts=true_texts,
        pred_texts=pred_texts,
        distance_fn=distance_fn,
    )

    actual = [
        {
            "match_type": match.match_type,
            "true_index": match.true_index,
            "pred_index": match.pred_index,
            "true_text": match.true_text,
            "pred_text": match.pred_text,
            "cost": match.cost,
        }
        for match in matches.matches
    ]

    assert actual == expected


def test_match_additional_texts_delegates_and_adds_diagram_metadata(
    monkeypatch,
):
    true_diagram = make_true_diagram(
        additional_texts=["True Figure 1", "True caption"],
    )
    pred_diagram = make_full_pred_diagram(
        additional_texts=["Pred Figure 1", "Pred caption"],
    )

    expected_matches = types_module.TextListMatches(
        matches=[
            types_module.TextListMatch(
                true_index=0,
                pred_index=0,
                true_text="True Figure 1",
                pred_text="Pred Figure 1",
                cost=1,
            ),
            types_module.TextListMatch(
                true_index=1,
                pred_index=1,
                true_text="True caption",
                pred_text="Pred caption",
                cost=2,
            ),
        ]
    )
    calls = []

    def mock_match_texts_lists(true_texts, pred_texts, distance_fn):
        calls.append(
            {
                "true_texts": true_texts,
                "pred_texts": pred_texts,
                "distance_fn": distance_fn,
            }
        )
        return expected_matches

    monkeypatch.setattr(match_module, "match_texts_lists", mock_match_texts_lists)

    result = match_module.match_additional_texts(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
    )

    assert result is expected_matches
    assert calls == [
        {
            "true_texts": ["True Figure 1", "True caption"],
            "pred_texts": ["Pred Figure 1", "Pred caption"],
            "distance_fn": levenshtein_fn,
        }
    ]
    assert [
        (match.parent_img_code, match.true_option_idx)
        for match in expected_matches.matches
    ] == [("diagram_1", 0), ("diagram_1", 0)]


@pytest.mark.parametrize(
    ("true_additional_texts", "pred_additional_texts"),
    [
        pytest.param(None, ["Pred figure text"], id="true-additional-texts-none"),
        pytest.param(["True figure text"], None, id="pred-additional-texts-none"),
        pytest.param(None, None, id="both-additional-texts-none"),
    ],
)
def test_match_additional_texts_raises_if_any_additional_texts_are_none(
    true_additional_texts: list[str] | None,
    pred_additional_texts: list[str] | None,
):
    true_diagram = make_true_diagram(
        additional_texts=["True figure text"],
    ).model_copy(update={"additional_texts": true_additional_texts})

    pred_diagram = make_full_pred_diagram(
        additional_texts=["Pred figure text"],
    ).model_copy(update={"additional_texts": pred_additional_texts})

    with pytest.raises(
        ValueError,
        match="Additional texts cannot be None when matching additional texts.",
    ):
        match_module.match_additional_texts(
            true_diagram=true_diagram,
            pred_diagram=pred_diagram,
            distance_fn=levenshtein_fn,
        )


@pytest.mark.parametrize(
    (
        "distance_fn",
        "true_nodes_factory",
        "pred_nodes_factory",
        "true_diagram_option_idx",
        "expected_pairs",
        "expected_total_cost",
        "expected_flow_jaccard",
        "expected_pred_details_by_true_node",
    ),
    [
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Beta",
                    labels=["b"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="Alpha",
                    labels=["a"],
                    points_to=[1],
                ),
            ],
            0,
            [
                (1, 2, 0),
                (2, 1, 0),
            ],
            0,
            0.5,
            None,
            id="levenshtein-swapped-exact-matches",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                ),
                make_pred_node(
                    node_number=2,
                    text="dig",
                    labels=["animal"],
                    points_to=[],
                ),
            ],
            1,
            [
                (1, 1, 0),
                (2, 2, 1),
            ],
            1,
            1.0,
            None,
            id="levenshtein-one-imperfect-match",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="dig",
                    labels=["animal"],
                    points_to=[1],
                ),
            ],
            2,
            [
                (1, 1, 5),
                (2, 2, 8),
            ],
            13,
            0.0,
            None,
            id="custom-one-imperfect-match",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="allocated",
                    labels=["label 1"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="randomised",
                    labels=["label 2"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="allocated",
                    labels=["label 1"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="randomised",
                    labels=["label 2"],
                    points_to=[1, 2],
                ),
            ],
            3,
            [
                (1, 2, 1),
                (2, 1, 1),
            ],
            2,
            1 / 3,
            None,
            id="custom-distance-changes-assignment",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Beta",
                    labels=["b"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="Alpha",
                    labels=["a"],
                    points_to=None,
                ),
            ],
            4,
            [
                (1, 2, 0),
                (2, 1, 0),
            ],
            0,
            None,
            None,
            id="levenshtein-swapped-exact-matches-pred-flow-not-done",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="dig",
                    labels=["animal"],
                    points_to=None,
                ),
            ],
            5,
            [
                (1, 1, 5),
                (2, 2, 8),
            ],
            13,
            None,
            None,
            id="custom-one-imperfect-match-pred-flow-not-done",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="allocated",
                    labels=["label 1"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="randomised",
                    labels=["label 2"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="allocated",
                    labels=["label 1"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="randomised",
                    labels=["label 2"],
                    points_to=None,
                ),
            ],
            6,
            [
                (1, 2, 1),
                (2, 1, 1),
            ],
            2,
            None,
            None,
            id="custom-distance-changes-assignment-pred-flow-not-done",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[],
                ),
            ],
            7,
            [
                (1, 1, 0),
                (2, None, 4),
            ],
            4,
            0.0,
            None,
            id="levenshtein-unmatched-true-node",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="mouse",
                    labels=["animal"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[],
                ),
            ],
            8,
            [
                (1, 1, 5),
                (2, None, 17),
            ],
            22,
            0.0,
            None,
            id="custom-unmatched-true-node",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[],
                ),
            ],
            9,
            [
                (1, 1, 0),
                (None, 2, 4),
            ],
            4,
            0.0,
            None,
            id="levenshtein-unmatched-pred-node",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="mouse",
                    labels=["animal"],
                    points_to=[],
                ),
            ],
            10,
            [
                (1, 1, 5),
                (None, 2, 23),
            ],
            28,
            0.0,
            None,
            id="custom-unmatched-pred-node",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Alpha",
                    labels=None,
                    points_to=[2, 1],
                ),
                make_pred_node(
                    node_number=2,
                    text="Beta",
                    labels=None,
                    points_to=[],
                ),
            ],
            11,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            1 / 3,
            None,
            id="levenshtein-pred-labels-not-done-flow-done",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=None,
                    points_to=[2, 1],
                ),
                make_pred_node(
                    node_number=2,
                    text="dog",
                    labels=None,
                    points_to=[2, 1],
                ),
            ],
            12,
            [
                (1, 1, 5),
                (2, 2, 6),
            ],
            11,
            1 / 4,
            None,
            id="custom-pred-labels-not-done-flow-done",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=None,
                ),
            ],
            13,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            None,
            None,
            id="levenshtein-pred-labels-done-flow-not-done",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["feline"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=None,
                ),
            ],
            14,
            [
                (1, 1, 5),
                (2, 2, 6),
            ],
            11,
            None,
            None,
            id="custom-pred-labels-done-flow-not-done",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Alpha",
                    labels=["a"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Beta",
                    labels=["b"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Alpha",
                    labels=None,
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="Beta",
                    labels=None,
                    points_to=None,
                ),
            ],
            15,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            None,
            None,
            id="levenshtein-pred-labels-and-flow-not-done",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["animal"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="dog",
                    labels=["animal"],
                    points_to=[1],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=None,
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="dog",
                    labels=None,
                    points_to=None,
                ),
            ],
            16,
            [
                (1, 1, 5),
                (2, 2, 6),
            ],
            11,
            None,
            None,
            id="custom-pred-labels-and-flow-not-done",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Duplicate",
                    labels=["true label 1"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Duplicate",
                    labels=["true label 2"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Duplicate",
                    labels=["pred label 1"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="Duplicate",
                    labels=["pred label 2"],
                    points_to=[1],
                ),
            ],
            17,
            [
                (1, 2, 0),
                (2, 1, 0),
            ],
            0,
            1.0,
            {
                1: {
                    "pred_labels": ["pred label 2"],
                    "pred_points_to": [1],
                },
                2: {
                    "pred_labels": ["pred label 1"],
                    "pred_points_to": [],
                },
            },
            id="levenshtein-duplicate-pred-nodes-swapped-to-maximise-flow",
        ),
        pytest.param(
            custom_distance_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="cat",
                    labels=["true label 1"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="cat",
                    labels=["true label 2"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="cat",
                    labels=["pred label 1"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=2,
                    text="cat",
                    labels=["pred label 2"],
                    points_to=[1],
                ),
            ],
            18,
            [
                (1, 2, 5),
                (2, 1, 5),
            ],
            10,
            1.0,
            {
                1: {
                    "pred_labels": ["pred label 2"],
                    "pred_points_to": [1],
                },
                2: {
                    "pred_labels": ["pred label 1"],
                    "pred_points_to": [],
                },
            },
            id="custom-duplicate-pred-nodes-swapped-to-maximise-flow",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Duplicate",
                    labels=["true label 1"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Duplicate",
                    labels=["true label 2"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=1,
                    text="Duplicate",
                    labels=["pred label 1"],
                    points_to=None,
                ),
                make_pred_node(
                    node_number=2,
                    text="Duplicate",
                    labels=["pred label 2"],
                    points_to=None,
                ),
            ],
            19,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            None,
            {
                1: {
                    "pred_labels": ["pred label 1"],
                    "pred_points_to": None,
                },
                2: {
                    "pred_labels": ["pred label 2"],
                    "pred_points_to": None,
                },
            },
            id="duplicate-pred-nodes-not-swapped-when-pred-flow-not-done",
        ),
        pytest.param(
            levenshtein_fn,
            lambda true_option_idx: [
                make_true_node(
                    node_number=1,
                    text="Start",
                    labels=["true start"],
                    points_to=[2],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=2,
                    text="Duplicate",
                    labels=["true duplicate 1"],
                    points_to=[3, 4],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=3,
                    text="Middle",
                    labels=["true middle"],
                    points_to=[5],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=4,
                    text="Duplicate",
                    labels=["true duplicate 2"],
                    points_to=[5, 6],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=5,
                    text="End",
                    labels=["true end"],
                    points_to=[],
                    true_option_idx=true_option_idx,
                ),
                make_true_node(
                    node_number=6,
                    text="Duplicate",
                    labels=["true duplicate 3"],
                    points_to=[5],
                    true_option_idx=true_option_idx,
                ),
            ],
            lambda: [
                make_pred_node(
                    node_number=47,
                    text="Middle",
                    labels=["pred middle"],
                    points_to=[91],
                ),
                make_pred_node(
                    node_number=12,
                    text="Duplicate",
                    labels=["pred duplicate right-for-true-4"],
                    points_to=[91, 37],
                ),
                make_pred_node(
                    node_number=83,
                    text="Start",
                    labels=["pred start"],
                    points_to=[64],
                ),
                make_pred_node(
                    node_number=91,
                    text="End",
                    labels=["pred end"],
                    points_to=[],
                ),
                make_pred_node(
                    node_number=64,
                    text="Duplicate",
                    labels=["pred duplicate right-for-true-2"],
                    points_to=[47, 12],
                ),
                make_pred_node(
                    node_number=37,
                    text="Duplicate",
                    labels=["pred duplicate right-for-true-6"],
                    points_to=[91],
                ),
            ],
            20,
            [
                (1, 83, 0),
                (2, 64, 0),
                (3, 47, 0),
                (4, 12, 0),
                (5, 91, 0),
                (6, 37, 0),
            ],
            0,
            1.0,
            {
                1: {
                    "pred_labels": ["pred start"],
                    "pred_points_to": [64],
                },
                2: {
                    "pred_labels": ["pred duplicate right-for-true-2"],
                    "pred_points_to": [47, 12],
                },
                3: {
                    "pred_labels": ["pred middle"],
                    "pred_points_to": [91],
                },
                4: {
                    "pred_labels": ["pred duplicate right-for-true-4"],
                    "pred_points_to": [91, 37],
                },
                5: {
                    "pred_labels": ["pred end"],
                    "pred_points_to": [],
                },
                6: {
                    "pred_labels": ["pred duplicate right-for-true-6"],
                    "pred_points_to": [91],
                },
            },
            id="levenshtein-six-nodes-three-duplicate-pred-nodes-swapped-to-maximise-flow",
        ),
    ],
)
def test_match_nodes_single_option_matches_nodes_by_minimum_text_cost(
    distance_fn,
    true_nodes_factory,
    pred_nodes_factory,
    true_diagram_option_idx,
    expected_pairs,
    expected_total_cost,
    expected_flow_jaccard,
    expected_pred_details_by_true_node,
):
    true_diagram = make_true_diagram(
        true_option_idx=true_diagram_option_idx,
        nodes=true_nodes_factory(true_diagram_option_idx),
    )
    pred_diagram = make_pred_diagram(
        nodes=pred_nodes_factory(),
        additional_texts=["Figure 1"],
    )

    node_matches = match_module.match_nodes_single_option(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
        true_diagram_option_idx=true_diagram_option_idx,
    )

    pairs = [
        (
            match.true_node.node_number if match.true_node is not None else None,
            match.pred_node.node_number if match.pred_node is not None else None,
            match.node_text_cost,
        )
        for match in node_matches.matches
    ]

    assert pairs == expected_pairs
    assert node_matches.total_node_text_cost == expected_total_cost
    assert node_matches.true_diagram_option_idx == true_diagram_option_idx

    if expected_flow_jaccard is not None:
        assert node_matches.flow_score.jaccard == pytest.approx(expected_flow_jaccard)

    if expected_pred_details_by_true_node is not None:
        pred_details_by_true_node = {
            match.true_node.node_number: {
                "pred_labels": match.pred_node.labels,
                "pred_points_to": match.pred_node.points_to,
            }
            for match in node_matches.matches
            if match.true_node is not None and match.pred_node is not None
        }

        assert pred_details_by_true_node == expected_pred_details_by_true_node


@pytest.mark.parametrize(
    (
        "true_node_specs",
        "pred_node_specs",
        "expected_true_to_pred",
        "expected_unmatched_pred_numbers",
        "expected_true_positives",
        "expected_false_positives",
        "expected_false_negatives",
        "expected_jaccard",
    ),
    [
        pytest.param(
            [(1, "Start", [2]), (2, "End", [])],
            [
                (10, "Start", [20]),
                (20, "End", []),
                (30, "Extra", [10]),
            ],
            {1: 10, 2: 20},
            {30},
            1,
            1,
            0,
            0.5,
            id="edge-from-extra-pred-node-to-matched-node",
        ),
        pytest.param(
            [(1, "Start", [2]), (2, "End", [])],
            [
                (10, "Start", [30]),
                (20, "End", []),
                (30, "Extra", []),
            ],
            {1: 10, 2: 20},
            {30},
            0,
            1,
            1,
            0.0,
            id="edge-from-matched-node-to-extra-pred-node",
        ),
        pytest.param(
            [(1, "Start", [2]), (2, "End", [])],
            [
                (10, "Start", [20]),
                (20, "End", []),
                (30, "Extra source", [40]),
                (40, "Extra target", []),
            ],
            {1: 10, 2: 20},
            {30, 40},
            1,
            1,
            0,
            0.5,
            id="edge-between-extra-pred-nodes",
        ),
        pytest.param(
            [(1, "Start", [2]), (2, "Middle", [3]), (3, "End", [])],
            [
                (10, "Start", [20, 30]),
                (20, "Middle", [30]),
                (30, "End", []),
            ],
            {1: 10, 2: 20, 3: 30},
            set(),
            2,
            1,
            0,
            2 / 3,
            id="extra-edge-between-matched-nodes",
        ),
        pytest.param(
            [(1, "Start", [2]), (2, "Middle", [3]), (3, "End", [])],
            [(10, "Start", []), (30, "End", [])],
            {1: 10, 2: None, 3: 30},
            set(),
            0,
            0,
            2,
            0.0,
            id="missing-true-node-and-its-edges",
        ),
    ],
)
def test_match_nodes_single_option_handles_edges_involving_unmatched_nodes(
    true_node_specs,
    pred_node_specs,
    expected_true_to_pred,
    expected_unmatched_pred_numbers,
    expected_true_positives,
    expected_false_positives,
    expected_false_negatives,
    expected_jaccard,
):
    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(
                node_number=node_number,
                text=text,
                labels=[f"true label {node_number}"],
                points_to=points_to,
            )
            for node_number, text, points_to in true_node_specs
        ]
    )
    pred_diagram = make_pred_diagram(
        nodes=[
            make_pred_node(
                node_number=node_number,
                text=text,
                labels=None,
                points_to=points_to,
            )
            for node_number, text, points_to in pred_node_specs
        ],
        additional_texts=["Figure 1"],
    )

    node_matches = match_module.match_nodes_single_option(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
        true_diagram_option_idx=0,
    )

    true_to_pred = {
        match.true_node.node_number: (
            match.pred_node.node_number if match.pred_node is not None else None
        )
        for match in node_matches.matches
        if match.true_node is not None
    }
    unmatched_pred_numbers = {
        match.pred_node.node_number
        for match in node_matches.matches
        if match.true_node is None and match.pred_node is not None
    }
    flow_score = node_matches.flow_score

    assert true_to_pred == expected_true_to_pred
    assert unmatched_pred_numbers == expected_unmatched_pred_numbers
    assert flow_score.tp == expected_true_positives
    assert flow_score.fp == expected_false_positives
    assert flow_score.fn == expected_false_negatives
    assert flow_score.jaccard == pytest.approx(expected_jaccard)


def test_match_nodes_single_option_returns_empty_node_matches_when_both_diagrams_have_no_nodes():
    true_diagram_option_idx = 21

    true_diagram = types_module.Diagram.model_construct(
        nodes=[],
        additional_texts=["Figure 1"],
        parent_img_code="diagram_1",
        diagram_type="pred",
        true_option_idx=None,
    )
    pred_diagram = types_module.Diagram.model_construct(
        nodes=[],
        additional_texts=["Figure 1"],
        parent_img_code="diagram_1",
        diagram_type="pred",
        true_option_idx=None,
    )

    node_matches = match_module.match_nodes_single_option(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
        true_diagram_option_idx=true_diagram_option_idx,
    )

    assert node_matches.matches == []
    assert node_matches.total_node_text_cost == 0
    assert node_matches.true_diagram_option_idx == true_diagram_option_idx


@pytest.mark.parametrize(
    ("true_nodes_missing", "pred_nodes_missing"),
    [
        pytest.param(True, False, id="true-nodes-missing"),
        pytest.param(False, True, id="pred-nodes-missing"),
        pytest.param(True, True, id="both-nodes-missing"),
    ],
)
def test_match_nodes_single_option_raises_if_nodes_missing(
    true_nodes_missing: bool,
    pred_nodes_missing: bool,
):
    true_diagram = make_true_diagram()
    pred_diagram = make_full_pred_diagram()

    if true_nodes_missing:
        true_diagram = true_diagram.model_copy(update={"nodes": None})

    if pred_nodes_missing:
        pred_diagram = pred_diagram.model_copy(update={"nodes": None})

    with pytest.raises(
        ValueError,
        match="Both diagrams must have nodes before matching nodes.",
    ):
        match_module.match_nodes_single_option(
            true_diagram=true_diagram,
            pred_diagram=pred_diagram,
            distance_fn=levenshtein_fn,
            true_diagram_option_idx=0,
        )


@pytest.mark.parametrize(
    ("true_text_done", "pred_text_done"),
    [
        pytest.param(False, True, id="true-text-not-done"),
        pytest.param(True, False, id="pred-text-not-done"),
        pytest.param(False, False, id="both-text-not-done"),
    ],
)
def test_match_nodes_single_option_raises_if_text_not_done(
    true_text_done: bool,
    pred_text_done: bool,
):
    true_diagram = make_true_diagram()
    pred_diagram = make_full_pred_diagram()

    if not true_text_done:
        true_node_without_text = true_diagram.nodes[0].model_copy(update={"text": None})
        true_diagram = true_diagram.model_copy(
            update={
                "nodes": [
                    true_node_without_text,
                    *true_diagram.nodes[1:],
                ]
            }
        )

    if not pred_text_done:
        pred_node_without_text = pred_diagram.nodes[0].model_copy(update={"text": None})
        pred_diagram = pred_diagram.model_copy(
            update={
                "nodes": [
                    pred_node_without_text,
                    *pred_diagram.nodes[1:],
                ]
            }
        )

    with pytest.raises(
        ValueError,
        match="Both diagrams must have text_done=True before matching nodes.",
    ):
        match_module.match_nodes_single_option(
            true_diagram=true_diagram,
            pred_diagram=pred_diagram,
            distance_fn=levenshtein_fn,
            true_diagram_option_idx=0,
        )


@pytest.mark.parametrize(
    (
        "distance_fn",
        "true_diagrams_factory",
        "pred_diagram_factory",
        "expected_true_diagram_option_idx",
        "expected_pairs",
        "expected_total_cost",
    ),
    [
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Wrong 1",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Wrong 2",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Node 2",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Node 1",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Node 2",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            1,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            id="levenshtein-selects-second-option",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Beta",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Gamma",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Delta",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=10,
                        text="Alpha",
                        labels=["label 1"],
                        points_to=[20],
                    ),
                    make_pred_node(
                        node_number=20,
                        text="Beta",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 10, 0),
                (2, 20, 0),
            ],
            0,
            id="levenshtein-selects-first-option-with-nonmatching-pred-node-numbers",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Beta",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Beta",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Alpha",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Beta",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            id="levenshtein-tie-selects-first-option",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Beta",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Beta",
                            labels=["label 2"],
                            points_to=[3],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=3,
                            text="Gamma",
                            labels=["label 3"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Alpha",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Beta",
                        labels=["label 2"],
                        points_to=[],
                    ),
                    make_pred_node(
                        node_number=3,
                        text="Gamma",
                        labels=["label 3"],
                        points_to=[],
                    ),
                ],
            ),
            1,
            [
                (1, 1, 0),
                (2, 2, 0),
                (3, 3, 0),
            ],
            0,
            id="levenshtein-selects-option-with-more-nodes-when-it-fits-best",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="allocated",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="randomised",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="cat",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="dog",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="allocated",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="randomised",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 2, 1),
                (2, 1, 1),
            ],
            2,
            id="custom-selects-first-option-with-crossed-assignment",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="allocated",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="randomised",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="cat",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="dog",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="cat",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="dog",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            1,
            [
                (1, 1, 5),
                (2, 2, 6),
            ],
            11,
            id="custom-selects-second-option",
        ),
    ],
)
def test_match_nodes_selects_true_option_with_lowest_node_text_cost(
    distance_fn,
    true_diagrams_factory,
    pred_diagram_factory,
    expected_true_diagram_option_idx,
    expected_pairs,
    expected_total_cost,
):
    true_diagrams = true_diagrams_factory()
    pred_diagram = pred_diagram_factory()

    true_diagram_options = make_diagram_options(
        options=true_diagrams,
        parent_img_code=pred_diagram.parent_img_code,
    )

    node_matches = match_module.match_nodes(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
    )

    pairs = [
        (
            match.true_node.node_number if match.true_node is not None else None,
            match.pred_node.node_number if match.pred_node is not None else None,
            match.node_text_cost,
        )
        for match in node_matches.matches
    ]

    assert node_matches.true_diagram_option_idx == expected_true_diagram_option_idx
    assert node_matches.total_node_text_cost == expected_total_cost
    assert pairs == expected_pairs


@pytest.mark.parametrize(
    (
        "true_diagram_options",
        "pred_diagram",
        "expected_true_diagram_option_idx",
    ),
    [
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        node_labels=("Wrong label 1", "Wrong label 2"),
                        node_points_to=((), (1,)),
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(
                        true_option_idx=1,
                        node_texts=("Other node 1", "Other node 2"),
                    ),
                ]
            ),
            make_option_selection_pred_diagram(),
            0,
            id="node-text-takes-priority-over-all-later-evidence",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        node_points_to=((), (1,)),
                    ),
                    make_option_selection_true_diagram(
                        true_option_idx=1,
                        node_labels=("Wrong label 1", "Wrong label 2"),
                        additional_text="Wrong caption",
                    ),
                ]
            ),
            make_option_selection_pred_diagram(),
            1,
            id="flow-breaks-node-text-tie-before-labels-and-additional-text",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(
                        true_option_idx=1,
                        node_labels=("Wrong label 1", "Wrong label 2"),
                    ),
                ]
            ),
            make_option_selection_pred_diagram(),
            0,
            id="labels-break-node-text-and-flow-tie-before-additional-text",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(true_option_idx=1),
                ]
            ),
            make_option_selection_pred_diagram(),
            1,
            id="additional-text-breaks-node-flow-and-label-tie",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(true_option_idx=0),
                    make_option_selection_true_diagram(true_option_idx=1),
                ]
            ),
            make_option_selection_pred_diagram(),
            0,
            id="first-option-wins-complete-tie",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        node_labels=("Wrong label 1", "Wrong label 2"),
                    ),
                    make_option_selection_true_diagram(
                        true_option_idx=1,
                        additional_text="Wrong caption",
                    ),
                ]
            ),
            make_option_selection_pred_diagram(node_points_to=None),
            1,
            id="labels-break-tie-when-flow-not-parsed",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(true_option_idx=1),
                ]
            ),
            make_option_selection_pred_diagram(node_labels=None),
            1,
            id="additional-text-breaks-tie-when-labels-not-parsed",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(true_option_idx=1),
                ]
            ),
            make_option_selection_pred_diagram(node_points_to=None),
            1,
            id="additional-text-breaks-tie-when-flow-not-parsed",
        ),
        pytest.param(
            make_diagram_options(
                options=[
                    make_option_selection_true_diagram(
                        true_option_idx=0,
                        additional_text="Wrong caption",
                    ),
                    make_option_selection_true_diagram(true_option_idx=1),
                ]
            ),
            make_option_selection_pred_diagram(
                node_labels=None,
                node_points_to=None,
            ),
            1,
            id="additional-text-breaks-tie-without-labels-or-flow",
        ),
    ],
)
def test_match_nodes_selects_true_option_using_available_evidence_in_order(
    true_diagram_options,
    pred_diagram,
    expected_true_diagram_option_idx,
):
    node_matches = match_module.match_nodes(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
    )

    assert node_matches.true_diagram_option_idx == expected_true_diagram_option_idx


def test_match_nodes_calls_all_true_options_and_returns_lowest_cost_match(
    monkeypatch,
):
    true_diagrams = [
        make_true_diagram(true_option_idx=0),
        make_true_diagram(true_option_idx=1),
        make_true_diagram(true_option_idx=2),
    ]
    true_diagram_options = make_diagram_options(options=true_diagrams)
    pred_diagram = make_full_pred_diagram()

    calls = []

    class FakeNodeMatches:
        def __init__(self, true_diagram_option_idx: int, total_node_text_cost: int):
            self.true_diagram_option_idx = true_diagram_option_idx
            self.total_node_text_cost = total_node_text_cost

    option_costs = {
        0: 10,
        1: 2,
        2: 7,
    }

    def fake_match_nodes_single_option(
        true_diagram,
        pred_diagram,
        distance_fn,
        true_diagram_option_idx,
    ):
        calls.append(
            {
                "true_diagram": true_diagram,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
                "true_diagram_option_idx": true_diagram_option_idx,
            }
        )

        return FakeNodeMatches(
            true_diagram_option_idx=true_diagram_option_idx,
            total_node_text_cost=option_costs[true_diagram_option_idx],
        )

    monkeypatch.setattr(
        match_module,
        "match_nodes_single_option",
        fake_match_nodes_single_option,
    )

    best_node_matches = match_module.match_nodes(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
    )

    assert best_node_matches.true_diagram_option_idx == 1
    assert best_node_matches.total_node_text_cost == 2

    assert len(calls) == 3

    assert [call["true_diagram_option_idx"] for call in calls] == [0, 1, 2]

    assert [call["true_diagram"] for call in calls] == true_diagrams

    assert all(call["pred_diagram"] is pred_diagram for call in calls)

    assert all(call["distance_fn"] is levenshtein_fn for call in calls)


@pytest.mark.parametrize(
    (
        "distance_fn",
        "node_matches_factory",
        "expected_label_matches_by_node_match_idx",
        "expected_total_label_error_cost",
    ),
    [
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                ],
            },
            0,
            id="levenshtein-single-exact-label-match",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["randomised"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["randomized"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
            },
            1,
            id="levenshtein-single-imperfect-label-match",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened", "excluded"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "unmatched_true",
                        "true_index": 1,
                        "pred_index": None,
                        "true_text": "excluded",
                        "pred_text": None,
                        "cost": 8,
                    },
                ],
            },
            8,
            id="levenshtein-matched-node-has-extra-true-label",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened", "excluded"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 1,
                        "true_text": None,
                        "pred_text": "excluded",
                        "cost": 8,
                    },
                ],
            },
            8,
            id="levenshtein-matched-node-has-extra-pred-label",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened", "randomised"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["randomized", "screened"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 1,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
            },
            1,
            id="levenshtein-labels-reordered",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=[],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=[],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [],
            },
            0,
            id="levenshtein-matched-node-both-label-lists-empty",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["excluded"],
                            points_to=[2],
                        ),
                        pred_node=None,
                        node_text_cost=6,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "unmatched_true",
                        "true_index": 0,
                        "pred_index": None,
                        "true_text": "excluded",
                        "pred_text": None,
                        "cost": 8,
                    },
                ],
            },
            8,
            id="levenshtein-unmatched-true-node-labels",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=None,
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Extra node",
                            labels=["extra"],
                            points_to=[],
                        ),
                        node_text_cost=10,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 0,
                        "true_text": None,
                        "pred_text": "extra",
                        "cost": 5,
                    },
                ],
            },
            5,
            id="levenshtein-unmatched-pred-node-labels",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=2,
                            text="Node two",
                            labels=["randomised", "excluded"],
                            points_to=[1],
                        ),
                        pred_node=make_pred_node(
                            node_number=2,
                            text="Node tw0",
                            labels=["randomized"],
                            points_to=[],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                    types_module.NodeMatch(
                        true_node=None,
                        pred_node=make_pred_node(
                            node_number=3,
                            text="Extra node",
                            labels=["extra"],
                            points_to=[],
                        ),
                        node_text_cost=10,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                ],
                1: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                    {
                        "match_type": "unmatched_true",
                        "true_index": 1,
                        "pred_index": None,
                        "true_text": "excluded",
                        "pred_text": None,
                        "cost": 8,
                    },
                ],
                2: [
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 0,
                        "true_text": None,
                        "pred_text": "extra",
                        "cost": 5,
                    },
                ],
            },
            14,
            id="levenshtein-multiple-node-matches-mixed-label-outcomes",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["cat", "dog"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["cat", "dig"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "cat",
                        "pred_text": "cat",
                        "cost": 5,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 1,
                        "true_text": "dog",
                        "pred_text": "dig",
                        "cost": 8,
                    },
                ],
            },
            13,
            id="custom-matched-node-labels-use-custom-costs",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["allocated", "randomised"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["allocated", "randomised"],
                            points_to=[2],
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 1,
                        "true_text": "allocated",
                        "pred_text": "randomised",
                        "cost": 1,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "allocated",
                        "cost": 1,
                    },
                ],
            },
            2,
            id="custom-distance-changes-label-assignment",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["mouse"],
                            points_to=[2],
                        ),
                        pred_node=None,
                        node_text_cost=5,
                        label_matches=None,
                    ),
                    types_module.NodeMatch(
                        true_node=None,
                        pred_node=make_pred_node(
                            node_number=2,
                            text="Extra node",
                            labels=["mouse"],
                            points_to=[],
                        ),
                        node_text_cost=5,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "unmatched_true",
                        "true_index": 0,
                        "pred_index": None,
                        "true_text": "mouse",
                        "pred_text": None,
                        "cost": 17,
                    },
                ],
                1: [
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 0,
                        "true_text": None,
                        "pred_text": "mouse",
                        "cost": 23,
                    },
                ],
            },
            40,
            id="custom-unmatched-true-and-pred-label-costs",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: types_module.NodeMatches(
                matches=[
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened", "randomised"],
                            points_to=[2],
                        ),
                        pred_node=make_pred_node(
                            node_number=1,
                            text="Node 1",
                            labels=["screened", "randomized"],
                            points_to=None,
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                    types_module.NodeMatch(
                        true_node=make_true_node(
                            node_number=2,
                            text="Node 2",
                            labels=["excluded"],
                            points_to=[1],
                        ),
                        pred_node=make_pred_node(
                            node_number=2,
                            text="Node 2",
                            labels=["excluded"],
                            points_to=None,
                        ),
                        node_text_cost=0,
                        label_matches=None,
                    ),
                ],
                true_diagram_option_idx=0,
            ),
            {
                0: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 1,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
                1: [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "excluded",
                        "pred_text": "excluded",
                        "cost": 0,
                    },
                ],
            },
            1,
            id="levenshtein-label-matching-works-when-pred-flow-not-done",
        ),
    ],
)
def test_match_node_labels_adds_expected_label_matches(
    distance_fn,
    node_matches_factory,
    expected_label_matches_by_node_match_idx,
    expected_total_label_error_cost,
):
    node_matches = node_matches_factory()

    result = match_module.match_node_labels(
        node_matches=node_matches,
        distance_fn=distance_fn,
    )

    assert result is node_matches
    assert node_matches.total_label_error_cost == expected_total_label_error_cost

    actual_label_matches_by_node_match_idx = {
        node_match_idx: [
            {
                "match_type": label_match.match_type,
                "true_index": label_match.true_index,
                "pred_index": label_match.pred_index,
                "true_text": label_match.true_text,
                "pred_text": label_match.pred_text,
                "cost": label_match.cost,
            }
            for label_match in node_match.label_matches.matches
        ]
        for node_match_idx, node_match in enumerate(node_matches.matches)
    }

    assert (
        actual_label_matches_by_node_match_idx
        == expected_label_matches_by_node_match_idx
    )


def test_match_node_labels_calls_match_texts_lists_with_expected_arguments_for_each_match_type(
    monkeypatch,
):
    matched_true_node = make_true_node(
        node_number=1,
        text="Matched true node",
        labels=["matched true label"],
        points_to=[2],
    )
    unmatched_true_node = make_true_node(
        node_number=2,
        text="Unmatched true node",
        labels=["unmatched true label"],
        points_to=[1],
    )

    matched_pred_node = make_pred_node(
        node_number=1,
        text="Matched pred node",
        labels=["matched pred label"],
        points_to=[],
    )
    unmatched_pred_node = make_pred_node(
        node_number=2,
        text="Unmatched pred node",
        labels=["unmatched pred label"],
        points_to=[],
    )

    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=matched_true_node,
                pred_node=matched_pred_node,
                node_text_cost=0,
                label_matches=None,
            ),
            types_module.NodeMatch(
                true_node=unmatched_true_node,
                pred_node=None,
                node_text_cost=5,
                label_matches=None,
            ),
            types_module.NodeMatch(
                true_node=None,
                pred_node=unmatched_pred_node,
                node_text_cost=5,
                label_matches=None,
            ),
        ],
        true_diagram_option_idx=0,
    )

    calls = []

    fake_label_matches = [
        types_module.TextListMatches(
            matches=[
                types_module.TextListMatch(
                    true_index=0,
                    pred_index=0,
                    true_text="matched true label",
                    pred_text="matched pred label",
                    cost=1,
                )
            ]
        ),
        types_module.TextListMatches(
            matches=[
                types_module.TextListMatch(
                    true_index=0,
                    pred_index=None,
                    true_text="unmatched true label",
                    pred_text=None,
                    cost=2,
                )
            ]
        ),
        types_module.TextListMatches(
            matches=[
                types_module.TextListMatch(
                    true_index=None,
                    pred_index=0,
                    true_text=None,
                    pred_text="unmatched pred label",
                    cost=3,
                )
            ]
        ),
    ]

    def fake_match_texts_lists(true_texts, pred_texts, distance_fn):
        calls.append(
            {
                "true_texts": true_texts,
                "pred_texts": pred_texts,
                "distance_fn": distance_fn,
            }
        )
        return fake_label_matches[len(calls) - 1]

    monkeypatch.setattr(
        match_module,
        "match_texts_lists",
        fake_match_texts_lists,
    )

    result = match_module.match_node_labels(
        node_matches=node_matches,
        distance_fn=levenshtein_fn,
    )

    assert result is node_matches

    assert calls == [
        {
            "true_texts": ["matched true label"],
            "pred_texts": ["matched pred label"],
            "distance_fn": levenshtein_fn,
        },
        {
            "true_texts": ["unmatched true label"],
            "pred_texts": [],
            "distance_fn": levenshtein_fn,
        },
        {
            "true_texts": [],
            "pred_texts": ["unmatched pred label"],
            "distance_fn": levenshtein_fn,
        },
    ]

    assert node_matches.matches[0].label_matches == fake_label_matches[0]
    assert node_matches.matches[1].label_matches == fake_label_matches[1]
    assert node_matches.matches[2].label_matches == fake_label_matches[2]


@pytest.mark.parametrize(
    ("true_labels", "pred_labels"),
    [
        pytest.param(None, ["pred label"], id="true-labels-none"),
        pytest.param(["true label"], None, id="pred-labels-none"),
        pytest.param(None, None, id="both-labels-none"),
    ],
)
def test_match_node_labels_raises_if_any_node_labels_are_none(
    true_labels: list[str] | None,
    pred_labels: list[str] | None,
):
    true_node = make_true_node(
        node_number=1,
        text="Node 1",
        labels=["true label"],
        points_to=[],
    ).model_copy(update={"labels": true_labels})

    pred_node = make_pred_node(
        node_number=1,
        text="Node 1",
        labels=["pred label"],
        points_to=[],
    ).model_copy(update={"labels": pred_labels})

    node_matches = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_node,
                pred_node=pred_node,
                node_text_cost=0,
                label_matches=None,
            )
        ],
        true_diagram_option_idx=0,
    )

    with pytest.raises(
        ValueError,
        match="Node labels cannot be None when matching node labels.",
    ):
        match_module.match_node_labels(
            node_matches=node_matches,
            distance_fn=levenshtein_fn,
        )


def test_match_additional_texts_returns_text_list_matches():
    true_diagram = make_true_diagram(additional_texts=["Figure 1", "Caption"])
    pred_diagram = make_full_pred_diagram(additional_texts=["Figure 1", "Other"])

    matches = match_module.match_additional_texts(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
    )

    assert len(matches.matches) == 2
    assert [match.match_type for match in matches.matches] == ["match", "match"]
    assert matches.matches[0].true_text == "Figure 1"
    assert matches.matches[0].pred_text == "Figure 1"
    assert matches.matches[0].cost == 0


def test_match_single_diagram_calls_helpers_with_expected_arguments_and_returns_diagram_match(
    monkeypatch,
):
    true_diagram_0 = make_true_diagram(
        true_option_idx=0,
        additional_texts=["Wrong Figure"],
        nodes=[
            make_true_node(
                node_number=1,
                text="Wrong node 1",
                labels=["wrong label 1"],
                points_to=[2],
                true_option_idx=0,
            ),
            make_true_node(
                node_number=2,
                text="Wrong node 2",
                labels=["wrong label 2"],
                points_to=[1],
                true_option_idx=0,
            ),
        ],
    )
    true_diagram_1 = make_true_diagram(
        true_option_idx=1,
        additional_texts=["True Figure 1"],
        nodes=[
            make_true_node(
                node_number=1,
                text="Node 1",
                labels=["true label 1"],
                points_to=[2],
                true_option_idx=1,
            ),
            make_true_node(
                node_number=2,
                text="Node 2",
                labels=["true label 2"],
                points_to=[1],
                true_option_idx=1,
            ),
        ],
    )
    true_diagram_options = make_diagram_options(
        options=[true_diagram_0, true_diagram_1],
    )
    pred_diagram = make_full_pred_diagram(
        additional_texts=["Pred Figure 1"],
        nodes=[
            make_pred_node(
                node_number=1,
                text="Node 1",
                labels=["pred label 1"],
                points_to=[2],
            ),
            make_pred_node(
                node_number=2,
                text="Node 2",
                labels=["pred label 2"],
                points_to=[],
            ),
        ],
    )

    node_matches_from_match_nodes = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_diagram_1.nodes[0],
                pred_node=pred_diagram.nodes[0],
                node_text_cost=0,
                label_matches=None,
            ),
            types_module.NodeMatch(
                true_node=true_diagram_1.nodes[1],
                pred_node=pred_diagram.nodes[1],
                node_text_cost=0,
                label_matches=None,
            ),
        ],
        true_diagram_option_idx=1,
    )

    label_matches = types_module.TextListMatches(
        matches=[
            types_module.TextListMatch(
                true_index=0,
                pred_index=0,
                true_text="true label 1",
                pred_text="pred label 1",
                cost=5,
            )
        ]
    )

    node_matches_with_labels = types_module.NodeMatches(
        matches=[
            types_module.NodeMatch(
                true_node=true_diagram_1.nodes[0],
                pred_node=pred_diagram.nodes[0],
                node_text_cost=0,
                label_matches=label_matches,
            ),
            types_module.NodeMatch(
                true_node=true_diagram_1.nodes[1],
                pred_node=pred_diagram.nodes[1],
                node_text_cost=0,
                label_matches=types_module.TextListMatches(matches=[]),
            ),
        ],
        true_diagram_option_idx=1,
    )

    additional_text_matches = types_module.TextListMatches(
        matches=[
            types_module.TextListMatch(
                true_index=0,
                pred_index=0,
                true_text="True Figure 1",
                pred_text="Pred Figure 1",
                cost=4,
            )
        ]
    )

    calls = []

    def fake_match_nodes(
        true_diagram_options,
        pred_diagram,
        distance_fn,
    ):
        calls.append(
            {
                "function": "match_nodes",
                "true_diagram_options": true_diagram_options,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )
        return node_matches_from_match_nodes

    def fake_match_node_labels(
        node_matches,
        distance_fn,
    ):
        calls.append(
            {
                "function": "match_node_labels",
                "node_matches": node_matches,
                "distance_fn": distance_fn,
            }
        )
        return node_matches_with_labels

    def fake_match_additional_texts(
        true_diagram,
        pred_diagram,
        distance_fn,
    ):
        calls.append(
            {
                "function": "match_additional_texts",
                "true_diagram": true_diagram,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )
        return additional_text_matches

    monkeypatch.setattr(match_module, "match_nodes", fake_match_nodes)
    monkeypatch.setattr(match_module, "match_node_labels", fake_match_node_labels)
    monkeypatch.setattr(
        match_module,
        "match_additional_texts",
        fake_match_additional_texts,
    )

    diagram_match = match_module.match_single_diagram(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=levenshtein_fn,
    )

    assert isinstance(diagram_match, types_module.DiagramMatch)

    assert calls == [
        {
            "function": "match_nodes",
            "true_diagram_options": true_diagram_options,
            "pred_diagram": pred_diagram,
            "distance_fn": levenshtein_fn,
        },
        {
            "function": "match_node_labels",
            "node_matches": node_matches_from_match_nodes,
            "distance_fn": levenshtein_fn,
        },
        {
            "function": "match_additional_texts",
            "true_diagram": true_diagram_1,
            "pred_diagram": pred_diagram,
            "distance_fn": levenshtein_fn,
        },
    ]

    assert diagram_match.node_matches == node_matches_with_labels
    assert diagram_match.additional_text_matches == additional_text_matches
    assert diagram_match.pred_diagram == pred_diagram
    assert diagram_match.true_diagram == true_diagram_1


@pytest.mark.parametrize(
    (
        "distance_fn",
        "true_diagrams_factory",
        "pred_diagram_factory",
        "expected_true_diagram_option_idx",
        "expected_node_pairs",
        "expected_total_node_text_cost",
        "expected_label_matches",
        "expected_total_label_error_cost",
        "expected_additional_text_matches",
        "expected_total_additional_text_cost",
    ),
    [
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Wrong Figure"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Wrong node 1",
                            labels=["wrong label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Wrong node 2",
                            labels=["wrong label 2"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    additional_texts=["Figure 1"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Node 2",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Figure 1"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Node 1",
                        labels=["label 1"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Node 2",
                        labels=["label 2"],
                        points_to=[],
                    ),
                ],
            ),
            1,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "label 1",
                        "pred_text": "label 1",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "label 2",
                        "pred_text": "label 2",
                        "cost": 0,
                    }
                ],
            ],
            0,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Figure 1",
                    "pred_text": "Figure 1",
                    "cost": 0,
                }
            ],
            0,
            id="levenshtein-selects-best-true-option-with-perfect-match",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Figure 1"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["cat"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    additional_texts=["Wrong Figure"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Wrong",
                            labels=["dog"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Figure X"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Alpha",
                        labels=["cat"],
                        points_to=[],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Beta",
                        labels=["mouse"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (None, 2, 4),
            ],
            4,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "cat",
                        "pred_text": "cat",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 0,
                        "true_text": None,
                        "pred_text": "mouse",
                        "cost": 5,
                    }
                ],
            ],
            5,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Figure 1",
                    "pred_text": "Figure X",
                    "cost": 1,
                }
            ],
            1,
            id="levenshtein-handles-unmatched-pred-node-labels-and-additional-text-cost",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Start",
                            labels=["screened", "randomised"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="End",
                            labels=["excluded"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Caption", "Extra note"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Start",
                        labels=["screened", "randomized"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="End",
                        labels=["included"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 1,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "excluded",
                        "pred_text": "included",
                        "cost": 2,
                    }
                ],
            ],
            3,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                },
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 1,
                    "true_text": None,
                    "pred_text": "Extra note",
                    "cost": 10,
                },
            ],
            10,
            id="levenshtein-label-costs-and-extra-pred-additional-text",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["allocated"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="allocated",
                            labels=["allocated"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="randomised",
                            labels=["randomised"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
                make_true_diagram(
                    true_option_idx=1,
                    additional_texts=["cat"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="cat",
                            labels=["cat"],
                            points_to=[2],
                            true_option_idx=1,
                        ),
                        make_true_node(
                            node_number=2,
                            text="dog",
                            labels=["dog"],
                            points_to=[1],
                            true_option_idx=1,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["randomised"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="allocated",
                        labels=["allocated"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="randomised",
                        labels=["randomised"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 2, 1),
                (2, 1, 1),
            ],
            2,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "allocated",
                        "pred_text": "randomised",
                        "cost": 1,
                    }
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "allocated",
                        "cost": 1,
                    }
                ],
            ],
            2,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "allocated",
                    "pred_text": "randomised",
                    "cost": 1,
                }
            ],
            1,
            id="custom-distance-used-for-node-label-and-additional-text-matching",
        ),
        pytest.param(
            custom_distance_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["extreme"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="abc",
                            labels=["abc"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["extreme"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="abc",
                        labels=["abc", "mouse"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 7),
            ],
            7,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "abc",
                        "pred_text": "abc",
                        "cost": 7,
                    },
                    {
                        "match_type": "unmatched_pred",
                        "true_index": None,
                        "pred_index": 1,
                        "true_text": None,
                        "pred_text": "mouse",
                        "cost": 23,
                    },
                ],
            ],
            30,
            [
                {
                    "match_type": "unmatched_true",
                    "true_index": 0,
                    "pred_index": None,
                    "true_text": "extreme",
                    "pred_text": None,
                    "cost": 89,
                },
                {
                    "match_type": "unmatched_pred",
                    "true_index": None,
                    "pred_index": 0,
                    "true_text": None,
                    "pred_text": "extreme",
                    "cost": 97,
                },
            ],
            186,
            id="custom-distance-with-extra-pred-label-and-none-matches-for-additional-text",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Start",
                            labels=["screened"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Missing true node",
                            labels=["missing label"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Caption"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Start",
                        labels=["screened"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (2, None, len("Missing true node")),
            ],
            len("Missing true node"),
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "unmatched_true",
                        "true_index": 0,
                        "pred_index": None,
                        "true_text": "missing label",
                        "pred_text": None,
                        "cost": len("missing label"),
                    }
                ],
            ],
            len("missing label"),
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                }
            ],
            0,
            id="levenshtein-unmatched-true-node-labels",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption", "Trial flow"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Start",
                            labels=["screened"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="End",
                            labels=["included"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Caption"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Start",
                        labels=["screened"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="End",
                        labels=["included"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "included",
                        "pred_text": "included",
                        "cost": 0,
                    }
                ],
            ],
            0,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                },
                {
                    "match_type": "unmatched_true",
                    "true_index": 1,
                    "pred_index": None,
                    "true_text": "Trial flow",
                    "pred_text": None,
                    "cost": len("Trial flow"),
                },
            ],
            len("Trial flow"),
            id="levenshtein-extra-true-additional-text",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Start",
                            labels=["screened", "randomised"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="End",
                            labels=["excluded", "included"],
                            points_to=[1],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Caption"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Start",
                        labels=["randomized", "screened"],
                        points_to=[2],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="End",
                        labels=["included", "excluded"],
                        points_to=[],
                    ),
                ],
            ),
            0,
            [
                (1, 1, 0),
                (2, 2, 0),
            ],
            0,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 1,
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 0,
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 1,
                        "true_text": "excluded",
                        "pred_text": "excluded",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_index": 1,
                        "pred_index": 0,
                        "true_text": "included",
                        "pred_text": "included",
                        "cost": 0,
                    },
                ],
            ],
            1,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                }
            ],
            0,
            id="levenshtein-reordered-labels",
        ),
        pytest.param(
            levenshtein_fn,
            lambda: [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Duplicate",
                            labels=["true label 1"],
                            points_to=[2],
                            true_option_idx=0,
                        ),
                        make_true_node(
                            node_number=2,
                            text="Duplicate",
                            labels=["true label 2"],
                            points_to=[],
                            true_option_idx=0,
                        ),
                    ],
                ),
            ],
            lambda: make_full_pred_diagram(
                additional_texts=["Caption"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Duplicate",
                        labels=["pred label 1"],
                        points_to=[],
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Duplicate",
                        labels=["pred label 2"],
                        points_to=[1],
                    ),
                ],
            ),
            0,
            [
                (1, 2, 0),
                (2, 1, 0),
            ],
            0,
            [
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "true label 1",
                        "pred_text": "pred label 2",
                        "cost": 4,
                    }
                ],
                [
                    {
                        "match_type": "match",
                        "true_index": 0,
                        "pred_index": 0,
                        "true_text": "true label 2",
                        "pred_text": "pred label 1",
                        "cost": 4,
                    }
                ],
            ],
            8,
            [
                {
                    "match_type": "match",
                    "true_index": 0,
                    "pred_index": 0,
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                }
            ],
            0,
            id="levenshtein-duplicate-pred-nodes-swapped-before-label-matching",
        ),
    ],
)
def test_match_single_diagram_returns_expected_diagram_match(
    distance_fn,
    true_diagrams_factory,
    pred_diagram_factory,
    expected_true_diagram_option_idx,
    expected_node_pairs,
    expected_total_node_text_cost,
    expected_label_matches,
    expected_total_label_error_cost,
    expected_additional_text_matches,
    expected_total_additional_text_cost,
):
    true_diagrams = true_diagrams_factory()
    pred_diagram = pred_diagram_factory()
    true_diagram_options = make_diagram_options(
        options=true_diagrams,
        parent_img_code=pred_diagram.parent_img_code,
    )

    diagram_match = match_module.match_single_diagram(
        true_diagram_options=true_diagram_options,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
    )

    assert isinstance(diagram_match, types_module.DiagramMatch)
    assert diagram_match.pred_diagram == pred_diagram
    assert diagram_match.true_diagram == true_diagrams[expected_true_diagram_option_idx]
    assert diagram_match.node_matches.true_diagram_option_idx == (
        expected_true_diagram_option_idx
    )

    node_pairs = [
        (
            node_match.true_node.node_number
            if node_match.true_node is not None
            else None,
            node_match.pred_node.node_number
            if node_match.pred_node is not None
            else None,
            node_match.node_text_cost,
        )
        for node_match in diagram_match.node_matches.matches
    ]

    assert node_pairs == expected_node_pairs
    assert diagram_match.node_matches.total_node_text_cost == (
        expected_total_node_text_cost
    )

    label_matches = [
        [
            {
                "match_type": label_match.match_type,
                "true_index": label_match.true_index,
                "pred_index": label_match.pred_index,
                "true_text": label_match.true_text,
                "pred_text": label_match.pred_text,
                "cost": label_match.cost,
            }
            for label_match in node_match.label_matches.matches
        ]
        for node_match in diagram_match.node_matches.matches
    ]

    assert label_matches == expected_label_matches
    assert diagram_match.total_label_error_cost == expected_total_label_error_cost

    additional_text_matches = [
        {
            "match_type": match.match_type,
            "true_index": match.true_index,
            "pred_index": match.pred_index,
            "true_text": match.true_text,
            "pred_text": match.pred_text,
            "cost": match.cost,
        }
        for match in diagram_match.additional_text_matches.matches
    ]

    assert additional_text_matches == expected_additional_text_matches
    assert (
        diagram_match.total_additional_text_cost == expected_total_additional_text_cost
    )
    assert diagram_match.total_text_cost == (
        expected_total_node_text_cost
        + expected_total_label_error_cost
        + expected_total_additional_text_cost
    )


def test_match_diagrams_calls_match_single_diagram_for_each_true_pred_pair(
    monkeypatch,
):
    true_options_1 = make_diagram_options(
        options=[make_true_diagram(parent_img_code="diagram_1")],
        parent_img_code="diagram_1",
    )
    true_options_2 = make_diagram_options(
        options=[make_true_diagram(parent_img_code="diagram_2")],
        parent_img_code="diagram_2",
    )

    pred_diagram_1 = make_full_pred_diagram(parent_img_code="diagram_1")
    pred_diagram_2 = make_full_pred_diagram(parent_img_code="diagram_2")

    expected_match_1 = types_module.DiagramMatch(
        node_matches=types_module.NodeMatches(
            matches=[
                types_module.NodeMatch(
                    true_node=true_options_1.options[0].nodes[0],
                    pred_node=pred_diagram_1.nodes[0],
                    node_text_cost=0,
                    label_matches=types_module.TextListMatches(
                        matches=[
                            types_module.TextListMatch(
                                true_index=0,
                                pred_index=0,
                                true_text="label 1",
                                pred_text="label 1",
                                cost=0,
                            )
                        ]
                    ),
                ),
                types_module.NodeMatch(
                    true_node=true_options_1.options[0].nodes[1],
                    pred_node=pred_diagram_1.nodes[1],
                    node_text_cost=0,
                    label_matches=types_module.TextListMatches(
                        matches=[
                            types_module.TextListMatch(
                                true_index=0,
                                pred_index=0,
                                true_text="label 2",
                                pred_text="label 2",
                                cost=0,
                            )
                        ]
                    ),
                ),
            ],
            true_diagram_option_idx=0,
        ),
        additional_text_matches=types_module.TextListMatches(
            matches=[
                types_module.TextListMatch(
                    true_index=0,
                    pred_index=0,
                    true_text="Figure 1",
                    pred_text="Figure 1",
                    cost=0,
                )
            ]
        ),
        pred_diagram=pred_diagram_1,
        true_diagram=true_options_1.options[0],
    )

    expected_match_2 = types_module.DiagramMatch(
        node_matches=types_module.NodeMatches(
            matches=[
                types_module.NodeMatch(
                    true_node=true_options_2.options[0].nodes[0],
                    pred_node=pred_diagram_2.nodes[0],
                    node_text_cost=0,
                    label_matches=types_module.TextListMatches(
                        matches=[
                            types_module.TextListMatch(
                                true_index=0,
                                pred_index=0,
                                true_text="label 1",
                                pred_text="label 1",
                                cost=0,
                            )
                        ]
                    ),
                ),
                types_module.NodeMatch(
                    true_node=true_options_2.options[0].nodes[1],
                    pred_node=pred_diagram_2.nodes[1],
                    node_text_cost=0,
                    label_matches=types_module.TextListMatches(
                        matches=[
                            types_module.TextListMatch(
                                true_index=0,
                                pred_index=0,
                                true_text="label 2",
                                pred_text="label 2",
                                cost=0,
                            )
                        ]
                    ),
                ),
            ],
            true_diagram_option_idx=0,
        ),
        additional_text_matches=types_module.TextListMatches(
            matches=[
                types_module.TextListMatch(
                    true_index=0,
                    pred_index=0,
                    true_text="Figure 1",
                    pred_text="Figure 1",
                    cost=0,
                )
            ]
        ),
        pred_diagram=pred_diagram_2,
        true_diagram=true_options_2.options[0],
    )

    expected_matches = [expected_match_1, expected_match_2]
    calls = []

    def fake_match_single_diagram(
        true_diagram_options,
        pred_diagram,
        distance_fn,
    ):
        calls.append(
            {
                "true_diagram_options": true_diagram_options,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )

        return expected_matches[len(calls) - 1]

    monkeypatch.setattr(
        match_module,
        "match_single_diagram",
        fake_match_single_diagram,
    )

    diagram_matches = match_module.match_diagrams(
        true_diagrams_options_list=[true_options_1, true_options_2],
        pred_diagrams=[pred_diagram_1, pred_diagram_2],
        distance_fn=levenshtein_fn,
    )

    assert calls == [
        {
            "true_diagram_options": true_options_1,
            "pred_diagram": pred_diagram_1,
            "distance_fn": levenshtein_fn,
        },
        {
            "true_diagram_options": true_options_2,
            "pred_diagram": pred_diagram_2,
            "distance_fn": levenshtein_fn,
        },
    ]

    assert diagram_matches == expected_matches


def test_match_diagrams_raises_if_true_and_pred_lists_have_different_lengths():
    true_options = make_diagram_options([make_true_diagram()])
    pred_diagram_1 = make_full_pred_diagram(parent_img_code="diagram_1")
    pred_diagram_2 = make_full_pred_diagram(parent_img_code="diagram_2")

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        match_module.match_diagrams(
            true_diagrams_options_list=[true_options],
            pred_diagrams=[pred_diagram_1, pred_diagram_2],
            distance_fn=levenshtein_fn,
        )


def test_match_diagrams_returns_expected_matches_for_multiple_diagrams():
    cases = [
        {
            "distance_fn": levenshtein_fn,
            "true_diagrams": [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Wrong Figure"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Wrong node 1",
                            labels=["wrong label 1"],
                            points_to=[2],
                            true_option_idx=0,
                            parent_img_code="diagram_1",
                        ),
                        make_true_node(
                            node_number=2,
                            text="Wrong node 2",
                            labels=["wrong label 2"],
                            points_to=[1],
                            true_option_idx=0,
                            parent_img_code="diagram_1",
                        ),
                    ],
                    parent_img_code="diagram_1",
                ),
                make_true_diagram(
                    true_option_idx=1,
                    additional_texts=["Figure 1"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Node 1",
                            labels=["label 1"],
                            points_to=[2],
                            true_option_idx=1,
                            parent_img_code="diagram_1",
                        ),
                        make_true_node(
                            node_number=2,
                            text="Node 2",
                            labels=["label 2"],
                            points_to=[1],
                            true_option_idx=1,
                            parent_img_code="diagram_1",
                        ),
                    ],
                    parent_img_code="diagram_1",
                ),
            ],
            "pred_diagram": make_full_pred_diagram(
                parent_img_code="diagram_1",
                additional_texts=["Figure 1"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Node 1",
                        labels=["label 1"],
                        points_to=[2],
                        parent_img_code="diagram_1",
                    ),
                    make_pred_node(
                        node_number=2,
                        text="Node 2",
                        labels=["label 2"],
                        points_to=[],
                        parent_img_code="diagram_1",
                    ),
                ],
            ),
            "expected_true_diagram_option_idx": 1,
            "expected_node_pairs": [
                (1, 1, 0),
                (2, 2, 0),
            ],
            "expected_total_node_text_cost": 0,
            "expected_label_matches": [
                [
                    {
                        "match_type": "match",
                        "true_text": "label 1",
                        "pred_text": "label 1",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "match",
                        "true_text": "label 2",
                        "pred_text": "label 2",
                        "cost": 0,
                    }
                ],
            ],
            "expected_total_label_error_cost": 0,
            "expected_additional_text_matches": [
                {
                    "match_type": "match",
                    "true_text": "Figure 1",
                    "pred_text": "Figure 1",
                    "cost": 0,
                }
            ],
            "expected_total_additional_text_error_cost": 0,
        },
        {
            "distance_fn": levenshtein_fn,
            "true_diagrams": [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Figure 1"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Alpha",
                            labels=["cat"],
                            points_to=[1],
                            true_option_idx=0,
                            parent_img_code="diagram_2",
                        ),
                    ],
                    parent_img_code="diagram_2",
                ),
                make_true_diagram(
                    true_option_idx=1,
                    additional_texts=["Wrong Figure"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Wrong",
                            labels=["dog"],
                            points_to=[1],
                            true_option_idx=1,
                            parent_img_code="diagram_2",
                        ),
                    ],
                    parent_img_code="diagram_2",
                ),
            ],
            "pred_diagram": make_full_pred_diagram(
                parent_img_code="diagram_2",
                additional_texts=["Figure X"],
                nodes=[
                    make_pred_node(
                        node_number=10,
                        text="Alpha",
                        labels=["cat"],
                        points_to=[],
                        parent_img_code="diagram_2",
                    ),
                    make_pred_node(
                        node_number=20,
                        text="Beta",
                        labels=["mouse"],
                        points_to=[],
                        parent_img_code="diagram_2",
                    ),
                ],
            ),
            "expected_true_diagram_option_idx": 0,
            "expected_node_pairs": [
                (1, 10, 0),
                (None, 20, 4),
            ],
            "expected_total_node_text_cost": 4,
            "expected_label_matches": [
                [
                    {
                        "match_type": "match",
                        "true_text": "cat",
                        "pred_text": "cat",
                        "cost": 0,
                    }
                ],
                [
                    {
                        "match_type": "unmatched_pred",
                        "true_text": None,
                        "pred_text": "mouse",
                        "cost": 5,
                    }
                ],
            ],
            "expected_total_label_error_cost": 5,
            "expected_additional_text_matches": [
                {
                    "match_type": "match",
                    "true_text": "Figure 1",
                    "pred_text": "Figure X",
                    "cost": 1,
                }
            ],
            "expected_total_additional_text_error_cost": 1,
        },
        {
            "distance_fn": levenshtein_fn,
            "true_diagrams": [
                make_true_diagram(
                    true_option_idx=0,
                    additional_texts=["Caption"],
                    nodes=[
                        make_true_node(
                            node_number=1,
                            text="Start",
                            labels=["screened", "randomised"],
                            points_to=[2],
                            true_option_idx=0,
                            parent_img_code="diagram_3",
                        ),
                        make_true_node(
                            node_number=2,
                            text="End",
                            labels=["excluded"],
                            points_to=[1],
                            true_option_idx=0,
                            parent_img_code="diagram_3",
                        ),
                    ],
                    parent_img_code="diagram_3",
                ),
            ],
            "pred_diagram": make_full_pred_diagram(
                parent_img_code="diagram_3",
                additional_texts=["Caption", "Extra note"],
                nodes=[
                    make_pred_node(
                        node_number=1,
                        text="Start",
                        labels=["screened", "randomized"],
                        points_to=[2],
                        parent_img_code="diagram_3",
                    ),
                    make_pred_node(
                        node_number=2,
                        text="End",
                        labels=["included"],
                        points_to=[],
                        parent_img_code="diagram_3",
                    ),
                ],
            ),
            "expected_true_diagram_option_idx": 0,
            "expected_node_pairs": [
                (1, 1, 0),
                (2, 2, 0),
            ],
            "expected_total_node_text_cost": 0,
            "expected_label_matches": [
                [
                    {
                        "match_type": "match",
                        "true_text": "screened",
                        "pred_text": "screened",
                        "cost": 0,
                    },
                    {
                        "match_type": "match",
                        "true_text": "randomised",
                        "pred_text": "randomized",
                        "cost": 1,
                    },
                ],
                [
                    {
                        "match_type": "match",
                        "true_text": "excluded",
                        "pred_text": "included",
                        "cost": 2,
                    }
                ],
            ],
            "expected_total_label_error_cost": 3,
            "expected_additional_text_matches": [
                {
                    "match_type": "match",
                    "true_text": "Caption",
                    "pred_text": "Caption",
                    "cost": 0,
                },
                {
                    "match_type": "unmatched_pred",
                    "true_text": None,
                    "pred_text": "Extra note",
                    "cost": 10,
                },
            ],
            "expected_total_additional_text_error_cost": 10,
        },
    ]

    true_diagrams_options_list = [
        make_diagram_options(
            options=case["true_diagrams"],
            parent_img_code=case["pred_diagram"].parent_img_code,
        )
        for case in cases
    ]
    pred_diagrams = [case["pred_diagram"] for case in cases]

    diagram_matches = match_module.match_diagrams(
        true_diagrams_options_list=true_diagrams_options_list,
        pred_diagrams=pred_diagrams,
        distance_fn=levenshtein_fn,
    )

    assert len(diagram_matches) == len(cases)

    for case, diagram_match in zip(cases, diagram_matches, strict=True):
        expected_true_diagram = case["true_diagrams"][
            case["expected_true_diagram_option_idx"]
        ]

        assert isinstance(diagram_match, types_module.DiagramMatch)
        assert diagram_match.pred_diagram == case["pred_diagram"]
        assert diagram_match.true_diagram == expected_true_diagram
        assert (
            diagram_match.node_matches.true_diagram_option_idx
            == case["expected_true_diagram_option_idx"]
        )

        node_pairs = [
            (
                node_match.true_node.node_number
                if node_match.true_node is not None
                else None,
                node_match.pred_node.node_number
                if node_match.pred_node is not None
                else None,
                node_match.node_text_cost,
            )
            for node_match in diagram_match.node_matches.matches
        ]

        assert node_pairs == case["expected_node_pairs"]
        assert (
            diagram_match.node_matches.total_node_text_cost
            == case["expected_total_node_text_cost"]
        )

        label_matches = [
            [
                {
                    "match_type": label_match.match_type,
                    "true_text": label_match.true_text,
                    "pred_text": label_match.pred_text,
                    "cost": label_match.cost,
                }
                for label_match in node_match.label_matches.matches
            ]
            for node_match in diagram_match.node_matches.matches
        ]

        assert label_matches == case["expected_label_matches"]
        assert (
            diagram_match.total_label_error_cost
            == case["expected_total_label_error_cost"]
        )

        additional_text_matches = [
            {
                "match_type": match.match_type,
                "true_text": match.true_text,
                "pred_text": match.pred_text,
                "cost": match.cost,
            }
            for match in diagram_match.additional_text_matches.matches
        ]

        assert additional_text_matches == case["expected_additional_text_matches"]
        assert (
            diagram_match.total_additional_text_cost
            == case["expected_total_additional_text_error_cost"]
        )

        assert (
            diagram_match.total_text_cost
            == diagram_match.total_node_text_cost
            + diagram_match.total_label_error_cost
            + diagram_match.total_additional_text_cost
        )


@pytest.mark.parametrize(
    (
        "distance_fn",
        "true_nodes",
        "pred_nodes",
        "expected_pairs",
        "expected_text_cost",
        "expected_flow_jaccard",
    ),
    [
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Alpha", ["left"], [2]),
                make_true_node(2, "Beta", ["right"], []),
            ],
            [
                make_pred_node(10, "Alpha", ["right"], []),
                make_pred_node(20, "Beta", ["left"], [10]),
            ],
            {(1, 10), (2, 20)},
            0,
            0.0,
            id="node-text-takes-priority-over-flow-and-labels",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["left"], [2]),
                make_true_node(2, "Duplicate", ["right"], []),
            ],
            [
                make_pred_node(10, "Duplicate", ["left"], []),
                make_pred_node(20, "Duplicate", ["right"], [10]),
            ],
            {(1, 20), (2, 10)},
            0,
            1.0,
            id="flow-breaks-node-text-tie-before-labels",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "ab", ["left"], [2]),
                make_true_node(2, "ac", ["right"], []),
            ],
            [
                make_pred_node(10, "ad", None, []),
                make_pred_node(20, "ae", None, [10]),
            ],
            {(1, 20), (2, 10)},
            2,
            1.0,
            id="flow-breaks-nonidentical-node-text-tie",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["one"], [2, 3]),
                make_true_node(2, "Duplicate", ["two"], [3]),
                make_true_node(3, "Duplicate", ["three"], []),
                make_true_node(4, "Duplicate", ["four"], [1]),
            ],
            [
                make_pred_node(10, "Duplicate", None, [20, 30]),
                make_pred_node(20, "Duplicate", None, [30]),
                make_pred_node(30, "Duplicate", None, []),
            ],
            {(1, 10), (2, 20), (3, 30), (4, None)},
            len("Duplicate"),
            0.75,
            id="flow-chooses-which-true-node-is-unmatched",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["one"], [2]),
                make_true_node(2, "Duplicate", ["two"], []),
            ],
            [
                make_pred_node(10, "Duplicate", None, [20]),
                make_pred_node(20, "Duplicate", None, []),
                make_pred_node(30, "Duplicate", None, []),
            ],
            {(1, 10), (2, 20), (None, 30)},
            len("Duplicate"),
            1.0,
            id="flow-chooses-which-predicted-node-is-unmatched",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["unmatched"], [2]),
                make_true_node(2, "Duplicate", ["middle"], [3]),
                make_true_node(3, "Duplicate", ["end"], []),
            ],
            [
                make_pred_node(10, "Duplicate", ["middle"], [20]),
                make_pred_node(20, "Duplicate", ["end"], []),
            ],
            {(1, None), (2, 10), (3, 20)},
            len("Duplicate"),
            0.5,
            id="labels-break-flow-tie-with-an-unmatched-true-node",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["left"], [2]),
                make_true_node(2, "Duplicate", ["right"], [1]),
            ],
            [
                make_pred_node(10, "Duplicate", ["right"], [20]),
                make_pred_node(20, "Duplicate", ["left"], [10]),
            ],
            {(1, 20), (2, 10)},
            0,
            1.0,
            id="labels-break-node-text-and-flow-tie",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["left"], [2]),
                make_true_node(2, "Duplicate", ["right"], [1]),
            ],
            [
                make_pred_node(10, "Duplicate", ["right"], None),
                make_pred_node(20, "Duplicate", ["left"], None),
            ],
            {(1, 20), (2, 10)},
            0,
            None,
            id="labels-break-node-text-tie-without-flow",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(
                    1,
                    "Duplicate",
                    ["left alpha", "left beta"],
                    [2],
                ),
                make_true_node(
                    2,
                    "Duplicate",
                    ["right alpha", "right beta"],
                    [],
                ),
            ],
            [
                make_pred_node(
                    10,
                    "Duplicate",
                    ["right beta", "right alpha"],
                    None,
                ),
                make_pred_node(
                    20,
                    "Duplicate",
                    ["left beta", "left alpha"],
                    None,
                ),
            ],
            {(1, 20), (2, 10)},
            0,
            None,
            id="multiple-labels-break-node-text-tie",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["left"], [2]),
                make_true_node(2, "Duplicate", ["right"], []),
            ],
            [
                make_pred_node(10, "Duplicate", ["right"], []),
                make_pred_node(20, "Duplicate", ["left"], []),
            ],
            {(1, 20), (2, 10)},
            0,
            0.0,
            id="labels-break-tie-when-parsed-flow-has-no-edges",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "ab", ["first"], [2]),
                make_true_node(2, "ac", ["second"], [1]),
            ],
            [
                make_pred_node(10, "ad", ["second"], None),
                make_pred_node(20, "ae", ["first"], None),
            ],
            {(1, 20), (2, 10)},
            2,
            None,
            id="labels-choose-between-nonidentical-text-optimal-pairs",
        ),
        pytest.param(
            fractional_levenshtein_fn,
            [
                make_true_node(1, "ab", ["first"], [2]),
                make_true_node(2, "ac", ["second"], [1]),
            ],
            [
                make_pred_node(10, "ad", ["second"], None),
                make_pred_node(20, "ae", ["first"], None),
            ],
            {(1, 20), (2, 10)},
            0.2,
            None,
            id="labels-break-fractional-node-text-cost-tie",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["one"], [2]),
                make_true_node(2, "Duplicate", ["two"], [3]),
                make_true_node(3, "Duplicate", ["xxxxxxxxxxxxxxxxxxxx"], [1]),
            ],
            [
                make_pred_node(10, "Duplicate", ["one"], None),
                make_pred_node(20, "Duplicate", ["two"], None),
            ],
            {(1, 10), (2, 20), (3, None)},
            len("Duplicate"),
            None,
            id="labels-choose-which-true-node-is-unmatched",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Duplicate", ["a"], [2]),
                make_true_node(2, "Duplicate", ["b"], []),
            ],
            [
                make_pred_node(10, "Duplicate", ["a"], None),
                make_pred_node(20, "Duplicate", ["b"], None),
                make_pred_node(30, "Duplicate", ["xxxxxxxxxx"], None),
            ],
            {(1, 10), (2, 20), (None, 30)},
            len("Duplicate"),
            None,
            id="labels-choose-which-predicted-node-is-unmatched",
        ),
        pytest.param(
            levenshtein_fn,
            [
                make_true_node(1, "Alpha", ["label 1"], [2]),
                make_true_node(2, "Beta", ["label 2"], []),
            ],
            [
                make_pred_node(10, "Alpha", None, None),
                make_pred_node(20, "Beta", None, None),
                make_pred_node(30, "Extra", None, None),
            ],
            {(1, 10), (2, 20), (None, 30)},
            len("Extra"),
            None,
            id="text-only-matching-includes-unmatched-predicted-node",
        ),
    ],
)
def test_match_nodes_single_option_uses_available_evidence_lexicographically(
    distance_fn,
    true_nodes,
    pred_nodes,
    expected_pairs,
    expected_text_cost,
    expected_flow_jaccard,
):
    true_diagram = make_true_diagram(nodes=true_nodes)
    pred_diagram = make_pred_diagram(
        nodes=pred_nodes,
        additional_texts=["Figure 1"],
    )

    node_matches = match_module.match_nodes_single_option(
        true_diagram=true_diagram,
        pred_diagram=pred_diagram,
        distance_fn=distance_fn,
        true_diagram_option_idx=0,
    )

    actual_pairs = {
        (
            match.true_node.node_number if match.true_node is not None else None,
            match.pred_node.node_number if match.pred_node is not None else None,
        )
        for match in node_matches.matches
    }
    assert actual_pairs == expected_pairs
    assert node_matches.total_node_text_cost == pytest.approx(expected_text_cost)
    if expected_flow_jaccard is not None:
        assert node_matches.flow_score.jaccard == pytest.approx(expected_flow_jaccard)


def test_match_nodes_single_option_finds_optimum_for_ten_duplicate_nodes():
    # There are 10! text-optimal correspondences. The directed cycle leaves ten
    # flow-optimal rotations, and the labels identify the intended one.
    pred_number_by_true_number = {
        1: 71,
        2: 23,
        3: 89,
        4: 14,
        5: 62,
        6: 35,
        7: 97,
        8: 46,
        9: 58,
        10: 11,
    }
    true_nodes = [
        make_true_node(
            node_number=node_number,
            text="Duplicate",
            labels=[f"node {node_number}"],
            points_to=[node_number % 10 + 1],
        )
        for node_number in range(1, 11)
    ]
    pred_nodes = [
        make_pred_node(
            node_number=pred_number_by_true_number[node_number],
            text="Duplicate",
            labels=[f"node {node_number}"],
            points_to=[pred_number_by_true_number[node_number % 10 + 1]],
        )
        for node_number in range(10, 0, -1)
    ]

    node_matches = match_module.match_nodes_single_option(
        true_diagram=make_true_diagram(nodes=true_nodes),
        pred_diagram=make_pred_diagram(
            nodes=pred_nodes,
            additional_texts=["Figure 1"],
        ),
        distance_fn=levenshtein_fn,
        true_diagram_option_idx=0,
    )

    mapping = {
        match.true_node.node_number: match.pred_node.node_number
        for match in node_matches.matches
        if match.true_node is not None and match.pred_node is not None
    }
    assert mapping == pred_number_by_true_number
    assert node_matches.total_node_text_cost == 0
    assert node_matches.flow_score.jaccard == pytest.approx(1.0)

    match_module.match_node_labels(
        node_matches=node_matches,
        distance_fn=levenshtein_fn,
    )
    assert node_matches.total_label_error_cost == 0


@pytest.mark.parametrize(
    ("pred_nodes", "expected_stage"),
    [
        pytest.param(
            [
                make_pred_node(10, "Duplicate", ["right"], None),
                make_pred_node(20, "Duplicate", ["left"], None),
            ],
            "label optimisation",
            id="label-optimisation",
        ),
        pytest.param(
            [
                make_pred_node(10, "Duplicate", None, [20]),
                make_pred_node(20, "Duplicate", None, [10]),
            ],
            "flow optimisation",
            id="flow-optimisation",
        ),
    ],
)
def test_match_nodes_single_option_raises_if_optimality_is_not_proven(
    monkeypatch: pytest.MonkeyPatch,
    pred_nodes,
    expected_stage,
):
    solver_calls = []

    def nonoptimal_milp(**kwargs):
        solver_calls.append(kwargs)
        return SimpleNamespace(
            status=1,
            success=False,
            x=None,
            message="Time limit reached",
        )

    monkeypatch.setattr(match_module, "milp", nonoptimal_milp)

    true_diagram = make_true_diagram(
        nodes=[
            make_true_node(1, "Duplicate", ["left"], [2]),
            make_true_node(2, "Duplicate", ["right"], [1]),
        ]
    )
    pred_diagram = make_pred_diagram(
        nodes=pred_nodes,
        additional_texts=["Figure 1"],
    )

    with pytest.raises(
        RuntimeError,
        match=f"Could not prove an optimal node matching during {expected_stage}",
    ):
        match_module.match_nodes_single_option(
            true_diagram=true_diagram,
            pred_diagram=pred_diagram,
            distance_fn=levenshtein_fn,
            true_diagram_option_idx=0,
        )

    assert len(solver_calls) == 1
    assert solver_calls[0]["options"] == {"mip_rel_gap": 0.0}
