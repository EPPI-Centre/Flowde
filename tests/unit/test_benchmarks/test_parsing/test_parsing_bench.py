from pathlib import Path
from types import SimpleNamespace

import pytest

import flowde.benchmarks.parsing.parsing_bench as benchmark_module
from flowde.benchmarks.parsing.parsing_bench_types import (
    PredDiagramSource,
    PredDiagramSources,
    PredDiagramStructure,
)

# TODO: Should add some actual non-monkeypatch orchestration tests with real values,
# just to be 100% sure we wired them properly.


def dummy_distance_fn(*, true_text: str | None, pred_text: str | None) -> int:
    return 0


def make_pred_diagram(parent_img_code: str):
    return SimpleNamespace(parent_img_code=parent_img_code)


def make_true_diagram(parent_img_code: str, option_idx: int = 0):
    return SimpleNamespace(
        parent_img_code=parent_img_code,
        option_idx=option_idx,
    )


def make_true_diagram_options(parent_img_code: str, n_options: int = 1):
    return SimpleNamespace(
        parent_img_code=parent_img_code,
        options=[
            make_true_diagram(parent_img_code=parent_img_code, option_idx=i)
            for i in range(n_options)
        ],
    )


def patch_successful_benchmark_init(
    monkeypatch,
    pred_diagrams,
    true_diagrams_options_list,
    structure=None,
):
    run_checks_calls = []
    validate_pred_sources_calls = []
    load_pred_diagrams_calls = []

    pred_diagrams_dir = Path("preds")
    if structure is None:
        structure = PredDiagramStructure(
            node_text=True,
            labels=True,
            flow=True,
            additional_texts=True,
        )
    pred_sources = PredDiagramSources(
        sources=(
            PredDiagramSource(
                diagrams_dir=pred_diagrams_dir,
                paths=tuple(
                    pred_diagrams_dir / f"{diagram.parent_img_code}.json"
                    for diagram in sorted(
                        pred_diagrams,
                        key=lambda diagram: diagram.parent_img_code,
                    )
                ),
                structure=structure,
            ),
        ),
        structure=structure,
    )

    def fake_run_all_benchmark_true_data_checks(**kwargs):
        run_checks_calls.append(kwargs)

    def fake_validate_pred_diagram_sources(**kwargs):
        validate_pred_sources_calls.append(kwargs)
        return pred_sources

    def fake_load_pred_diagrams_from_sources(sources):
        load_pred_diagrams_calls.append(sources)
        return pred_diagrams

    def fake_load_true_diagram_options(
        true_nodes_dir,
        true_labels_dir,
        true_additional_texts_dir,
        true_flow_dir,
    ):
        return true_diagrams_options_list

    monkeypatch.setattr(
        benchmark_module,
        "run_all_benchmark_true_data_checks",
        fake_run_all_benchmark_true_data_checks,
    )
    monkeypatch.setattr(
        benchmark_module,
        "validate_pred_diagram_sources",
        fake_validate_pred_diagram_sources,
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_pred_diagrams_from_sources",
        fake_load_pred_diagrams_from_sources,
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_true_diagram_options",
        fake_load_true_diagram_options,
    )
    return (
        run_checks_calls,
        validate_pred_sources_calls,
        load_pred_diagrams_calls,
        pred_sources,
    )


def make_benchmark(monkeypatch):
    pred_diagrams = [
        make_pred_diagram("diagram_2"),
        make_pred_diagram("diagram_1"),
    ]
    true_diagrams_options_list = [
        make_true_diagram_options("diagram_2"),
        make_true_diagram_options("diagram_1"),
    ]

    patch_successful_benchmark_init(
        monkeypatch=monkeypatch,
        pred_diagrams=pred_diagrams,
        true_diagrams_options_list=true_diagrams_options_list,
    )

    return benchmark_module.ParsingBenchmark(
        pred_diagrams_dir=Path("preds"),
        distance_fn=dummy_distance_fn,
    )


def test_parsing_benchmark_init_runs_checks_loads_diagrams_and_sorts_by_img_code(
    monkeypatch,
):
    pred_diagrams = [
        make_pred_diagram("diagram_2"),
        make_pred_diagram("diagram_1"),
    ]
    true_diagrams_options_list = [
        make_true_diagram_options("diagram_2"),
        make_true_diagram_options("diagram_1"),
    ]

    (
        run_checks_calls,
        validate_pred_sources_calls,
        load_pred_diagrams_calls,
        pred_sources,
    ) = patch_successful_benchmark_init(
        monkeypatch=monkeypatch,
        pred_diagrams=pred_diagrams,
        true_diagrams_options_list=true_diagrams_options_list,
    )

    benchmark = benchmark_module.ParsingBenchmark(
        pred_diagrams_dir=Path("preds"),
        distance_fn=dummy_distance_fn,
        allow_missing_pred_diagrams=False,
        true_nodes_dir=Path("true_nodes"),
        true_labels_dir=Path("true_labels"),
        true_additional_texts_dir=Path("true_additional_texts"),
        true_flow_dir=Path("true_flow"),
        expected_num_diagrams=2,
    )

    assert run_checks_calls == [
        {
            "pred_sources": pred_sources,
            "true_nodes_dir": Path("true_nodes"),
            "true_labels_dir": Path("true_labels"),
            "true_additional_texts_dir": Path("true_additional_texts"),
            "true_flow_dir": Path("true_flow"),
            "allow_missing_pred_diagrams": False,
            "expected_num_diagrams": 2,
        }
    ]

    assert [pd.parent_img_code for pd in benchmark.pred_diagrams] == [
        "diagram_1",
        "diagram_2",
    ]
    assert [tdo.parent_img_code for tdo in benchmark.true_diagrams_options_list] == [
        "diagram_1",
        "diagram_2",
    ]

    assert benchmark.allow_missing_pred_diagrams is False
    assert benchmark.distance_fn is dummy_distance_fn

    assert validate_pred_sources_calls == [{"pred_diagrams_dirs": Path("preds")}]
    assert load_pred_diagrams_calls == [pred_sources]


def test_parsing_benchmark_init_filters_true_diagram_options_when_missing_preds_allowed(
    monkeypatch,
):
    pred_diagrams = [
        make_pred_diagram("diagram_2"),
        make_pred_diagram("diagram_1"),
    ]
    true_diagrams_options_list = [
        make_true_diagram_options("diagram_3"),
        make_true_diagram_options("diagram_2"),
        make_true_diagram_options("diagram_1"),
    ]

    patch_successful_benchmark_init(
        monkeypatch=monkeypatch,
        pred_diagrams=pred_diagrams,
        true_diagrams_options_list=true_diagrams_options_list,
    )

    benchmark = benchmark_module.ParsingBenchmark(
        pred_diagrams_dir=Path("preds"),
        distance_fn=dummy_distance_fn,
        allow_missing_pred_diagrams=True,
    )

    assert [pd.parent_img_code for pd in benchmark.pred_diagrams] == [
        "diagram_1",
        "diagram_2",
    ]
    assert [tdo.parent_img_code for tdo in benchmark.true_diagrams_options_list] == [
        "diagram_1",
        "diagram_2",
    ]


# TODO: Are we checking anywhere that they are only allowed when valid is true and
# not when it isn't?
def test_diagram_matches_calls_match_diagrams_with_expected_arguments(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    expected_result = [SimpleNamespace(name="diagram match")]
    calls = []

    def fake_match_diagrams(
        true_diagrams_options_list,
        pred_diagrams,
        distance_fn,
    ):
        calls.append(
            {
                "true_diagrams_options_list": true_diagrams_options_list,
                "pred_diagrams": pred_diagrams,
                "distance_fn": distance_fn,
            }
        )
        return expected_result

    monkeypatch.setattr(
        benchmark_module,
        "match_diagrams",
        fake_match_diagrams,
    )

    result = benchmark.diagram_matches()

    assert result is expected_result
    assert calls == [
        {
            "true_diagrams_options_list": benchmark.true_diagrams_options_list,
            "pred_diagrams": benchmark.pred_diagrams,
            "distance_fn": dummy_distance_fn,
        }
    ]


def test_node_matches_calls_match_nodes_for_each_true_pred_pair(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    expected_node_matches = [
        SimpleNamespace(name="node matches 1"),
        SimpleNamespace(name="node matches 2"),
    ]
    calls = []

    def fake_match_nodes(
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
        return expected_node_matches[len(calls) - 1]

    monkeypatch.setattr(
        benchmark_module,
        "match_nodes",
        fake_match_nodes,
    )

    result = benchmark.node_matches()

    assert result == expected_node_matches
    assert calls == [
        {
            "true_diagram_options": benchmark.true_diagrams_options_list[0],
            "pred_diagram": benchmark.pred_diagrams[0],
            "distance_fn": dummy_distance_fn,
        },
        {
            "true_diagram_options": benchmark.true_diagrams_options_list[1],
            "pred_diagram": benchmark.pred_diagrams[1],
            "distance_fn": dummy_distance_fn,
        },
    ]


def test_node_and_label_matches_calls_node_matches_then_match_node_labels(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    node_matches = [
        SimpleNamespace(name="node matches 1"),
        SimpleNamespace(name="node matches 2"),
    ]
    expected_labelled_node_matches = [
        SimpleNamespace(name="labelled node matches 1"),
        SimpleNamespace(name="labelled node matches 2"),
    ]

    node_match_calls = []
    label_match_calls = []

    def fake_match_nodes(
        true_diagram_options,
        pred_diagram,
        distance_fn,
    ):
        node_match_calls.append(
            {
                "true_diagram_options": true_diagram_options,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )
        return node_matches[len(node_match_calls) - 1]

    def fake_match_node_labels(
        node_matches,
        distance_fn,
    ):
        label_match_calls.append(
            {
                "node_matches": node_matches,
                "distance_fn": distance_fn,
            }
        )
        return expected_labelled_node_matches[len(label_match_calls) - 1]

    monkeypatch.setattr(benchmark_module, "match_nodes", fake_match_nodes)
    monkeypatch.setattr(benchmark_module, "match_node_labels", fake_match_node_labels)

    result = benchmark.node_and_label_matches()

    assert result == expected_labelled_node_matches

    assert node_match_calls == [
        {
            "true_diagram_options": benchmark.true_diagrams_options_list[0],
            "pred_diagram": benchmark.pred_diagrams[0],
            "distance_fn": dummy_distance_fn,
        },
        {
            "true_diagram_options": benchmark.true_diagrams_options_list[1],
            "pred_diagram": benchmark.pred_diagrams[1],
            "distance_fn": dummy_distance_fn,
        },
    ]

    assert label_match_calls == [
        {
            "node_matches": node_matches[0],
            "distance_fn": dummy_distance_fn,
        },
        {
            "node_matches": node_matches[1],
            "distance_fn": dummy_distance_fn,
        },
    ]


def test_additional_text_matches_selects_true_option_from_node_matches_and_calls_match_additional_texts(
    monkeypatch,
):
    pred_diagram_1 = make_pred_diagram("diagram_1")
    pred_diagram_2 = make_pred_diagram("diagram_2")

    true_options_1 = make_true_diagram_options("diagram_1", n_options=2)
    true_options_2 = make_true_diagram_options("diagram_2", n_options=2)

    patch_successful_benchmark_init(
        monkeypatch=monkeypatch,
        pred_diagrams=[pred_diagram_2, pred_diagram_1],
        true_diagrams_options_list=[true_options_2, true_options_1],
    )

    benchmark = benchmark_module.ParsingBenchmark(
        pred_diagrams_dir=Path("preds"),
        distance_fn=dummy_distance_fn,
    )

    node_matches = [
        SimpleNamespace(true_diagram_option_idx=1),
        SimpleNamespace(true_diagram_option_idx=0),
    ]
    expected_additional_text_matches = [
        [SimpleNamespace(name="additional text matches 1")],
        [SimpleNamespace(name="additional text matches 2")],
    ]

    node_match_calls = []
    additional_text_calls = []

    def fake_match_nodes(
        true_diagram_options,
        pred_diagram,
        distance_fn,
    ):
        node_match_calls.append(
            {
                "true_diagram_options": true_diagram_options,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )
        return node_matches[len(node_match_calls) - 1]

    def fake_match_additional_texts(
        true_diagram,
        pred_diagram,
        distance_fn,
    ):
        additional_text_calls.append(
            {
                "true_diagram": true_diagram,
                "pred_diagram": pred_diagram,
                "distance_fn": distance_fn,
            }
        )
        return expected_additional_text_matches[len(additional_text_calls) - 1]

    monkeypatch.setattr(benchmark_module, "match_nodes", fake_match_nodes)
    monkeypatch.setattr(
        benchmark_module,
        "match_additional_texts",
        fake_match_additional_texts,
    )

    result = benchmark.additional_text_matches()

    assert result == expected_additional_text_matches

    assert node_match_calls == [
        {
            "true_diagram_options": true_options_1,
            "pred_diagram": pred_diagram_1,
            "distance_fn": dummy_distance_fn,
        },
        {
            "true_diagram_options": true_options_2,
            "pred_diagram": pred_diagram_2,
            "distance_fn": dummy_distance_fn,
        },
    ]

    assert additional_text_calls == [
        {
            "true_diagram": true_options_1.options[1],
            "pred_diagram": pred_diagram_1,
            "distance_fn": dummy_distance_fn,
        },
        {
            "true_diagram": true_options_2.options[0],
            "pred_diagram": pred_diagram_2,
            "distance_fn": dummy_distance_fn,
        },
    ]


def test_node_matches_with_flow_checks_flow_validity_then_returns_node_matches(
    monkeypatch,
):
    benchmark = make_benchmark(monkeypatch)

    expected_node_matches = [
        SimpleNamespace(name="node matches 1"),
        SimpleNamespace(name="node matches 2"),
    ]

    def fake_node_matches():
        return expected_node_matches

    monkeypatch.setattr(
        benchmark,
        "node_matches",
        fake_node_matches,
    )

    result = benchmark.node_matches_with_flow()

    assert result == expected_node_matches


def test_total_node_text_cost_sums_node_match_costs(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_node_matches():
        return [
            SimpleNamespace(total_node_text_cost=2),
            SimpleNamespace(total_node_text_cost=3),
        ]

    monkeypatch.setattr(benchmark, "node_matches", fake_node_matches)

    assert benchmark.total_node_text_cost() == 5


def test_total_label_cost_sums_node_match_label_costs(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_node_and_label_matches():
        return [
            SimpleNamespace(total_label_error_cost=4),
            SimpleNamespace(total_label_error_cost=6),
        ]

    monkeypatch.setattr(
        benchmark,
        "node_and_label_matches",
        fake_node_and_label_matches,
    )

    assert benchmark.total_label_cost() == 10


def test_total_additional_text_cost_sums_costs_from_each_diagram(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_additional_text_matches():
        return [
            SimpleNamespace(total_cost=3),
            SimpleNamespace(total_cost=3),
        ]

    monkeypatch.setattr(
        benchmark,
        "additional_text_matches",
        fake_additional_text_matches,
    )

    assert benchmark.total_additional_text_cost() == 6


def test_all_flow_scores_returns_flow_scores_from_node_matches_with_flow(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    flow_score_1 = SimpleNamespace(jaccard=0.25)
    flow_score_2 = SimpleNamespace(jaccard=0.75)

    def fake_node_matches_with_flow():
        return [
            SimpleNamespace(flow_score=flow_score_1),
            SimpleNamespace(flow_score=flow_score_2),
        ]

    monkeypatch.setattr(
        benchmark,
        "node_matches_with_flow",
        fake_node_matches_with_flow,
    )

    assert benchmark.all_flow_scores() == [flow_score_1, flow_score_2]


def test_all_flow_jaccard_scores_returns_jaccard_from_all_flow_scores(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_all_flow_scores():
        return [
            SimpleNamespace(jaccard=0.25),
            SimpleNamespace(jaccard=0.75),
        ]

    monkeypatch.setattr(benchmark, "all_flow_scores", fake_all_flow_scores)

    assert benchmark.all_flow_jaccard_scores() == [0.25, 0.75]


def test_avg_flow_jaccard_returns_mean_flow_jaccard(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_all_flow_jaccard_scores():
        return [0.25, 0.75, 0.5]

    monkeypatch.setattr(
        benchmark,
        "all_flow_jaccard_scores",
        fake_all_flow_jaccard_scores,
    )

    assert benchmark.avg_flow_jaccard() == 0.5


def test_avg_flow_jaccard_raises_if_there_are_no_scores(monkeypatch):
    benchmark = make_benchmark(monkeypatch)

    def fake_all_flow_jaccard_scores():
        return []

    monkeypatch.setattr(
        benchmark,
        "all_flow_jaccard_scores",
        fake_all_flow_jaccard_scores,
    )

    with pytest.raises(
        ValueError,
        match="Cannot calculate average flow Jaccard because there are no flow scores.",
    ):
        benchmark.avg_flow_jaccard()


@pytest.mark.parametrize(
    ("structure", "method_name", "operation", "missing_part"),
    [
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=True,
                flow=True,
                additional_texts=False,
            ),
            "diagram_matches",
            "Diagram matching",
            "additional_texts",
            id="diagram-matches-without-additional-text",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "node_and_label_matches",
            "Node and label matching",
            "labels",
            id="node-and-label-matches-without-labels",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "total_label_cost",
            "Node and label matching",
            "labels",
            id="total-label-cost-without-labels",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "additional_text_matches",
            "Additional text matching",
            "additional_texts",
            id="additional-text-matches-without-additional-text",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "total_additional_text_cost",
            "Additional text matching",
            "additional_texts",
            id="total-additional-text-cost-without-additional-text",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "all_flow_scores",
            "Flow scoring",
            "flow",
            id="flow-scores-without-flow",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "all_flow_jaccard_scores",
            "Flow scoring",
            "flow",
            id="flow-jaccard-scores-without-flow",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "avg_flow_jaccard",
            "Flow scoring",
            "flow",
            id="average-flow-jaccard-without-flow",
        ),
        pytest.param(
            PredDiagramStructure(
                node_text=True,
                labels=False,
                flow=False,
                additional_texts=False,
            ),
            "node_matches_with_flow",
            "Node matching with flow",
            "flow",
            id="node-matches-with-flow-without-flow",
        ),
    ],
)
def test_methods_raise_if_pred_structure_lacks_required_component(
    monkeypatch,
    structure,
    method_name,
    operation,
    missing_part,
):
    pred_diagrams = [make_pred_diagram("diagram_1")]
    true_diagrams_options_list = [make_true_diagram_options("diagram_1")]

    patch_successful_benchmark_init(
        monkeypatch=monkeypatch,
        pred_diagrams=pred_diagrams,
        true_diagrams_options_list=true_diagrams_options_list,
        structure=structure,
    )

    benchmark = benchmark_module.ParsingBenchmark(
        pred_diagrams_dir=Path("preds"),
        distance_fn=dummy_distance_fn,
    )

    assert method_name not in benchmark.capabilities.available_methods

    with pytest.raises(ValueError) as exc_info:
        getattr(benchmark, method_name)()

    msg = str(exc_info.value)
    assert f"{operation} is unavailable" in msg
    assert f"Missing required components: {missing_part}." in msg


@pytest.mark.parametrize(
    ("labels", "flow", "additional_texts", "expected_extra_methods"),
    [
        pytest.param(False, False, False, set(), id="node-text-only"),
        pytest.param(
            True,
            False,
            False,
            {"node_and_label_matches", "total_label_cost"},
            id="with-labels",
        ),
        pytest.param(
            False,
            True,
            False,
            {
                "node_matches_with_flow",
                "all_flow_scores",
                "all_flow_jaccard_scores",
                "avg_flow_jaccard",
            },
            id="with-flow",
        ),
        pytest.param(
            False,
            False,
            True,
            {"additional_text_matches", "total_additional_text_cost"},
            id="with-additional-text",
        ),
        pytest.param(
            True,
            True,
            False,
            {
                "node_and_label_matches",
                "total_label_cost",
                "node_matches_with_flow",
                "all_flow_scores",
                "all_flow_jaccard_scores",
                "avg_flow_jaccard",
            },
            id="with-labels-and-flow",
        ),
        pytest.param(
            True,
            False,
            True,
            {
                "node_and_label_matches",
                "total_label_cost",
                "additional_text_matches",
                "total_additional_text_cost",
            },
            id="with-labels-and-additional-text",
        ),
        pytest.param(
            False,
            True,
            True,
            {
                "node_matches_with_flow",
                "all_flow_scores",
                "all_flow_jaccard_scores",
                "avg_flow_jaccard",
                "additional_text_matches",
                "total_additional_text_cost",
            },
            id="with-flow-and-additional-text",
        ),
        pytest.param(
            True,
            True,
            True,
            {
                "node_and_label_matches",
                "total_label_cost",
                "node_matches_with_flow",
                "all_flow_scores",
                "all_flow_jaccard_scores",
                "avg_flow_jaccard",
                "additional_text_matches",
                "total_additional_text_cost",
                "diagram_matches",
            },
            id="complete-structure",
        ),
    ],
)
def test_available_methods_matches_pred_structure(
    labels,
    flow,
    additional_texts,
    expected_extra_methods,
):
    structure = PredDiagramStructure(
        node_text=True,
        labels=labels,
        flow=flow,
        additional_texts=additional_texts,
    )

    capabilities = benchmark_module.ParsingBenchmarkCapabilities.from_pred_structure(
        structure
    )

    assert set(capabilities.available_methods) == {
        "node_matches",
        "total_node_text_cost",
        *expected_extra_methods,
    }


def test_parsing_benchmark_init_raises_if_predictions_do_not_contain_node_text(
    tmp_path,
):
    pred_diagrams_dir = tmp_path / "preds"
    pred_diagrams_dir.mkdir()
    (pred_diagrams_dir / "diagram_1.json").write_text(
        '{"nodes": [{"node_number": 1, "labels": []}]}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Predicted node text is required to run a parsing benchmark",
    ):
        benchmark_module.ParsingBenchmark(
            pred_diagrams_dir=pred_diagrams_dir,
            distance_fn=dummy_distance_fn,
        )
