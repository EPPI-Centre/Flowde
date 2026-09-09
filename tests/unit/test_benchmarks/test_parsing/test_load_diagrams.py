from pathlib import Path

import pytest

import flowde.benchmarks.parsing.load_diagrams as load_module


class FakeDiagram:
    calls = []

    @classmethod
    def from_path(cls, path: Path, diagram_type: str, true_option_idx):
        cls.calls.append(
            {
                "path": path,
                "diagram_type": diagram_type,
                "true_option_idx": true_option_idx,
            }
        )
        return {
            "path": path,
            "diagram_type": diagram_type,
            "true_option_idx": true_option_idx,
        }


class FakeDiagramOptions:
    calls = []

    @classmethod
    def from_true_paths(
        cls,
        true_nodes_path: Path,
        true_labels_path: Path,
        true_additional_texts_path: Path,
        true_flow_path: Path,
    ):
        cls.calls.append(
            {
                "true_nodes_path": true_nodes_path,
                "true_labels_path": true_labels_path,
                "true_additional_texts_path": true_additional_texts_path,
                "true_flow_path": true_flow_path,
            }
        )
        return {
            "true_nodes_path": true_nodes_path,
            "true_labels_path": true_labels_path,
            "true_additional_texts_path": true_additional_texts_path,
            "true_flow_path": true_flow_path,
        }


def reset_fakes() -> None:
    FakeDiagram.calls = []
    FakeDiagramOptions.calls = []


def create_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def create_text(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")


def test_load_pred_diagrams_from_paths_calls_diagram_from_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    diagram_paths = [
        Path("pred/diagram_1.json"),
        Path("pred/diagram_2.json"),
    ]

    monkeypatch.setattr(load_module, "Diagram", FakeDiagram)

    result = load_module.load_pred_diagrams_from_paths(diagram_paths)

    expected = [
        {
            "path": diagram_paths[0],
            "diagram_type": "pred",
            "true_option_idx": None,
        },
        {
            "path": diagram_paths[1],
            "diagram_type": "pred",
            "true_option_idx": None,
        },
    ]

    assert result == expected
    assert FakeDiagram.calls == expected


def test_load_pred_diagrams_from_paths_returns_empty_list_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    monkeypatch.setattr(load_module, "Diagram", FakeDiagram)

    result = load_module.load_pred_diagrams_from_paths([])

    assert result == []
    assert FakeDiagram.calls == []


def test_load_pred_diagrams_loads_sorted_json_paths_from_single_dir_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    diagrams_dir = tmp_path / "pred"

    path_b = diagrams_dir / "b.json"
    path_a = diagrams_dir / "a.json"
    non_json_path = diagrams_dir / "c.txt"

    nested_json_path = diagrams_dir / "nested" / "nested.json"

    create_json(path_b)
    create_json(path_a)
    create_text(non_json_path)
    create_json(nested_json_path)

    monkeypatch.setattr(load_module, "Diagram", FakeDiagram)

    result = load_module.load_pred_diagrams(diagrams_dir)

    expected = [
        {
            "path": path_a,
            "diagram_type": "pred",
            "true_option_idx": None,
        },
        {
            "path": path_b,
            "diagram_type": "pred",
            "true_option_idx": None,
        },
    ]

    assert result == expected
    assert FakeDiagram.calls == expected

    expected_paths = {
        diagrams_dir,
        path_a,
        path_b,
        non_json_path,
        nested_json_path.parent,
        nested_json_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_load_pred_diagrams_returns_empty_list_if_no_json_files_in_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    diagrams_dir = tmp_path / "pred"
    nested_json_path = diagrams_dir / "nested" / "nested.json"

    diagrams_dir.mkdir(parents=True)
    create_text(diagrams_dir / "not_json.txt")
    create_json(nested_json_path)

    monkeypatch.setattr(load_module, "Diagram", FakeDiagram)

    result = load_module.load_pred_diagrams(diagrams_dir)

    assert result == []
    assert FakeDiagram.calls == []

    expected_paths = {
        diagrams_dir,
        diagrams_dir / "not_json.txt",
        nested_json_path.parent,
        nested_json_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_load_true_diagram_options_from_paths_calls_diagram_options_from_true_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    true_nodes_paths = [
        Path("nodes/diagram_1.json"),
        Path("nodes/diagram_2.json"),
    ]
    true_labels_paths = [
        Path("labels/diagram_1.json"),
        Path("labels/diagram_2.json"),
    ]
    true_additional_texts_paths = [
        Path("additional_texts/diagram_1.json"),
        Path("additional_texts/diagram_2.json"),
    ]
    true_flow_paths = [
        Path("flow/diagram_1.json"),
        Path("flow/diagram_2.json"),
    ]

    monkeypatch.setattr(load_module, "DiagramOptions", FakeDiagramOptions)

    result = load_module.load_true_diagram_options_from_paths(
        true_nodes_paths=true_nodes_paths,
        true_labels_paths=true_labels_paths,
        true_additional_texts_paths=true_additional_texts_paths,
        true_flow_paths=true_flow_paths,
    )

    expected = [
        {
            "true_nodes_path": true_nodes_paths[0],
            "true_labels_path": true_labels_paths[0],
            "true_additional_texts_path": true_additional_texts_paths[0],
            "true_flow_path": true_flow_paths[0],
        },
        {
            "true_nodes_path": true_nodes_paths[1],
            "true_labels_path": true_labels_paths[1],
            "true_additional_texts_path": true_additional_texts_paths[1],
            "true_flow_path": true_flow_paths[1],
        },
    ]

    assert result == expected
    assert FakeDiagramOptions.calls == expected


def test_load_true_diagram_options_from_paths_returns_empty_list_for_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    monkeypatch.setattr(load_module, "DiagramOptions", FakeDiagramOptions)

    result = load_module.load_true_diagram_options_from_paths(
        true_nodes_paths=[],
        true_labels_paths=[],
        true_additional_texts_paths=[],
        true_flow_paths=[],
    )

    assert result == []
    assert FakeDiagramOptions.calls == []


@pytest.mark.parametrize(
    (
        "true_nodes_paths",
        "true_labels_paths",
        "true_additional_texts_paths",
        "true_flow_paths",
    ),
    [
        pytest.param(
            [Path("nodes/diagram_1.json"), Path("nodes/diagram_2.json")],
            [Path("labels/diagram_1.json")],
            [Path("additional_texts/diagram_1.json")],
            [Path("flow/diagram_1.json")],
            id="labels-shorter",
        ),
        pytest.param(
            [Path("nodes/diagram_1.json")],
            [Path("labels/diagram_1.json")],
            [
                Path("additional_texts/diagram_1.json"),
                Path("additional_texts/diagram_2.json"),
            ],
            [Path("flow/diagram_1.json")],
            id="additional-texts-longer",
        ),
        pytest.param(
            [Path("nodes/diagram_1.json")],
            [Path("labels/diagram_1.json")],
            [Path("additional_texts/diagram_1.json")],
            [Path("flow/diagram_1.json"), Path("flow/diagram_2.json")],
            id="flow-longer",
        ),
    ],
)
def test_load_true_diagram_options_from_paths_raises_for_mismatched_lengths(
    monkeypatch: pytest.MonkeyPatch,
    true_nodes_paths: list[Path],
    true_labels_paths: list[Path],
    true_additional_texts_paths: list[Path],
    true_flow_paths: list[Path],
) -> None:
    reset_fakes()
    monkeypatch.setattr(load_module, "DiagramOptions", FakeDiagramOptions)

    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        load_module.load_true_diagram_options_from_paths(
            true_nodes_paths=true_nodes_paths,
            true_labels_paths=true_labels_paths,
            true_additional_texts_paths=true_additional_texts_paths,
            true_flow_paths=true_flow_paths,
        )


def test_load_true_diagram_options_loads_sorted_json_paths_from_single_dirs_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    true_nodes_dir = tmp_path / "nodes"
    true_labels_dir = tmp_path / "labels"
    true_additional_texts_dir = tmp_path / "additional_texts"
    true_flow_dir = tmp_path / "flow"

    nodes_path_b = true_nodes_dir / "b.json"
    nodes_path_a = true_nodes_dir / "a.json"

    labels_path_b = true_labels_dir / "b.json"
    labels_path_a = true_labels_dir / "a.json"

    additional_texts_path_b = true_additional_texts_dir / "b.json"
    additional_texts_path_a = true_additional_texts_dir / "a.json"

    flow_path_b = true_flow_dir / "b.json"
    flow_path_a = true_flow_dir / "a.json"

    nested_nodes_path = true_nodes_dir / "nested" / "nested.json"
    nested_labels_path = true_labels_dir / "nested" / "nested.json"
    nested_additional_texts_path = true_additional_texts_dir / "nested" / "nested.json"
    nested_flow_path = true_flow_dir / "nested" / "nested.json"

    for path in [
        nodes_path_b,
        nodes_path_a,
        labels_path_b,
        labels_path_a,
        additional_texts_path_b,
        additional_texts_path_a,
        flow_path_b,
        flow_path_a,
        nested_nodes_path,
        nested_labels_path,
        nested_additional_texts_path,
        nested_flow_path,
    ]:
        create_json(path)

    create_text(true_nodes_dir / "not_included.txt")
    create_text(true_labels_dir / "not_included.txt")
    create_text(true_additional_texts_dir / "not_included.txt")
    create_text(true_flow_dir / "not_included.txt")

    monkeypatch.setattr(load_module, "DiagramOptions", FakeDiagramOptions)

    result = load_module.load_true_diagram_options(
        true_nodes_dir=true_nodes_dir,
        true_labels_dir=true_labels_dir,
        true_additional_texts_dir=true_additional_texts_dir,
        true_flow_dir=true_flow_dir,
    )

    expected = [
        {
            "true_nodes_path": nodes_path_a,
            "true_labels_path": labels_path_a,
            "true_additional_texts_path": additional_texts_path_a,
            "true_flow_path": flow_path_a,
        },
        {
            "true_nodes_path": nodes_path_b,
            "true_labels_path": labels_path_b,
            "true_additional_texts_path": additional_texts_path_b,
            "true_flow_path": flow_path_b,
        },
    ]

    assert result == expected
    assert FakeDiagramOptions.calls == expected

    expected_paths = {
        true_nodes_dir,
        nodes_path_a,
        nodes_path_b,
        true_nodes_dir / "not_included.txt",
        nested_nodes_path.parent,
        nested_nodes_path,
        true_labels_dir,
        labels_path_a,
        labels_path_b,
        true_labels_dir / "not_included.txt",
        nested_labels_path.parent,
        nested_labels_path,
        true_additional_texts_dir,
        additional_texts_path_a,
        additional_texts_path_b,
        true_additional_texts_dir / "not_included.txt",
        nested_additional_texts_path.parent,
        nested_additional_texts_path,
        true_flow_dir,
        flow_path_a,
        flow_path_b,
        true_flow_dir / "not_included.txt",
        nested_flow_path.parent,
        nested_flow_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_load_true_diagram_options_returns_empty_list_if_no_json_files_in_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_fakes()

    true_nodes_dir = tmp_path / "nodes"
    true_labels_dir = tmp_path / "labels"
    true_additional_texts_dir = tmp_path / "additional_texts"
    true_flow_dir = tmp_path / "flow"

    nested_nodes_path = true_nodes_dir / "nested" / "nested.json"
    nested_labels_path = true_labels_dir / "nested" / "nested.json"
    nested_additional_texts_path = true_additional_texts_dir / "nested" / "nested.json"
    nested_flow_path = true_flow_dir / "nested" / "nested.json"

    for base in [
        true_nodes_dir,
        true_labels_dir,
        true_additional_texts_dir,
        true_flow_dir,
    ]:
        base.mkdir(parents=True)

    for path in [
        nested_nodes_path,
        nested_labels_path,
        nested_additional_texts_path,
        nested_flow_path,
    ]:
        create_json(path)

    monkeypatch.setattr(load_module, "DiagramOptions", FakeDiagramOptions)

    result = load_module.load_true_diagram_options(
        true_nodes_dir=true_nodes_dir,
        true_labels_dir=true_labels_dir,
        true_additional_texts_dir=true_additional_texts_dir,
        true_flow_dir=true_flow_dir,
    )

    assert result == []
    assert FakeDiagramOptions.calls == []

    expected_paths = {
        true_nodes_dir,
        nested_nodes_path.parent,
        nested_nodes_path,
        true_labels_dir,
        nested_labels_path.parent,
        nested_labels_path,
        true_additional_texts_dir,
        nested_additional_texts_path.parent,
        nested_additional_texts_path,
        true_flow_dir,
        nested_flow_path.parent,
        nested_flow_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths
