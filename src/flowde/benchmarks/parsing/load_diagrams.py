from pathlib import Path

from flowde.benchmarks.parsing.parsing_bench_types import (
    Diagram,
    DiagramOptions,
    PredDiagramSources,
)
from flowde.utils import load_json


def load_pred_diagrams_from_paths(
    diagram_paths: list[Path],
) -> list[Diagram]:
    return [
        Diagram.from_path(
            path=p,
            diagram_type="pred",
            true_option_idx=None,
        )
        for p in diagram_paths
    ]


def load_pred_diagrams(
    diagrams_dir: Path,
) -> list[Diagram]:
    diagram_paths = list(diagrams_dir.glob("*.json"))
    diagram_paths.sort()
    return load_pred_diagrams_from_paths(diagram_paths)


def load_pred_diagrams_from_sources(
    pred_sources: PredDiagramSources,
) -> list[Diagram]:
    """
    Join validated prediction components before creating benchmark diagrams.

    Parameters
    ----------
    pred_sources
        Sources returned by ``validate_pred_diagram_sources``.

    Returns
    -------
    list[Diagram]
        Joined predicted diagrams in sorted image-code order.

    """
    pred_diagrams = []

    path_groups = zip(
        *(source.paths for source in pred_sources.sources),
        strict=True,
    )
    for paths in path_groups:
        img_code = paths[0].stem
        nodes_by_number: dict[int, dict[str, object]] = {}
        diagram_dict: dict[str, object] = {
            "parent_img_code": img_code,
            "diagram_type": "pred",
            "true_option_idx": None,
        }

        for source, path in zip(pred_sources.sources, paths, strict=True):
            data = load_json(path)

            if source.structure.has_nodes:
                for node in data["nodes"]:
                    node_number = node["node_number"]
                    nodes_by_number.setdefault(node_number, {}).update(node)

            if source.structure.additional_texts:
                diagram_dict["additional_texts"] = data["additional_texts"]

        diagram_dict["nodes"] = [
            {
                **nodes_by_number[node_number],
                "parent_img_code": img_code,
                "diagram_type": "pred",
                "true_option_idx": None,
            }
            for node_number in sorted(nodes_by_number)
        ]

        pred_diagrams.append(Diagram.model_validate(diagram_dict))

    return pred_diagrams


def load_true_diagram_options_from_paths(
    true_nodes_paths: list[Path],
    true_labels_paths: list[Path],
    true_additional_texts_paths: list[Path],
    true_flow_paths: list[Path],
) -> list[DiagramOptions]:
    return [
        DiagramOptions.from_true_paths(
            true_nodes_path=true_nodes_path,
            true_labels_path=true_labels_path,
            true_additional_texts_path=true_additional_texts_path,
            true_flow_path=true_flow_path,
        )
        for (
            true_nodes_path,
            true_labels_path,
            true_additional_texts_path,
            true_flow_path,
        ) in zip(
            true_nodes_paths,
            true_labels_paths,
            true_additional_texts_paths,
            true_flow_paths,
            strict=True,
        )
    ]


def load_true_diagram_options(
    true_nodes_dir: Path,
    true_labels_dir: Path,
    true_additional_texts_dir: Path,
    true_flow_dir: Path,
) -> list[DiagramOptions]:
    true_nodes_paths = list(true_nodes_dir.glob("*.json"))
    true_labels_paths = list(true_labels_dir.glob("*.json"))
    true_additional_texts_paths = list(true_additional_texts_dir.glob("*.json"))
    true_flow_paths = list(true_flow_dir.glob("*.json"))

    true_nodes_paths.sort()
    true_labels_paths.sort()
    true_flow_paths.sort()
    true_additional_texts_paths.sort()

    return load_true_diagram_options_from_paths(
        true_nodes_paths=true_nodes_paths,
        true_labels_paths=true_labels_paths,
        true_additional_texts_paths=true_additional_texts_paths,
        true_flow_paths=true_flow_paths,
    )
