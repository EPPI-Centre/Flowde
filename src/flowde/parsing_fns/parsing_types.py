from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, create_model

from flowde.utils import validate_matching_stems

ParseType = Literal["node_text", "labels", "flow", "additional_texts"]


class Node(BaseModel):
    node_number: int
    text: str
    labels: list[str]
    points_to: list[int]


class Flowchart(BaseModel):
    nodes: list[Node]
    additional_texts: list[str]


class ParsingFunction(Protocol):
    result_structure: type[BaseModel]

    def __call__(
        self,
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> BaseModel: ...


def build_a_partial_flowchart(
    nodes_path: Path,
    labels_path: Path | None = None,
    additional_texts_path: Path | None = None,
    flow_path: Path | None = None,
) -> BaseModel:
    if nodes_path is None:
        msg = (
            "nodes_path must be provided to build a partial flowchart, "
            "as node numbers are needed to join the different parts together."
        )
        raise ValueError(msg)

    parsing_types = set()
    if nodes_path is not None:
        parsing_types.add("node_text")
    if labels_path is not None:
        parsing_types.add("labels")
    if flow_path is not None:
        parsing_types.add("flow")
    if additional_texts_path is not None:
        parsing_types.add("additional_texts")

    active_paths = [
        path
        for path in [
            nodes_path,
            labels_path,
            additional_texts_path,
            flow_path,
        ]
        if path is not None
    ]

    validate_matching_stems(*[[p] for p in active_paths])

    node_based_parts = []

    if nodes_path is not None:
        JustNodesSchema = build_partial_flowchart_schema({"node_text"})
        node_texts = JustNodesSchema.model_validate_json(
            nodes_path.read_text(encoding="utf-8")
        ).model_dump()["nodes"]
        node_texts.sort(key=lambda n: n["node_number"])
        node_based_parts.append(node_texts)

    if labels_path is not None:
        JustLabelsSchema = build_partial_flowchart_schema({"labels"})
        node_labels = JustLabelsSchema.model_validate_json(
            labels_path.read_text(encoding="utf-8")
        ).model_dump()["nodes"]
        node_labels.sort(key=lambda n: n["node_number"])
        node_based_parts.append(node_labels)

    if flow_path is not None:
        JustFlowsSchema = build_partial_flowchart_schema({"flow"})
        node_flows = JustFlowsSchema.model_validate_json(
            flow_path.read_text(encoding="utf-8")
        ).model_dump()["nodes"]
        node_flows.sort(key=lambda n: n["node_number"])
        node_based_parts.append(node_flows)

    node_numbers = [n["node_number"] for n in node_based_parts[0]]
    if node_numbers != list(range(1, len(node_numbers) + 1)):
        msg = (
            "Node numbers must be consecutive integers starting from 1. "
            f"Found node numbers {node_numbers}."
        )
        raise ValueError(msg)

    if len(node_numbers) == 0:
        msg = "At least one node must be provided to build a flowchart."
        raise ValueError(msg)

    for part in node_based_parts:
        part_node_numbers = [n["node_number"] for n in part]
        if part_node_numbers != node_numbers:
            msg = (
                "All provided node-based parts must have the same node numbers "
                "in the same order. "
                f"Expected node numbers {node_numbers} but got {part_node_numbers}."
            )
            raise ValueError(msg)

    joined_nodes = []
    for i in range(len(node_numbers)):
        joined_node = {"node_number": node_numbers[i]}
        for part in node_based_parts:
            joined_node.update(part[i])
        joined_nodes.append(joined_node)

    flowchart_dict = {"nodes": joined_nodes}

    if additional_texts_path is not None:
        JustAdditionalTextSchema = build_partial_flowchart_schema({"additional_texts"})
        parsed_additional_texts = JustAdditionalTextSchema.model_validate_json(
            additional_texts_path.read_text(encoding="utf-8")
        )
        flowchart_dict["additional_texts"] = parsed_additional_texts.additional_texts

    PartialFlowchartSchema = build_partial_flowchart_schema(parsing_types)
    return PartialFlowchartSchema.model_validate(flowchart_dict)


# TODO: Check the models are not getting anything for the properties that are None
def build_partial_flowcharts(
    nodes_paths: list[Path],
    labels_paths: list[Path] | None = None,
    additional_texts_paths: list[Path] | None = None,
    flow_paths: list[Path] | None = None,
) -> list[BaseModel]:
    if nodes_paths is None or len(nodes_paths) == 0:
        msg = (
            "nodes_paths must be provided to build partial flowcharts, "
            "as node numbers are needed to join the different parts together."
        )
        raise ValueError(msg)
    return [
        build_a_partial_flowchart(
            nodes_path=nodes_path,
            labels_path=labels_path,
            additional_texts_path=additional_texts_path,
            flow_path=flow_path,
        )
        for nodes_path, labels_path, additional_texts_path, flow_path in zip(
            nodes_paths,
            labels_paths or [None] * len(nodes_paths),
            additional_texts_paths or [None] * len(nodes_paths),
            flow_paths or [None] * len(nodes_paths),
            strict=True,
        )
    ]


def build_partial_flowchart_schema(
    parts_to_parse: set[ParseType] | None = None,
) -> type[BaseModel]:
    if parts_to_parse is None:
        parts_to_parse = {"node_text", "labels", "flow", "additional_texts"}

    node_fields = {}
    flowchart_fields = {}

    if {"node_text", "labels", "flow"} & parts_to_parse:
        node_fields["node_number"] = (int, ...)
        if "node_text" in parts_to_parse:
            node_fields["text"] = (str, ...)
        if "labels" in parts_to_parse:
            node_fields["labels"] = (list[str], ...)
        if "flow" in parts_to_parse:
            node_fields["points_to"] = (list[int], ...)

        PartialNodeDynamic = create_model(
            "PartialNodeDynamic",
            __config__=ConfigDict(extra="forbid"),
            **node_fields,
        )

        flowchart_fields["nodes"] = (list[PartialNodeDynamic], ...)

    if "additional_texts" in parts_to_parse:
        flowchart_fields["additional_texts"] = (list[str], ...)

    PartialFlowchartDynamic = create_model(
        "PartialFlowchartDynamic",
        __config__=ConfigDict(extra="forbid"),
        **flowchart_fields,
    )

    return PartialFlowchartDynamic
