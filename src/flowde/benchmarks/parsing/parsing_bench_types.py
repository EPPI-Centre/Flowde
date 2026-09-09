from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    number_only_levenshtein,
)
from flowde.utils import load_json, pretty_json

if TYPE_CHECKING:
    from pathlib import Path

TextMatchType = Literal["match", "unmatched_true", "unmatched_pred"]
ParsingPart = Literal["node_text", "labels", "flow", "additional_texts"]
NonEmptyStr = Annotated[str, Field(min_length=1)]

COMMON_CONFIG = ConfigDict(extra="forbid", validate_assignment=True, strict=True)


@dataclass(frozen=True)
class PredDiagramStructure:
    """The parsing components present in a prediction source or joined dataset."""

    node_text: bool
    labels: bool
    flow: bool
    additional_texts: bool

    @property
    def parts(self) -> frozenset[ParsingPart]:
        parts: set[ParsingPart] = set()
        if self.node_text:
            parts.add("node_text")
        if self.labels:
            parts.add("labels")
        if self.flow:
            parts.add("flow")
        if self.additional_texts:
            parts.add("additional_texts")
        return frozenset(parts)

    @property
    def has_nodes(self) -> bool:
        return self.node_text or self.labels or self.flow

    @property
    def top_level_keys(self) -> set[str]:
        keys = set()
        if self.has_nodes:
            keys.add("nodes")
        if self.additional_texts:
            keys.add("additional_texts")
        return keys

    @property
    def node_keys(self) -> set[str]:
        if not self.has_nodes:
            return set()

        keys = {"node_number"}
        if self.node_text:
            keys.add("text")
        if self.labels:
            keys.add("labels")
        if self.flow:
            keys.add("points_to")
        return keys

    def describe(self) -> str:
        if not self.parts:
            return "no parsing components"
        return ", ".join(sorted(self.parts))


@dataclass(frozen=True)
class PredDiagramSource:
    """One validated directory of consistently structured prediction files."""

    diagrams_dir: Path
    paths: tuple[Path, ...]
    structure: PredDiagramStructure


@dataclass(frozen=True)
class PredDiagramSources:
    """Validated prediction sources that can be joined into benchmark diagrams."""

    sources: tuple[PredDiagramSource, ...]
    structure: PredDiagramStructure

    @property
    def node_text_source(self) -> PredDiagramSource:
        for source in self.sources:
            if source.structure.node_text:
                return source

        msg = "Validated prediction sources should contain a node-text source."
        raise RuntimeError(msg)


# TODO: I should add some text normalisation in a text scoring function of my own.
# TODO: Add the parent image code to the nodes so that is correctly added when
# making a diagram or diagram options.
# TODO: Remind myself why we have diagram options as it's own class and should
# each diagram have it's type (pred or true) and the option index if true
# TODO: We have a counter that starts at 10,000. Add some validation to make sure
# that all nodes are numbered less than 10,000

# TODO: Do we really need them to be sequential. Like what's wrong with them not
# being. Check all uses to make sure we are not looping through the ndoes
# assuming that they are sequential anywhere.


class Node(BaseModel):
    model_config = COMMON_CONFIG

    node_number: int
    text: NonEmptyStr
    labels: list[NonEmptyStr] | None = None
    points_to: list[int] | None = None
    diagram_type: Literal["pred", "true"]
    true_option_idx: int | None = None
    parent_img_code: str

    @property
    def labels_done(self) -> bool:
        return self.labels is not None

    @property
    def text_done(self) -> bool:
        return self.text is not None

    @property
    def flow_done(self) -> bool:
        return self.points_to is not None

    @property
    def is_complete(self) -> bool:
        return self.labels_done and self.text_done and self.flow_done

    @model_validator(mode="after")
    def validate_option_idx_set(self) -> Node:
        if self.diagram_type == "true" and self.true_option_idx is None:
            msg = (
                f"True nodes should have a true option index.\n"
                f"In node {self.node_number} from {self.parent_img_code}\n"
                "No true option index found"
            )
            raise ValueError(msg)
        if self.diagram_type == "pred" and self.true_option_idx is not None:
            msg = (
                f"Pred nodes should not have a true option index.\n"
                f"In node {self.node_number} from {self.parent_img_code}\n"
                "A true option index was found"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_node_number_small(self) -> Node:
        if self.node_number >= 1000:  # noqa: PLR2004
            msg = (
                f"In flow scoring, we assume all node numbers are less than 1000.\n"
                f"In node from {self.parent_img_code}\n"
                f"Found node number {self.node_number}."
            )
            raise ValueError(msg)

        return self


class Diagram(BaseModel):
    model_config = COMMON_CONFIG

    nodes: list[Node] | None = Field(default=None, min_length=1)
    additional_texts: list[NonEmptyStr] | None = None
    parent_img_code: str
    diagram_type: Literal["pred", "true"]
    true_option_idx: int | None

    @property
    def labels_done(self) -> bool:
        return self.nodes is not None and all(node.labels_done for node in self.nodes)

    @property
    def text_done(self) -> bool:
        return self.nodes is not None and all(node.text_done for node in self.nodes)

    @property
    def flow_done(self) -> bool:
        return self.nodes is not None and all(node.flow_done for node in self.nodes)

    @property
    def additional_texts_done(self) -> bool:
        return self.additional_texts is not None

    @property
    def is_complete(self) -> bool:
        return (
            self.nodes is not None
            and all(node.is_complete for node in self.nodes)
            and self.additional_texts_done
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        diagram_type: Literal["pred", "true"],
        true_option_idx: int | None,
    ) -> Diagram:
        try:
            diagram_dict = load_json(path)
            diagram_dict["parent_img_code"] = path.stem
            diagram_dict["diagram_type"] = diagram_type
            diagram_dict["true_option_idx"] = true_option_idx

            for node_dict in diagram_dict["nodes"]:
                node_dict["parent_img_code"] = path.stem
                node_dict["diagram_type"] = diagram_type
                node_dict["true_option_idx"] = true_option_idx
            return cls.model_validate(diagram_dict)
        except Exception as e:
            msg = (
                f"Failed to create {cls.__name__}\n for img_parent_code="
                f"{path.stem!s}\n From path={path}.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e

    @model_validator(mode="after")
    def validate_true_diagrams_complete(self) -> Diagram:
        if self.diagram_type == "true" and not self.is_complete:
            msg = (
                "True diagrams should be complete.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Diagram is not complete.\n text_done: {self.text_done}\n"
                f"additional_texts_done: {self.additional_texts_done}\n "
                f"labels_done: {self.labels_done}\nflow_done: {self.flow_done}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_node_numbers_unique(self) -> Diagram:
        if self.nodes is None:
            return self

        node_numbers = [node.node_number for node in self.nodes]
        if len(node_numbers) != len(set(node_numbers)):
            msg = (
                f"Node numbers should be unique within a diagram.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Found duplicate node numbers: {node_numbers}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)
        return self

    # TODO: we changed this to only true, should it be both still
    @model_validator(mode="after")
    def validate_sequential_node_numbers(self) -> Diagram:
        if self.nodes is None:
            return self

        node_numbers = [node.node_number for node in self.nodes]
        if (
            node_numbers != list(range(1, len(node_numbers) + 1))
            and self.diagram_type == "true"
        ):
            msg = (
                "For true diagrams, node numbers should be sequential starting from 1\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Found node numbers: {node_numbers}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_option_idx_set(self) -> Diagram:
        if self.diagram_type == "true" and self.true_option_idx is None:
            msg = (
                "True diagrams should have a true option index.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Found true option index: {self.true_option_idx}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)
        if self.diagram_type == "pred" and self.true_option_idx is not None:
            msg = (
                "Pred diagrams should not have a true option index.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Found true option index: {self.true_option_idx}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_diagram_types_match(self) -> Diagram:
        if self.nodes is None:
            return self

        for node in self.nodes:
            if node.diagram_type != self.diagram_type:
                msg = (
                    "All nodes in a diagram should have the same diagram type "
                    "as the diagram.\n"
                    f"In diagram from: {self.parent_img_code}\n"
                    f"Found diagram type: {self.diagram_type}\n"
                    f"Found node diagram type: {node.diagram_type}\n"
                    f"In node number: {node.node_number}\n"
                    f"Diagram: \n{pretty_json(self)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_all_nodes_have_parent_img_code(self) -> Diagram:
        if self.nodes is None:
            return self

        for node in self.nodes:
            if node.parent_img_code != self.parent_img_code:
                msg = (
                    "All nodes in a diagram should have the same parent img "
                    "code as the diagram.\n"
                    f"In diagram from: {self.parent_img_code}\n"
                    f"Found diagram parent img code: {self.parent_img_code}\n"
                    f"Found node parent img code: {node.parent_img_code}\n"
                    f"In node number: {node.node_number}\n"
                    f"Diagram: \n{pretty_json(self)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_all_nodes_have_option_idx(self) -> Diagram:
        if self.nodes is None:
            return self

        for node in self.nodes:
            if node.true_option_idx != self.true_option_idx:
                msg = (
                    "All nodes in a diagram should have the same true option "
                    "index as the diagram.\n"
                    f"In diagram from: {self.parent_img_code}\n"
                    f"Found diagram true option index: {self.true_option_idx}\n"
                    f"Found node true option index: {node.true_option_idx}\n"
                    f"In node number: {node.node_number}\n"
                    f"Diagram: \n{pretty_json(self)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_one_attribute_done(self) -> Diagram:
        if not (self.text_done or self.additional_texts_done):
            msg = (
                "Either text or additional texts should be done for a diagram.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Found text_done: {self.text_done}\n"
                f"Found additional_texts_done: {self.additional_texts_done}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_all_nodes_are_in_flow(self) -> Diagram:
        if self.diagram_type == "true":
            points_out = set()
            is_pointed_to = set()
            node_numbers = {node.node_number for node in self.nodes}
            for node in self.nodes:
                if len(node.points_to) > 0:
                    points_out.add(node.node_number)
                    is_pointed_to.update(node.points_to)

            union = points_out.union(is_pointed_to)
            if union != node_numbers:
                msg = (
                    "In a true diagram, all nodes should point to or be pointed "
                    "to by another node\n"
                    f"In true diagram from: {self.parent_img_code}\n"
                    f"The nodes that were not included in the flow:\n"
                    f"{node_numbers - union}\n"
                )
                raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_no_fake_nodes_in_flow(self) -> Diagram:
        if not self.flow_done:
            return self

        node_numbers = {node.node_number for node in self.nodes}
        for node in self.nodes:
            for pointed_to_node_number in node.points_to:
                if pointed_to_node_number not in node_numbers:
                    msg = (
                        "Nodes should not point to nodes that do not exist.\n"
                        f"In diagram from: {self.parent_img_code}\n"
                        f"Node {node.node_number} points to non-existent node "
                        f"{pointed_to_node_number}\n"
                        f"Node numbers in diagram: {node_numbers}\n"
                        f"Diagram: \n{pretty_json(self)}"
                    )
                    raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_labels_all_or_none_done(self) -> Diagram:
        if self.nodes is None:
            return self

        labels_done_by_node = [node.labels_done for node in self.nodes]

        if any(labels_done_by_node) and not all(labels_done_by_node):
            nodes_with_labels = [
                node.node_number for node in self.nodes if node.labels_done
            ]
            nodes_without_labels = [
                node.node_number for node in self.nodes if not node.labels_done
            ]
            msg = (
                "If labels are done for one node in a diagram, labels should be done "
                "for all nodes in that diagram.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Nodes with labels done: {nodes_with_labels}\n"
                f"Nodes without labels done: {nodes_without_labels}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_flow_all_or_none_done(self) -> Diagram:
        if self.nodes is None:
            return self

        flow_done_by_node = [node.flow_done for node in self.nodes]

        if any(flow_done_by_node) and not all(flow_done_by_node):
            nodes_with_flow = [
                node.node_number for node in self.nodes if node.flow_done
            ]
            nodes_without_flow = [
                node.node_number for node in self.nodes if not node.flow_done
            ]
            msg = (
                "If flow is done for one node in a diagram, flow should be done "
                "for all nodes in that diagram.\n"
                f"In diagram from: {self.parent_img_code}\n"
                f"Nodes with flow done: {nodes_with_flow}\n"
                f"Nodes without flow done: {nodes_without_flow}\n"
                f"Diagram: \n{pretty_json(self)}"
            )
            raise ValueError(msg)

        return self


class DiagramOptions(BaseModel):
    model_config = COMMON_CONFIG

    options: list[Diagram] = Field(min_length=1)
    parent_img_code: str
    parent_nodes_path: Path
    parent_labels_path: Path
    parent_additional_texts_path: Path
    parent_flow_path: Path

    @classmethod
    def from_true_paths(
        cls,
        true_nodes_path: Path,
        true_labels_path: Path,
        true_additional_texts_path: Path,
        true_flow_path: Path,
    ) -> DiagramOptions:
        try:
            true_nodes_dict = load_json(true_nodes_path)
            true_labels_dict = load_json(true_labels_path)
            true_additional_texts_dict = load_json(true_additional_texts_path)
            true_flow_dict = load_json(true_flow_path)

            diagram_options = {
                "options": [],
                "parent_img_code": true_nodes_path.stem,
                "parent_nodes_path": true_nodes_path,
                "parent_labels_path": true_labels_path,
                "parent_additional_texts_path": true_additional_texts_path,
                "parent_flow_path": true_flow_path,
            }

            for i, (
                nodes_option,
                labels_option,
                additional_texts_option,
                flow_option,
            ) in enumerate(
                zip(
                    true_nodes_dict["options"],
                    true_labels_dict["options"],
                    true_additional_texts_dict["options"],
                    true_flow_dict["options"],
                    strict=True,
                )
            ):
                diagram = {
                    "nodes": [],
                    "additional_texts": additional_texts_option["additional_texts"],
                    "parent_img_code": true_nodes_path.stem,
                    "diagram_type": "true",
                    "true_option_idx": i,
                }

                nodes_option["nodes"].sort(key=lambda x: x["node_number"])
                labels_option["nodes"].sort(key=lambda x: x["node_number"])
                flow_option["nodes"].sort(key=lambda x: x["node_number"])

                for node_w_text, node_w_labels, node_w_flow in zip(
                    nodes_option["nodes"],
                    labels_option["nodes"],
                    flow_option["nodes"],
                    strict=True,
                ):
                    diagram["nodes"].append(
                        {
                            "node_number": node_w_text["node_number"],
                            "text": node_w_text["text"],
                            "labels": node_w_labels["labels"],
                            "points_to": node_w_flow["points_to"],
                            "parent_img_code": true_nodes_path.stem,
                            "diagram_type": "true",
                            "true_option_idx": i,
                        }
                    )

                diagram_options["options"].append(diagram)

            return cls.model_validate(diagram_options)
        except Exception as e:
            msg = (
                f"Failed to create {cls.__name__}\nFor img_parent_code="
                f"{true_nodes_path.stem!s}.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e

    @model_validator(mode="after")
    def validate_true_paths_are_from_same_img(self) -> DiagramOptions:
        if not (
            self.parent_nodes_path.stem
            == self.parent_labels_path.stem
            == self.parent_additional_texts_path.stem
            == self.parent_flow_path.stem
        ):
            msg = f"""The files
            {self.parent_nodes_path}
            {self.parent_labels_path}
            {self.parent_additional_texts_path}
            {self.parent_flow_path}
              should have the same stem.
              for img {self.parent_img_code}."""
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_jsons_have_same_node_numbers(self) -> DiagramOptions:
        true_nodes_dict = load_json(self.parent_nodes_path)
        true_labels_dict = load_json(self.parent_labels_path)
        true_flow_dict = load_json(self.parent_flow_path)

        for i, (nodes_option, labels_option, flow_option) in enumerate(
            zip(
                true_nodes_dict["options"],
                true_labels_dict["options"],
                true_flow_dict["options"],
                strict=True,
            )
        ):
            node_numbers_from_nodes = [
                node["node_number"] for node in nodes_option["nodes"]
            ]
            node_numbers_from_nodes.sort()

            node_numbers_from_labels = [
                node["node_number"] for node in labels_option["nodes"]
            ]
            node_numbers_from_labels.sort()

            node_numbers_from_flow = [
                node["node_number"] for node in flow_option["nodes"]
            ]
            node_numbers_from_flow.sort()

            if not (
                node_numbers_from_nodes
                == node_numbers_from_flow
                == node_numbers_from_labels
            ):
                msg = (
                    f"For {self.__class__.__name__}, the node numbers in nodes, "
                    "flows,and labels files should match\n"
                    f"For option {i},\n"
                    f"Nodes had: {node_numbers_from_nodes}\n"
                    f"Labels had: {node_numbers_from_labels}\n"
                    f"Flow had: {node_numbers_from_flow}\n"
                    f"From files:\n"
                    f"Nodes file: {self.parent_nodes_path}\n"
                    f"Labels file: {self.parent_labels_path}\n"
                    f"Flow file: {self.parent_flow_path}\n"
                    f"For img {self.parent_img_code}.\n"
                    f"Diagram: \n{pretty_json(self.options[i])}"
                )
                raise ValueError(msg)

        return self


class TextListMatch(BaseModel):
    model_config = COMMON_CONFIG

    true_index: int | None
    pred_index: int | None
    true_text: NonEmptyStr | None
    pred_text: NonEmptyStr | None
    parent_img_code: str | None = None
    true_option_idx: int | None = None
    cost: float

    @property
    def match_type(self) -> TextMatchType:
        if self.true_index is not None and self.pred_index is not None:
            return "match"
        if self.true_index is not None:
            return "unmatched_true"
        return "unmatched_pred"

    @model_validator(mode="after")
    def validate_index_and_text_match(self) -> TextListMatch:
        if (self.true_index is None) != (self.true_text is None):
            msg = "True index and true text should both be None or both be set."
            raise ValueError(msg)

        if (self.pred_index is None) != (self.pred_text is None):
            msg = "Pred index and pred text should both be None or both be set."
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_not_both_unmatched(self) -> TextListMatch:
        if self.true_index is None and self.pred_index is None:
            msg = "At least one of true index or pred index should be set."
            raise ValueError(msg)

        return self


# TODO: Store the original true and predicted text lists and validate that the
# matches account for every item in them exactly once.
class TextListMatches(BaseModel):
    model_config = COMMON_CONFIG

    matches: list[TextListMatch]

    @property
    def total_cost(self) -> float:
        return sum(match.cost for match in self.matches)


class NodeMatch(BaseModel):
    model_config = COMMON_CONFIG

    true_node: Node | None
    pred_node: Node | None
    node_text_cost: int | float
    label_matches: TextListMatches | None

    @property
    def match_type(self) -> TextMatchType:
        if self.true_node is not None and self.pred_node is not None:
            return "match"
        if self.true_node is not None:
            return "unmatched_true"
        return "unmatched_pred"

    @model_validator(mode="after")
    def validate_nodes_have_same_parent_img_code(self) -> NodeMatch:
        if (
            self.true_node is not None
            and self.pred_node is not None
            and self.true_node.parent_img_code != self.pred_node.parent_img_code
        ):
            msg = (
                f"True node and pred node in a node match should have the same "
                "parent img code. "
                f"True node came from: {self.true_node.parent_img_code}"
                f"Pred node came from: {self.pred_node.parent_img_code}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_one_node_is_not_none(self) -> NodeMatch:
        if self.true_node is None and self.pred_node is None:
            msg = "At least one of true node or pred node should be not None."
            raise ValueError(msg)
        return self

    @property
    def parent_img_code(self) -> str:
        if self.true_node is not None:
            return self.true_node.parent_img_code

        return self.pred_node.parent_img_code

    @property
    def numbers_only_node_text_cost(self) -> float:
        return number_only_levenshtein(
            true_text=self.true_node.text if self.true_node is not None else None,
            pred_text=self.pred_node.text if self.pred_node is not None else None,
        )

    # TODO: validate that the text is only allowed to be None if the text is None
    # and the match type is correct.


class NodeMatches(BaseModel):
    model_config = COMMON_CONFIG

    matches: list[NodeMatch]
    true_diagram_option_idx: int

    def add(
        self,
        true_node: Node | None,
        pred_node: Node | None,
        node_text_cost: float,
        label_matches: TextListMatches | None = None,
    ) -> None:
        self.matches.append(
            NodeMatch(
                true_node=true_node,
                pred_node=pred_node,
                node_text_cost=node_text_cost,
                label_matches=label_matches,
            )
        )

    def get_matched_true_node(
        self,
        pred_node_number: int,
        allow_fake_pred_node: bool = False,
    ) -> Node | None:
        for match in self.matches:
            if (
                match.pred_node is not None
                and match.pred_node.node_number == pred_node_number
            ):
                return match.true_node

        if allow_fake_pred_node:
            return None

        msg = (
            f"Pred node {pred_node_number} not found in node matches,\n"
            f"for img {self.parent_img_code}.\n"
            f"Pred nodes in matches: {pretty_json(self.pred_nodes)}\n"
        )
        raise ValueError(msg)

    def get_matched_pred_node(self, true_node_number: int) -> Node | None:
        for match in self.matches:
            if (
                match.true_node is not None
                and match.true_node.node_number == true_node_number
            ):
                return match.pred_node

        msg = (
            f"True node {true_node_number} not found in node matches,\n"
            f"for img {self.parent_img_code}.\n"
            f"Pred nodes in matches: {pretty_json(self.true_nodes)}\n"
        )
        raise ValueError(msg)

    @property
    def parent_img_code(self) -> str:
        if len(self.matches) == 0:
            msg = "There should be at least one node match to get parent img code."
            raise ValueError(msg)

        return self.matches[0].parent_img_code

    @property
    def total_node_text_cost(self) -> float:
        return sum(match.node_text_cost for match in self.matches)

    @property
    def total_numbers_only_node_text_cost(self) -> float:
        return sum(match.numbers_only_node_text_cost for match in self.matches)

    @property
    def total_label_error_cost(self) -> float:
        total_cost = 0.0
        for node_match in self.matches:
            if node_match.label_matches is None:
                msg = (
                    "All node matches must have label matches before calculating "
                    "the total label cost."
                )
                raise ValueError(msg)
            total_cost += node_match.label_matches.total_cost
        return total_cost

    @property
    def true_nodes(self) -> list[Node]:
        return [
            match.true_node for match in self.matches if match.true_node is not None
        ]

    @property
    def pred_nodes(self) -> list[Node]:
        return [
            match.pred_node for match in self.matches if match.pred_node is not None
        ]

    @property
    def flow_score(self) -> FlowScores:
        true_edges = [
            (node.node_number, next_node_number)
            for node in self.true_nodes
            for next_node_number in node.points_to
        ]

        # TODO: We need to decide how to solve this so that jaccard is still
        # calculated correctly. Because we are getting an edge (node_number,
        # None) when the nexxt Node is not matched? Or something like that?
        # TODO: So both matched node and matched_next node can be None. But we
        # don't want multiplee (None, None) because that breaks the result.
        counter = count(start=10000)
        transformed_pred_edges = []
        for node in self.pred_nodes:
            for next_node_number in node.points_to:
                matched_node = self.get_matched_true_node(
                    pred_node_number=node.node_number,
                    allow_fake_pred_node=True,
                )
                matched_next_node = self.get_matched_true_node(
                    pred_node_number=next_node_number,
                    allow_fake_pred_node=True,
                )

                matched_node_number = (
                    next(counter) if matched_node is None else matched_node.node_number
                )
                matched_next_node_number = (
                    next(counter)
                    if matched_next_node is None
                    else matched_next_node.node_number
                )

                transformed_pred_edges.append(
                    (matched_node_number, matched_next_node_number)
                )

        true_edges = set(true_edges)
        transformed_pred_edges = set(transformed_pred_edges)

        # TODO: Check that these are all correct.
        tp = len(transformed_pred_edges & true_edges)
        fp = len(transformed_pred_edges - true_edges)
        fn = len(true_edges - transformed_pred_edges)

        precision = tp / len(transformed_pred_edges) if transformed_pred_edges else 0.0
        recall = tp / len(true_edges) if true_edges else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        jaccard = (
            tp / len(transformed_pred_edges | true_edges)
            if (transformed_pred_edges | true_edges)
            else 1.0
        )

        return FlowScores(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            jaccard=jaccard,  # (intersection over union)
            missing_edges=true_edges - transformed_pred_edges,
            extra_edges=transformed_pred_edges - true_edges,
        )

    @model_validator(mode="after")
    def validate_no_node_matched_twice(self) -> NodeMatches:
        matched_true_node_numbers = set()
        matched_pred_node_numbers = set()

        for match in self.matches:
            if match.true_node is not None:
                if match.true_node.node_number in matched_true_node_numbers:
                    msg = (
                        f"True node {match.true_node.node_number} is matched more "
                        "than once.\n"
                        f"In NodeMatches from img {self.parent_img_code}.\n"
                    )
                    raise ValueError(msg)
                matched_true_node_numbers.add(match.true_node.node_number)

            if match.pred_node is not None:
                if match.pred_node.node_number in matched_pred_node_numbers:
                    msg = (
                        f"Pred node {match.pred_node.node_number} is matched more "
                        "than once.\n"
                        f"In NodeMatches from img {self.parent_img_code}.\n"
                    )
                    raise ValueError(msg)
                matched_pred_node_numbers.add(match.pred_node.node_number)

        return self


class DiagramMatch(BaseModel):
    model_config = COMMON_CONFIG

    node_matches: NodeMatches
    additional_text_matches: TextListMatches
    pred_diagram: Diagram
    true_diagram: Diagram

    @property
    def total_node_text_cost(self) -> float:
        return self.node_matches.total_node_text_cost

    @property
    def total_label_error_cost(self) -> float:
        return self.node_matches.total_label_error_cost

    @property
    def total_text_cost(self) -> float:
        return (
            self.total_node_text_cost
            + self.total_label_error_cost
            + self.total_additional_text_cost
        )

    @property
    def total_additional_text_cost(self) -> float:
        return self.additional_text_matches.total_cost

    @property
    def flow_score(self) -> FlowScores:
        return self.node_matches.flow_score

    @model_validator(mode="after")
    def validate_diagrams_complete(self) -> DiagramMatch:
        if not self.pred_diagram.is_complete:
            msg = (
                "Diagrams should be complete for a diagram match.\n"
                "Pred diagram is not complete.\n"
                f"Parent img code: {self.pred_diagram.parent_img_code}\n"
                f"text_done: {self.pred_diagram.text_done}\n"
                f"labels_done: {self.pred_diagram.labels_done}\n"
                f"flow_done: {self.pred_diagram.flow_done}\n"
                f"additional_texts_done: {self.pred_diagram.additional_texts_done}\n"
                f"Diagram: \n{pretty_json(self.pred_diagram)}"
            )
            raise ValueError(msg)

        if not self.true_diagram.is_complete:
            msg = (
                "Diagrams should be complete for a diagram match.\n"
                "True diagram is not complete.\n"
                f"Parent img code: {self.true_diagram.parent_img_code}\n"
                f"text_done: {self.true_diagram.text_done}\n"
                f"labels_done: {self.true_diagram.labels_done}\n"
                f"flow_done: {self.true_diagram.flow_done}\n"
                f"additional_texts_done: {self.true_diagram.additional_texts_done}\n"
                f"Diagram: \n{pretty_json(self.true_diagram)}"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_every_node_in_matches(self) -> DiagramMatch:
        true_nodes_from_matches = [
            match.true_node
            for match in self.node_matches.matches
            if match.true_node is not None
        ]
        true_nodes_from_matches.sort(key=lambda x: x.node_number)

        pred_nodes_from_matches = [
            match.pred_node
            for match in self.node_matches.matches
            if match.pred_node is not None
        ]
        pred_nodes_from_matches.sort(key=lambda x: x.node_number)

        if self.true_diagram.nodes != true_nodes_from_matches:
            msg = (
                f"True nodes do not match nodes from matches list."
                f"For the DiagramMatch from {self.true_diagram.parent_img_code},\n"
                f"True nodes from diagram:\n{pretty_json(self.true_diagram.nodes)}\n"
                f"True nodes from matches:\n{pretty_json(true_nodes_from_matches)}\n"
            )
            raise ValueError(msg)

        if self.pred_diagram.nodes != pred_nodes_from_matches:
            msg = (
                "Pred nodes do not match nodes from matches list.\n"
                f"For the DiagramMatch from {self.pred_diagram.parent_img_code},\n"
                f"Pred nodes from diagram:\n{pretty_json(self.pred_diagram.nodes)}\n"
                f"Pred nodes from matches:\n{pretty_json(pred_nodes_from_matches)}\n"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_every_additional_text_in_matches(self) -> DiagramMatch:
        pred_additional_texts_from_diagram = sorted(self.pred_diagram.additional_texts)
        true_additional_texts_from_diagram = sorted(self.true_diagram.additional_texts)

        pred_additional_texts_from_matches = sorted(
            match.pred_text
            for match in self.additional_text_matches.matches
            if match.pred_text is not None
        )
        true_additional_texts_from_matches = sorted(
            match.true_text
            for match in self.additional_text_matches.matches
            if match.true_text is not None
        )

        if pred_additional_texts_from_diagram != pred_additional_texts_from_matches:
            msg = (
                "Pred additional texts do not match additional texts from matches "
                "list.\n"
                f"For the DiagramMatch from {self.pred_diagram.parent_img_code},\n"
                "Pred additional texts from diagram:\n"
                f"{pretty_json(pred_additional_texts_from_diagram)}\n"
                "Pred additional texts from matches:\n"
                f"{pretty_json(pred_additional_texts_from_matches)}\n"
            )
            raise ValueError(msg)

        if true_additional_texts_from_diagram != true_additional_texts_from_matches:
            msg = (
                "True additional texts do not match additional texts from matches "
                "list.\n"
                f"For the DiagramMatch from {self.true_diagram.parent_img_code},\n"
                "True additional texts from diagram:\n"
                f"{pretty_json(true_additional_texts_from_diagram)}\n"
                "True additional texts from matches:\n"
                f"{pretty_json(true_additional_texts_from_matches)}\n"
            )
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_both_diagrams_have_same_origin(self) -> DiagramMatch:
        if self.pred_diagram.parent_img_code != self.true_diagram.parent_img_code:
            msg = (
                "The pred and true diagrams in a diagram match should have the "
                "same parent img code.\n"
                "Found pred diagram parent img code: "
                f"{self.pred_diagram.parent_img_code}\n"
                "Found true diagram parent img code: "
                f"{self.true_diagram.parent_img_code}\n"
                f"Pred diagram: \n{pretty_json(self.pred_diagram)}\n\n"
                f"True diagram: \n{pretty_json(self.true_diagram)}\n"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_diagram_pred_true_type(self) -> DiagramMatch:
        if self.pred_diagram.diagram_type != "pred":
            msg = (
                "The pred diagram in a diagram match should have diagram type "
                "'pred'.\n"
                f"For the DiagramMatch from {self.pred_diagram.parent_img_code},\n"
                f"Found pred diagram type: {self.pred_diagram.diagram_type}\n"
                f"Pred diagram: \n{pretty_json(self.pred_diagram)}\n"
            )
            raise ValueError(msg)

        if self.true_diagram.diagram_type != "true":
            msg = (
                "The true diagram in a diagram match should have diagram type "
                "'true'.\n"
                f"For the DiagramMatch from {self.true_diagram.parent_img_code},\n"
                f"Found true diagram type: {self.true_diagram.diagram_type}\n"
                f"True diagram: \n{pretty_json(self.true_diagram)}\n"
            )
            raise ValueError(msg)

        return self


class FlowScores(BaseModel):
    model_config = COMMON_CONFIG

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    missing_edges: set[tuple[int, int]]
    extra_edges: set[tuple[int, int]]
