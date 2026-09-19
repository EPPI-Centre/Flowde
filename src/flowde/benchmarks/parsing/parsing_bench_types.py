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
    """
    Represent one predicted or ground-truth node inside a parsing benchmark.

    Benchmark loaders attach the flowchart identifier and Ground-Truth Option
    index to the node fields read from JSON. Node numbers identify nodes within
    a diagram; predicted and ground-truth versions of a node may have different
    numbers.

    Attributes
    ----------
    node_number : int
        Node identifier. Must be less than `1000`; the containing `Diagram`
        checks uniqueness and applies the ground-truth numbering rules.
    text : str
        Non-empty node text as loaded, before any distance-function
        normalisation.
    labels : list[str] | None
        Non-empty label strings. `[]` means labels were parsed and none were
        found; `None`, the default, means labels were not parsed.
    points_to : list[int] | None
        Destination node numbers for outgoing connections. `[]` means flow
        was parsed and the node has no outgoing connections; `None`, the
        default, means flow was not parsed.
    diagram_type : {"pred", "true"}
        Whether the node belongs to a prediction or to ground truth.
    true_option_idx : int | None
        Zero-based Ground-Truth Option index. Required for a ground-truth
        node; must be `None` for a predicted node. Defaults to `None`.
    parent_img_code : str
        Filename stem identifying the source flowchart.
    text_done : bool
        Whether node text is present; always `True` for a valid `Node`.
    labels_done : bool
        Whether `labels` is present, including an empty list.
    flow_done : bool
        Whether `points_to` is present, including an empty list.
    is_complete : bool
        Whether text, labels and flow are all present.

    Raises
    ------
    pydantic.ValidationError
        If a field has an invalid type, text or a label string is empty,
        `node_number` is at least `1000`, or `true_option_idx` conflicts with
        `diagram_type`.

    """

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
    """
    Represent one prediction or one accepted ground-truth interpretation.

    A `DiagramMatch` exposes the compared diagrams through `pred_diagram` and
    `true_diagram`. Diagram fields retain the original text and node numbers;
    matching and text normalisation do not rewrite the loaded diagrams.

    Attributes
    ----------
    nodes : list[Node] | None
        Non-empty list of nodes, or `None` when no node-based parts were
        supplied. Defaults to `None`. Parsing benchmarks require predicted
        node text, so diagrams returned by benchmark methods contain nodes.
    additional_texts : list[str] | None
        Non-empty strings outside the nodes. `[]` means additional text was
        parsed and none was found; `None`, the default, means the part was
        not parsed.
    parent_img_code : str
        Filename stem identifying the source flowchart.
    diagram_type : {"pred", "true"}
        Whether the diagram is a prediction or a Ground-Truth Option.
    true_option_idx : int | None
        Zero-based Ground-Truth Option index for a ground-truth diagram;
        `None` for a predicted diagram.
    text_done : bool
        Whether nodes and their text are present.
    labels_done : bool
        Whether every node has a labels list, including empty lists.
    flow_done : bool
        Whether every node has a `points_to` list, including empty lists.
    additional_texts_done : bool
        Whether `additional_texts` is present, including an empty list.
    is_complete : bool
        Whether node text, labels, flow and additional text are all present.

    Raises
    ------
    pydantic.ValidationError
        If fields are invalid, node numbers repeat, connections refer to
        absent nodes, or node metadata disagrees with the diagram. Also
        raised if only some nodes supply labels or flow, or a ground-truth
        diagram is incomplete, has non-sequential node numbers, or contains
        a node absent from every connection.

    Notes
    -----
    Ground-truth nodes must appear in consecutive node-number order starting
    at `1`. Predictions may use different numbering. Ground-truth diagrams
    must contain all four parts; predictions may omit labels, flow or
    additional text.

    """

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
    """
    Record one paired or unmatched label or additional-text string.

    A pair records which predicted string represents which ground-truth
    string, even when their text differs. An unmatched record contains text
    on only one side.

    Attributes
    ----------
    true_index : int | None
        Zero-based position in the ground-truth text list. `None` when a
        predicted string has no ground-truth match.
    pred_index : int | None
        Zero-based position in the predicted text list. `None` when a
        ground-truth string has no predicted match.
    true_text : str | None
        Original ground-truth string, before normalisation; `None` exactly
        when `true_index` is `None`. Present strings must be non-empty.
    pred_text : str | None
        Original predicted string, before normalisation; `None` exactly when
        `pred_index` is `None`. Present strings must be non-empty.
    parent_img_code : str | None
        Source flowchart's filename stem. Additional-text matching sets this
        field; label matching leaves the default `None` because the containing
        `NodeMatch` identifies the flowchart.
    true_option_idx : int | None
        Selected Ground-Truth Option index, starting at `0`. Additional-text
        matching sets this field; label matching leaves the default `None`.
    cost : float
        Distance-function cost for this string pair or unmatched string.
        An unmatched string is compared with `None`; the built-in distance
        functions treat the missing side as an empty string.
    match_type : {"match", "unmatched_true", "unmatched_pred"}
        `"match"` when both strings are present; `"unmatched_true"` for a
        ground-truth string without a prediction; `"unmatched_pred"` for a
        predicted string without ground truth. `"match"` does not imply
        identical text or zero cost.

    Raises
    ------
    pydantic.ValidationError
        If a field is invalid, an index and its text disagree about whether
        the corresponding side is absent, or both sides are absent.

    """

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
    """
    Collect comparisons for one node's labels or a diagram's additional text.

    The benchmark matches strings to minimise total cost rather than pairing
    strings by their list positions. Each record retains the original indices
    so the source strings can be identified.

    Attributes
    ----------
    matches : list[TextListMatch]
        Paired and unmatched string records. Benchmark-generated collections
        account for every string in both source lists. Two empty source lists
        produce an empty collection. Use each record's `true_index` and
        `pred_index` to identify its original positions.
    total_cost : float
        Sum of every record's `cost`, including unmatched strings. An empty
        collection has cost zero. The property reports a total, not an average.

    Notes
    -----
    This class stores the supplied records and computes their total; the
    class does not perform string matching when constructed. The container
    does not store source lists or flowchart metadata. For additional text,
    the individual records hold the flowchart and Ground-Truth Option IDs.

    """

    model_config = COMMON_CONFIG

    matches: list[TextListMatch]

    @property
    def total_cost(self) -> float:
        return sum(match.cost for match in self.matches)


class NodeMatch(BaseModel):
    """
    Record one pair of corresponding nodes or one unmatched node.

    A paired prediction and ground-truth node may have different text or node
    numbers. The benchmark stores text cost separately from any label costs.

    Attributes
    ----------
    true_node : Node | None
        Node from the selected Ground-Truth Option, or `None` when a predicted
        node has no ground-truth match.
    pred_node : Node | None
        Predicted node, or `None` when a ground-truth node has no prediction.
    node_text_cost : int | float
        Cost returned by the benchmark's `distance_fn` for the node text.
        An unmatched node is compared with a missing text value; built-in
        distance functions treat the missing value as an empty string.
        Excludes label and flow costs.
    label_matches : TextListMatches | None
        Label comparisons for this Node Match, or `None` before labels have
        been matched. An empty collection means both nodes have no labels.
        Label comparisons include labels belonging to unmatched nodes.
    match_type : {"match", "unmatched_true", "unmatched_pred"}
        `"match"` for two present nodes; `"unmatched_true"` for a ground-truth
        node without a prediction; `"unmatched_pred"` for a predicted node
        without ground truth. A pair need not have zero text cost.
    parent_img_code : str
        Source flowchart's filename stem, taken from the ground-truth node
        when present, otherwise from the predicted node.
    numbers_only_node_text_cost : float
        Cost from comparing only the numbers in the two nodes' text with
        [`number_only_levenshtein()`][flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.number_only_levenshtein].
        Uses this existing Node Match without changing the nodes or their
        pairing.

    Raises
    ------
    pydantic.ValidationError
        If a field is invalid, both nodes are absent, or two present nodes
        have different `parent_img_code` values.

    """

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
    """
    Collect the Node Matches for one flowchart and one Ground-Truth Option.

    Benchmark-generated collections include every ground-truth and predicted
    node exactly once, either in a pair or as an unmatched node.

    Attributes
    ----------
    matches : list[NodeMatch]
        Paired and unmatched node records. Inspect `true_node` and `pred_node`
        on each record to identify the nodes; list positions are not node
        numbers.
    true_diagram_option_idx : int
        Index of the selected Ground-Truth Option, starting at `0`.
    parent_img_code : str
        Source flowchart's filename stem, taken from the first Node Match.
        Access raises `ValueError` if `matches` is empty.
    true_nodes : list[Node]
        All present ground-truth nodes, including unmatched nodes, in the
        order of their records in `matches`.
    pred_nodes : list[Node]
        All present predicted nodes, including unmatched nodes, in the
        order of their records in `matches`.
    total_node_text_cost : float
        Sum of the records' `node_text_cost` values, including unmatched nodes.
    total_numbers_only_node_text_cost : float
        Sum of the records' numbers-only text costs. Uses the existing Node
        Matches without choosing different pairings or a different option.
    total_label_error_cost : float
        Sum of label costs across all Node Matches. Access raises `ValueError`
        if any record has `label_matches=None`; an empty collection costs zero.
    flow_score : FlowScores
        Directed-connection scores calculated from these Node Matches.
        Requires a `points_to` list on every present node. Matched endpoints
        use ground-truth node numbers; unmatched predicted endpoints receive
        generated identifiers starting at `10000`.

    Raises
    ------
    pydantic.ValidationError
        If a field is invalid or a ground-truth or predicted node number
        appears in more than one record on the same side during validation.

    Notes
    -----
    Text costs include unmatched nodes and labels. Flow uses the same Node
    Matches; accessing `flow_score` does not select new pairings. Cost
    properties sum the stored records rather than rerunning model requests
    or matching.

    """

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
        """
        Find the ground-truth node paired with a predicted node number.

        Parameters
        ----------
        pred_node_number : int
            Identifier of a predicted node recorded in `matches`.
        allow_fake_pred_node : bool, optional
            Whether an absent predicted node number should return `None`
            instead of raising an error. Defaults to `False`. Flow scoring
            uses this option when looking up predicted connection endpoints.

        Returns
        -------
        Node | None
            The paired ground-truth node. Returns `None` for a recorded
            unmatched predicted node, or for an absent predicted node number
            when `allow_fake_pred_node=True`.

        Raises
        ------
        ValueError
            If `pred_node_number` is absent and `allow_fake_pred_node=False`.

        """
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
        """
        Find the predicted node paired with a ground-truth node number.

        Parameters
        ----------
        true_node_number : int
            Identifier of a ground-truth node recorded in `matches`.

        Returns
        -------
        Node | None
            The paired predicted node, or `None` when the recorded ground-truth
            node is unmatched.

        Raises
        ------
        ValueError
            If `true_node_number` does not appear in `matches`.

        """
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
    """
    Collect a complete prediction's comparison with one Ground-Truth Option.

    Both diagrams contain node text, labels, flow and additional text. The
    result exposes the loaded diagrams, their matching records, and the text
    and flow scores calculated from those records.

    Attributes
    ----------
    node_matches : NodeMatches
        Every paired and unmatched node, with label matches for each record.
        `true_diagram_option_idx` identifies the selected Ground-Truth Option.
    additional_text_matches : TextListMatches
        Paired and unmatched additional-text strings from both diagrams.
    pred_diagram : Diagram
        Complete predicted diagram, retaining its original node numbers and
        text.
    true_diagram : Diagram
        Complete ground-truth diagram selected from the accepted options.
        `true_option_idx` is its zero-based option index.
    total_node_text_cost : float
        Sum of paired and unmatched node-text costs.
    total_label_error_cost : float
        Sum of paired and unmatched label costs, including labels on unmatched
        nodes.
    total_additional_text_cost : float
        Sum of paired and unmatched additional-text costs.
    total_text_cost : float
        Sum of node-text, label and additional-text costs. Excludes flow scores.
    flow_score : FlowScores
        Directed-connection scores after applying `node_matches`.

    Raises
    ------
    pydantic.ValidationError
        If a field is invalid, either diagram is incomplete, the diagrams
        refer to different flowcharts or have incorrect `diagram_type`
        values, or the node and additional-text records do not account for
        the contents of the compared diagrams.

    """

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
    """
    Store directed-connection counts and scores for one matched flowchart.

    The benchmark calculates these fields after matching predicted nodes to
    ground-truth nodes. Connection direction matters: `1 -> 2` differs from
    `2 -> 1`.

    Attributes
    ----------
    tp : int
        Number of connections present in both the prediction and ground truth.
    fp : int
        Number of predicted connections absent from ground truth.
    fn : int
        Number of ground-truth connections absent from the prediction.
    precision : float
        Correct connections divided by predicted connections:
        `tp / (tp + fp)`. Zero when the prediction has no connections.
    recall : float
        Correct connections divided by ground-truth connections:
        `tp / (tp + fn)`. Zero when ground truth has no connections.
    f1 : float
        Harmonic mean of precision and recall. Zero when both are zero.
    jaccard : float
        Correct connections divided by all distinct connections in either
        diagram: `tp / (tp + fp + fn)`. Ranges from `0.0` to `1.0`; higher is
        better. Two empty connection sets score `1.0`.
    missing_edges : set[tuple[int, int]]
        Missing ground-truth connections as `(source, destination)` pairs
        of ground-truth node numbers.
    extra_edges : set[tuple[int, int]]
        Predicted connections absent from ground truth, expressed using
        matched ground-truth node numbers. Unmatched predicted endpoints use
        generated identifiers starting at `10000`, not original node numbers.

    Notes
    -----
    A `FlowScores` object stores supplied values; constructing the object does
    not calculate metrics from `tp`, `fp` and `fn`. Benchmark methods calculate
    the metrics before constructing the object.

    """

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
