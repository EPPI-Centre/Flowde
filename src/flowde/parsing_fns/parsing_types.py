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
    """
    Describe a callable that parses one image into a Pydantic result.

    Attributes
    ----------
    result_structure : type[BaseModel]
        Pydantic class describing the parser's output. Every successful call
        must return an instance of this class. The attribute holds the class,
        not an already parsed result. The class can describe a complete
        flowchart, selected flowchart parts or a custom output format.

    Notes
    -----
    This protocol describes an interface; it does not implement image parsing.
    A function or callable object can satisfy the interface without inheriting
    from `ParsingFunction`. The callable accepts `img_path` and optional
    `partial_flowchart` arguments, as described by
    [`__call__()`][flowde.parsing_fns.parsing_types.ParsingFunction.__call__].

    The OpenAI and Gemini parsing factories return callables with this
    interface. A returned parser processes one image per call and returns its
    Pydantic result. You can pass the parser to
    [`parse_imgs()`][flowde.parse_imgs.parse_imgs] to process a directory and
    save one JSON file per image.

    A custom parser used by Flowde must also expose its declared `run_settings`.
    You can attach these settings and the required `result_structure` class with
    [`model_function()`][flowde.model_function]. These settings allow saved-run
    compatibility checks;
    `run_settings` is a pipeline requirement beyond the attributes declared
    by this protocol. Built-in factories attach these settings automatically.

    The protocol itself performs no runtime validation. The directory pipeline
    checks that `result_structure` is a Pydantic class and that each returned
    result is an instance of that class. A dictionary, JSON string or `None`
    is not an accepted parsing result.

    """

    result_structure: type[BaseModel]

    def __call__(
        self,
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> BaseModel:
        """
        Parse one image, optionally using previously parsed data as context.

        Parameters
        ----------
        img_path : Path
            Path to the target image. The parser implementation must support
            the image's format. Flowde supplies this argument by keyword.
        partial_flowchart : BaseModel | None, optional
            Previously parsed data for the same image, such as node numbers
            and text when requesting labels. Defaults to `None`, meaning no
            earlier results are supplied. Flowde supplies this argument by
            keyword when context files are provided to
            [`parse_imgs()`][flowde.parse_imgs.parse_imgs].

        Returns
        -------
        BaseModel
            Parsed result for the target image, as an instance of the callable's
            `result_structure` class. The output schema determines which fields
            the result contains. Supplying partial context does not require
            those context fields to appear in the result.

        Notes
        -----
        This method defines the call signature; a parser implementation performs
        the actual work. The implementation can raise an exception when parsing
        fails. Returning `None` does not mark an image as successfully parsed.

        """
        ...


def build_a_partial_flowchart(
    nodes_path: Path,
    labels_path: Path | None = None,
    additional_texts_path: Path | None = None,
    flow_path: Path | None = None,
) -> BaseModel:
    """
    Combine saved parsing parts for one image into a Pydantic model.

    Load the node text and any supplied labels, additional text or flow from
    separate JSON files. Node numbers identify which labels and outgoing
    connections belong to each node. Supplying all four files produces a
    complete flowchart; optional files can be omitted to build a partial one.

    Parameters
    ----------
    nodes_path : Path
        Path to the UTF-8 JSON file containing node numbers and text, with the
        structure `{"nodes": [{"node_number": 1, "text": "..."}]}`. At least
        one node is required. Node numbers must be unique consecutive integers
        starting at `1`, although nodes may appear in any order in the file.
    labels_path : Path | None, optional
        Path to the UTF-8 JSON file containing labels for the same nodes, with
        the structure `{"nodes": [{"node_number": 1, "labels": ["..."]}]}`.
        Each node's `labels` value is a list of strings, which may be empty.
        Defaults to `None`, which omits labels from the combined model.
    additional_texts_path : Path | None, optional
        Path to the UTF-8 JSON file containing text outside the nodes, with
        the structure `{"additional_texts": ["..."]}`. The list may be empty.
        Defaults to `None`, which omits `additional_texts` from the combined
        model.
    flow_path : Path | None, optional
        Path to the UTF-8 JSON file containing outgoing connections for the
        same nodes, with the structure
        `{"nodes": [{"node_number": 1, "points_to": [2]}]}`. Each node's
        `points_to` list contains the destination node numbers and may be
        empty. Defaults to `None`, which omits `points_to` from the combined
        model.

    Returns
    -------
    BaseModel
        Instance of a generated Pydantic model containing `nodes`, sorted by
        `node_number`. Every node has `node_number` and `text`, plus `labels`
        and `points_to` when their files are supplied. The model also contains
        the top-level `additional_texts` list when its file is supplied.
        Fields for omitted parts are absent. Use `model_dump()` to obtain a
        dictionary or `model_dump_json()` to obtain JSON text.

    Raises
    ------
    ValueError
        If `nodes_path` is `None`, supplied filenames have different stems,
        no nodes are provided, node numbers are not consecutive starting at
        `1`, or the node-text, label and flow files contain different node
        numbers.
    pydantic.ValidationError
        If a file contains invalid JSON or does not match its part's schema,
        including missing required fields or extra fields.
    OSError
        If a supplied file is missing or cannot be read.
    UnicodeDecodeError
        If a supplied file cannot be decoded as UTF-8.

    Notes
    -----
    All supplied files must have the same filename stem, such as
    `paper-1_0.json` in separate part directories. Each file must contain only
    the fields for its own part, in the format written by parsing. Benchmark
    ground-truth files with an `options` wrapper are not accepted.

    Supplied node-text, label and flow files must contain exactly the same
    node numbers. Each file's nodes are sorted before joining, so their
    original order may differ. Additional text belongs to the whole
    flowchart and is not matched to individual nodes. The helper does not
    check whether the numbers in `points_to` refer to existing nodes.

    The function reads local files and makes no model requests. It returns
    the combined model without writing a file or changing the supplied files.

    Examples
    --------
    Combine four existing parsing results for `paper-1_0.png`, then save the
    combined JSON:

    ```python
    from pathlib import Path

    from flowde.parsing_fns.parsing_types import build_a_partial_flowchart

    parts_dir = Path("results/parsing")
    filename = "paper-1_0.json"

    diagram = build_a_partial_flowchart(
        nodes_path=parts_dir / "node_text" / filename,
        labels_path=parts_dir / "labels" / filename,
        additional_texts_path=parts_dir / "additional_texts" / filename,
        flow_path=parts_dir / "flow" / filename,
    )

    output_path = parts_dir / "combined" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diagram.model_dump_json(indent=2), encoding="utf-8")
    ```

    """
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
