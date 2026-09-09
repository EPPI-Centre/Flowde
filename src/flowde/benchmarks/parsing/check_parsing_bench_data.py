import inspect
import sys
from collections.abc import Sequence
from pathlib import Path

from flowde.benchmarks.parsing.parsing_bench_types import (
    ParsingPart,
    PredDiagramSource,
    PredDiagramSources,
    PredDiagramStructure,
)
from flowde.utils import load_json, pretty_json

N_DIAGRAMS_IN_BENCHMARK = 346
# Change all the new funcs so that they use the new diagram and node stuff,
# including in their checks.


# TODO: No duplicate path_ids when back, in any list.
def check_benchmark_files_match(
    pred_paths: Sequence[Path],
    nodes_paths: Sequence[Path],
    labels_paths: Sequence[Path],
    additional_texts_paths: Sequence[Path],
    flow_paths: Sequence[Path],
    allow_missing_pred_diagrams: bool = False,
    expected_num_diagrams: int = N_DIAGRAMS_IN_BENCHMARK,
) -> None:
    if allow_missing_pred_diagrams:
        if not (
            len(nodes_paths)
            == len(labels_paths)
            == len(additional_texts_paths)
            == len(flow_paths)
            == expected_num_diagrams
        ):
            msg = f"""Expected {expected_num_diagrams} diagrams, but found
            Number of true nodes files: {len(nodes_paths)}
            Number of true labels files: {len(labels_paths)}
            Number of true additional texts files: {len(additional_texts_paths)}
            Number of true flow files: {len(flow_paths)}
            """
            raise ValueError(msg)
    elif not (
        len(pred_paths)
        == len(nodes_paths)
        == len(labels_paths)
        == len(additional_texts_paths)
        == len(flow_paths)
        == expected_num_diagrams
    ):
        msg = f"""Expected {expected_num_diagrams} diagrams, but found
            Number of predicted diagrams: {len(pred_paths)}
            Number of true nodes files: {len(nodes_paths)}
            Number of true labels files: {len(labels_paths)}
            Number of true additional texts files: {len(additional_texts_paths)}
            Number of true flow files: {len(flow_paths)}
            """
        raise ValueError(msg)

    for nodes_path, labels_path, additional_texts_path, flow_path in zip(
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
        strict=True,
    ):
        if not (
            nodes_path.stem
            == labels_path.stem
            == additional_texts_path.stem
            == flow_path.stem
        ):
            msg = f"""The files
            {nodes_path}
            {labels_path}
            {additional_texts_path}
            {flow_path}
              should have the same stem"""
            raise ValueError(msg)

    if allow_missing_pred_diagrams:
        for pred_path in pred_paths:
            if not any(pred_path.stem == nodes_path.stem for nodes_path in nodes_paths):
                msg = (
                    f"The predicted file {pred_path} should have the same stem "
                    "and parent directory as one of the true nodes files."
                )
                raise ValueError(msg)
    else:
        for pred_path, nodes_path in zip(pred_paths, nodes_paths, strict=True):
            if pred_path.stem != nodes_path.stem:
                msg = (
                    f"The predicted file {pred_path} should have the same stem "
                    "and parent directory as the true nodes file "
                    f"{nodes_path}."
                )
                raise ValueError(msg)

    pred_path_ids = [p.stem for p in pred_paths]
    if len(set(pred_path_ids)) != len(pred_paths):
        msg = (
            "Duplicate predicted files found. Each file should have a unique "
            "combination of parent directory and stem. Found duplicates: "
            f"{[str(p) for p in set(pred_path_ids) if pred_path_ids.count(p) > 1]}"
        )
        raise ValueError(msg)

    # We check all true files have same path_id, so this covers all of them
    nodes_path_ids = [p.stem for p in nodes_paths]
    if len(set(nodes_path_ids)) != len(nodes_paths):
        msg = (
            "Duplicate true nodes files found. Each file should have a unique "
            "combination of parent directory and stem. Found duplicates: "
            f"{[str(p) for p in set(nodes_path_ids) if nodes_path_ids.count(p) > 1]}"
        )
        raise ValueError(msg)


# TODO: Change to files and add it to the run_benchmark
def check_benchmark_options_match(
    nodes_paths: Sequence[Path],
    labels_paths: Sequence[Path],
    additional_texts_paths: Sequence[Path],
    flow_paths: Sequence[Path],
) -> None:
    for nodes_path, labels_path, additional_texts_path, flow_path in zip(
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
        strict=True,
    ):
        nodes = load_json(nodes_path)
        labels = load_json(labels_path)
        additional_texts = load_json(additional_texts_path)
        flow = load_json(flow_path)

        # The number of options should match
        if not (
            len(nodes["options"])
            == len(flow["options"])
            == len(labels["options"])
            == len(additional_texts["options"])
        ):
            msg = (
                f"The number of options in true_diagrams, nodes, flows, labels, "
                "and additional texts should be the same for the same flowchart.\n"
                f"Found {len(nodes['options'])} options in {nodes_path}\n"
                f"Found {len(labels['options'])} options in {labels_path}\n"
                "Found "
                f"{len(additional_texts['options'])} options in "
                f"{additional_texts_path}\n"
                f"Found {len(flow['options'])} options in {flow_path}"
            )
            raise ValueError(msg)

        # Node numbers match for labels and flow.
        for nodes_option, flow_option, label_option in zip(
            nodes["options"],
            flow["options"],
            labels["options"],
            strict=True,
        ):
            node_numbers = [node["node_number"] for node in nodes_option["nodes"]]
            node_numbers.sort()

            flow_node_numbers = [node["node_number"] for node in flow_option["nodes"]]
            flow_node_numbers.sort()

            label_node_numbers = [node["node_number"] for node in label_option["nodes"]]
            label_node_numbers.sort()

            if not (node_numbers == flow_node_numbers == label_node_numbers):
                msg = (
                    f"The node numbers in nodes, flows, and labels should match\n"
                    f"In {nodes_path}\n"
                    f"Found {node_numbers}\n"
                    f"In {labels_path}\n"
                    f"Found {label_node_numbers}\n"
                    f"In {flow_path}\n"
                    f"Found {flow_node_numbers}\n"
                )
                raise ValueError(msg)


def check_true_nodes_structure(true_nodes_paths: Sequence[Path]) -> None:
    for true_nodes_path in true_nodes_paths:
        true_nodes = load_json(true_nodes_path)

        check_true_diagram_component_generic_structure(
            true_diagram_path=true_nodes_path
        )
        check_true_diagram_generic_nodes_list_structure(
            true_diagram_path=true_nodes_path
        )

        for i, option in enumerate(true_nodes["options"]):
            for node in option["nodes"]:
                # Check each node has node_number and text as its only keys
                if node.keys() != {"node_number", "text"}:
                    msg = (
                        "Each node should only have 'node_number' and 'text' as keys.\n"
                        f"In file: {true_nodes_path}\n"
                        f"In option: {i}\n"
                        f"Found keys: {node.keys()}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise ValueError(msg)

                # Check node text is a string
                if not isinstance(node["text"], str):
                    msg = (
                        "The 'text' for each node should be a string.\n"
                        f"In file: {true_nodes_path}\n"
                        f"In option: {i}\n"
                        f"Found type: {type(node['text'])}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)


def check_true_labels_structure(true_labels_paths: Sequence[Path]) -> None:
    for true_labels_path in true_labels_paths:
        true_labels = load_json(true_labels_path)

        check_true_diagram_component_generic_structure(
            true_diagram_path=true_labels_path
        )
        check_true_diagram_generic_nodes_list_structure(
            true_diagram_path=true_labels_path
        )

        for i, option in enumerate(true_labels["options"]):
            # Check each labels option has nodes as its only key

            for node in option["nodes"]:
                # Check each node has node_number and labels as its only keys
                if node.keys() != {"node_number", "labels"}:
                    msg = (
                        f"Each node should only have 'node_number' and 'labels' as "
                        f"keys.\n"
                        f"In file: {true_labels_path}\n"
                        f"In option: {i}\n"
                        f"Found keys: {node.keys()}\n"
                        f"In node: \n{pretty_json(node)}\n"
                    )
                    raise ValueError(msg)

                # Check labels is a list
                if not isinstance(node["labels"], list):
                    msg = (
                        f"'labels' should be a list.\n"
                        f"In file: {true_labels_path}\n"
                        f"In option: {i}\n"
                        f"Found type: {type(node['labels'])}\n"
                        f"In node: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)

                for label in node["labels"]:
                    # Check each label is a string
                    if not isinstance(label, str):
                        msg = (
                            f"Each label should be a string.\n"
                            f"In file: {true_labels_path}\n"
                            f"In option: {i}\n"
                            f"Found type: {type(label)}\n"
                            f"In node: \n{pretty_json(node)}\n"
                        )

                        raise TypeError(msg)


def check_true_additional_texts_structure(
    true_additional_texts_paths: Sequence[Path],
) -> None:
    for true_additional_texts_path in true_additional_texts_paths:
        true_additional_texts = load_json(true_additional_texts_path)

        check_true_diagram_component_generic_structure(
            true_diagram_path=true_additional_texts_path
        )

        for i, option in enumerate(true_additional_texts["options"]):
            # Check each additional texts option has additional_texts as its only key
            if option.keys() != {"additional_texts"}:
                msg = (
                    "Each 'additional texts' JSON file should have 'additional_texts'"
                    " as its only key.\n"
                    f"In file: {true_additional_texts_path}\n"
                    f"In option: {i}\n"
                    f"Found keys: {option.keys()}\n"
                    f"JSON data: \n{pretty_json(true_additional_texts)}\n"
                )
                raise ValueError(msg)

            # Check additional_texts is a list
            if not isinstance(option["additional_texts"], list):
                msg = (
                    "The 'additional_texts' key should be a list.\n"
                    f"In file: {true_additional_texts_path}\n"
                    f"In option: {i}\n"
                    f"Found type: {type(option['additional_texts'])}\n"
                    f"Option data: \n{pretty_json(option)}\n"
                )
                raise TypeError(msg)

            for text in option["additional_texts"]:
                # Check each additional text is a string
                if not isinstance(text, str):
                    msg = (
                        "Each item in 'additional_texts' should be a string.\n"
                        f"In file: {true_additional_texts_path}\n"
                        f"In option: {i}\n"
                        f"Found type: {type(text)}\n"
                        f"Option data: \n{pretty_json(option)}\n"
                    )
                    raise TypeError(msg)


def check_true_flow_structure(true_flow_paths: Sequence[Path]) -> None:
    for true_flow_path in true_flow_paths:
        true_flow_diagram = load_json(true_flow_path)

        check_true_diagram_component_generic_structure(true_diagram_path=true_flow_path)
        check_true_diagram_generic_nodes_list_structure(
            true_diagram_path=true_flow_path
        )

        for i, option in enumerate(true_flow_diagram["options"]):
            for node in option["nodes"]:
                # Check each node has node_number and points_to as its only keys
                if node.keys() != {"node_number", "points_to"}:
                    msg = (
                        "Each node should only have 'node_number' and 'points_to' "
                        "as keys.\n"
                        f"In file: {true_flow_path}\n"
                        f"In option: {i}\n"
                        f"Found keys: {node.keys()}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise ValueError(msg)

                # Check points_to is a list
                if not isinstance(node["points_to"], list):
                    msg = (
                        "The 'points_to' key should be a list.\n"
                        f"In file: {true_flow_path}\n"
                        f"In option: {i}\n"
                        f"Found type: {type(node['points_to'])}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)

                for node_number in node["points_to"]:
                    # Check each item in points_to is an integer
                    if not isinstance(node_number, int):
                        msg = (
                            "Each item in 'points_to' should be an integer.\n"
                            f"In file: {true_flow_path}\n"
                            f"In option: {i}\n"
                            f"Found type: {type(node_number)}\n"
                            f"Node data: \n{pretty_json(node)}\n"
                        )
                        raise TypeError(msg)

                        # Every node should point to, or be pointed to by another node

                # Check there are no duplicate edges
                if len(node["points_to"]) != len(set(node["points_to"])):
                    duplicate_edges = [
                        num
                        for num in set(node["points_to"])
                        if node["points_to"].count(num) > 1
                    ]
                    msg = (
                        "There should be no duplicate edges in 'points_to'.\n"
                        f"In file: {true_flow_path}\n"
                        f"In option: {i}\n"
                        f"Found duplicates: {duplicate_edges}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise ValueError(msg)

        # Check every node should point to, or be pointed to by another node
        for i, option in enumerate(true_flow_diagram["options"]):
            points_out = set()
            is_pointed_to = set()
            node_numbers = {node["node_number"] for node in option["nodes"]}
            for node in option["nodes"]:
                if len(node["points_to"]) > 0:
                    points_out.add(node["node_number"])
                    is_pointed_to.update(node["points_to"])

            union = points_out.union(is_pointed_to)
            if union != node_numbers:
                msg = (
                    "Every node should point to or be pointed to by another node.\n"
                    f"In file: {true_flow_path}\n"
                    f"In option: {i}\n"
                    "Found nodes that do not point to or are not pointed to by another "
                    f"node: {node_numbers - union}\n"
                    f"Option data: \n{pretty_json(option)}\n"
                )
                raise ValueError(msg)


def check_true_diagram_component_generic_structure(true_diagram_path: Path) -> None:
    true_diagram_component: object = load_json(true_diagram_path)

    # Check that the component is a dict
    if not isinstance(true_diagram_component, dict):
        msg = (
            "The JSON file should be a dictionary.\n"
            f"In file: {true_diagram_path}\n"
            f"Found type: {type(true_diagram_component)}\n"
            f"JSON data: \n{pretty_json(true_diagram_component)}\n"
        )
        raise TypeError(msg)

    # Check that the top-level key is "options"
    if true_diagram_component.keys() != {"options"}:
        msg = (
            "The JSON file should have 'options' as its only top level key.\n"
            f"In file: {true_diagram_path}\n"
            f"Found keys: {true_diagram_component.keys()}\n"
            f"JSON data: \n{pretty_json(true_diagram_component)}\n"
        )
        raise ValueError(msg)

    # Check that there is at least one option
    if len(true_diagram_component["options"]) <= 0:
        msg = (
            "The JSON file should have at least one option.\n"
            f"In file: {true_diagram_path}\n"
            f"Found {len(true_diagram_component['options'])} options.\n"
            f"JSON data: \n{pretty_json(true_diagram_component)}\n"
        )
        raise ValueError(msg)

    # Check that the options are a list:
    if not isinstance(true_diagram_component["options"], list):
        msg = (
            "The 'options' key should be a list.\n"
            f"In file: {true_diagram_path}\n"
            f"Found type: {type(true_diagram_component['options'])}\n"
            f"JSON data: \n{pretty_json(true_diagram_component)}\n"
        )
        raise TypeError(msg)

    # Check that each option is a dict
    for i, option in enumerate(true_diagram_component["options"]):
        if not isinstance(option, dict):
            msg = (
                "Each option should be a dictionary.\n"
                f"In file: {true_diagram_path}\n"
                f"In option: {i}\n"
                f"Found type: {type(option)}\n"
                f"Option data: \n{pretty_json(option)}\n"
            )
            raise TypeError(msg)


def check_true_diagram_generic_nodes_list_structure(true_diagram_path: Path) -> None:
    true_diagram_component = load_json(true_diagram_path)

    for i, option in enumerate(true_diagram_component["options"]):
        # Check each nodes option has nodes as its only key
        if option.keys() != {"nodes"}:
            msg = (
                "Each option should have 'nodes' as its only key.\n"
                f"In file: {true_diagram_path}\n"
                f"In option: {i}\n"
                f"Found keys: {option.keys()}\n"
                f"Option data: \n{pretty_json(option)}\n"
            )
            raise ValueError(msg)

        for node in option["nodes"]:
            # Check each node has node_number
            if "node_number" not in node:
                msg = (
                    "Each node should have a 'node_number' key.\n"
                    f"In file: {true_diagram_path}\n"
                    f"In option: {i}\n"
                    f"Found keys: {node.keys()}\n"
                    f"Node data: \n{pretty_json(node)}\n"
                )
                raise ValueError(msg)

            # Check node_number is an integer
            if not isinstance(node["node_number"], int):
                msg = (
                    "The 'node_number' should be an integer.\n"
                    f"In file: {true_diagram_path}\n"
                    f"In option: {i}\n"
                    f"Found type: {type(node['node_number'])}\n"
                    f"Node data: \n{pretty_json(node)}\n"
                )
                raise TypeError(msg)

        node_numbers = [node["node_number"] for node in option["nodes"]]

        # Check node numbers are unique
        if len(node_numbers) != len(set(node_numbers)):
            duplicate_node_numbers = [
                num for num in set(node_numbers) if node_numbers.count(num) > 1
            ]
            msg = (
                "The 'node_number' values should be unique.\n"
                f"In file: {true_diagram_path}\n"
                f"In option: {i}\n"
                f"Found duplicates: {duplicate_node_numbers}\n"
                f"Option data: \n{pretty_json(option)}\n"
            )
            raise ValueError(msg)

        # Check node numbers are sequential starting from 1
        if set(node_numbers) != set(range(1, len(node_numbers) + 1)):
            msg = (
                "The 'node_number' values should be sequential starting from 1.\n"
                f"In file: {true_diagram_path}\n"
                f"In option: {i}\n"
                f"Found node numbers: {node_numbers}\n"
                f"Option data: \n{pretty_json(option)}\n"
            )
            raise ValueError(msg)

        # Check there is at least one node
        if len(node_numbers) <= 0:
            msg = (
                "There should be at least one node.\n"
                f"In file: {true_diagram_path}\n"
                f"In option: {i}\n"
                f"Found {len(node_numbers)} nodes.\n"
                f"Option data: \n{pretty_json(option)}\n"
            )
            raise ValueError(msg)


def check_pred_structure(
    pred_diagrams_dir: Path,
) -> PredDiagramStructure:
    """Validate one prediction directory and infer its common structure."""
    pred_paths = sorted(pred_diagrams_dir.glob("*.json"))
    if not pred_paths:
        msg = f"No predicted diagram JSON files found in {pred_diagrams_dir}."
        raise ValueError(msg)

    preds = []
    for pred_path in pred_paths:
        pred: object = load_json(pred_path)
        if not isinstance(pred, dict):
            msg = (
                "Each predicted diagram JSON file should be a dictionary.\n"
                f"In file: {pred_path}\n"
                f"Found type: {type(pred)}\n"
                f"JSON data: \n{pretty_json(pred)}\n"
            )
            raise TypeError(msg)
        preds.append((pred_path, pred))

    first_path, first_pred = preds[0]
    allowed_top_level_keys = {"nodes", "additional_texts"}
    if not first_pred.keys() or not first_pred.keys() <= allowed_top_level_keys:
        msg = (
            "The JSON file has invalid top-level keys.\n"
            f"In file: {first_path}\n"
            "Expected one or both of: {'nodes', 'additional_texts'}\n"
            f"Found keys: {first_pred.keys()}\n"
            f"JSON data: \n{pretty_json(first_pred)}\n"
        )
        raise ValueError(msg)

    expected_top_level_keys = set(first_pred)
    for pred_path, pred in preds[1:]:
        if pred.keys() != expected_top_level_keys:
            msg = (
                "All predicted diagram files should have the same top-level keys.\n"
                f"Expected keys: {expected_top_level_keys}\n"
                f"In file: {pred_path}\n"
                f"Found keys: {pred.keys()}\n"
                f"JSON data: \n{pretty_json(pred)}\n"
            )
            raise ValueError(msg)

    inferred_node_keys: set[str] = set()
    if "nodes" in expected_top_level_keys:
        allowed_node_keys = {"node_number", "text", "labels", "points_to"}
        first_nodes = first_pred["nodes"]
        if not isinstance(first_nodes, list):
            msg = (
                "The 'nodes' key should be a list.\n"
                f"In file: {first_path}\n"
                f"Found type: {type(first_nodes)}\n"
                f"JSON data: \n{pretty_json(first_pred)}\n"
            )
            raise TypeError(msg)

        if not first_nodes:
            msg = (
                "The 'nodes' list should contain at least one node.\n"
                f"In file: {first_path}\n"
                f"JSON data: \n{pretty_json(first_pred)}\n"
            )
            raise ValueError(msg)

        first_node = first_nodes[0]
        if not isinstance(first_node, dict):
            msg = (
                "Each item in 'nodes' should be a dictionary.\n"
                f"In file: {first_path}\n"
                f"Found type: {type(first_node)}\n"
                f"Node data: \n{pretty_json(first_node)}\n"
            )
            raise TypeError(msg)

        inferred_node_keys = set(first_node)
        if (
            "node_number" not in inferred_node_keys
            or not inferred_node_keys <= allowed_node_keys
        ):
            msg = (
                "Node has invalid keys.\n"
                f"In file: {first_path}\n"
                "Expected 'node_number' and one or more of 'text', 'labels', "
                "and 'points_to'.\n"
                f"Found keys: {first_node.keys()}\n"
                f"Node data: \n{pretty_json(first_node)}\n"
            )
            raise ValueError(msg)

        if inferred_node_keys == {"node_number"}:
            msg = (
                "Predicted nodes should contain at least one benchmarkable "
                "component: 'text', 'labels', or 'points_to'."
            )
            raise ValueError(msg)

    expected_structure = PredDiagramStructure(
        node_text="text" in inferred_node_keys,
        labels="labels" in inferred_node_keys,
        flow="points_to" in inferred_node_keys,
        additional_texts="additional_texts" in expected_top_level_keys,
    )

    expected_pred_keys = expected_structure.top_level_keys
    expected_node_keys = expected_structure.node_keys

    if not expected_structure.parts:
        msg = "The expected predicted structure has no parsing components."
        raise ValueError(msg)

    for pred_path, pred in preds:
        # Check top-level keys
        if pred.keys() != expected_pred_keys:
            msg = (
                "The JSON file has invalid top-level keys.\n"
                f"In file: {pred_path}\n"
                f"Expected keys: {expected_pred_keys}\n"
                f"Found keys: {pred.keys()}\n"
                f"JSON data: \n{pretty_json(pred)}\n"
            )
            raise ValueError(msg)

        # Check nodes structure
        if expected_structure.has_nodes:
            if not isinstance(pred["nodes"], list):
                msg = (
                    "The 'nodes' key should be a list.\n"
                    f"In file: {pred_path}\n"
                    f"Found type: {type(pred['nodes'])}\n"
                    f"JSON data: \n{pretty_json(pred)}\n"
                )
                raise TypeError(msg)

            if not pred["nodes"]:
                msg = (
                    "The 'nodes' list should contain at least one node.\n"
                    f"In file: {pred_path}\n"
                    f"JSON data: \n{pretty_json(pred)}\n"
                )
                raise ValueError(msg)

            for node in pred["nodes"]:
                if not isinstance(node, dict):
                    msg = (
                        "Each item in 'nodes' should be a dictionary.\n"
                        f"In file: {pred_path}\n"
                        f"Found type: {type(node)}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)

                if node.keys() != expected_node_keys:
                    msg = (
                        "Node has invalid keys.\n"
                        f"In file: {pred_path}\n"
                        f"Expected keys: {expected_node_keys}\n"
                        f"Found keys: {node.keys()}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise ValueError(msg)

                if not isinstance(node["node_number"], int):
                    msg = (
                        "The 'node_number' should be an integer.\n"
                        f"In file: {pred_path}\n"
                        f"Found type: {type(node['node_number'])}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)

                if expected_structure.node_text and not isinstance(node["text"], str):
                    msg = (
                        "The 'text' should be a string.\n"
                        f"In file: {pred_path}\n"
                        f"Found type: {type(node['text'])}\n"
                        f"Node data: \n{pretty_json(node)}\n"
                    )
                    raise TypeError(msg)

                if expected_structure.labels:
                    if not isinstance(node["labels"], list):
                        msg = (
                            "The 'labels' key should be a list.\n"
                            f"In file: {pred_path}\n"
                            f"Found type: {type(node['labels'])}\n"
                            f"Node data: \n{pretty_json(node)}\n"
                        )
                        raise TypeError(msg)

                    for label in node["labels"]:
                        if not isinstance(label, str):
                            msg = (
                                "Each label should be a string.\n"
                                f"In file: {pred_path}\n"
                                f"Found type: {type(label)}\n"
                                f"Node data: \n{pretty_json(node)}\n"
                            )
                            raise TypeError(msg)

                if expected_structure.flow:
                    if not isinstance(node["points_to"], list):
                        msg = (
                            "The 'points_to' key should be a list.\n"
                            f"In file: {pred_path}\n"
                            f"Found type: {type(node['points_to'])}\n"
                            f"Node data: \n{pretty_json(node)}\n"
                        )
                        raise TypeError(msg)

                    for node_number in node["points_to"]:
                        if not isinstance(node_number, int):
                            msg = (
                                "Each item in 'points_to' should be an integer.\n"
                                f"In file: {pred_path}\n"
                                f"Found type: {type(node_number)}\n"
                                f"Node data: \n{pretty_json(node)}\n"
                            )
                            raise TypeError(msg)

            # Check node numbers are unique
            node_numbers = [node["node_number"] for node in pred["nodes"]]
            if len(node_numbers) != len(set(node_numbers)):
                duplicate_node_numbers = [
                    num for num in set(node_numbers) if node_numbers.count(num) > 1
                ]
                msg = (
                    "The 'node_number' values should be unique.\n"
                    f"In file: {pred_path}\n"
                    f"Found duplicates: {duplicate_node_numbers}\n"
                    f"JSON data: \n{pretty_json(pred)}\n"
                )
                raise ValueError(msg)

            # TODO: should we add back the check that pred nodes should be sequential?
            # Find in repo history
            # Check node numbers sequential starting from 1

        if expected_structure.additional_texts:
            # Check additional_text structure
            if not isinstance(pred["additional_texts"], list):
                msg = (
                    "The 'additional_texts' key should be a list.\n"
                    f"In file: {pred_path}\n"
                    f"Found type: {type(pred['additional_texts'])}\n"
                    f"JSON data: \n{pretty_json(pred)}\n"
                )
                raise TypeError(msg)

            for text in pred["additional_texts"]:
                if not isinstance(text, str):
                    msg = (
                        "Each item in 'additional_texts' should be a string.\n"
                        f"In file: {pred_path}\n"
                        f"Found type: {type(text)}\n"
                        f"JSON data: \n{pretty_json(pred)}\n"
                    )
                    raise TypeError(msg)

        if expected_structure.flow:
            for node in pred["nodes"]:
                for node_number in node["points_to"]:
                    if node_number not in node_numbers:
                        msg = (
                            "For pred diagrams, each item in 'points_to' should be "
                            "a valid node_number.\n"
                            f"In file: {pred_path}\n"
                            f"Found invalid node_number: {node_number}\n"
                            f"Node data: \n{pretty_json(node)}\n"
                            f"JSON data: \n{pretty_json(pred)}\n"
                        )
                        raise ValueError(msg)

    return expected_structure


def validate_pred_diagram_sources(
    pred_diagrams_dirs: Path | Sequence[Path],
) -> PredDiagramSources:
    """
    Validate prediction directories and collect their source information.

    Parameters
    ----------
    pred_diagrams_dirs
        One prediction directory or a sequence of component directories.

    Returns
    -------
    PredDiagramSources
        The validated source paths and their combined parsing structure.

    """
    directories: tuple[Path, ...]
    if isinstance(pred_diagrams_dirs, Path):
        directories = (pred_diagrams_dirs,)
    else:
        directories = tuple(pred_diagrams_dirs)

    if not directories:
        msg = "At least one predicted diagram directory should be provided."
        raise ValueError(msg)

    if len(set(directories)) != len(directories):
        msg = (
            "Each predicted diagram directory should be provided only once.\n"
            f"Found directories: {directories}"
        )
        raise ValueError(msg)

    sources = tuple(
        PredDiagramSource(
            diagrams_dir=directory,
            paths=tuple(sorted(directory.glob("*.json"))),
            structure=check_pred_structure(pred_diagrams_dir=directory),
        )
        for directory in directories
    )

    part_sources: dict[ParsingPart, Path] = {}
    for source in sources:
        for part in source.structure.parts:
            if part in part_sources:
                msg = (
                    f"The predicted component {part!r} was supplied more than once.\n"
                    f"First directory: {part_sources[part]}\n"
                    f"Duplicate directory: {source.diagrams_dir}"
                )
                raise ValueError(msg)
            part_sources[part] = source.diagrams_dir

    structure = PredDiagramStructure(
        node_text="node_text" in part_sources,
        labels="labels" in part_sources,
        flow="flow" in part_sources,
        additional_texts="additional_texts" in part_sources,
    )

    if not structure.node_text:
        msg = (
            "Predicted node text is required to run a parsing benchmark. "
            "Labels, flow, and additional text cannot select and match ground-truth "
            "nodes on their own."
        )
        raise ValueError(msg)

    node_text_source = next(source for source in sources if source.structure.node_text)
    expected_stems = tuple(path.stem for path in node_text_source.paths)
    expected_stem_set = set(expected_stems)

    for source in sources:
        stems = tuple(path.stem for path in source.paths)
        if stems != expected_stems:
            stem_set = set(stems)
            msg = (
                "All predicted component directories should contain files for the "
                "same diagrams.\n"
                f"Node-text directory: {node_text_source.diagrams_dir}\n"
                f"Compared directory: {source.diagrams_dir}\n"
                f"Missing stems: {sorted(expected_stem_set - stem_set)}\n"
                f"Extra stems: {sorted(stem_set - expected_stem_set)}"
            )
            raise ValueError(msg)

    source_paths_by_stem = [
        {path.stem: path for path in source.paths} for source in sources
    ]
    node_text_source_idx = sources.index(node_text_source)

    for node_text_path in node_text_source.paths:
        img_code = node_text_path.stem
        node_text_data = load_json(source_paths_by_stem[node_text_source_idx][img_code])
        expected_node_numbers = {
            node["node_number"] for node in node_text_data["nodes"]
        }

        for source_idx, source in enumerate(sources):
            if source is node_text_source or not source.structure.has_nodes:
                continue

            source_path = source_paths_by_stem[source_idx][img_code]
            source_data = load_json(source_path)
            found_node_numbers = {node["node_number"] for node in source_data["nodes"]}
            if found_node_numbers != expected_node_numbers:
                msg = (
                    "All node-based prediction sources should contain the same node "
                    "numbers for each diagram.\n"
                    f"Diagram: {img_code}\n"
                    f"Node-text file: {node_text_path}\n"
                    f"Expected node numbers: {sorted(expected_node_numbers)}\n"
                    f"Compared file: {source_path}\n"
                    f"Found node numbers: {sorted(found_node_numbers)}"
                )
                raise ValueError(msg)

    return PredDiagramSources(sources=sources, structure=structure)


def count_functions_in_this_module() -> int:
    module = sys.modules[__name__]
    return sum(
        1
        for _, obj in inspect.getmembers(module, inspect.isfunction)
        if obj.__module__ == __name__
    )


def run_all_benchmark_true_data_checks(
    pred_sources: PredDiagramSources,
    true_nodes_dir: Path,
    true_labels_dir: Path,
    true_additional_texts_dir: Path,
    true_flow_dir: Path,
    allow_missing_pred_diagrams: bool,
    expected_num_diagrams: int = N_DIAGRAMS_IN_BENCHMARK,
) -> None:
    # validate_pred_diagram_sources has already proved that every prediction
    # source contains the same diagram stems. The node-text paths are therefore
    # the canonical prediction set for comparison with the ground-truth files.
    pred_paths = list(pred_sources.node_text_source.paths)
    true_nodes_paths = sorted(true_nodes_dir.glob("*.json"))
    true_labels_paths = sorted(true_labels_dir.glob("*.json"))
    true_additional_texts_paths = sorted(true_additional_texts_dir.glob("*.json"))
    true_flow_paths = sorted(true_flow_dir.glob("*.json"))

    check_benchmark_files_match(
        pred_paths=pred_paths,
        nodes_paths=true_nodes_paths,
        labels_paths=true_labels_paths,
        additional_texts_paths=true_additional_texts_paths,
        flow_paths=true_flow_paths,
        allow_missing_pred_diagrams=allow_missing_pred_diagrams,
        expected_num_diagrams=expected_num_diagrams,
    )
    check_benchmark_options_match(
        nodes_paths=true_nodes_paths,
        labels_paths=true_labels_paths,
        additional_texts_paths=true_additional_texts_paths,
        flow_paths=true_flow_paths,
    )
    check_true_nodes_structure(
        true_nodes_paths=true_nodes_paths,
    )
    check_true_labels_structure(
        true_labels_paths=true_labels_paths,
    )
    check_true_additional_texts_structure(
        true_additional_texts_paths=true_additional_texts_paths,
    )
    check_true_flow_structure(
        true_flow_paths=true_flow_paths,
    )

    checks_run = 6
    # check_true_diagram_component_generic_structure,
    # check_true_diagram_generic_nodes_list_structure,
    # count_functions_in_this_module, run_all_benchmark_checks, _path_id,
    # check_pred_structure, validate_pred_diagram_sources
    non_base_check_funcs_in_module = 6

    if checks_run + non_base_check_funcs_in_module != count_functions_in_this_module():
        expected_checks = (
            count_functions_in_this_module() - non_base_check_funcs_in_module
        )
        msg = (
            "Not all benchmark check functions are being run. Expected "
            f"{expected_checks} checks to be run, but only {checks_run} "
            "are being run."
        )
        raise ValueError(msg)
