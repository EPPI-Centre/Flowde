from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from flowde._batch import run_batch
from flowde._run_settings import file_digest, function_settings
from flowde._run_state import ExistingRun, json_bytes, protect_inputs
from flowde.parsing_fns.parsing_types import ParsingFunction, build_partial_flowcharts
from flowde.utils import (
    validate_matching_stems,
    validate_same_path_lengths,
    validate_unique_stems,
)


def parse_imgs_from_paths(
    parse_fn: ParsingFunction,
    img_paths: list[Path],
    save_dir: Path,
    nodes_paths: list[Path] | None = None,
    labels_paths: list[Path] | None = None,
    additional_texts_paths: list[Path] | None = None,
    flow_paths: list[Path] | None = None,
    n_jobs: int = cpu_count(),
    *,
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[BaseModel]:
    """Parse explicit image paths into same-stem JSON files inside `save_dir`."""
    validate_unique_stems(img_paths)

    validate_same_path_lengths(
        img_paths,
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
    )

    validate_matching_stems(
        img_paths,
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
    )

    items_for_model_by_param: dict[str, list[Any]] = {
        "img_path": img_paths,
    }

    if nodes_paths is None and any(
        paths is not None
        for paths in [
            labels_paths,
            additional_texts_paths,
            flow_paths,
        ]
    ):
        msg = (
            "nodes_paths must be provided if any of labels_paths, "
            "additional_texts_paths, or flow_paths are provided, since node "
            "numbers are needed to join the different parts together."
        )
        raise ValueError(msg)

    if nodes_paths is not None:
        str_parts_included = []
        str_parts_included.append("nodes")

        if labels_paths is not None:
            str_parts_included.append("labels")
        if additional_texts_paths is not None:
            str_parts_included.append("additional_texts")
        if flow_paths is not None:
            str_parts_included.append("flow")
        print(f"Included partial flowchart parts: {', '.join(str_parts_included)}")

        items_for_model_by_param["partial_flowchart"] = build_partial_flowcharts(
            nodes_paths=nodes_paths,
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
        )
    else:
        print("No partial flowchart data provided, parsing with just images...")

    if not img_paths:
        msg = "img_paths cannot be empty."
        raise ValueError(msg)
    protect_inputs(
        save_dir,
        [
            path
            for paths in (
                img_paths,
                nodes_paths,
                labels_paths,
                flow_paths,
                additional_texts_paths,
            )
            if paths
            for path in paths
        ],
    )
    structure = getattr(parse_fn, "result_structure", None)
    if not isinstance(structure, type) or not issubclass(structure, BaseModel):
        msg = "Parsers must expose a Pydantic class as parse_fn.result_structure."
        raise TypeError(msg)
    settings = function_settings(parse_fn)
    inputs = []
    for index, path in enumerate(img_paths):
        kwargs = {
            name: values[index] for name, values in items_for_model_by_param.items()
        }
        partial = kwargs.get("partial_flowchart")
        inputs.append(
            {
                "key": str(path.resolve()),
                "path": str(path),
                "output": f"{path.stem}.json",
                "input": {
                    "image": file_digest(path),
                    "partial_flowchart": partial.model_dump(mode="json")
                    if partial is not None
                    else None,
                },
                "kwargs": kwargs,
            }
        )

    def encode(result: Any) -> dict[str, Any]:
        if not isinstance(result, BaseModel):
            msg = "A parser must return a Pydantic result; None is a failure."
            raise TypeError(msg)
        if not isinstance(result, structure):
            msg = "Parser returned a different class from parse_fn.result_structure."
            raise TypeError(msg)
        return result.model_dump(mode="json", round_trip=True)

    responses = run_batch(
        parse_fn,
        inputs,
        save_dir,
        kind="parsing",
        settings=settings,
        encode=encode,
        decode=lambda value: structure.model_validate_json(json_bytes(value)),
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
    print("PARSING COMPLETE")
    return responses


def parse_imgs(
    parse_fn: ParsingFunction,
    img_dir: Path,
    save_dir: Path,
    nodes_dir: Path | None = None,
    labels_dir: Path | None = None,
    additional_texts_dir: Path | None = None,
    flow_dir: Path | None = None,
    range_indices: tuple[int | None, int | None] | None = None,
    n_jobs: int = cpu_count(),
    img_extensions: set[str] | None = None,
    *,
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[BaseModel]:
    if not img_dir.is_dir():
        msg = f"Image directory {img_dir} does not exist or is not a directory."
        raise ValueError(msg)

    effective_img_extensions = {"png"} if img_extensions is None else img_extensions

    img_paths: list[Path] = []
    for ext in effective_img_extensions:
        img_paths.extend(img_dir.glob(f"*.{ext}"))

    img_paths.sort()

    if len(img_paths) == 0:
        extensions = ", ".join(sorted(effective_img_extensions))
        msg = (
            f"No image files matching extensions [{extensions}] were found "
            f"in directory {img_dir}."
        )
        raise ValueError(msg)

    # Check before slicing so separate runs cannot overwrite the same JSON file.
    validate_unique_stems(img_paths)

    nodes_paths = None
    if nodes_dir is not None:
        nodes_paths = list(nodes_dir.glob("*.json"))
        nodes_paths.sort()

    labels_paths = None
    if labels_dir is not None:
        labels_paths = list(labels_dir.glob("*.json"))
        labels_paths.sort()

    additional_texts_paths = None
    if additional_texts_dir is not None:
        additional_texts_paths = list(additional_texts_dir.glob("*.json"))
        additional_texts_paths.sort()

    flow_paths = None
    if flow_dir is not None:
        flow_paths = list(flow_dir.glob("*.json"))
        flow_paths.sort()

    validate_same_path_lengths(
        img_paths,
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
    )

    validate_matching_stems(
        img_paths,
        nodes_paths,
        labels_paths,
        additional_texts_paths,
        flow_paths,
    )

    if range_indices is None:
        range_indices = (0, len(img_paths))

    img_paths = img_paths[range_indices[0] : range_indices[1]]
    if nodes_paths is not None:
        nodes_paths = nodes_paths[range_indices[0] : range_indices[1]]
    if labels_paths is not None:
        labels_paths = labels_paths[range_indices[0] : range_indices[1]]
    if additional_texts_paths is not None:
        additional_texts_paths = additional_texts_paths[
            range_indices[0] : range_indices[1]
        ]
    if flow_paths is not None:
        flow_paths = flow_paths[range_indices[0] : range_indices[1]]

    return parse_imgs_from_paths(
        parse_fn=parse_fn,
        img_paths=img_paths,
        save_dir=save_dir,
        nodes_paths=nodes_paths,
        labels_paths=labels_paths,
        additional_texts_paths=additional_texts_paths,
        flow_paths=flow_paths,
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
