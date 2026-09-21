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
    max_concurrent_jobs: int = 10,
    *,
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[BaseModel]:
    """
    Parse explicit image paths into same-stem JSON files inside `save_dir`.

    Input filename stems must be unique ignoring case across the run, including
    inputs from earlier resumed calls. Each image has a separate saved input
    record in `.flowde/input_records`; keep the whole `.flowde` directory with
    the outputs for resume and overwrite.

    Image functions run in threads in a separate worker process, even with
    `max_concurrent_jobs=1`. See [`parse_imgs()`][flowde.parse_imgs.parse_imgs]
    for concurrency, custom-function requirements and saved-run compatibility.
    """
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
                "outputs": [f"{path.stem}.json"],
                "input": {
                    "file_digest": file_digest(path),
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
        n_jobs=max_concurrent_jobs,
        threaded=True,
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
    max_concurrent_jobs: int = 10,
    img_extensions: set[str] | None = None,
    *,
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[BaseModel]:
    """
    Parse images in a directory into JSON files, optionally using saved context.

    Parameters
    ----------
    parse_fn : ParsingFunction
        Function accepting an `img_path` keyword argument and, when context is
        supplied, a `partial_flowchart` keyword argument containing a Pydantic
        model. `img_path` is a `Path` to the image being parsed; the partial
        model contains previously parsed parts for that same image. Without
        saved context, Flowde passes only `img_path`.

        The function must return an instance of the Pydantic class exposed
        as `parse_fn.result_structure`, not a dictionary, JSON string or `None`.
        The class can describe standard flowchart parts or a custom output
        format. The parser must raise an exception if parsing fails. Flowde
        saves each returned model as a same-stem JSON file in `save_dir`.

        Built-in factories declare their settings;
        custom functions must declare their settings and result class with
        [`model_function()`][flowde.model_function].
    img_dir : Path
        Directory containing input images. Subdirectories are not searched.
        Matching paths are sorted before applying `range_indices`. Filename stems
        must be unique across all matching images, including different extensions,
        because each stem determines an output JSON filename.
        Selected stems must also remain unique ignoring case across the run,
        including images from earlier resumed calls.
    save_dir : Path
        Dedicated output directory, created if needed. Each image produces a JSON
        file with the same stem, such as `diagram.png` producing `diagram.json`.
        Run metadata is saved in `.flowde`. Input images, partial JSONs
        and files referenced by the parser's declared settings must be outside
        this directory.
    nodes_dir : Path | None, optional
        Directory of node-text JSONs supplied as context alongside the images.
        Each JSON contains `nodes` with `node_number` and `text` fields.
        Defaults to `None`, meaning no saved context is supplied. Required if
        `labels_dir`, `additional_texts_dir` or `flow_dir` is provided.
    labels_dir : Path | None, optional
        Directory of label JSONs to add to the context from `nodes_dir`.
        Each JSON contains `nodes` with `node_number` and `labels` fields.
        Defaults to `None`, meaning no saved labels are included in the context.
    additional_texts_dir : Path | None, optional
        Directory of additional-text JSONs to add to the context from `nodes_dir`.
        Each JSON contains an `additional_texts` list. Defaults to `None`, meaning
        no saved additional text is included in the context.
    flow_dir : Path | None, optional
        Directory of connection JSONs to add to the context from `nodes_dir`.
        Each JSON contains `nodes` with `node_number` and `points_to` fields.
        Defaults to `None`, meaning no saved connections are included in the
        context.
    range_indices : tuple[int | None, int | None] | None, optional
        A `(start, stop)` slice of the sorted image paths. `start` is included
        and `stop` is excluded; `(0, 10)` selects up to the first ten images.
        Either bound can be `None`, and negative indices follow Python slicing
        rules. Defaults to `None`, which selects all matching images. The same
        slice is applied to supplied context files after checking their filenames
        against all matching images. An empty selection raises an error.
    max_concurrent_jobs : int, optional
        Maximum number of images handled concurrently by threads in a separate
        worker process, by default 10. Must be a positive integer; 1 handles one
        image at a time.
        Custom functions must support concurrent calls when this exceeds 1.
        Even at 1, worker mutations do not update objects in the caller; see
        [custom functions](../pipeline/custom-functions.md#concurrent-image-calls).
        This setting can change when resuming a run.
    img_extensions : set[str] | None, optional
        Image extensions to select, without leading dots, such as
        `{"png", "jpg", "webp"}`. Defaults to `None`, which selects PNG files.
        The supplied parser must support the selected image formats.
    show_usage : bool, optional
        Whether to display token usage and estimated costs reported by the
        parser. Defaults to `True`. `False` hides those figures while retaining
        the progress display and saved usage reports. This value can change
        when resuming a run.
    on_existing : {"error", "resume", "overwrite"}, optional
        How to handle an existing run in `save_dir`. Defaults to `"error"`.

        - `"error"`: start in a new or empty directory; reject existing work.
        - `"resume"`: require a saved parsing run with matching parser settings.
          Restore saved results for unchanged images and assembled partial
          context, and parse selected images without saved results.
        - `"overwrite"`: remove the previous run's tracked outputs and saved state,
          then start a new run. Also works in a new or empty directory.
          Unrelated files in an existing output directory cause an error.

        Resume and overwrite require all internal records. Missing or invalid
        records raise an error.

    Returns
    -------
    list[BaseModel]
        One Pydantic result per image selected by this call, in sorted image-path
        order. Each result uses `parse_fn.result_structure`. Includes results
        restored from a previous run. With `range_indices`, the returned list
        covers only the selected slice; JSON files for earlier slices remain
        saved in `save_dir`.

    Raises
    ------
    FileExistsError
        If `save_dir` contains existing work and `on_existing="error"`.
    ValueError
        If `img_dir` is invalid, no images are selected, image stems conflict
        (including case-only differences across the run), context files do not
        match the images or required part schemas,
        node numbers cannot be joined, inputs are inside `save_dir`, or the saved
        run fails compatibility or integrity checks.
    TypeError
        If `parse_fn.result_structure` is not a Pydantic class or the parser
        returns a value that is not an instance of that class, including `None`.
    RuntimeError
        If another Flowde call is already using the same `save_dir`.

    Notes
    -----
    Each supplied context directory must contain exactly one top-level `*.json`
    file per matching input image, with the same stem and no extra JSON files.
    Filename and file-count checks cover all matching images before slicing,
    including images outside `range_indices`.

    Context files contain the fields for their individual parts, without the
    benchmark ground truth's `options` wrapper. Selected node-text files must
    contain at least one node, numbered consecutively from `1`. Selected label
    and flow files must contain the same node numbers as the node-text file.
    The parts are joined into the `partial_flowchart` passed to the parser.

    Context files supply input to the parser; they are not automatically merged
    into its output. The output fields are defined by `parse_fn.result_structure`.
    For built-in parsers, the factory's `parts_to_parse` or `result_structure`
    argument chooses those fields.

    Flowde saves settings in `.flowde/run_metadata.state` and each image's result,
    usage and progress in `.flowde/input_records/<stem>.state`. Progress updates
    rewrite only that image's record, without rewriting other images' results
    or the run metadata. Keep the whole `.flowde` directory with the outputs.

    Flowde records each result before writing its output JSON. On resume, missing
    output JSONs are recreated from these records without another parsing request.
    Edited saved outputs or changes to previously parsed input images or their
    assembled partial context cause an error. Missing or invalid internal records
    cause an error even when the output JSONs still exist.

    """
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
        max_concurrent_jobs=max_concurrent_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
