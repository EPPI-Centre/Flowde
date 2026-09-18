"""Combine directories of saved parsing results."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from flowde._run_state import atomic_write, json_bytes, protect_inputs
from flowde.parsing_fns.parsing_types import build_partial_flowcharts
from flowde.utils import validate_matching_stems


def _json_files(directory: Path | None) -> list[Path]:
    if directory is None:
        msg = "An input directory is required."
        raise ValueError(msg)
    if not directory.is_dir():
        msg = f"Input directory does not exist or is not a directory: {directory}."
        raise NotADirectoryError(msg)
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def _recorded_outputs(save_dir: Path) -> set[str]:
    """Validate ownership and every existing entry before permitting writes."""
    metadata = save_dir / ".flowde"
    record_path = metadata / "combined.state"
    entries = list(save_dir.iterdir()) if save_dir.exists() else []
    if not entries:
        return set()
    if metadata.is_symlink() or (metadata.exists() and not metadata.is_dir()):
        msg = f"Combining metadata must be an ordinary directory: {metadata}."
        raise ValueError(msg)
    # An initial inventory write can fail before leaving a record or any output.
    if entries == [metadata] and not any(metadata.iterdir()):
        return set()
    msg = (
        f"Cannot overwrite {save_dir}: a valid Flowde .flowde/combined.state "
        "is required. Use a new output directory."
    )
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError(msg)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(msg) from error
    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record["version"] != 1
        or record.get("kind") != "combined_parsed_parts"
        or not isinstance(record.get("outputs"), list)
    ):
        raise ValueError(msg)
    names = record["outputs"]
    if any(
        not isinstance(name, str)
        or not name.endswith(".json")
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        for name in names
    ) or len(set(names)) != len(names):
        raise ValueError(msg)
    outputs = set(names)
    for path in entries + list(metadata.iterdir()):
        if path == metadata:
            continue
        if (
            not path.is_symlink()
            and path.is_file()
            and (
                path == record_path
                or (path.parent == save_dir and path.name in outputs)
            )
        ):
            continue
        msg = (
            f"Cannot overwrite {save_dir}: unexpected file or directory "
            f"{path.relative_to(save_dir)}. No files have been changed. "
            "Move it elsewhere or use a new output directory."
        )
        raise ValueError(msg)
    return outputs


def _save_outputs(record_path: Path, names: set[str]) -> None:
    atomic_write(
        record_path,
        json_bytes(
            {"version": 1, "kind": "combined_parsed_parts", "outputs": sorted(names)}
        ),
    )


def combine_parsed_parts(
    nodes_dir: Path,
    save_dir: Path,
    *,
    labels_dir: Path | None = None,
    flow_dir: Path | None = None,
    additional_texts_dir: Path | None = None,
    on_existing: Literal["error", "overwrite"] = "error",
) -> list[BaseModel]:
    """
    Combine saved parsing parts into one JSON file per image.

    Match top-level JSON files by filename stem, join their nodes by node
    number, and save the combined results in `save_dir`. Node text is required;
    labels, flow and additional text are optional. All inputs are validated
    before any output file is written or removed.

    Parameters
    ----------
    nodes_dir : Path
        Directory containing node-text JSON files, each with a `nodes` list
        of `node_number` and `text` values. At least one top-level `*.json`
        file is required. Files are processed in sorted path order.
    save_dir : Path
        Directory for the combined JSON files. Created, with missing parent
        directories, after validation succeeds. Output filenames match the
        node-text filenames. The directory must not be a symbolic link, be
        an input directory, or contain any supplied input directory or file.
    labels_dir : Path | None, optional
        Directory containing label JSON files with `node_number` and `labels`
        for every node. Defaults to `None`, which omits labels from the results.
    flow_dir : Path | None, optional
        Directory containing flow JSON files with `node_number` and
        `points_to` for every node. Defaults to `None`, which omits outgoing
        connections from the results.
    additional_texts_dir : Path | None, optional
        Directory containing JSON files with an `additional_texts` list.
        Defaults to `None`, which omits additional text from the results.
    on_existing : {"error", "overwrite"}, optional
        How to handle an existing output directory. The default, `"error"`,
        raises an error if `save_dir` contains any entries. `"overwrite"`
        replaces recorded combined JSON files and, after all new results are
        saved, removes recorded outputs absent from the current inputs.
        An occupied directory requires a valid `.flowde/combined.state` record.
        Unrecorded files or directories cause an error before any files are
        changed, even if an unrecorded filename matches a new result. Existing
        output JSON files and metadata must not be symbolic links. An empty
        output directory is allowed with either option. Resuming is not supported.

    Returns
    -------
    list[BaseModel]
        Combined Pydantic models in sorted node-text path order. Each model
        contains node numbers and text, plus the parts whose directories were
        supplied. Nodes are sorted by `node_number`. The saved JSON files
        contain the same data as the returned models.

    Raises
    ------
    ValueError
        If node text is omitted, `nodes_dir` contains no JSON files,
        `on_existing` is unsupported, output paths violate the protections
        above, the output record is missing or invalid in an occupied directory,
        unrecorded output-directory entries are present, input filenames do not
        match, or node numbers are invalid or inconsistent across parts.
    NotADirectoryError
        If a supplied input directory does not exist or is not a directory,
        or an existing `save_dir` is not a directory.
    FileExistsError
        If `save_dir` is occupied and `on_existing="error"`.
    IsADirectoryError
        If a subdirectory in `save_dir` has a required output filename.
    pydantic.ValidationError
        If an input JSON is malformed or does not match its part's schema.
    UnicodeDecodeError
        If an input file cannot be decoded as UTF-8.
    OSError
        If an input cannot be read or an output cannot be written or removed.

    Notes
    -----
    Only top-level `*.json` input files are read; other input files and
    subdirectories, including `.flowde`, are ignored. Every supplied part directory must
    contain exactly the same filename stems. Each JSON must contain only its
    declared part, without a benchmark ground-truth `options` wrapper. The
    formats and node-number checks are those of
    [`build_a_partial_flowchart()`][flowde.parsing_fns.parsing_types.build_a_partial_flowchart].

    Input-validation errors leave existing output files unchanged and do not
    create a new output directory. Each JSON is written atomically, but the
    whole directory is not a single transaction. Before saving new results,
    `.flowde/combined.state` records both previous and intended output filenames.
    A write failure or interruption can leave some complete new results saved.
    After saving all results and removing obsolete recorded outputs, the record
    is reduced to the current output filenames. Rerun with
    `on_existing="overwrite"` to regenerate the combined results. Avoid
    simultaneous calls writing to the same output directory.

    Combining makes no model requests and does not modify the source files.
    The separate `.flowde/combined.state` file records only output ownership;
    no `.flowde/run.state` is created. Source run records, token usage and costs
    are not combined. Overwrite permits replacing manually edited recorded
    outputs. Older combined directories without an output record cannot be
    overwritten; use a new output directory.

    Examples
    --------
    Combine existing node-text and label results:

    ```python
    from pathlib import Path

    from flowde.combine_parsed_parts import combine_parsed_parts

    combined = combine_parsed_parts(
        nodes_dir=Path("results/parsing/node_text"),
        labels_dir=Path("results/parsing/labels"),
        save_dir=Path("results/parsing/combined"),
    )
    ```

    Add `flow_dir` and `additional_texts_dir` to include those parts. To replace
    a previous set of combined results, pass `on_existing="overwrite"`.

    """
    if on_existing not in {"error", "overwrite"}:
        msg = "on_existing must be 'error' or 'overwrite'."
        raise ValueError(msg)
    nodes_paths = _json_files(nodes_dir)
    if not nodes_paths:
        msg = f"No JSON files were found in nodes_dir: {nodes_dir}."
        raise ValueError(msg)
    labels_paths = _json_files(labels_dir) if labels_dir is not None else None
    flow_paths = _json_files(flow_dir) if flow_dir is not None else None
    additional_texts_paths = (
        _json_files(additional_texts_dir) if additional_texts_dir is not None else None
    )
    path_lists = [nodes_paths, labels_paths, flow_paths, additional_texts_paths]
    validate_matching_stems(*path_lists)

    root = save_dir.resolve()
    for directory in (nodes_dir, labels_dir, flow_dir, additional_texts_dir):
        if directory is not None and directory.resolve().is_relative_to(root):
            msg = (
                f"Output directory {save_dir} contains input directory {directory}. "
                "Use a separate output directory."
            )
            raise ValueError(msg)
    protect_inputs(save_dir, [path for paths in path_lists if paths for path in paths])
    if save_dir.is_symlink():
        msg = f"Output directory must not be a symbolic link: {save_dir}."
        raise ValueError(msg)
    if save_dir.exists() and not save_dir.is_dir():
        msg = f"Output path is not a directory: {save_dir}."
        raise NotADirectoryError(msg)
    if on_existing == "error" and save_dir.exists() and any(save_dir.iterdir()):
        msg = f"Output directory {save_dir} already contains files or subdirectories."
        msg += " Use on_existing='overwrite' or a new output directory."
        raise FileExistsError(msg)

    for path in save_dir.glob("*.json"):
        if path.is_symlink():
            msg = f"Output JSON must not be a symbolic link: {path}."
            raise ValueError(msg)
    output_paths = [save_dir / path.name for path in nodes_paths]
    for path in output_paths:
        if path.is_dir():
            msg = f"A directory occupies the required output filename: {path}."
            raise IsADirectoryError(msg)

    results = build_partial_flowcharts(
        nodes_paths=nodes_paths,
        labels_paths=labels_paths,
        flow_paths=flow_paths,
        additional_texts_paths=additional_texts_paths,
    )
    contents = [json_bytes(result.model_dump(mode="json")) for result in results]
    previous_names = _recorded_outputs(save_dir)
    current_names = {path.name for path in output_paths}
    record_path = save_dir / ".flowde" / "combined.state"
    # Record ownership before publishing any new file so failed writes can be retried.
    _save_outputs(record_path, previous_names | current_names)
    for path, content in zip(output_paths, contents, strict=True):
        atomic_write(path, content)
    for name in sorted(previous_names - current_names):
        (save_dir / name).unlink(missing_ok=True)
    _save_outputs(record_path, current_names)
    return results
