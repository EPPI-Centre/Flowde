import json
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, TypeVar

from joblib import Parallel, delayed
from pydantic import BaseModel

from flowde._progress import progress_reporting, run_with_usage

ItemType = TypeVar("ItemType")
ReturnType = TypeVar("ReturnType")

VISION_FEW_SHOT_INSTRUCTIONS = (
    "The following exchanges are worked examples of the task above. "
    "Each example image is followed by its expected JSON answer. "
    "Any already-parsed information belongs only to the image in the same message. "
    "Use them as guidance and return an answer only for the final target image."
)


@dataclass(frozen=True, kw_only=True)
class VisionFewShotExample:
    """
    Describe one example image and its expected JSON answer for a model prompt.

    Parameters
    ----------
    img_path : Path
        Path to an existing example image. The image is included in the prompt
        as a worked example, before the target image being classified or parsed.
        The image format must be supported by the selected provider.
    expected_output_path : Path
        Path to an existing UTF-8 JSON file containing the desired response for
        the example image. For binary classification, the file could contain
        `{"label": 1}`. For parsing, the JSON should match the requested parts
        or custom response schema, without a benchmark ground truth's `options`
        wrapper. Schema compatibility is the caller's responsibility; this class
        does not validate the example answer against the model's response schema.
    partial_flowchart : BaseModel | None, optional
        Previously parsed data supplied as input context for this example image,
        such as node numbers and text when demonstrating label parsing.
        Defaults to `None`, meaning no partial context accompanies the example.
        This context belongs to `img_path`, not to the target image that the
        returned classifier or parser will later process.

    Raises
    ------
    ValueError
        If `img_path` or `expected_output_path` does not point to an existing
        file when the example is created.

    Notes
    -----
    Constructor arguments must be passed by name. The dataclass is frozen, so
    its fields cannot be reassigned after construction. The referenced files
    and the supplied Pydantic model are not made immutable.

    Creating an example checks that both paths point to files; it does not
    decode the image, read the expected JSON or make an API request.
    [`expected_output_text()`][flowde.utils.VisionFewShotExample.expected_output_text]
    reads and formats the expected JSON when a request is prepared. File
    contents are read from disk rather than cached in this object.

    You can pass a list of these instances as `few_shot_examples` to an OpenAI
    or Gemini classification or parsing factory. Each request includes the
    worked examples in list order, followed by the target image. Example
    answers demonstrate the desired response; they are not predictions for
    the target image.

    When a factory's callable is used in a saved Flowde run, the example image,
    expected-response file and partial context contribute to the settings
    checked on resume. Changes to those files or that context require a new
    run or explicit overwrite.

    Examples
    --------
    Given an existing `flowchart.png` and a `flowchart.json` containing
    `{"label": 1}`, create a classification example:

    ```python
    from pathlib import Path

    from flowde.utils import VisionFewShotExample

    example = VisionFewShotExample(
        img_path=Path("data/examples/flowchart.png"),
        expected_output_path=Path("data/examples/flowchart.json"),
    )
    ```

    The factory's `few_shot_examples` argument can then receive `[example]`.

    """

    img_path: Path
    expected_output_path: Path
    partial_flowchart: BaseModel | None = None

    def __post_init__(self) -> None:
        if not self.img_path.is_file():
            msg = f"img_path must be a file. Got {self.img_path}"
            raise ValueError(msg)

        if not self.expected_output_path.is_file():
            msg_0 = (
                f"expected_output_path must be a file. Got {self.expected_output_path}"
            )
            raise ValueError(msg_0)

    def expected_output_text(self) -> str:
        """
        Read the expected-response file and return its contents as formatted JSON.

        Returns
        -------
        str
            JSON text with two-space indentation. Non-ASCII characters are
            preserved rather than escaped. The file is read afresh on every
            call; no response-schema validation is performed.

        Raises
        ------
        OSError
            If `expected_output_path` cannot be opened or read, including when
            the file has been removed after the example was created.
        UnicodeDecodeError
            If the expected-response file is not valid UTF-8 text.
        json.JSONDecodeError
            If the expected-response file does not contain valid JSON.

        """
        with self.expected_output_path.open(encoding="utf-8") as f:
            expected_output = json.load(f)

        return json.dumps(expected_output, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def parse_bool(value: str) -> bool:
    normalized_value = value.lower()
    if normalized_value not in {"true", "false"}:
        msg = "expected true or false"
        raise ValueError(msg)
    return normalized_value == "true"


def _run_parallel_fn_with_better_errors(
    fn: Callable[[ItemType], ReturnType], item: ItemType
) -> ReturnType:
    try:
        return fn(item)
    except Exception as e:
        tb = traceback.format_exc()
        msg = (
            f"Error while processing item {item!r}\n"
            f"Original exception: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{tb}"
        )
        raise RuntimeError(msg) from e


def apply_fn_parallel_on_list(
    fn: Callable[[ItemType], ReturnType],
    items: Sequence[ItemType],
    msg: str = "Applying function in parallel",
    n_jobs: int = cpu_count(),
    *,
    show_usage: bool = True,
) -> list[ReturnType]:
    with progress_reporting(
        len(items), msg, parallel=n_jobs > 1, show_usage=show_usage
    ) as events:
        if n_jobs > 1:
            tasks = (
                delayed(run_with_usage)(
                    events, index, _run_parallel_fn_with_better_errors, fn, item
                )
                for index, item in enumerate(items)
            )
            response_list = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
        else:
            response_list = [
                run_with_usage(events, index, fn, item)
                for index, item in enumerate(items)
            ]
    return list(response_list)


def _run_parallel_fn_with_kwargs_better_errors(
    fn: Callable[..., ReturnType],
    kwargs: dict[str, Any],
) -> ReturnType:
    try:
        return fn(**kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        msg = (
            f"Error while processing kwargs {kwargs!r}\n"
            f"Original exception: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{tb}"
        )
        raise RuntimeError(msg) from e


def apply_fn_parallel_on_dict_of_lists(
    fn: Callable[..., ReturnType],
    items_by_param: dict[str, Sequence[Any]],
    msg: str = "Applying function in parallel",
    n_jobs: int = cpu_count(),
    *,
    show_usage: bool = True,
) -> list[ReturnType]:
    if not items_by_param:
        msg = "items_by_param cannot be empty."
        raise ValueError(msg)

    n_items = len(next(iter(items_by_param.values())))
    if not all(len(v) == n_items for v in items_by_param.values()):
        msg = "All parameter lists must have the same length."
        raise ValueError(msg)

    kwargs_list = [
        {key: value[i] for key, value in items_by_param.items()} for i in range(n_items)
    ]

    with progress_reporting(
        n_items, msg, parallel=n_jobs > 1, show_usage=show_usage
    ) as events:
        if n_jobs > 1:
            tasks = (
                delayed(run_with_usage)(
                    events,
                    index,
                    _run_parallel_fn_with_kwargs_better_errors,
                    fn,
                    kwargs,
                )
                for index, kwargs in enumerate(kwargs_list)
            )
            response_list = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
        else:
            response_list = [
                run_with_usage(events, index, fn, **kwargs)
                for index, kwargs in enumerate(kwargs_list)
            ]

    return list(response_list)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_same_path_lengths(*path_lists: list[Path] | None) -> None:
    provided_lists = [paths for paths in path_lists if paths is not None]

    if not provided_lists:
        return

    expected_len = len(provided_lists[0])

    for i, paths in enumerate(provided_lists):
        if len(paths) != expected_len:
            msg = (
                "All provided path lists must have the same length. "
                f"List 0 has length {expected_len}, list {i} has length {len(paths)}."
            )
            raise ValueError(msg)


def validate_unique_stems(paths: Sequence[Path]) -> None:
    """Raise an error if two paths have the same filename stem."""
    paths_by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in paths_by_stem:
            msg = (
                f"Duplicate file stem {path.stem!r}: "
                f"{paths_by_stem[path.stem]} and {path}. "
                "Input files must have unique stems."
            )
            raise ValueError(msg)
        paths_by_stem[path.stem] = path


def validate_matching_stems(*path_lists: list[Path] | None) -> None:
    provided_lists = [paths for paths in path_lists if paths is not None]

    if not provided_lists:
        return

    expected_len = len(provided_lists[0])
    for i, paths in enumerate(provided_lists):
        if len(paths) != expected_len:
            msg = (
                f"All provided path lists must have the same length. "
                f"List 0 has length {expected_len}, list {i} has length {len(paths)}."
            )
            raise ValueError(msg)

    for i in range(expected_len):
        stems = [paths[i].stem for paths in provided_lists]
        if len(set(stems)) != 1:
            msg = f"Mismatched stems at index {i}: {stems}"
            raise ValueError(msg)


def pretty_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    elif isinstance(obj, list):
        obj = [
            item.model_dump() if isinstance(item, BaseModel) else item for item in obj
        ]
    return json.dumps(obj, indent=2, ensure_ascii=False)
