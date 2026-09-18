from collections.abc import Sequence
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

from flowde._batch import run_batch
from flowde._run_settings import file_digest, function_settings, settings_value
from flowde._run_state import ExistingRun, protect_inputs
from flowde.classify_fns.classify_types import (
    ClassificationFunction,
    ClassificationLabel,
    LabelType,
)
from flowde.utils import save_json, validate_unique_stems


def classify_imgs_from_paths(
    classify_fn: ClassificationFunction[LabelType],
    img_paths: list[Path],
    save_dir: Path,
    *,
    positive_classes: set[ClassificationLabel] | None = None,
    n_jobs: int = cpu_count(),
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[LabelType]:
    """
    Classify images, saving each answer in a dedicated run directory.

    `classifications.json` accumulates results across resumed calls. If
    `positive_classes` is supplied, matching images go in `positive_images`.
    Existing output requires explicit `on_existing="resume"` or `"overwrite"`.
    Custom functions declare settings with `model_function()`; built-in factories
    already attach their settings.
    Return labels in the supplied image order, including restored results.
    """
    if not img_paths:
        msg = "img_paths cannot be empty."
        raise ValueError(msg)
    validate_unique_stems(img_paths)
    protect_inputs(save_dir, img_paths)
    settings = function_settings(classify_fn)
    settings["positive_classes"] = settings_value(positive_classes)
    structure = getattr(classify_fn, "result_structure", None)

    def label(value: Any) -> ClassificationLabel:
        if not isinstance(value, (str, int, bool)):
            msg = (
                "A classifier must return a string, integer, or boolean label; "
                "None is a failure."
            )
            raise TypeError(msg)
        return structure.model_validate({"label": value}).label if structure else value

    inputs = [
        {
            "key": str(path.resolve()),
            "path": str(path),
            "output": path.name,
            "input": {"image": file_digest(path)},
            "kwargs": {"img_path": path},
        }
        for path in img_paths
    ]

    def classify_one(img_path: Path) -> LabelType:
        return classify_fn(img_path)

    results = run_batch(
        classify_one,
        inputs,
        save_dir,
        kind="classification",
        settings=settings,
        encode=label,
        decode=label,
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
    print("CLASSIFICATION COMPLETE")
    return results


def save_classifications_to_json(
    json_path: Path, img_paths: list[Path], labels: Sequence[ClassificationLabel]
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_content = [
        {"img_path": str(img_path), "label": label}
        for img_path, label in zip(img_paths, labels, strict=True)
    ]
    save_json(json_content, json_path)


def classify_imgs(
    classify_fn: ClassificationFunction[LabelType],
    img_dir: Path,
    save_dir: Path,
    *,
    positive_classes: set[ClassificationLabel] | None = None,
    range_indices: tuple[int | None, int | None] | None = None,
    n_jobs: int = cpu_count(),
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[LabelType]:
    """
    Classify PNG images in a directory and save their labels in a resumable run.

    Parameters
    ----------
    classify_fn : ClassificationFunction[LabelType]
        Function accepting one image path and returning a string, integer or
        boolean label. The function returns the label itself, not a dictionary
        or Pydantic model. Built-in factories declare their settings;
        custom functions must declare their settings with
        [`model_function()`][flowde.model_function]. If the classifier exposes
        a `result_structure`, returned labels are validated against that schema.
    img_dir : Path
        Directory containing the input `*.png` files. Subdirectories are not
        searched. Image paths are sorted before applying `range_indices`.
    save_dir : Path
        Dedicated output directory, created if needed. Labels are saved in
        `classifications.json`, run metadata in `.flowde/run.state`, and optional
        image copies in `positive_images`. Input images and files referenced by
        the classifier's declared settings must be outside this directory.
    positive_classes : set[ClassificationLabel] | None, optional
        Labels whose images are copied into `save_dir / "positive_images"`,
        preserving filenames and leaving the original images unchanged.
        For example, `{1}` copies images labelled `1`. Defaults to `None`,
        which saves labels without copying images. Must remain unchanged when
        resuming the run.
    range_indices : tuple[int | None, int | None] | None, optional
        A `(start, stop)` slice of the sorted image paths. `start` is included
        and `stop` is excluded; `(0, 10)` selects up to the first ten images.
        Either bound can be `None`, and negative indices follow Python slicing
        rules. Defaults to `None`, which selects all matching images. An empty
        selection raises an error.
    n_jobs : int, optional
        Number of images that can be classified concurrently. Defaults to the
        number of CPU cores. `1` processes images sequentially in the calling
        process; larger values use worker processes. This value can change
        when resuming a run.
    show_usage : bool, optional
        Whether to display token usage and estimated costs reported by the
        classifier. Defaults to `True`. `False` hides those figures while
        retaining the progress display and saved usage reports. This value
        can change when resuming a run.
    on_existing : {"error", "resume", "overwrite"}, optional
        How to handle an existing run in `save_dir`. Defaults to `"error"`.

        - `"error"`: start in a new or empty directory; reject existing work.
        - `"resume"`: require a saved classification run with matching classifier
          settings and `positive_classes`. Reuse saved labels for unchanged
          inputs and classify selected images without saved labels.
        - `"overwrite"`: remove the previous run's tracked outputs and saved state,
          then start a new run. Also works in a new or empty directory.
          Unrelated files in an existing output directory cause an error.

    Returns
    -------
    list[LabelType]
        One label per image selected by this call, in sorted image-path order.
        Includes labels restored from a previous run. With `range_indices`, the
        returned list covers only the selected slice; labels for earlier slices
        remain saved in `classifications.json`.

    Raises
    ------
    FileExistsError
        If `save_dir` contains existing work and `on_existing="error"`.
    ValueError
        If `img_dir` is missing or is not a directory, no PNGs are selected,
        inputs are inside `save_dir`, or the saved run fails compatibility or
        integrity checks.
    TypeError
        If the classifier returns a label that is not a string, integer or
        boolean, including `None`.
    RuntimeError
        If another Flowde call is already using the same `save_dir`.

    Notes
    -----
    `classifications.json` contains objects with `img_path` and `label` fields.
    The JSON file accumulates labels across resumed calls and orders entries
    by resolved input paths. Run metadata records classifier settings, labels
    and file fingerprints so resume can detect changed inputs or saved outputs.

    On resume, missing `classifications.json` or selected `positive_images`
    copies are recreated from saved labels and unchanged input images without
    calling the classifier again. Manually edited saved outputs raise an error.

    """
    if not img_dir.is_dir():
        msg = f"Image directory {img_dir} does not exist or is not a directory."
        raise ValueError(msg)
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        msg = f"No PNG image files found in directory {img_dir}."
        raise ValueError(msg)
    if range_indices is not None:
        img_paths = img_paths[range_indices[0] : range_indices[1]]
    return classify_imgs_from_paths(
        classify_fn=classify_fn,
        img_paths=img_paths,
        save_dir=save_dir,
        positive_classes=positive_classes,
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
