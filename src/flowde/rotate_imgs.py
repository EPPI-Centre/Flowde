from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

from flowde._batch import run_batch
from flowde._run_settings import file_digest, function_settings
from flowde._run_state import ExistingRun, protect_inputs
from flowde.classify_fns.classify_types import (
    ClassificationFunction,
    RotationClassification,
    RotationLabel,
)
from flowde.utils import validate_unique_stems

INPUT_TEXT = """
Here is a flowchart image.
Return the angle of clockwise rotation needed to correct its orientation, such that the
majority of the text reads left to right and top to bottom.
Only give an angle that is one of the following: 0, 90, 180, or 270 degrees.
"""


def rotate_imgs_from_paths(
    classify_fn: ClassificationFunction[RotationLabel],
    img_paths: list[Path],
    save_dir: Path,
    *,
    n_jobs: int = cpu_count(),
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[RotationLabel]:
    """
    Predict clockwise corrections and save copies in a dedicated run directory.

    Each valid angle is saved before writing its corrected image. Resume reuses
    saved angles and recreates missing copies without repeating model requests.
    Original images are never modified, including for a zero-degree correction.

    Parameters
    ----------
    classify_fn : ClassificationFunction[RotationLabel]
        Return one of `0`, `90`, `180`, or `270`. Custom functions declare their
        settings with `model_function()`; built-in factories already do this.
    img_paths : list[Path]
        Input images in the requested result order, with unique stems.
    save_dir : Path
        Dedicated directory containing `rotations.json`, corrected copies in
        `rotated_images`, and run metadata in `.flowde`. Inputs must be outside it.
    n_jobs : int, optional
        Number of parallel model calls. Defaults to the number of CPU cores.
    show_usage : bool, optional
        Show reported token usage and estimated costs. Defaults to `True`.
    on_existing : {"error", "resume", "overwrite"}, optional
        How to handle an existing run. Defaults to `"error"`.

    Returns
    -------
    list[RotationLabel]
        Clockwise angles in input order, including restored results.

    """
    if not img_paths:
        msg = "img_paths cannot be empty."
        raise ValueError(msg)
    validate_unique_stems(img_paths)
    protect_inputs(save_dir, img_paths)
    settings = function_settings(classify_fn)

    def angle(value: Any) -> RotationLabel:
        if value is None:
            msg = "A rotation classifier must return an angle; None is a failure."
            raise TypeError(msg)
        return RotationClassification.model_validate({"label": value}).label

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

    def classify_one(img_path: Path) -> RotationLabel:
        return classify_fn(img_path)

    results = run_batch(
        classify_one,
        inputs,
        save_dir,
        kind="rotation",
        settings=settings,
        encode=angle,
        decode=angle,
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
    print("ROTATION COMPLETE")
    return results


def rotate_imgs(
    classify_fn: ClassificationFunction[RotationLabel],
    img_dir: Path,
    save_dir: Path,
    *,
    range_indices: tuple[int | None, int | None] | None = None,
    n_jobs: int = cpu_count(),
    show_usage: bool = True,
    on_existing: ExistingRun = "error",
) -> list[RotationLabel]:
    """
    Rotate PNG images in a directory and save corrected copies in a resumable run.

    Parameters
    ----------
    classify_fn : ClassificationFunction[RotationLabel]
        Function accepting one image path and returning the clockwise correction
        in degrees: `0`, `90`, `180` or `270`. The function returns the angle
        itself, not a dictionary or Pydantic model. Built-in factories declare
        their settings; custom functions must declare their settings with
        [`model_function()`][flowde.model_function].
    img_dir : Path
        Directory containing the original `*.png` files. Subdirectories are not
        searched. Image paths are sorted before applying `range_indices`.
    save_dir : Path
        Dedicated output directory, created if needed. Correction angles are
        saved in `rotations.json`, corrected copies in `rotated_images`, and run
        metadata in `.flowde/run.state`. Original images remain unchanged.
        Input images and files referenced by the classifier's declared settings
        must be outside this directory.
    range_indices : tuple[int | None, int | None] | None, optional
        A `(start, stop)` slice of the sorted image paths. `start` is included
        and `stop` is excluded; `(0, 10)` selects up to the first ten images.
        Either bound can be `None`, and negative indices follow Python slicing
        rules. Defaults to `None`, which selects all matching images. An empty
        selection raises an error.
    n_jobs : int, optional
        Number of angle predictions that can run concurrently. Defaults to the
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
        - `"resume"`: require a saved rotation run with matching classifier
          settings. Reuse saved angles for unchanged inputs and predict angles
          for selected images without saved angles.
        - `"overwrite"`: remove the previous run's tracked outputs and saved state,
          then start a new run. Also works in a new or empty directory.
          Unrelated files in an existing output directory cause an error.

    Returns
    -------
    list[RotationLabel]
        One clockwise correction angle per image selected by this call, in sorted
        image-path order. Includes angles restored from a previous run. With
        `range_indices`, the returned list covers only the selected slice;
        angles for earlier slices remain saved in `rotations.json`.

    Raises
    ------
    FileExistsError
        If `save_dir` contains existing work and `on_existing="error"`.
    ValueError
        If `img_dir` is missing or is not a directory, no PNGs are selected,
        an angle is unsupported, inputs are inside `save_dir`, or the saved
        run fails compatibility or integrity checks.
    TypeError
        If the classifier returns `None` instead of an angle.
    RuntimeError
        If another Flowde call is already using the same `save_dir`.

    Notes
    -----
    `rotations.json` contains objects with `img_path` and `label` fields, where
    `label` is the clockwise correction angle. The JSON file accumulates angles
    across resumed calls and orders entries by resolved input paths.

    Every successfully processed image has a copy in `rotated_images`, including
    images with a zero-degree correction. Copies retain their filenames and
    expand to fit the rotated image without cropping.

    On resume, missing `rotations.json` or selected `rotated_images` copies are
    recreated from saved angles and unchanged original images without another
    prediction. Corrected copies are always made from the original inputs, so
    resuming does not rotate an already-corrected copy again. Edited previously
    processed input images or saved outputs cause an error.

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
    return rotate_imgs_from_paths(
        classify_fn=classify_fn,
        img_paths=img_paths,
        save_dir=save_dir,
        n_jobs=n_jobs,
        show_usage=show_usage,
        on_existing=on_existing,
    )
