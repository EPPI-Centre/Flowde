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
    Rotate sorted top-level PNGs, saving corrected copies and resumable results.

    Parameters
    ----------
    classify_fn : ClassificationFunction[RotationLabel]
        Predict the clockwise correction for one image.
    img_dir : Path
        Directory containing the original PNG images.
    save_dir : Path
        Dedicated run directory; corrected copies go in `rotated_images` and
        angles in `rotations.json`. Original images are preserved.
    range_indices : tuple[int | None, int | None] | None, optional
        Slice the sorted images using an inclusive start and exclusive stop.
    n_jobs : int, optional
        Number of parallel model calls. Defaults to the number of CPU cores.
    show_usage : bool, optional
        Show reported token usage and estimated costs. Defaults to `True`.
    on_existing : {"error", "resume", "overwrite"}, optional
        How to handle existing work. Defaults to `"error"`. Resume requires
        matching settings and unchanged previously answered input images.

    Returns
    -------
    list[RotationLabel]
        Clockwise angles in the selected sorted-image order.

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
