from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image

from flowde.classify_fns.classify_types import ClassificationFunction, RotationLabel
from flowde.classify_imgs import save_classifications_to_json
from flowde.utils import apply_fn_parallel_on_list

INPUT_TEXT = """
Here is a flowchart image.
Return the angle of clockwise rotation needed to correct its orientation, such that the
majority of the text reads left to right and top to bottom.
Only give an angle that is one of the following: 0, 90, 180, or 270 degrees.
"""


def rotate_imgs_from_paths(
    classify_fn: ClassificationFunction[RotationLabel],
    img_paths: list[Path],
    json_path: Path | None = None,
    save_paths: list[Path] | None = None,
    save_in_place: bool = False,
    n_jobs: int = cpu_count(),
) -> list[RotationLabel]:
    if save_in_place and save_paths is not None:
        msg = "Cannot specify save_paths if save_in_place is True."
        raise ValueError(msg)

    if save_paths is not None and len(save_paths) != len(img_paths):
        msg = "Length of save_paths must match length of img_paths."
        raise ValueError(msg)

    if len(img_paths) == 0:
        msg = "img_paths cannot be empty."
        raise ValueError(msg)

    responses = apply_fn_parallel_on_list(
        fn=classify_fn,
        items=img_paths,
        msg="Classifying images...",
        n_jobs=n_jobs,
    )

    for img_path, angle in zip(img_paths, responses, strict=True):
        if angle not in {0, 90, 180, 270}:
            msg = (
                f"Invalid angle {angle} for image {img_path}. "
                "Angle must be one of the following: 0, 90, 180, or 270 degrees."
            )
            raise ValueError(msg)

    if json_path is not None:
        print(f"Saving rotation results to JSON: {json_path!s}")
        save_classifications_to_json(
            json_path=json_path, img_paths=img_paths, labels=responses
        )

    if save_in_place:
        print("Rotating images in place...")
        for img_path, angle in zip(img_paths, responses, strict=True):
            if angle != 0:
                with Image.open(img_path) as img:
                    rotated_img = img.rotate(angle, expand=True)
                rotated_img.save(img_path)

    if save_paths is not None:
        print("Rotating images and saving to new paths...")
        for img_path, save_path, angle in zip(
            img_paths, save_paths, responses, strict=True
        ):
            with Image.open(img_path) as img:
                rotated_img = img.rotate(angle, expand=True)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rotated_img.save(save_path)

    return responses


def rotate_imgs(
    classify_fn: ClassificationFunction[RotationLabel],
    img_dir: Path,
    save_dir: Path | None = None,
    json_path: Path | None = None,
    save_in_place: bool = False,
    n_jobs: int = cpu_count(),
) -> list[RotationLabel]:
    if not img_dir.is_dir():
        msg = f"Image directory {img_dir} does not exist or is not a directory."
        raise ValueError(msg)

    if save_in_place and save_dir is not None:
        msg = "Cannot specify save_dir if save_in_place is True."
        raise ValueError(msg)

    img_paths = list(img_dir.glob("*.png"))
    img_paths.sort()

    if len(img_paths) == 0:
        msg = f"No PNG image files found in directory {img_dir}."
        raise ValueError(msg)

    save_paths = None
    if save_dir is not None:
        save_paths = [save_dir / img_path.name for img_path in img_paths]

    return rotate_imgs_from_paths(
        classify_fn=classify_fn,
        img_paths=img_paths,
        json_path=json_path,
        save_paths=save_paths,
        save_in_place=save_in_place,
        n_jobs=n_jobs,
    )
