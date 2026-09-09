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
    """Classify sorted top-level PNGs; resume or overwrite only when requested."""
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
