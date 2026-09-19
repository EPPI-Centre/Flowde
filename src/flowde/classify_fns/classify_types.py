from pathlib import Path
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict

ClassificationLabel: TypeAlias = str | int | bool

LabelType = TypeVar("LabelType", bound=ClassificationLabel)
LabelType_co = TypeVar(
    "LabelType_co",
    bound=ClassificationLabel,
    covariant=True,
)

BinaryLabel: TypeAlias = Literal[0, 1]
RotationLabel: TypeAlias = Literal[0, 90, 180, 270]


class Classification(BaseModel, Generic[LabelType]):
    model_config = ConfigDict(frozen=True, strict=True)

    label: LabelType


class ConsortClassification(Classification[BinaryLabel]):
    pass


class BinaryClassification(Classification[BinaryLabel]):
    pass


class RotationClassification(Classification[RotationLabel]):
    pass


class ClassificationFunction(Protocol[LabelType_co]):
    """
    Describe a callable that returns one classification label for one image.

    Pass a classifier with this interface as `classify_fn` to
    [`classify_imgs()`][flowde.classify_imgs.classify_imgs]. The generic label
    type describes the callable's return value: a `str`, `int` or `bool`.
    For example, `ClassificationFunction[str]` returns string labels.

    Rotation uses the same interface. A `ClassificationFunction[RotationLabel]`
    passed to [`rotate_imgs()`][flowde.rotate_imgs.rotate_imgs] must return
    the clockwise correction angle as an integer: `0`, `90`, `180` or `270`.
    The classifier predicts the angle; Flowde applies the rotation and saves
    corrected image copies.

    Notes
    -----
    This protocol describes an interface; it does not implement classification
    or perform runtime validation. A function or callable object can satisfy
    the interface without inheriting from `ClassificationFunction`.

    Custom classifiers must also declare their settings with
    [`model_function()`][flowde.model_function]. The resulting `run_settings`
    attribute lets Flowde check compatibility when resuming a saved run.
    This requirement applies in addition to the call signature defined here.
    Built-in classifier factories declare their settings automatically.

    A classifier can optionally expose a `result_structure` class with a
    `label` field to restrict the permitted labels. For example,
    [`BinaryClassification`](data-types.md#flowde.classify_fns.classify_types.BinaryClassification)
    permits only the integers `0` and `1`. The classification pipeline checks
    each returned label against the supplied schema. Without a schema, the
    classification pipeline accepts any `str`, `int` or `bool` label. The
    rotation pipeline always checks for one of the four correction angles.

    """

    def __call__(self, img_path: Path) -> LabelType_co:
        """
        Classify one image and return its label or clockwise correction angle.

        Parameters
        ----------
        img_path : Path
            Path to the target image. Flowde supplies the path as the first
            positional argument. The classifier must support the image's
            format and leave the source image unchanged.

        Returns
        -------
        LabelType_co
            One `str`, `int` or `bool` label matching the callable's declared
            generic label type. Return the label itself, not a dictionary,
            JSON string containing an object, or Pydantic model. A rotation
            classifier must return the integer clockwise correction angle
            `0`, `90`, `180` or `270`; `0` means no correction is needed.

        Notes
        -----
        Raise an exception if classification or angle prediction fails.
        Returning `None` causes a pipeline error and does not mark the image
        as successfully processed.

        """
        ...
