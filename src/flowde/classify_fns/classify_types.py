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
    def __call__(self, img_path: Path) -> LabelType_co: ...
