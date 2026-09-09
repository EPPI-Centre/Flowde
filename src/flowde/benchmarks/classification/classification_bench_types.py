from dataclasses import dataclass
from typing import Generic

from flowde.classify_fns.classify_types import LabelType


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleClassificationResult(Generic[LabelType]):
    img_path: str
    pred: LabelType
    true: LabelType
