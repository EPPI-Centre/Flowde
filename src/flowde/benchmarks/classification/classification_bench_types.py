from dataclasses import dataclass
from typing import Generic

from flowde.classify_fns.classify_types import LabelType


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleClassificationResult(Generic[LabelType]):
    """
    Store the true and predicted labels for one benchmark image.

    Parameters
    ----------
    img_path : str
        Path to the classified image. Benchmarks take this path from the
        ground-truth classification file.
    pred : LabelType
        Predicted label for the image.
    true : LabelType
        Ground-truth label for the image.

    """

    img_path: str
    pred: LabelType
    true: LabelType
