from dataclasses import dataclass

from flowde.classify_fns.classify_types import ClassificationLabel


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleClassificationResult[LabelType: ClassificationLabel]:
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
