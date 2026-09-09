from dataclasses import dataclass, field

from flowde.benchmarks.classification.classification_benchmark import (
    MulticlassClassificationBenchmark,
)
from flowde.classify_fns.classify_types import RotationLabel


@dataclass(frozen=True, kw_only=True)
class RotationBenchmark(MulticlassClassificationBenchmark[RotationLabel]):
    labels: tuple[RotationLabel, ...] = field(
        default=(0, 90, 180, 270),
        init=False,
    )
