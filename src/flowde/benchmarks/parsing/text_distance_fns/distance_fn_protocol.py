from typing import Protocol


class DistanceFnProtocol(Protocol):
    def __call__(
        self,
        *,
        true_text: str | None,
        pred_text: str | None,
    ) -> int | float: ...
