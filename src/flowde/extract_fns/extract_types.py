from pathlib import Path
from typing import Protocol


class ExtractImgsFunction(Protocol):
    def __call__(
        self,
        *,
        pdf_path: Path,
        save_dir: Path,
    ) -> None: ...
