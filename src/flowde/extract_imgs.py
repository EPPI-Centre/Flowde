from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from flowde._batch import run_batch
from flowde._extraction_state import (
    ExtractionRunState,
    image_manifest,
    staging_directory,
)
from flowde._run_settings import file_digest, function_settings
from flowde._run_state import ExistingRun, protect_inputs
from flowde.extract_fns.extract_types import ExtractImgsFunction
from flowde.utils import validate_unique_stems


def extract_imgs_pdf_list(
    pdf_paths: list[Path],
    save_dir: Path,
    extract_fn: ExtractImgsFunction,
    n_jobs: int = 1,
    *,
    on_existing: ExistingRun = "error",
) -> None:
    """
    Extract PDFs into one directory, recording each complete image set.

    Parameters
    ----------
    pdf_paths : list[Path]
        PDFs in the requested processing order, with unique filename stems.
    save_dir : Path
        Dedicated run directory. PNGs go directly here; tracking information goes
        in `.flowde`. Input PDFs must be outside this directory.
    extract_fn : ExtractImgsFunction
        Save top-level PNGs for one PDF into the supplied temporary directory.
        Custom functions declare their settings with `model_function()`.
    n_jobs : int, optional
        Number of PDFs processed concurrently. Defaults to `1`.
    on_existing : {"error", "resume", "overwrite"}, optional
        Defaults to `"error"`. Resume skips completed PDFs and restarts unfinished
        PDFs. Missing PNGs trigger re-extraction; edited PNGs cause an error.

    Returns
    -------
    None
        Images are saved only after extraction of their entire PDF succeeds.

    """
    if not pdf_paths:
        msg = "pdf_paths cannot be empty."
        raise ValueError(msg)
    validate_unique_stems(pdf_paths)
    if len({path.stem.casefold() for path in pdf_paths}) != len(pdf_paths):
        msg = "PDF filename stems must also be unique ignoring case."
        raise ValueError(msg)
    protect_inputs(save_dir, pdf_paths)
    settings = function_settings(extract_fn)
    inputs = [
        {
            "key": str(path.resolve()),
            "path": str(path),
            "output": path.name,
            "input": {"pdf": file_digest(path)},
            "kwargs": {"pdf_path": path},
        }
        for path in pdf_paths
    ]

    # The parent owns staging, so worker shutdown finishes before cleanup.
    with TemporaryDirectory(prefix="flowde-extraction-") as temporary:
        staging_root = Path(temporary)

        def extract_one(pdf_path: Path) -> dict[str, str]:
            stage = staging_directory(staging_root, str(pdf_path.resolve()))
            stage.mkdir()
            extract_fn(pdf_path=pdf_path, save_dir=stage)
            return image_manifest(stage)

        run_batch(
            extract_one,
            inputs,
            save_dir,
            kind="extraction",
            settings=settings,
            encode=dict,
            decode=dict,
            n_jobs=n_jobs,
            show_usage=False,
            on_existing=on_existing,
            state_factory=partial(ExtractionRunState, staging_root=staging_root),
        )


def extract_imgs(
    pdf_dir: Path,
    save_dir: Path,
    extract_fn: ExtractImgsFunction,
    n_jobs: int = 1,
    *,
    on_existing: ExistingRun = "error",
) -> None:
    """
    Extract sorted top-level PDFs, saving complete image sets and resumable state.

    Parameters
    ----------
    pdf_dir : Path
        Directory containing input PDFs. Subdirectories are not searched.
    save_dir : Path
        Dedicated output directory containing PNGs and `.flowde` run metadata.
    extract_fn : ExtractImgsFunction
        Extractor created by a built-in factory or declared with
        [`model_function()`][flowde.model_function].
    n_jobs : int, optional
        Number of PDFs processed concurrently. Defaults to `1`.
    on_existing : {"error", "resume", "overwrite"}, optional
        Defaults to `"error"`. Resume skips completed PDFs and restarts unfinished
        PDFs. Missing PNGs trigger re-extraction; edited PNGs cause an error.

    Returns
    -------
    None
        Extracted images are written into `save_dir`.

    """
    if not pdf_dir.is_dir():
        msg = f"PDF directory {pdf_dir} does not exist or is not a directory."
        raise ValueError(msg)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        msg = f"No PDF files found in directory {pdf_dir}."
        raise ValueError(msg)
    extract_imgs_pdf_list(
        pdf_paths, save_dir, extract_fn, n_jobs, on_existing=on_existing
    )
