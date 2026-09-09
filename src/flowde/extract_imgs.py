from pathlib import Path

from flowde.extract_fns.extract_types import ExtractImgsFunction
from flowde.utils import apply_fn_parallel_on_dict_of_lists


# TODO: Need to specify that the true_consort_ones have been rotated.
# TODO: Put classification benchmark into the benchmark directory, with a rotation one.
# TODO: Probably need to move the benchmark data and pdfs to another location
# also, so it isn't taking up bandwidth.
def extract_imgs_pdf_list(
    pdf_paths: list[Path],
    save_dirs: list[Path],
    extract_fn: ExtractImgsFunction,
    n_jobs: int = 1,
) -> None:
    if len(save_dirs) != len(pdf_paths):
        msg = "Length of save_dirs must match length of pdf_paths"
        raise ValueError(msg)

    for save_dir in save_dirs:
        save_dir.mkdir(parents=True, exist_ok=True)

    apply_fn_parallel_on_dict_of_lists(
        fn=extract_fn,
        items_by_param={
            "pdf_path": pdf_paths,
            "save_dir": save_dirs,
        },
        msg="Extracting images from PDFs",
        n_jobs=n_jobs,
    )


def extract_imgs(
    pdf_dir: Path,
    save_dir: Path,
    extract_fn: ExtractImgsFunction,
    n_jobs: int = 1,
) -> None:
    if not pdf_dir.is_dir():
        msg = f"PDF directory {pdf_dir} does not exist or is not a directory."
        raise ValueError(msg)

    pdf_paths = list(pdf_dir.glob("*.pdf"))
    if len(pdf_paths) == 0:
        msg = f"No PDF files found in directory {pdf_dir}."
        raise ValueError(msg)
    pdf_paths.sort()

    save_dir.mkdir(parents=True, exist_ok=True)

    save_dirs = [save_dir for _ in pdf_paths]

    extract_imgs_pdf_list(
        pdf_paths=pdf_paths,
        save_dirs=save_dirs,
        extract_fn=extract_fn,
        n_jobs=n_jobs,
    )
