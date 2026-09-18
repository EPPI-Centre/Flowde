from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pymupdf
from paddleocr import LayoutDetection

from flowde._run_settings import model_function
from flowde.extract_fns.extract_types import ExtractImgsFunction


@lru_cache(maxsize=1)
def _get_layout_detection_pipeline(
    device: str,
    cpu_threads: int,
) -> LayoutDetection:
    return LayoutDetection(device=device, cpu_threads=cpu_threads)


def make_paddle_layout_extract_fn(
    device: str = "gpu",
    cpu_threads: int = 10,
    batch_size: int = 1,
    dpi: int = 144,
    padding: int = 3,
) -> ExtractImgsFunction:
    """
    Create a PaddleOCR layout extractor that saves image regions from PDFs.

    Parameters
    ----------
    device : str, optional
        Device passed to PaddleOCR, for example `"cpu"` or `"gpu"`. Defaults
        to `"gpu"`. GPU execution requires a compatible GPU build of
        PaddlePaddle and an available GPU.
    cpu_threads : int, optional
        CPU inference thread setting passed to PaddleOCR. Defaults to `10`.
    batch_size : int, optional
        Number of page images per layout prediction batch. Defaults to `1`.
        This does not control the number of PDFs processed concurrently;
        `n_jobs` in [`extract_imgs()`][flowde.extract_imgs.extract_imgs]
        controls concurrent PDFs.
    dpi : int, optional
        Resolution used to render PDF pages for the saved crops. Defaults
        to `144`. This does not set the layout model's input resolution.
    padding : int, optional
        Margin around each detected box, in pixels of PaddleOCR's page image
        before scaling to the rendered page. Defaults to `3`. Crops are
        clipped to the rendered page boundaries.

    Returns
    -------
    ExtractImgsFunction
        Callable accepting `pdf_path` and `save_dir` as `Path` arguments.
        The callable saves regions labelled `image` as top-level PNGs named
        `<pdf_stem>_<counter>.png` and returns `None`. The counter starts at
        zero and runs across all extracted images in the PDF, rather than
        restarting on each page.

    Notes
    -----
    The layout model is loaded when the returned callable first processes a
    PDF; creating the callable does not load the model. The first extraction
    may download model files. Each worker process caches one model for reuse
    while `device` and `cpu_threads` remain unchanged.

    Start with `n_jobs=1` in [`extract_imgs()`][flowde.extract_imgs.extract_imgs]:
    each additional worker may load its own model and increase memory use.
    Larger batch sizes or thread counts
    do not guarantee faster extraction because page rendering and PNG writing
    also contribute to processing time.

    The returned callable records all five settings and the installed
    PaddleOCR and PaddleX versions. When resuming with
    [`extract_imgs()`][flowde.extract_imgs.extract_imgs], changes to these
    recorded settings require a new run or `on_existing="overwrite"`.
    Changing `n_jobs` is allowed when resuming.

    """

    def extract_imgs(pdf_path: Path, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        pipeline = _get_layout_detection_pipeline(
            device=device, cpu_threads=cpu_threads
        )

        output = pipeline.predict(
            str(pdf_path),
            batch_size=batch_size,
        )

        doc = pymupdf.open(pdf_path)
        try:
            img_counter = 0

            for page_num, res in enumerate(output):
                paddle_page_img = res.img["res"]
                page = doc.load_page(page_num)

                rendered_page_img = page.get_pixmap(dpi=dpi).pil_image()

                x_scale = rendered_page_img.width / paddle_page_img.width
                y_scale = rendered_page_img.height / paddle_page_img.height

                for box in res["boxes"]:
                    if box.get("label") != "image":
                        continue

                    bbox = box["coordinate"]

                    x0 = max(0, round((bbox[0] - padding) * x_scale))
                    y0 = max(0, round((bbox[1] - padding) * y_scale))
                    x1 = min(
                        rendered_page_img.width,
                        round((bbox[2] + padding) * x_scale),
                    )
                    y1 = min(
                        rendered_page_img.height,
                        round((bbox[3] + padding) * y_scale),
                    )

                    crop = rendered_page_img.crop((x0, y0, x1, y1))

                    crop.save(save_dir / f"{pdf_path.stem}_{img_counter}.png")
                    img_counter += 1
        finally:
            doc.close()

    return model_function(
        extract_imgs,
        extractor="paddle_layout_detection",
        paddleocr_version=version("paddleocr"),
        paddlex_version=version("paddlex"),
        device=device,
        cpu_threads=cpu_threads,
        batch_size=batch_size,
        dpi=dpi,
        padding=padding,
    )
