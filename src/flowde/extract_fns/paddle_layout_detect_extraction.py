from functools import lru_cache
from pathlib import Path

import pymupdf
from paddleocr import LayoutDetection

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

    return extract_imgs
