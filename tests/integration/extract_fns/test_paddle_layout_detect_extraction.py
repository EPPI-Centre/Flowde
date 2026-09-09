from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import flowde.extract_fns.paddle_layout_detect_extraction as paddle_extract_module

pytestmark = pytest.mark.integration


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
PDF_DIR = FIXTURE_DIR / "pdfs"
EXPECTED_DIR = FIXTURE_DIR / "paddle_expected"


def load_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def assert_images_exactly_equal(actual_path: Path, expected_path: Path) -> None:
    actual = load_rgb_array(actual_path)
    expected = load_rgb_array(expected_path)

    assert actual.shape == expected.shape, (
        f"Image shapes differ for {actual_path.name}. "
        f"Actual shape: {actual.shape}. Expected shape: {expected.shape}."
    )

    assert np.array_equal(actual, expected), (
        f"Image pixels differ for {actual_path.name}. "
        f"Actual: {actual_path}. Expected: {expected_path}."
    )


def test_paddle_layout_extract_fn_exactly_matches_expected_extracted_images(tmp_path):
    pdf_path = PDF_DIR / "Bush_2018.pdf"
    expected_paths = sorted(EXPECTED_DIR.glob(f"{pdf_path.stem}_*.png"))

    if not pdf_path.exists():
        raise AssertionError(f"Test fixture PDF does not exist: {pdf_path}")

    if len(expected_paths) == 0:
        raise AssertionError(
            f"No expected extracted PNG fixtures found in {EXPECTED_DIR} "
            f"for PDF stem {pdf_path.stem}."
        )

    save_dir = tmp_path / "extracted"
    save_dir.mkdir()

    extract_imgs = paddle_extract_module.make_paddle_layout_extract_fn(
        device="cpu",
        cpu_threads=2,
        batch_size=1,
    )

    result = extract_imgs(pdf_path=pdf_path, save_dir=save_dir)

    assert result is None

    actual_paths = sorted(save_dir.glob("*.png"))

    assert [path.name for path in actual_paths] == [
        path.name for path in expected_paths
    ]

    for actual_path, expected_path in zip(actual_paths, expected_paths, strict=True):
        assert_images_exactly_equal(
            actual_path=actual_path,
            expected_path=expected_path,
        )
