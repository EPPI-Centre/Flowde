from io import BytesIO
from unittest.mock import Mock, call

import pymupdf
import pytest
from PIL import Image, ImageChops

import flowde.extract_fns.paddle_layout_detect_extraction as paddle_extract
from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)


class PaddleResult(dict):
    """Provide Paddle's boxes and page-image size without running its model."""

    def __init__(self, *, size, boxes):
        super().__init__(boxes=boxes)
        self.img = {"res": Image.new("RGB", size)}


@pytest.fixture
def pipeline():
    model = Mock(spec=["predict"])
    model.predict.return_value = []
    return model


@pytest.fixture(autouse=True)
def layout_detection_constructor(monkeypatch, pipeline):
    """Mock `LayoutDetection` construction and isolate its cache for every test."""
    constructor = Mock(return_value=pipeline)
    monkeypatch.setattr(paddle_extract, "LayoutDetection", constructor)
    paddle_extract._get_layout_detection_pipeline.cache_clear()
    yield constructor
    paddle_extract._get_layout_detection_pipeline.cache_clear()


@pytest.fixture
def page_image():
    """Give each pixel a distinct colour so incorrect crop positions fail."""
    image = Image.new("RGB", (100, 100))
    image.putdata([(x, y, 0) for y in range(100) for x in range(100)])
    return image


def write_pdf(path, images):
    """Put each image on a PDF page of the same size at 72 DPI."""
    with pymupdf.open() as document:
        for image in images:
            png = BytesIO()
            image.save(png, format="PNG")
            page = document.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, stream=png.getvalue())
        document.save(path)


@pytest.fixture
def pdf_path(tmp_path, page_image):
    path = tmp_path / "paper.pdf"
    write_pdf(path, [page_image])
    return path


@pytest.fixture
def save_dir(tmp_path):
    return tmp_path / "nested" / "extracted"


@pytest.fixture
def opened_pdf(monkeypatch, pdf_path):
    """Keep access to the real document so cleanup tests can inspect it."""
    document = pymupdf.open(pdf_path)
    monkeypatch.setattr(paddle_extract.pymupdf, "open", Mock(return_value=document))
    yield document
    if not document.is_closed:
        document.close()


def assert_saved_image(path, expected):
    with Image.open(path) as actual:
        assert actual.format == "PNG"
        assert actual.size == expected.size
        difference = ImageChops.difference(actual.convert("RGB"), expected)
        assert difference.getbbox() is None, f"Incorrect pixels in {path.name}"


def test_model_is_created_only_when_extraction_starts(
    layout_detection_constructor, pdf_path, save_dir
):
    extract = make_paddle_layout_extract_fn(device="cpu", cpu_threads=4)

    layout_detection_constructor.assert_not_called()

    extract(pdf_path, save_dir)

    layout_detection_constructor.assert_called_once_with(
        device="cpu", cpu_threads=4, enable_mkldnn=True
    )


def test_model_is_reused_for_subsequent_pdfs(
    layout_detection_constructor, pipeline, pdf_path, save_dir
):
    second_pdf = pdf_path.with_name("second.pdf")
    second_pdf.write_bytes(pdf_path.read_bytes())
    extract = make_paddle_layout_extract_fn()

    extract(pdf_path, save_dir)
    extract(second_pdf, save_dir)

    layout_detection_constructor.assert_called_once_with(
        device="gpu", cpu_threads=10, enable_mkldnn=True
    )
    assert pipeline.predict.call_args_list == [
        call(str(pdf_path), batch_size=1),
        call(str(second_pdf), batch_size=1),
    ]


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="identical-settings"),
        pytest.param({"batch_size": 4}, id="different-batch-size"),
        pytest.param({"dpi": 300}, id="different-dpi"),
        pytest.param({"padding": 20}, id="different-padding"),
    ],
)
def test_extractors_share_the_model_when_model_settings_match(
    layout_detection_constructor, pdf_path, save_dir, settings
):
    first_extract = make_paddle_layout_extract_fn()
    second_extract = make_paddle_layout_extract_fn(**settings)

    first_extract(pdf_path, save_dir)
    second_extract(pdf_path, save_dir)

    layout_detection_constructor.assert_called_once_with(
        device="gpu", cpu_threads=10, enable_mkldnn=True
    )


@pytest.mark.parametrize(
    ("device", "cpu_threads", "enable_mkldnn"),
    [
        pytest.param("cpu", 10, True, id="different-device"),
        pytest.param("gpu", 4, True, id="different-thread-count"),
        pytest.param("gpu", 10, False, id="disable-mkldnn"),
    ],
)
def test_changing_model_settings_creates_a_new_model(
    layout_detection_constructor, pdf_path, save_dir, device, cpu_threads, enable_mkldnn
):
    first_extract = make_paddle_layout_extract_fn()
    second_extract = make_paddle_layout_extract_fn(
        device=device, cpu_threads=cpu_threads, enable_mkldnn=enable_mkldnn
    )

    first_extract(pdf_path, save_dir)
    second_extract(pdf_path, save_dir)

    assert layout_detection_constructor.call_args_list == [
        call(device="gpu", cpu_threads=10, enable_mkldnn=True),
        call(device=device, cpu_threads=cpu_threads, enable_mkldnn=enable_mkldnn),
    ]


def test_only_the_most_recent_model_settings_remain_cached(
    layout_detection_constructor, pdf_path, save_dir
):
    gpu_extract = make_paddle_layout_extract_fn(device="gpu")
    cpu_extract = make_paddle_layout_extract_fn(device="cpu")

    # Repeated GPU extractions reuse the first model.
    gpu_extract(pdf_path, save_dir)
    gpu_extract(pdf_path, save_dir)

    layout_detection_constructor.assert_called_once_with(
        device="gpu", cpu_threads=10, enable_mkldnn=True
    )

    # Switching to CPU replaces the cached model; repeated CPU extractions reuse it.
    cpu_extract(pdf_path, save_dir)
    cpu_extract(pdf_path, save_dir)

    assert layout_detection_constructor.call_args_list == [
        call(device="gpu", cpu_threads=10, enable_mkldnn=True),
        call(device="cpu", cpu_threads=10, enable_mkldnn=True),
    ]

    # Switching back recreates the displaced GPU model, which is then reused.
    gpu_extract(pdf_path, save_dir)
    gpu_extract(pdf_path, save_dir)

    assert layout_detection_constructor.call_args_list == [
        call(device="gpu", cpu_threads=10, enable_mkldnn=True),
        call(device="cpu", cpu_threads=10, enable_mkldnn=True),
        call(device="gpu", cpu_threads=10, enable_mkldnn=True),
    ]


def test_prediction_receives_the_pdf_path_and_requested_batch_size(
    pipeline, pdf_path, save_dir
):
    extract = make_paddle_layout_extract_fn(batch_size=3)

    extract(pdf_path, save_dir)

    pipeline.predict.assert_called_once_with(str(pdf_path), batch_size=3)


def test_non_image_boxes_are_ignored_without_leaving_filename_gaps(
    pipeline, pdf_path, save_dir, page_image
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100),
            boxes=[
                {"label": "text", "coordinate": [0, 0, 10, 10]},
                {"label": "image", "coordinate": [10, 10, 20, 20]},
                {"coordinate": [20, 20, 30, 30]},
                {"label": "image", "coordinate": [30, 30, 40, 40]},
            ],
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=0)

    extract(pdf_path, save_dir)

    assert sorted(path.name for path in save_dir.iterdir()) == [
        "paper_0.png",
        "paper_1.png",
    ]
    assert_saved_image(save_dir / "paper_0.png", page_image.crop((10, 10, 20, 20)))
    assert_saved_image(save_dir / "paper_1.png", page_image.crop((30, 30, 40, 40)))


def test_image_numbering_continues_across_pages_in_detection_order(
    pipeline, pdf_path, save_dir, page_image
):
    blue_page = Image.new("RGB", (100, 100), "blue")
    write_pdf(pdf_path, [page_image, blue_page])
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100),
            boxes=[
                {"label": "image", "coordinate": [10, 10, 20, 20]},
                {"label": "image", "coordinate": [30, 30, 40, 40]},
            ],
        ),
        PaddleResult(
            size=(100, 100),
            boxes=[{"label": "image", "coordinate": [0, 0, 100, 100]}],
        ),
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=0)

    extract(pdf_path, save_dir)

    assert sorted(path.name for path in save_dir.iterdir()) == [
        "paper_0.png",
        "paper_1.png",
        "paper_2.png",
    ]
    assert_saved_image(save_dir / "paper_0.png", page_image.crop((10, 10, 20, 20)))
    assert_saved_image(save_dir / "paper_1.png", page_image.crop((30, 30, 40, 40)))
    assert_saved_image(save_dir / "paper_2.png", blue_page)


def test_each_page_is_rendered_once_at_the_requested_dpi(
    monkeypatch, pipeline, pdf_path, save_dir, page_image
):
    write_pdf(pdf_path, [page_image, page_image])
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100),
            boxes=[
                {"label": "image", "coordinate": [10, 10, 20, 20]},
                {"label": "image", "coordinate": [30, 30, 40, 40]},
            ],
        ),
        PaddleResult(size=(100, 100), boxes=[]),
    ]
    extract = make_paddle_layout_extract_fn(dpi=216)

    rendered_pages = []
    render_page = pymupdf.Page.get_pixmap

    def record_render(page, *, dpi):
        rendered_pages.append((page.number, dpi))
        return render_page(page, dpi=dpi)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", record_render)

    extract(pdf_path, save_dir)

    # Both pages render once, including the page with no image boxes.
    assert rendered_pages == [(0, 216), (1, 216)]


def test_coordinates_scale_independently_in_each_direction(
    pipeline, pdf_path, save_dir, page_image
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(50, 25),
            boxes=[{"label": "image", "coordinate": [5, 5, 15, 10]}],
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=0)

    extract(pdf_path, save_dir)

    # The PDF renders to 100 x 100: multiply x by 2 and y by 4.
    assert_saved_image(save_dir / "paper_0.png", page_image.crop((10, 20, 30, 40)))


def test_padding_is_added_in_paddle_coordinates_before_scaling(
    pipeline, pdf_path, save_dir, page_image
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(50, 50),
            boxes=[{"label": "image", "coordinate": [10, 10, 20, 20]}],
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=5)

    extract(pdf_path, save_dir)

    # Padding gives (5, 5, 25, 25); doubling gives (10, 10, 50, 50).
    assert_saved_image(save_dir / "paper_0.png", page_image.crop((10, 10, 50, 50)))


def test_fractional_coordinates_are_rounded_after_scaling(
    pipeline, pdf_path, save_dir, page_image
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(50, 25),
            boxes=[{"label": "image", "coordinate": [5.2, 5.4, 15.4, 10.2]}],
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=0)

    extract(pdf_path, save_dir)

    # Scaling gives (10.4, 21.6, 30.8, 40.8); then round to whole pixels.
    assert_saved_image(save_dir / "paper_0.png", page_image.crop((10, 22, 31, 41)))


@pytest.mark.parametrize(
    "coordinate",
    [
        pytest.param([-5, -5, 105, 105], id="box-extends-outside-page"),
        pytest.param([2, 3, 98, 97], id="padding-extends-outside-page"),
    ],
)
def test_crops_are_clipped_to_all_four_page_edges(
    pipeline, pdf_path, save_dir, page_image, coordinate
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100), boxes=[{"label": "image", "coordinate": coordinate}]
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=5)

    extract(pdf_path, save_dir)

    assert_saved_image(save_dir / "paper_0.png", page_image)


def test_existing_output_is_overwritten(pipeline, pdf_path, save_dir, page_image):
    save_dir.mkdir(parents=True)
    Image.new("RGB", (1, 1), "blue").save(save_dir / "paper_0.png")
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100),
            boxes=[{"label": "image", "coordinate": [0, 0, 100, 100]}],
        )
    ]
    extract = make_paddle_layout_extract_fn(dpi=72, padding=0)

    extract(pdf_path, save_dir)

    assert_saved_image(save_dir / "paper_0.png", page_image)


@pytest.mark.parametrize(
    "results",
    [
        pytest.param([], id="no-page-results"),
        pytest.param([PaddleResult(size=(100, 100), boxes=[])], id="no-boxes"),
        pytest.param(
            [
                PaddleResult(
                    size=(100, 100),
                    boxes=[
                        {"label": "text", "coordinate": [0, 0, 10, 10]},
                        {"coordinate": [20, 20, 30, 30]},
                    ],
                )
            ],
            id="no-image-boxes",
        ),
    ],
)
def test_no_images_creates_an_empty_output_directory_and_returns_none(
    pipeline, pdf_path, save_dir, opened_pdf, results
):
    pipeline.predict.return_value = results
    extract = make_paddle_layout_extract_fn()

    result = extract(pdf_path, save_dir)

    assert result is None
    assert save_dir.is_dir()
    assert list(save_dir.iterdir()) == []
    assert opened_pdf.is_closed


def test_prediction_errors_propagate_without_opening_the_pdf(
    monkeypatch, pipeline, pdf_path, save_dir
):
    error = RuntimeError("prediction failed")
    pipeline.predict.side_effect = error
    open_pdf = Mock()
    monkeypatch.setattr(paddle_extract.pymupdf, "open", open_pdf)
    extract = make_paddle_layout_extract_fn()

    with pytest.raises(RuntimeError, match="prediction failed") as raised:
        extract(pdf_path, save_dir)

    assert raised.value is error
    open_pdf.assert_not_called()


@pytest.mark.parametrize(
    ("owner", "method"),
    [
        pytest.param(None, None, id="successful-extraction"),
        pytest.param(pymupdf.Page, "get_pixmap", id="rendering-fails"),
        pytest.param(Image.Image, "crop", id="cropping-fails"),
        pytest.param(Image.Image, "save", id="saving-fails"),
    ],
)
def test_pdf_is_closed_after_extraction(
    monkeypatch, pipeline, pdf_path, save_dir, opened_pdf, owner, method
):
    pipeline.predict.return_value = [
        PaddleResult(
            size=(100, 100),
            boxes=[{"label": "image", "coordinate": [10, 10, 20, 20]}],
        )
    ]
    extract = make_paddle_layout_extract_fn()

    if method is None:
        result = extract(pdf_path, save_dir)
        assert result is None
    else:
        error = RuntimeError("processing failed")
        monkeypatch.setattr(owner, method, Mock(side_effect=error))

        with pytest.raises(RuntimeError, match="processing failed") as raised:
            extract(pdf_path, save_dir)

        assert raised.value is error

    assert opened_pdf.is_closed


def test_pdf_is_closed_if_prediction_results_fail_during_iteration(
    pipeline, pdf_path, save_dir, opened_pdf
):
    error = RuntimeError("prediction stream failed")

    def predictions():
        yield PaddleResult(size=(100, 100), boxes=[])
        raise error

    pipeline.predict.return_value = predictions()
    extract = make_paddle_layout_extract_fn()

    with pytest.raises(RuntimeError, match="prediction stream failed") as raised:
        extract(pdf_path, save_dir)

    assert raised.value is error
    assert opened_pdf.is_closed


def test_malformed_boxes_raise_naturally_and_the_pdf_is_closed(
    pipeline, pdf_path, save_dir, opened_pdf
):
    pipeline.predict.return_value = [
        PaddleResult(size=(100, 100), boxes=[{"label": "image"}])
    ]
    extract = make_paddle_layout_extract_fn()

    with pytest.raises(KeyError, match="coordinate"):
        extract(pdf_path, save_dir)

    assert opened_pdf.is_closed
