# Custom processing functions

You can replace the default extraction, classification, rotation or parsing
functions with your own while still using Flowde's directory handling, parallel
processing, automatic result saving and support for stopping and resuming runs.

## Declare settings

To use Flowde's automatic result saving and support for stopping and resuming
runs, **you must declare the settings that affect your custom function's
results** using
[`model_function()`](../reference/helpers.md#flowde.model_function). For
example, the [page extractor below](#an-extraction-function) uses this
declaration:

```python
from flowde import model_function

extract_fn = model_function(render_pages, extractor="whole-pages", dpi=DPI, version=1)
```

Flowde stores the declared settings as metadata about the run in the
`.flowde` directory inside `save_dir`. If any declared setting changes,
Flowde raises an error when you try to resume the existing run.

## Examples

### An extraction function

You can pass a custom extractor as `extract_fn` to
[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs).
For example, the following extractor saves each complete PDF page as a PNG:

```python
from pathlib import Path

import pymupdf

from flowde import model_function
from flowde.extract_imgs import extract_imgs

DPI = 144


def render_pages(*, pdf_path: Path, save_dir: Path) -> None:
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document):
            page.get_pixmap(dpi=DPI).save(save_dir / f"{pdf_path.stem}_{index}.png")


extract_fn = model_function(render_pages, extractor="whole-pages", dpi=DPI, version=1)

extract_imgs(
    pdf_dir=Path("data/pdfs"),
    save_dir=Path("results/page-images"),
    extract_fn=extract_fn,
    n_jobs=1,
)
```

The custom extractor must:

- Accept `pdf_path` and `save_dir` as keyword arguments.
- Save only PNG files directly inside the supplied `save_dir`, without
  subdirectories.
- Use filenames that remain unique across the input PDFs.
- Finish writing and close every output file before returning `None`.

See the
[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)
API reference for full details of the `extract_fn` requirements.

### A classification function

You can pass a custom classifier as `classify_fn` to
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs).
For example, the following classifier returns `1` for images that are at least
500 pixels wide and 500 pixels tall, and `0` if either dimension is below 500
pixels:

```python
from pathlib import Path

from PIL import Image

from flowde import model_function
from flowde.classify_fns.classify_types import BinaryClassification
from flowde.classify_imgs import classify_imgs

MIN_DIMENSION = 500


def classify_by_size(img_path: Path) -> int:
    with Image.open(img_path) as image:
        width, height = image.size
    return int(width >= MIN_DIMENSION and height >= MIN_DIMENSION)


classify_fn = model_function(
    classify_by_size,
    result_structure=BinaryClassification,
    method="image-size",
    min_dimension=MIN_DIMENSION,
    version=1,
)

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("results/extraction"),
    save_dir=Path("results/size-classification"),
    positive_classes={1},
    max_concurrent_jobs=1,
)
```

The custom classifier must:

- Accept an `img_path` argument.
- Return a `str`, `int` or `bool` label.
- Raise an exception if classification fails. Returning `None` raises an error.

See the
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)
API reference for full details of the `classify_fn` requirements.

Flowde saves the returned labels in `classifications.json` inside `save_dir`.

You can supply `result_structure` through
[`model_function()`](../reference/helpers.md#flowde.model_function) to validate
the returned labels. The example uses
[`BinaryClassification`](../reference/data-types.md#flowde.classify_fns.classify_types.BinaryClassification),
which permits only the integers `0` and `1`. Without `result_structure`, Flowde
checks that the label is a string, integer or boolean but does not restrict the
allowed values.

### A rotation function

You can pass a custom rotation classifier as `classify_fn` to
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs). For
example, the following function returns `90` for images that are wider than they
are tall, and `0` for portrait or square images. Flowde uses these angles to
rotate landscape images into portrait orientation:

```python
from pathlib import Path

from PIL import Image

from flowde import model_function
from flowde.rotate_imgs import rotate_imgs


def portrait_rotation_angle(img_path: Path) -> int:
    with Image.open(img_path) as image:
        width, height = image.size
    return 90 if width > height else 0


rotation_fn = model_function(
    portrait_rotation_angle,
    method="portrait-by-dimensions",
    version=1,
)

angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=Path("results/extraction"),
    save_dir=Path("results/portrait-rotation"),
    max_concurrent_jobs=1,
)
```

The custom rotation classifier must:

- Accept an `img_path` argument.
- Return a clockwise correction in degrees as an `int`: `0`, `90`, `180` or
  `270`.
- Raise an exception if angle prediction fails. Returning `None` raises an
  error.

See the
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs) API
reference for full details of the `classify_fn` requirements.

Flowde applies the returned angles, saves image copies in `rotated_images` and
records the angles in `rotations.json`, both inside `save_dir`.

### A parsing function

You can pass a custom parser as `parse_fn` to
[`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs). For
example, the following parser reads each image's width, height and colour mode
into a custom Pydantic result:

```python
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from flowde import model_function
from flowde.parse_imgs import parse_imgs


class ImageDetails(BaseModel):
    width: int
    height: int
    colour_mode: str


def parse_image_details(img_path: Path) -> ImageDetails:
    with Image.open(img_path) as image:
        return ImageDetails(
            width=image.width,
            height=image.height,
            colour_mode=image.mode,
        )


parse_fn = model_function(
    parse_image_details,
    result_structure=ImageDetails,
    method="image-details",
    version=1,
)

results = parse_imgs(
    parse_fn=parse_fn,
    img_dir=Path("results/extraction"),
    save_dir=Path("results/image-details"),
    max_concurrent_jobs=1,
)
```

The custom parser must:

- Accept `img_path` as a keyword argument.
- Declare a Pydantic result class through `result_structure` in
  [`model_function()`](../reference/helpers.md#flowde.model_function).
- Return an instance of the declared result class, rather than a dictionary or
  JSON string.
- Raise an exception if parsing fails. Returning `None` raises an error.

See the [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)
API reference for full details of the `parse_fn` requirements.

The example uses a custom result class as its `result_structure`. For the
standard flowchart format, you can use
[`build_partial_flowchart_schema()`](../reference/helpers.md#flowde.parsing_fns.parsing_types.build_partial_flowchart_schema)
to create a result class containing node text, labels, flow or additional text.

To support [previously parsed parts](parsing.md#parse-parts-separately), your
parser must also accept `partial_flowchart` as a keyword argument. Flowde
supplies a Pydantic model containing the saved parts for the image being parsed.
Without saved parts, Flowde passes only `img_path`.

## Errors, workers and usage

Your function must raise an exception if processing fails. Flowde stops starting
further PDFs or images after detecting the error.

You can use
[`report_usage()`](../reference/data-types.md#flowde.usage.report_usage) inside
your classification, rotation or parsing function to include token counts and
estimated costs in the progress display. See
[report usage from a custom function](usage.md#report-usage-from-a-custom-function)
for an example.
