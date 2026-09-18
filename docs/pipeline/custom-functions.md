# Custom processing functions

You can replace the default extraction, classification, rotation or parsing
helper while keeping Flowde's directory handling, parallelism and saved runs.

A custom function must have the interface expected by its stage and declare
the settings that affect its results. Use the public
[`model_function()`](../reference/helpers.md#flowde.model_function) helper
to attach those settings.

## Declare settings

```python
from flowde import model_function
```

[`model_function(fn, **settings)`](../reference/helpers.md#flowde.model_function)
returns the same function with saved-run
settings attached. The settings describe the function; they do not configure
the function's code. Built-in factories already make this declaration.

Declare the model, prompt, threshold, algorithm version and other choices that
affect the answer. If you change your algorithm, change its declared version.
Flowde does not infer changes from the Python function's source code.

Settings must be serialisable. Ordinary strings, numbers, lists and dictionaries
work. Pydantic schemas and instances are supported. Pass referenced **files**
as `Path` objects so Flowde records both their paths and contents. A directory
is not a file setting; provide a list or mapping of the actual files instead.

## An extraction function

An extractor accepts `pdf_path` and `save_dir` as keyword arguments. It writes
top-level PNG files into `save_dir` and returns `None`. This complete example
renders each PDF page, rather than detecting figure regions:

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

During extraction, the supplied `save_dir` is the temporary directory for one
PDF. Flowde validates and publishes the PNGs after your function returns
successfully. Close all files before returning. Use filenames unique across
PDFs, such as the PDF stem plus a counter.

## A classification function

A classifier accepts `img_path` and returns a string, integer or boolean label.
It returns the label itself, rather than `{"label": ...}`. Returning `None`
is an error.

For example, to use labels you have already reviewed, create
`data/reviewed-labels.json`:

```json
{ "paper-1_0": 1, "paper-1_1": 0, "paper-2_0": 1 }
```

Then define:

```python
import json
from pathlib import Path

from flowde import model_function
from flowde.classify_fns.classify_types import BinaryClassification
from flowde.classify_imgs import classify_imgs

labels_path = Path("data/reviewed-labels.json")
reviewed_labels = json.loads(labels_path.read_text(encoding="utf-8"))


def reviewed_classifier(img_path: Path) -> int:
    return reviewed_labels[img_path.stem]


classify_fn = model_function(
    reviewed_classifier,
    result_structure=BinaryClassification,
    labels_file=labels_path,
    version=1,
)

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("results/extraction"),
    save_dir=Path("results/reviewed-classification"),
    positive_classes={1},
    n_jobs=1,
)
```

This example demonstrates the interface using reviewed answers. Replace the
lookup with your own image classifier when predictions are required.
`result_structure` is optional for a custom classifier; providing the schema
lets Flowde validate the returned label against that schema.

## A rotation function

A rotation function has the classification interface, but returns a clockwise
correction of `0`, `90`, `180` or `270`. Declare its settings with
[`model_function()`](../reference/helpers.md#flowde.model_function) and pass the
function to [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs) .

For example, if every input image is known to be correctly oriented:

```python
from pathlib import Path

from flowde import model_function
from flowde.rotate_imgs import rotate_imgs


def already_upright(img_path: Path) -> int:
    return 0


rotation_fn = model_function(already_upright, method="known-upright", version=1)

angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=Path("data/upright-flowcharts"),
    save_dir=Path("results/upright-copies"),
    n_jobs=1,
)
```

Flowde still saves a copy of every image under `rotated_images`, together with
the recorded zero-degree angles.

## A parsing function

A parser accepts `img_path` and an optional `partial_flowchart` and returns a
Pydantic model. Declare the result class with `result_structure`. Returning a
plain dictionary or `None` is not accepted.

This example loads previously prepared full-flowchart JSON files to demonstrate
the interface without making model requests:

```python
from pathlib import Path

from pydantic import BaseModel

from flowde import model_function
from flowde.parse_imgs import parse_imgs
from flowde.parsing_fns.parsing_types import build_partial_flowchart_schema

FullFlowchart = build_partial_flowchart_schema()
annotation_files = {
    path.stem: path for path in Path("data/prepared-parses").glob("*.json")
}


def prepared_parser(
    img_path: Path,
    partial_flowchart: BaseModel | None = None,
) -> BaseModel:
    text = annotation_files[img_path.stem].read_text(encoding="utf-8")
    return FullFlowchart.model_validate_json(text)


parse_fn = model_function(
    prepared_parser,
    result_structure=FullFlowchart,
    annotation_files=annotation_files,
    version=1,
)

results = parse_imgs(
    parse_fn=parse_fn,
    img_dir=Path("data/upright-flowcharts"),
    save_dir=Path("results/prepared-parsing"),
    n_jobs=1,
)
```

Your model-based parser can use `partial_flowchart` as context and construct
the declared Pydantic result from its response. When no partial context is
provided, Flowde calls the parser with only the image path.

Keep annotation files and other referenced inputs outside the run's `save_dir`.
The result schema is also part of the settings checked on resume.

## Errors, workers and usage

Raise an exception when your function cannot produce a valid result. Flowde
stops starting new inputs after noticing the failure and lets other active
workers save successful results. A resumed run retries inputs without saved
answers.

Custom functions can run in separate processes when `n_jobs` is greater than
one. Avoid relying on mutable global state shared between workers. For
extraction, each worker writes only into its supplied temporary directory.
Classification and parsing functions return answers; Flowde writes the run's
output files.

Use [`report_usage()`](../reference/data-types.md#flowde.usage.report_usage) inside the
function if you want token counts and cost estimates in the progress display.
See [`model_function()`](../reference/helpers.md#flowde.model_function) for
the declaration reference.
