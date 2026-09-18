# Image extraction

Image extraction is the first stage of the core pipeline. It takes PDFs as input
and saves candidate flowchart images as PNG files.

The default PaddleOCR extractor detects regions labelled as images. These crops
can include figures that are not flowcharts. Use the
[classification stage](classification.md) to select the images you want to keep.

## Extract images from a directory of PDFs

[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)
takes a directory of PDFs and saves the extracted images into
one dedicated output directory.

First complete the [PaddleOCR setup](setup-default-funcs.md#set-up-image-extraction).
Then run:

```python
from pathlib import Path

from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)
from flowde.extract_imgs import extract_imgs

pdf_dir = Path("data/pdfs")
save_dir = Path("results/extraction")

extract_fn = make_paddle_layout_extract_fn(device="cpu", cpu_threads=1)

extract_imgs(
    pdf_dir=pdf_dir,
    save_dir=save_dir,
    extract_fn=extract_fn,
    n_jobs=1,
)
```

[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)
searches for `*.pdf` files directly inside `pdf_dir`, sorts the
paths and processes those PDFs. Subdirectories are not searched.

See the
[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)
and
[`make_paddle_layout_extract_fn()`](../reference/helpers.md#flowde.extract_fns.paddle_layout_detect_extraction.make_paddle_layout_extract_fn)
API references for full details of all parameters.

## Saved images

The PaddleOCR extractor names each PNG using the PDF stem and an image counter
starting at zero. The counter runs across all detected images in that PDF; it
is not a page number.

```text
results/
└── extraction/
    ├── paper-1_0.png
    ├── paper-1_1.png
    ├── paper-2_0.png
    ├── paper-3_0.png
    └── .flowde/
        ├── run.state
        └── run.lock
```

`.flowde/run.state` stores run metadata, including extraction settings, completed
PDFs and file fingerprints. Flowde uses this metadata to resume the run and
detect changes to the input PDFs or saved PNGs.

## Resume extraction

Repeat the call with the same settings and `on_existing="resume"`:

```python
extract_imgs(
    pdf_dir=pdf_dir,
    save_dir=save_dir,
    extract_fn=extract_fn,
    n_jobs=1,
    on_existing="resume",
)
```

Completed PDFs with all their unchanged PNGs are skipped. An unfinished PDF is
extracted again from its first page. If a completed PDF's saved PNG is missing,
Flowde also extracts that entire PDF again.

If a PNG in `save_dir` has been edited, resume raises an error. It does not
silently replace the edited PNG. Use a new output directory or explicitly
choose `on_existing="overwrite"` to start the extraction again.

See [managing runs](resuming.md) for interruption handling, settings checks and
what overwrite removes.

## Configure the PaddleOCR extractor

See the
[`make_paddle_layout_extract_fn()`](../reference/helpers.md#flowde.extract_fns.paddle_layout_detect_extraction.make_paddle_layout_extract_fn)
API reference for all parameters, defaults and configuration details.

## Bring your own extraction function

You can write your own extractor and pass it to
[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)
as `extract_fn`.
See the [custom extraction tutorial](custom-functions.md#an-extraction-function)
for the requirements your function must meet and a complete working example.

## Next step

[Classify the extracted images](classification.md) to select flowcharts or a
particular type of flowchart.
