# Image extraction

Image extraction is the first stage of the core pipeline. It takes PDFs as input
and saves candidate flowchart images as PNG files.

This stage only extracts images from PDFs. It does not decide whether an
extracted image is a flowchart. Use the classification stage to classify the
extracted images.

<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

!!! note "Default PaddleOCR extraction"
    `flowde` includes a default extraction function based on PaddleOCR layout
    detection. This method detects image regions in each PDF page and saves the
    cropped images to a directory.

    You can use this default function or provide your own image extraction
    function.

    To use the default method, follow the
    [setup default image extraction](./setup-default-funcs.md#setup-default-image-extraction)
    steps.

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD046 -->

## Extract images from a directory of PDFs

Most users should use `extract_imgs`. This takes a directory of PDFs and saves
the extracted images into a single output directory.

```python
from pathlib import Path

from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)
from flowde.extract_imgs import extract_imgs

pdf_dir = Path("data/pdfs")
save_dir = Path("data/extracted-images")

extract_fn = make_paddle_layout_extract_fn(device="cpu")

# Use gpu if you have installed gpu support:
# extract_fn = make_paddle_layout_extract_fn(device="gpu")

extract_imgs(
    pdf_dir=pdf_dir,
    save_dir=save_dir,
    extract_fn=extract_fn,
    n_jobs=1,
)
```

The input directory should contain only the PDFs you want to process, and those
PDFs should be at the top level of the directory. `extract_imgs` searches for
files matching `*.pdf` directly inside `pdf_dir`.

For example:

```text
data/
└── pdfs/
    ├── paper-1.pdf
    ├── paper-2.pdf
    └── paper-3.pdf
```

Extracted images are saved as PNG files in `save_dir`. The output filenames use
the PDF stem and an image counter:

```text
data/
└── extracted-images/
    ├── paper-1_0.png
    ├── paper-1_1.png
    ├── paper-2_0.png
    └── paper-3_0.png
```

## Extract from an explicit list of PDFs

If you do not want to process every PDF in a directory, use
`extract_imgs_pdf_list`. This lets you pass the PDF paths and output directories
directly.

```python
from pathlib import Path

from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)
from flowde.extract_imgs import extract_imgs_pdf_list

pdf_paths = [
    Path("data/pdfs/paper-1.pdf"),
    Path("data/pdfs/paper-2.pdf"),
]

save_dirs = [
    Path("data/extracted-images/paper-1"),
    Path("data/extracted-images/paper-2"),
]

extract_fn = make_paddle_layout_extract_fn(device="cpu")

extract_imgs_pdf_list(
    pdf_paths=pdf_paths,
    save_dirs=save_dirs,
    extract_fn=extract_fn,
    n_jobs=1,
)
```

`pdf_paths` and `save_dirs` must have the same length. Each PDF is processed
with the corresponding output directory.

## Bring your own extraction function

You do not have to use the default PaddleOCR extraction function. Any function
with the same interface can be used.

An extraction function must accept a `pdf_path` and a `save_dir`, and save any
extracted images into `save_dir`.

```python
from pathlib import Path


def my_extract_fn(*, pdf_path: Path, save_dir: Path) -> None:
    ...
```

You can then pass it to `extract_imgs`:

```python
from pathlib import Path

from flowde.extract_imgs import extract_imgs

extract_imgs(
    pdf_dir=Path("data/pdfs"),
    save_dir=Path("data/extracted-images"),
    extract_fn=my_extract_fn,
)
```

### Parallelism

Both `extract_imgs` and `extract_imgs_from_paths` accept an `n_jobs` parameter
to control the number of parallel processes of `extract_fn` to run. Image
extraction functions tend to be memory heavy and better parallelised internally;
we reccommend keeping `n_jobs=1` for most users.

```python
from pathlib import Path

from flowde.extract_imgs import extract_imgs

labels = extract_imgs(
    extract_fn=extract_fn,
    img_dir=Path("data/extracted-images"),
    json_path=Path("data/classifications.json"),
    n_jobs=3,
)

## Next step

After extracting images, classify the images as flowcharts or a specific type of
flowchart.
```
