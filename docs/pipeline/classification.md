# Image classification

Classification is the second stage of the core pipeline. It takes extracted
images as input and assigns a label to each image.

In a typical workflow, we
[extract images](./image-extraction.md#image-extraction) from PDFs, and use
classification to decide which of those images are relevant. For this workflow,
you will probably want to classify each extracted image as a flowchart or some
specific type of flowchart, such as a CONSORT diagram.

## Define a classification function

Before running classification, define the function that will classify each
image. This can be one of the default LLM-based classification functions, or a
custom function that you provide.

A classification function takes the path to one image and returns a label.

### Use a default classification function

`flowde` includes default classification functions using OpenAI and Gemini
models.

<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

!!! note "Default classification with Gemini or OpenAI"
    To use either of the default classification functions, follow the
    [setup default parsing and classification](./setup-default-funcs.md#setup-default-parsing-and-classification)
    steps.

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD046 -->

To define an OpenAI classification function:

```python
from flowde.classify_fns.classify_types import Classification
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn

input_text = """
Classify whether this image is a flowchart.

Return 1 if it is a flowchart.
Return 0 if it is not a flowchart.
"""

classify_fn = make_openai_classify_fn(
    input_text=input_text,
    model="gpt-5.4-mini",
    effort="high",
)
```

To use Gemini instead, use `make_gemini_classify_fn`:

```python
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

classify_fn = make_gemini_classify_fn(
    input_text=input_text,
    model="gemini-3.1-flash-lite",
    result_structure=FlowchartClassification,
    effort="high",
)
```

See (insert api ref docs here) for details about params.

#### Multiclass classification

By default, these functions do binary classification, outputting `0` for
negative and `1` for positive. You can also use multiclass classification by
passing a Pydantic model with a `label` field to the `result_structure`
parameter.

For example, you could classify extracted images by image type:

```python
from typing import Literal

from pydantic import BaseModel

from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn


class ImageTypeClassification(BaseModel):
    label: Literal[
        "consort_flowchart",
        "other_flowchart",
        "table",
        "graph",
        "other",
    ]


input_text = """
Classify the extracted image into exactly one of the following categories:

- consort_flowchart: a CONSORT-style participant flow diagram
- other_flowchart: a flowchart that is not a CONSORT-style participant flow diagram
- table: a table or tabular display
- graph: a chart, plot, or graph
- other: anything else

Return only the selected label.
"""

classify_fn = make_openai_classify_fn(
    input_text=input_text,
    model="gpt-5.4-mini",
    result_structure=ImageTypeClassification,
    effort="high",
)
```

(add rotation defaults so user doesn't need to edit the result structure.)

### Bring your own classification function

You do not have to use the default OpenAI or Gemini classification helpers. Any
function with the same interface can be used.

A custom classification function must accept an image path and return a label:

```python
from pathlib import Path
from typing import Literal

FlowchartLabel = Literal[0, 1]


def classify_flowchart(img_path: Path) -> FlowchartLabel:
    ...
```

## Run classification

After defining `classify_fn`, pass it to one of the classification pipeline
functions.

Most users should use `classify_imgs`. This takes a directory of PNG images,
classifies each image, and returns the predicted labels.

### Classify images from a directory

```python
from pathlib import Path

from flowde.classify_imgs import classify_imgs

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/extracted-images"),
    json_path=Path("data/classifications.json"),
)
```

The input directory should contain the extracted PNG images at the top level.
`classify_imgs` searches for files matching `*.png` directly inside `img_dir`.

For example:

```text
data/
└── extracted-images/
    ├── paper-1_0.png
    ├── paper-1_1.png
    ├── paper-2_0.png
    └── paper-3_0.png
```

### Save classifications to JSON

Pass `json_path` to save the classification results:

```python
labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/extracted-images"),
    json_path=Path("data/classifications.json"),
)
```

The JSON file contains the image path and predicted label for each image:

```json
[
  {
    "img_path": "data/extracted-images/paper-1_0.png",
    "label": 1
  },
  {
    "img_path": "data/extracted-images/paper-1_1.png",
    "label": 0
  }
]
```

### Save positive images

You can copy positively classified images into a separate directory. This is
useful when you want to keep only images with particular labels.

```python
labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/extracted-images"),
    json_path=Path("data/classifications.json"),
    positive_img_save_dir=Path("data/flowchart-images"),
    positive_classes={1},
)
```

When `positive_img_save_dir` and `positive_classes` are provided, each image
with a label in `positive_classes` is copied to `positive_img_save_dir`.

Both arguments must be provided together.

### Classify an explicit list of images

If you do not want to classify every PNG in a directory, use
`classify_imgs_from_paths`. This lets you pass the image paths directly.

```python
from pathlib import Path

from flowde.classify_imgs import classify_imgs_from_paths

img_paths = [
    Path("data/extracted-images/paper-1_0.png"),
    Path("data/extracted-images/paper-2_0.png"),
]

labels = classify_imgs_from_paths(
    classify_fn=classify_fn,
    img_paths=img_paths,
    json_path=Path("data/classifications.json"),
)
```

You can also provide explicit save paths for positive images:

```python
labels = classify_imgs_from_paths(
    classify_fn=classify_fn,
    img_paths=img_paths,
    json_path=Path("data/classifications.json"),
    positive_img_save_paths=[
        Path("data/flowchart-images/paper-1_0.png"),
        Path("data/flowchart-images/paper-2_0.png"),
    ],
    positive_clsses={1},
)
```

`img_paths` and `positive_img_save_paths` must have the same length. Each image
is copied to the corresponding save path if its predicted label is in
`positive_clsses`.

### Parallelism

Both `classify_imgs` and `classify_imgs_from_paths` accept an `n_jobs` parameter
to control the number of parallel processes of `classify_fn` to run. By default,
they will try to use all cpu cores available. If your `classify_fn` is memory
heavy, you may need to manually reduce the number of jobs.

```python
from pathlib import Path

from flowde.classify_imgs import classify_imgs

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/extracted-images"),
    json_path=Path("data/classifications.json"),
    n_jobs=1,
)
```

## Next step

After classification, use the rotation stage to rotate the classified flowchart
images into the correct orientation.
