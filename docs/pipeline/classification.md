# Image classification

Classification takes images as input and assigns a label to each image. In the
core pipeline, classification selects the relevant images after
[extraction](image-extraction.md).

For example, label a flowchart `1` and any other figure `0`. You can also define
several classes, such as CONSORT diagram, other flowchart, table and graph.

## Classify images from a directory

[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)
takes a directory of images and saves their labels into one dedicated output
directory.

```python
from pathlib import Path

from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.classify_imgs import classify_imgs

img_dir = Path("results/extraction")
save_dir = Path("results/classification")

input_text = """
Classify whether this image is a flowchart.
Return 1 if it is a flowchart.
Return 0 if it is not a flowchart.
"""

classify_fn = make_openai_classify_fn(
    input_text=input_text,
    model="gpt-5.6-luna",
    effort="medium",
)

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=img_dir,
    save_dir=save_dir,
    positive_classes={1},
    max_concurrent_jobs=1,
)
```

[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)
processes `*.png` files inside `img_dir`, in sorted path order.
Subdirectories are not searched. The returned `labels` list contains one label
per processed image: `labels[0]` corresponds to the first sorted image path,
`labels[1]` to the second sorted image path, and so on.

The example above uses OpenAI and requires the
[OpenAI setup](setup-default-funcs.md#set-up-openai).
For `classify_fn`, you can use an OpenAI classifier created with
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn),
a Gemini classifier created with
[`make_gemini_classify_fn()`](../reference/helpers.md#flowde.classify_fns.gemini_classify_fn.make_gemini_classify_fn),
or [your own classification function](custom-functions.md#a-classification-function).

You can use `positive_classes` to save copies of images with selected labels.
For example, `positive_classes={1}` copies images labelled `1` into
`save_dir / "positive_images"` for use in the core pipeline's
[image rotation](rotation.md) or [image parsing](parsing.md) stage.
Without `positive_classes`, Flowde saves the labels without copying images.

[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)
creates a binary classification function by default, returning `0` or `1`.
You can change the allowed labels through its `result_structure` parameter;
see [multiclass classification](#multiclass-classification).

See the
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)
and
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)
API references for full details of all parameters.

## Classification results

In the example above, Flowde saves the labels and copies images labelled `1`:

```text
results/classification/
├── classifications.json
├── positive_images/
│   ├── paper-1_0.png
│   └── paper-2_0.png
└── .flowde/
    ├── run.state
    └── run.lock
```

`classifications.json` contains image paths and labels:

```json
[
  { "img_path": "results/extraction/paper-1_0.png", "label": 1 },
  { "img_path": "results/extraction/paper-1_1.png", "label": 0 },
  { "img_path": "results/extraction/paper-2_0.png", "label": 1 }
]
```

`.flowde/run.state` records which images have been classified, their labels,
the classification settings and file fingerprints. Flowde uses this metadata
to resume the run and detect changes to the input images or saved results.

## Resume classification

To continue an unfinished classification run whose results are saved in
`save_dir`, you can set `on_existing="resume"`:

```python
labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=img_dir,
    save_dir=save_dir,
    positive_classes={1},
    max_concurrent_jobs=1,
    on_existing="resume",
)
```

The `.flowde/run.state` file inside `save_dir` stores the classification run's
state. Flowde uses this record to resume unfinished work and check the integrity
of the run.
Resume raises an error if the classifier's settings or `positive_classes` have
changed, or if previously classified input images or saved outputs have been
edited.

To use different settings, you can start a run in a new `save_dir` or replace
the previous run with `on_existing="overwrite"`. See
[managing runs](resuming.md) for the full rules.

## Multiclass classification

You can classify images into more than two categories. This example creates
a classifier with five possible labels:

```python
from typing import Literal

from flowde.classify_fns.classify_types import Classification
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn


class ImageTypeClassification(
    Classification[Literal["consort", "other_flowchart", "table", "graph", "other"]]
):
    pass


image_type_fn = make_openai_classify_fn(
    input_text=(
        "Classify this image as consort, other_flowchart, table, graph, or other. "
        "Use consort for a CONSORT-style participant flow diagram."
    ),
    model="gpt-5.6-luna",
    effort="medium",
    result_structure=ImageTypeClassification,
)
```

`ImageTypeClassification` defines the allowed labels using
[`Classification`](../reference/data-types.md#flowde.classify_fns.classify_types.Classification).
Passing this schema as `result_structure` to
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)
creates a classifier that returns one of those five labels.

## Bring your own classification function

You can write your own classifier and pass it to
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)
as `classify_fn`. See the
[custom classification tutorial](custom-functions.md#a-classification-function)
for the requirements your function must meet and a complete working example.

## Next step

You can use `save_dir / "positive_images"` as the input directory for
[image rotation](rotation.md) or, if the copied images are already correctly
oriented, [image parsing](parsing.md).
