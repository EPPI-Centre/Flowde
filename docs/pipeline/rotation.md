# Image rotation

Rotation is the third stage of the core pipeline. It takes images as input,
predicts the clockwise rotation needed to orient each image correctly, and then
optionally saves the rotated images.

In a typical workflow, we
[classify images](./classification.md#image-classification) to keep only the
relevant flowchart images, then use rotation to correct their orientation before
parsing. Of course, the order of rotation and classification is interchangeable.

## Define a rotation classification function

Before running rotation, define the function that will predict the rotation
angle for each image.

A rotation classification function takes the path to one image and returns one
of the following labels: `0`, `90`, `180`, `270`. These labels represent the
**clockwise angle** used to rotate the image into the correct orientation.

Rotation uses the same classification-function pattern as
[image classification](./classification.md#image-classification). The difference
is that the output label is always a rotation angle.

### Use a default LLM classification function

<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

!!! note "Default classification with Gemini or OpenAI"
    To use the default OpenAI or Gemini helpers, follow the
    [setup default parsing and classification](./setup-default-funcs.md#setup-default-parsing-and-classification)
    steps.

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD046 -->

For example, to define an OpenAI rotation classification function:

```python
from pydantic import BaseModel

from flowde.classify_fns.classify_types import RotationLabel
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn


class RotationClassification(BaseModel):
    label: RotationLabel # RotationLabel = Literal[0, 90, 180, 270]

input_text = """
Return the clockwise angle required to correctly orient the image,
such that the majority of text reads left to right, top to bottom.
"""

classify_fn = make_openai_classify_fn(
    input_text=input_text,
    model="gpt-5.4-mini",
    result_structure=RotationClassification,
    effort="high",
)
```

To use Gemini instead, use `make_gemini_classify_fn`:

```python
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

classify_fn = make_gemini_classify_fn(
    input_text=input_text,
    model="gemini-3.1-flash-lite",
    result_structure=RotationClassification,
    effort="high",
)
```

See (insert API ref docs here) for details about params.

### Bring your own rotation classification function

A custom rotation classification function must accept an image path and return
one of `0`, `90`, `180`, or `270`:

```python
from pathlib import Path

from flowde.classify_fns.classify_types import RotationLabel


def classify_rotation(img_path: Path) -> RotationLabel:
    ...
```

## Run rotation

After defining `classify_fn`, pass it to one of the rotation pipeline functions.

Most users should use `rotate_imgs`. This takes a directory of PNG images,
predicts the rotation angle for each image, and optionally saves the rotated
images.

### Rotate images from a directory

```python
from pathlib import Path

from flowde.rotate_imgs import rotate_imgs

rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    save_dir=Path("data/rotated-flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
)
```

The input directory should contain the PNG images at the top level.
`rotate_imgs` searches for files matching `*.png` directly inside `img_dir`.

For example:

```text
data/
└── flowchart-images/
    ├── paper-1_0.png
    ├── paper-2_0.png
    └── paper-3_0.png
```

When `save_dir` is provided, rotated images are saved to that directory using
the same filenames:

```text
data/
└── rotated-flowchart-images/
    ├── paper-1_0.png
    ├── paper-2_0.png
    └── paper-3_0.png
```

### Save rotation labels to JSON

Pass `json_path` to save the predicted rotation labels:

```python
rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
)
```

The JSON file contains the image path and predicted rotation label for each
image:

```json
[
  {
    "img_path": "data/flowchart-images/paper-1_0.png",
    "label": 0
  },
  {
    "img_path": "data/flowchart-images/paper-2_0.png",
    "label": 90
  }
]
```

### Save rotated images to a new directory

To save rotated images without modifying the original images, pass `save_dir`:

```python
rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    save_dir=Path("data/rotated-flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
)
```

This is usually the safest option because it keeps the original extracted images
unchanged.

### Rotate images in place

If you want to overwrite the original image files, pass `save_in_place=True`:

```python
rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
    save_in_place=True,
)
```

When `save_in_place=True`, you cannot also provide `save_dir`.

<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

!!! warning "Rotating in place overwrites the original files"
    Use `save_in_place=True` only if you are sure you do not need to keep the
    original image orientation.

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD046 -->

### Predict rotation without saving images

If you only want the predicted rotation labels, omit both `save_dir` and
`save_in_place`:

```python
rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
)
```

This classifies the rotation angle for each image and optionally saves the
labels to JSON, but does not write rotated image files.

### Rotate an explicit list of images

If you do not want to process every PNG in a directory, use
`rotate_imgs_from_paths`. This lets you pass the image paths directly.

```python
from pathlib import Path

from flowde.rotate_imgs import rotate_imgs_from_paths

img_paths = [
    Path("data/flowchart-images/paper-1_0.png"),
    Path("data/flowchart-images/paper-2_0.png"),
]

save_paths = [
    Path("data/rotated-flowchart-images/paper-1_0.png"),
    Path("data/rotated-flowchart-images/paper-2_0.png"),
]

rotation_labels = rotate_imgs_from_paths(
    classify_fn=classify_fn,
    img_paths=img_paths,
    save_paths=save_paths,
    json_path=Path("data/rotation-labels.json"),
)
```

`img_paths` and `save_paths` must have the same length. Each input image is
rotated and saved to the corresponding save path.

You can also rotate the explicit paths in place:

```python
rotation_labels = rotate_imgs_from_paths(
    classify_fn=classify_fn,
    img_paths=img_paths,
    json_path=Path("data/rotation-labels.json"),
    save_in_place=True,
)
```

When `save_in_place=True`, you cannot also provide `save_paths`.

### Parallelism

Both `rotate_imgs` and `rotate_imgs_from_paths` accept an `n_jobs` parameter to
control the number of parallel processes used when predicting rotation labels.
By default, they will try to use all CPU cores available.

```python
from pathlib import Path

from flowde.rotate_imgs import rotate_imgs

rotation_labels = rotate_imgs(
    classify_fn=classify_fn,
    img_dir=Path("data/flowchart-images"),
    save_dir=Path("data/rotated-flowchart-images"),
    json_path=Path("data/rotation-labels.json"),
    n_jobs=1,
)
```

## Next step

After rotation, use the parsing stage to extract structured data from the
correctly oriented flowchart images.
