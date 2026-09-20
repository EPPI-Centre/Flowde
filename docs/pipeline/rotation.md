# Image rotation

Rotation predicts the correction needed to orient an image correctly and saves
a corrected copy. In the core pipeline, rotation follows
[classification](classification.md) and prepares the selected images for
parsing.

## Rotate images from a directory

[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)
takes a directory of images and saves correction angles and corrected copies
into one dedicated output directory.

```python
from pathlib import Path

from flowde.classify_fns.classify_types import RotationClassification
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.rotate_imgs import rotate_imgs

img_dir = Path("results/classification/positive_images")
save_dir = Path("results/rotation")

rotation_fn = make_openai_classify_fn(
    input_text=(
        "Return the clockwise angle needed to orient this image correctly, "
        "so the majority of text reads left to right and top to bottom. "
        "Return one of 0, 90, 180 or 270 degrees."
    ),
    model="gpt-5.6-luna",
    effort="medium",
    result_structure=RotationClassification,
)

angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=img_dir,
    save_dir=save_dir,
    max_concurrent_jobs=1,
)
```

[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)
processes `*.png` files inside `img_dir`, in sorted path order.
Subdirectories are not searched. The returned `angles` list contains one
correction angle per processed image: `angles[0]` corresponds to the first
sorted image path, `angles[1]` to the second sorted image path, and so on.

The example above uses OpenAI and requires the
[OpenAI setup](setup-default-funcs.md#set-up-openai).
For `classify_fn`, you can use an OpenAI classifier created with
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn),
a Gemini classifier created with
[`make_gemini_classify_fn()`](../reference/helpers.md#flowde.classify_fns.gemini_classify_fn.make_gemini_classify_fn),
or [your own rotation classifier](custom-functions.md#a-rotation-function).
The [provider setup guide](setup-default-funcs.md#use-azure-openai) also covers
Azure OpenAI.

The example sets `result_structure` to
[`RotationClassification`](../reference/data-types.md#flowde.classify_fns.classify_types.RotationClassification)
to restrict the classifier's labels to `0`, `90`, `180` or `270`. These labels
specify **clockwise corrections in degrees**: `90` means rotate the input image
clockwise by 90 degrees.

You can also rotate extracted images before classification by using the
extraction output directory as `img_dir`.

<!-- prettier-ignore-start -->
<!-- markdownlint-disable MD046 -->

!!! note "Note: concurrency"

    `max_concurrent_jobs` controls how many calls to `classify_fn` can run
    concurrently and defaults to `10`. Parallel execution in
    [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)
    uses threads, allowing concurrency to greatly exceed the CPU core count
    when waiting for API responses. Choose `max_concurrent_jobs` based on your
    API limits.

<!-- markdownlint-enable MD046 -->
<!-- prettier-ignore-end -->

See the
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)
and
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)
API references for full details of all parameters.

## Rotation results

Flowde saves the correction angles and corrected image copies:

```text
results/rotation/
├── rotations.json
├── rotated_images/
│   ├── paper-1_0.png
│   └── paper-2_0.png
└── .flowde/
    ├── run.state
    └── run.lock
```

`rotations.json` records the input image path and correction angle:

```json
[
  {
    "img_path": "results/classification/positive_images/paper-1_0.png",
    "label": 0
  },
  {
    "img_path": "results/classification/positive_images/paper-2_0.png",
    "label": 90
  }
]
```

Every successfully processed image has a copy in
`save_dir / "rotated_images"`, including images labelled `0` that need no
rotation. The copies retain their filenames, and the original images in
`img_dir` remain unchanged.

`.flowde/run.state` records which images have been processed, their correction
angles, the rotation settings and file fingerprints. Flowde uses this metadata
to resume the run and detect changes to the input images or saved results.

## Resume rotation

To continue an unfinished rotation run whose results are saved in `save_dir`,
you can set `on_existing="resume"`:

```python
angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=img_dir,
    save_dir=save_dir,
    max_concurrent_jobs=1,
    on_existing="resume",
)
```

The `.flowde/run.state` file inside `save_dir` stores the rotation run's state.
Flowde uses this record to resume unfinished work and check the integrity of the run.
Resume raises an error if the classifier's settings have changed, or if
previously processed input images or saved outputs have been edited.

To use different settings, you can start a run in a new `save_dir` or replace
the previous run with `on_existing="overwrite"`. See
[managing runs](resuming.md) for the full rules.

## Bring your own rotation function

You can write your own rotation classifier and pass it to
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)
as `classify_fn`. See the
[custom rotation tutorial](custom-functions.md#a-rotation-function)
for the requirements your function must meet and a complete working example.

## Next step

You can use `save_dir / "rotated_images"` as the input directory for
[image parsing](parsing.md).
