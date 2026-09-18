# Core pipeline

The core pipeline turns PDF figures into structured flowchart data:

1. **Extract** candidate images from PDFs.
2. **Classify** the images you want to keep.
3. **Rotate** those images into the correct orientation.
4. **Parse** the corrected images into JSON.

Benchmarking is optional and follows these steps. You can also use each stage
independently. For example, start at parsing if you already have correctly
oriented flowchart images. Rotation and classification can be run in either
order.

## Functions and processing stages

Each stage combines two functions. A helper such as
[`make_openai_classify_fn()`](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)
creates a function that processes one image.
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs) then
applies that function to a directory, manages workers and
saves the results.

| Stage          | Function                                                                         |
| -------------- | -------------------------------------------------------------------------------- |
| Extraction     | [`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)    |
| Classification | [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs) |
| Rotation       | [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs)       |
| Parsing        | [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)          |

## Pipeline outputs

After running the pipeline, your output directories might look like this:

```text
results/
├── extraction/
│   ├── paper-1_0.png
│   ├── paper-2_0.png
│   └── .flowde/
├── classification/
│   ├── classifications.json
│   ├── positive_images/
│   │   └── paper-1_0.png
│   └── .flowde/
├── rotation/
│   ├── rotations.json
│   ├── rotated_images/
│   │   └── paper-1_0.png
│   └── .flowde/
└── parsing/
    ├── paper-1_0.json
    └── .flowde/
```

Each `.flowde` directory contains the saved run record (`run.state`), including
run settings and processing progress, and a lock file (`run.lock`). Keep
`.flowde` with the outputs so Flowde can resume the run.

## Next step

[Set up the default functions](setup-default-funcs.md), then follow
[image extraction](image-extraction.md). For a single worked example using the
supplied data, use the [CONSORT recipe](../recipes/consort.md).
