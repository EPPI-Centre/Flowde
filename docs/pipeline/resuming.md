# Stopping and resuming

When you run image extraction, classification, rotation or parsing, Flowde
records the settings, input files and output files belonging to the run in the
`.flowde/run.state` file inside `save_dir`. Flowde can use this record to
continue unfinished work with `on_existing="resume"`, or to identify the
previous run's output files before replacing the run with
`on_existing="overwrite"`.

## Existing run options

| `on_existing`                                           | With a saved run in `save_dir`                                    | With a new or empty `save_dir`               |
| ------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------- |
| `"error"` <span class="default-option">(Default)</span> | Raises an error.                                                  | Starts a new run.                            |
| `"resume"`                                              | Reuses completed work and continues the run.                      | Raises an error because no saved run exists. |
| `"overwrite"`                                           | Removes the previous run's recorded outputs and starts a new run. | Starts a new run.                            |

## Resume an unfinished run

A run can stop before completion if an API request fails. After resolving the
error, you can continue the unfinished run by setting `on_existing="resume"` in
any of the core pipeline functions:
[`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs),
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs),
[`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs) and
[`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs). Each
function records completed work in the `.flowde/run.state` file inside
`save_dir` as the run progresses. Flowde uses the saved record to recover
completed work and check that the run can safely continue.

For example, using `classify_fn`, `img_dir` and `save_dir` from the
[classification example](classification.md#classify-images-from-a-directory):

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

### Checks before resuming

Before resuming, Flowde raises an error if the model, prompt or previously
processed input data have changed. This prevents a single `save_dir` from
containing results produced using different models, prompts or versions of the
input data.

#### Settings checked when resuming

In the table below, `classify_fn` and `parse_fn` are created with the
[OpenAI or Gemini helpers](setup-default-funcs.md). `extract_fn` is created with
[`make_paddle_layout_extract_fn()`](../reference/helpers.md#flowde.extract_fns.paddle_layout_detect_extraction.make_paddle_layout_extract_fn).

<!-- prettier-ignore-start -->
<!-- markdownlint-disable MD046 -->

!!! important

    When defining a custom function, you must manually declare the settings
    that cannot change when resuming a run. See
    [Declare settings](custom-functions.md#declare-settings) for how to do this.

<!-- markdownlint-enable MD046 -->
<!-- prettier-ignore-end -->

| Setting                        | Cannot change                                                                 | Applies to                                                                                                                                                                                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pipeline stage                 | Extraction, classification, rotation or parsing.                              | [`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs), [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs), [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs), [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs) |
| Prompt                         | `input_text`                                                                  | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Model                          | `model`                                                                       | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Provider                       | Choice of OpenAI or Gemini helper, and `from_azure` for OpenAI.               | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Reasoning effort               | `effort`                                                                      | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Output schema                  | The schema defined by `result_structure`, or by `parts_to_parse` for parsing. | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Few-shot examples              | `few_shot_examples`, including the example images and expected responses.     | `classify_fn`, `parse_fn`                                                                                                                                                                                                                                                                                            |
| Paddle device                  | `device`                                                                      | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| Paddle CPU threads             | `cpu_threads`                                                                 | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| Paddle batch size              | `batch_size`                                                                  | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| PDF rendering resolution       | `dpi`                                                                         | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| Image-crop padding             | `padding`                                                                     | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| Paddle software versions       | Installed PaddleOCR and PaddleX versions.                                     | `extract_fn`                                                                                                                                                                                                                                                                                                         |
| Positive classification labels | `positive_classes`                                                            | [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs)                                                                                                                                                                                                                                     |
| Custom function settings       | Declared settings.                                                            | [`model_function()`](../reference/helpers.md#flowde.model_function)                                                                                                                                                                                                                                                  |

#### Files checked when resuming

| Files                                              | Cannot change                              | Applies to                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PDFs selected from `pdf_dir`                       | PDF file paths and contents.               | [`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs)                                                                                                                                                                                                                                        |
| Images selected from `img_dir`                     | Image file paths and contents.             | [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs), [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs), [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)                                                                                |
| JSON files supplied through `nodes_dir`            | Previously parsed node numbers and text.   | [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)                                                                                                                                                                                                                                              |
| JSON files supplied through `labels_dir`           | Previously parsed node numbers and labels. | [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)                                                                                                                                                                                                                                              |
| JSON files supplied through `flow_dir`             | Previously parsed node numbers and flow.   | [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)                                                                                                                                                                                                                                              |
| JSON files supplied through `additional_texts_dir` | Previously parsed additional text.         | [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs)                                                                                                                                                                                                                                              |
| All output files in `save_dir`                     | Saved file contents.                       | [`extract_imgs()`](../reference/pipeline.md#flowde.extract_imgs.extract_imgs), [`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs), [`rotate_imgs()`](../reference/pipeline.md#flowde.rotate_imgs.rotate_imgs), [`parse_imgs()`](../reference/pipeline.md#flowde.parse_imgs.parse_imgs) |

## Overwrite an existing run

With `on_existing="overwrite"`, Flowde removes all output files recorded during
the previous run. Flowde then starts a new run with the current inputs and
settings. The previous run's saved results and usage history are cleared.
Unrecorded files or unrelated directories in `save_dir` cause an error before
any previous outputs are deleted.

## Stop a run

When a pipeline function is running in the main Python thread, you can stop
processing from the terminal:

1. Press **Ctrl+C once** to stop Flowde processing any new PDFs or images.
   Flowde finishes processing the PDFs or images it has already started, saves
   the results, then raises `KeyboardInterrupt`.
2. Press **Ctrl+C again** to force an immediate stop.
