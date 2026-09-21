# Pipeline functions

All four stages save run state in the `.flowde` directory inside
`save_dir`. Each stage needs its own output directory. Do not put source files
or unrelated results in that directory. See
[stopping and resuming](../pipeline/resuming.md) for the complete recovery
rules.

## Shared parameters

| Parameter             | Meaning                                                                                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `save_dir`            | Directory for this run's results and `.flowde` records. Created when needed.                                                                                                                                             |
| `on_existing`         | `"error"` (default) rejects an existing run; `"resume"` restores a compatible run; `"overwrite"` starts a fresh run and replaces the previous run's tracked outputs. Overwrite also works with a new or empty directory. |
| `n_jobs`              | Extraction only: number of worker processes, default `1`.                                                                                                                                                                |
| `max_concurrent_jobs` | Classification, rotation and parsing: maximum image jobs in a separate process's thread pool, default `10`. Must be a positive integer.                                                                                  |
| `range_indices`       | On classification, rotation and parsing, a `(start, stop)` slice of sorted input paths, with `stop` excluded. `None` selects all paths. Extraction does not take this parameter.                                         |
| `show_usage`          | On classification, rotation and parsing, show the token and estimated cost summary. `False` hides the summary without disabling saved usage reports.                                                                     |

`max_concurrent_jobs` can change when resuming a run.

Image functions return results for the selected inputs in sorted input order.
Resuming an overlapping slice restores completed answers within that slice.
Answers outside the current slice remain saved, but are not included in the
returned list.

Input filename stems must be unique ignoring case across the run, including
inputs from earlier resumed calls. For example, `diagram.png` and `DIAGRAM.jpg`
cannot belong to the same run, even in separate calls.

## Saved run state

Keep the whole `.flowde` directory with the outputs. Its files have these roles:

| Path inside `.flowde`              | Contents                                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `run_metadata.state`               | Format version, pipeline stage, settings and input paths.                                      |
| `input_records/<stem>.state`       | One input's result, reported usage, error, completion status and output fingerprints.          |
| `shared_output_fingerprints.state` | Fingerprints for `classifications.json` or `rotations.json`; empty for parsing and extraction. |
| `run.lock`                         | Lock that prevents concurrent runs from writing to the same output directory.                  |

Progress updates rewrite only the affected input record. Run metadata is written
when inputs are first registered or added on resume. Parsing saves each result
before publishing its output JSON, so resume can recover a saved answer without
another parsing request. Classification and rotation still rewrite their combined
output JSON and its fingerprints as results are published.

On resume and overwrite, Flowde validates all expected internal records,
including records outside the selected slice. Missing or invalid records cause
an error; Flowde does not reconstruct them from output files. This differs from a missing public
output: saved image results can recreate missing JSONs or image copies, while
missing extracted PNGs require re-extracting the source PDF.

## Extraction

::: flowde.extract_imgs.extract_imgs

| Parameter    | Meaning                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `pdf_dir`    | Directory containing top-level `*.pdf` files with distinct filename stems, including when compared without letter case.                   |
| `extract_fn` | Callable accepting `pdf_path` and `save_dir` keyword arguments. The callable writes top-level PNGs into the supplied temporary directory. |

Returns `None`. After an entire PDF succeeds, Flowde publishes that PDF's PNGs
into the run's `save_dir`. The run record includes successful PDFs that produced
no images. Missing PNGs cause the whole source PDF to be extracted again on
resume; modified saved PNGs cause an error.

For the Paddle factory's crop names and settings, see
[image extraction](../pipeline/image-extraction.md).

## Classification

::: flowde.classify_imgs.classify_imgs

See the [classification guide](../pipeline/classification.md) for worked
examples.

## Rotation

::: flowde.rotate_imgs.rotate_imgs

See the [rotation guide](../pipeline/rotation.md) for worked examples.

## Parsing

::: flowde.parse_imgs.parse_imgs

See the [parsing guide](../pipeline/parsing.md) for full, partial and
custom-schema examples.
