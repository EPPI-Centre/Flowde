# Benchmarking

Benchmarking compares saved predictions with manually annotated ground truth.
It is optional: you can extract and parse flowcharts without running a
benchmark.

The benchmark classes read JSON files and calculate results locally. They do
not call the classification or parsing model again.

## Choose a benchmark

| Task                      | Class                               | Predictions                                                | Ground truth                                                     |
| ------------------------- | ----------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Binary classification     | `BinaryClassificationBenchmark`     | `classifications.json` with integer `0`/`1` labels         | JSON with the same images and reviewed labels                    |
| Multiclass classification | `MulticlassClassificationBenchmark` | `classifications.json`                                     | JSON with the same images and reviewed class labels              |
| Rotation                  | `RotationBenchmark`                 | `rotations.json`                                           | JSON with the clockwise correction for each original input image |
| Parsing                   | `ParsingBenchmark`                  | Combined diagram JSONs or separate parsed-part directories | Four directories containing accepted Ground-Truth Options        |

Install the [benchmark extra](../installation.md#benchmarking) before using
these classes.

## Prepare comparable inputs

Classification and rotation compare the image filenames in two JSON lists.
The lists must have the same length and the same filenames in the same order;
the parent directories can differ. A saved run may contain answers from
several resumed slices, so use the complete saved JSON when preparing the
matching ground truth.

Parsing compares diagrams by filename stem. Each ground-truth diagram needs
node text, labels, flow and additional-text files with aligned `options` lists.
The predictions can contain a subset of the supported fields, but must include
node text. The available scoring methods depend on the predicted fields.

Ground-truth annotations should be prepared independently of the predictions
you are evaluating. If a diagram has multiple accepted interpretations, record
the alternatives as Ground-Truth Options rather than changing the ground truth
to match one model's answer.

## Read scores alongside the evaluated sample

A classification score describes the supplied images, and a parsing score
describes the supplied diagrams. Report the number of evaluated examples with
the scores.

For an end-to-end pipeline, also report diagrams missed during extraction or
classification. A parsing benchmark restricted to successfully selected
diagrams does not measure those earlier failures. In particular,
`allow_missing_pred_diagrams=True` evaluates available predictions; missing
diagrams are not assigned a failure score.

## Next step

- [Classification benchmarking](classification.md): accuracy, precision,
  recall, confusion matrices and class-level results.
- [Rotation benchmarking](rotation.md): correct correction angles and which
  angles are confused.
- [Parsing benchmarking](parsing.md): input files, text costs and flow scores.
- [Parsing matching](matching.md): Node Matches and Ground-Truth Option selection.

The [benchmark API reference](../reference/benchmarks.md) lists the public
classes, methods and result objects.
