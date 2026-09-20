# Flowde

Flowde extracts structured data from flowcharts. It was developed for extracting
information from CONSORT diagrams in systematic reviews, and can also be used
with other flowcharts. The supplied prompts and example dataset focus on
CONSORT diagrams.

The package takes you from PDFs to images, from images to structured JSON, and
optionally from predictions to benchmark scores:

[![Flowde workflow: extract figures from PDFs, select relevant flowcharts,
correct their orientation, parse to JSON, and optionally compare with ground
truth.](images/flowde-workflow.svg)](images/flowde-workflow.svg)
{ .workflow-graphic }

You can use the default PaddleOCR and LLM helpers, or provide your own
functions. Each stage saves its results and supports resuming work after an
interruption.

## Start here

1. [Install Flowde](installation.md).
2. [Set up the default functions](pipeline/setup-default-funcs.md).
3. Follow the [core pipeline](pipeline/index.md) with your own files, or run the
   complete [CONSORT example](recipes/consort.md).

To try the workshop demo in a browser, use
[Open demo in Google Colab](https://colab.research.google.com/github/EPPI-Centre/Flowde/blob/main/notebooks/CONSORT_demo.ipynb).
The notebook uses Azure OpenAI and asks for the endpoint and API key.

## The pipeline

| Stage                                            | Input                                        | Result                                       |
| ------------------------------------------------ | -------------------------------------------- | -------------------------------------------- |
| [Image extraction](pipeline/image-extraction.md) | PDFs                                         | Candidate PNG images                         |
| [Classification](pipeline/classification.md)     | Images                                       | Labels and selected image copies             |
| [Rotation](pipeline/rotation.md)                 | Images                                       | Correction angles and corrected image copies |
| [Parsing](pipeline/parsing.md)                   | Images, optionally with earlier parsed parts | Structured JSON                              |
| [Benchmarking](benchmarking/classification.md)   | Predictions and ground truth                 | Scores and detailed matches                  |

If you already have flowchart images, start with rotation or parsing.
Classification and rotation can be performed in either order.

See [managing runs](pipeline/resuming.md) for stopping, resuming and
overwriting, and [usage reporting](pipeline/usage.md) for tokens and estimated
costs. The [API reference](reference/index.md) describes function parameters and
results.
