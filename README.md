# Flowde

[![Tests](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml?query=branch%3Amain)
[![Pre-commit](https://github.com/EPPI-Centre/Flowde/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/EPPI-Centre/Flowde/actions/workflows/pre-commit.yml?query=branch%3Amain)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/EPPI-Centre/Flowde/badges/coverage.svg)](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml?query=branch%3Amain)

Flowde is a Python package for extracting structured data from flowcharts.
Developed for CONSORT diagrams in systematic reviews, Flowde also supports other
types of flowchart.

[![Open demo in Google
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EPPI-Centre/Flowde/blob/main/notebooks/CONSORT_demo.ipynb)

Try the complete CONSORT workflow in your browser. The demo requires a Google
account, an Azure OpenAI endpoint and an API key.

**[Read the documentation](https://eppi-centre.github.io/Flowde/)** for setup,
usage guides and the API reference.

## Features

- **Extract** candidate images from PDFs with PaddleOCR.
- **Classify** images and keep the diagrams you need.
- **Rotate** diagrams into the correct orientation.
- **Parse** node text, labels, flow and additional text into structured JSON.
- **Benchmark** classification, rotation and parsing against ground truth.

Use the full pipeline or individual stages, with OpenAI, Azure OpenAI, Gemini or
your own processing functions. Flowde saves completed work and supports stopping
and resuming runs. The package includes prompts and example data for CONSORT
diagrams.

## Installation

Requires Python 3.11 or later. To install all features with CPU extraction:

```bash
python -m pip install "flowde[all] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

See the [installation guide](https://eppi-centre.github.io/Flowde/installation/)
for individual features and GPU setup.

## Documentation

- [Pipeline guide](https://eppi-centre.github.io/Flowde/pipeline/): work through
  extraction, classification, rotation and parsing.
- [CONSORT recipe](https://eppi-centre.github.io/Flowde/recipes/consort/): copy
  a complete workflow using the supplied prompts.
- [API reference](https://eppi-centre.github.io/Flowde/reference/): inspect
  functions, parameters and result types.
- [Example data](data/consort-demo/README.md): view the included studies and
  article licences.

For bug reports and feature requests,
[open an issue](https://github.com/EPPI-Centre/Flowde/issues).
