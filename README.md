# Flowde

[![Tests](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml?query=branch%3Amain)
[![Pre-commit](https://github.com/EPPI-Centre/Flowde/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/EPPI-Centre/Flowde/actions/workflows/pre-commit.yml?query=branch%3Amain)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/EPPI-Centre/Flowde/actions/workflows/tests.yml)
[![Coverage](https://eppi-centre.github.io/Flowde/badges/coverage.svg)](https://github.com/EPPI-Centre/Flowde/actions)
[![License](https://img.shields.io/github/license/EPPI-Centre/Flowde.svg?label=License)](https://github.com/EPPI-Centre/Flowde/blob/main/LICENSE)

This repo contains the code for extracting structured data from CONSORT flow
diagrams in PDF files reporting randomized trials.

**Try the complete CONSORT demo in your browser:**

[![Open demo in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EPPI-Centre/Flowde/blob/main/notebooks/CONSORT_demo.ipynb)

Follow six open-access studies from PDFs to extracted images, orientation
correction, CONSORT classification, structured parsing and evaluation against
manual annotations. Requires a Google account and the workshop's Azure endpoint
and API key. No installation on your computer is needed. The notebook also works
locally with Azure settings in your `.env` file.

[Example data and article licences](data/consort-demo/README.md)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Setup](#setup)
  - [Create environment](#create-environment)
  - [Install](#install)
  - [Assign OpenAI API Key](#assign-openai-api-key)
- [Quickstart](#quickstart)
  - [Extract Figures From PDF](#extract-figures-from-pdf)
  - [Extract CONSORT From Images Dir](#extract-consort-from-images-dir)
  - [Parse CONSORT From Images Dir](#parse-consort-from-images-dir)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Setup

### Create environment

```bash
conda create -n flow python==3.11.11 -y
conda activate flow
```

### Install

```bash
git clone https://github.com/EPPI-Centre/Flowde.git
cd Flowde
pip install -e .
```

### Assign OpenAI API Key

In the root of this repo, you must create a file called `.env`. In this file you
will register your OpenAI API key as so:

```txt
OPENAI_API_KEY=COPY_AND_PASTE_YOUR_API_KEY_HERE
```

## Quickstart

### Extract Figures From PDF

**Windows (Powershell):**

```powershell
$Env:OUTPUT_IMAGE_FORMAT = "PNG"
marker --output_dir OUTPUT_DIR INPUT_DIR
```

**Mac/Linux:**

```bash
export OUTPUT_IMAGE_FORMAT="PNG"
marker --output_dir OUTPUT_DIR INPUT_DIR
```

### Extract CONSORT From Images Dir

```bash
python classify_images_as_flowchart.py
```

### Parse CONSORT From Images Dir

```bash
python parse_flowchart_images.py
```
