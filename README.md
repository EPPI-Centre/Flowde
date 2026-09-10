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
and API key. The notebook runs in Google Colab; no installation on your computer
is needed.

[Example data and article licences](data/consort-demo/README.md)
