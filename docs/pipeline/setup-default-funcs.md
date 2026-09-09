# Setup default pipeline functions

`flowde` contains a pipeline for extracting, classifying, rotating and parsing
images from pdfs and evaluating the performance of each step. There are two main
ways to use the package:

- **Use Case 1:** bring your own extraction, classification and parsing
  functions and use `flowde` to manage the files, parallelise the functions and
  evaluate the performance.
- **Use Case 2:** use the package as an end-to-end pipeline for extracting,
  classifying, rotating, parsing and evaluating performance by using all the
  default functions.

If you intend to use the package for **Use Case 2**, you will need to setup the
default functions by following the steps on this page.

## Setup default image extraction

The default image extraction task function is created with
`make_paddle_layout_extract_fn`.

Before using it, install the required PaddleOCR dependencies.

<div class="indent-section" markdown>

### 1. Install the extraction dependencies

Choose one of the following options:

- use the [full installation](../installation.md#full-installation), which
  installs all optional extras; or
- install only the
  [PaddleOCR-specific dependencies](../installation.md#paddleocr-helpers).

### 2. Optional: enable GPU support

If you want to run the default PaddleOCR extractor with a GPU, follow the
[GPU support instructions](../installation.md#gpu-support).

You can skip this step if you are using CPU-only extraction.

</div>

## Setup default parsing and classification

The default parsing and classification functions rely on making API calls to
OpenAI and Gemini.

You can run the entire pipeline by using only OpenAI functions, only Gemini
functions, or interchanging between the two.

<div class="indent-section" markdown>

### 1. Install the dependencies

Choose one of the following options:

- use the [full installation](../installation.md#full-installation), which
  installs all optional extras; or
- install the [OpenAI dependencies](../installation.md#openai-helpers) if you
  wish to use only the OpenAI default functions; or
- install the [Gemini dependencies](../installation.md#gemini-helpers) if you
  wish to use only the Gemini default functions

### 2. Add a `.env`

In either your current working directory, or at the root of your workspace,
create a `.env` file with your OpenAI and Gemini API keys.

```bash
# If you are using OpenAI:
OPENAI_API_KEY=your-openai-api-key

# If you are using Gemini:
GEMINI_API_KEY=your-gemini-api-key
```

</div>
