# Set up the default functions

Flowde handles directories, saved results, parallel workers and benchmarking.
The function supplied to each pipeline stage does the actual extraction,
classification or parsing. You can use a default helper or
[bring your own function](custom-functions.md).

## Set up image extraction

Install the [PaddleOCR dependencies](../installation.md#paddleocr-helpers), then
create an extractor:

```python
from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)

extract_fn = make_paddle_layout_extract_fn(device="cpu", cpu_threads=1)
```

The layout model is loaded when extraction starts. The first run may download
model files. Use `device="gpu"` only after completing the
[GPU setup](../installation.md#gpu-support).

See [image extraction](image-extraction.md) for the function that processes your
PDF directory.

## Set up OpenAI

Install the [OpenAI extra](../installation.md#openai-helpers). In the directory
where you will run your script or notebook, create a `.env` file containing:

```text
OPENAI_API_KEY=your-openai-api-key
```

Load that file explicitly at the start of your example:

```python
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))
```

Flowde's helpers also load environment settings. Explicitly loading the file
above makes its location clear when your script and working directory differ.

With your API key loaded, you can create functions that use OpenAI's Responses
API to classify, rotate or parse images:

<!-- markdownlint-disable MD033 -->

| Stage                               | Create the function with                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Classification](classification.md) | <code class="function-call">[make_openai_classify_fn](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)(...)</code>                                                                                                                                                                                              |
| [Rotation](rotation.md)             | <code class="function-call">[make_openai_classify_fn](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)(<br>&nbsp;&nbsp;&nbsp;&nbsp;...,<br>&nbsp;&nbsp;&nbsp;&nbsp;result_structure=[RotationClassification](../reference/data-types.md#flowde.classify_fns.classify_types.RotationClassification),<br>)</code> |
| [Parsing](parsing.md)               | <code class="function-call">[make_openai_parse_fn](../reference/helpers.md#flowde.parsing_fns.openai_parse.make_openai_parse_fn)(...)</code>                                                                                                                                                                                                           |

<!-- markdownlint-enable MD033 -->

For example, create a classifier:

```python
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn

classify_fn = make_openai_classify_fn(
    input_text="Return 1 if the image is a flowchart, otherwise return 0.",
    model="gpt-5.6-luna",
    effort="medium",
)
```

Creating the function does not classify any images. Pass `classify_fn` to
[`classify_imgs()`](../reference/pipeline.md#flowde.classify_imgs.classify_imgs) to run
the requests. Model requests are
billable through your provider account.

## Use Azure OpenAI

Install the [OpenAI extra](../installation.md#openai-helpers). In the directory
where you will run your script or notebook, create a `.env` file containing:

```text
AZURE_API_KEY=your-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/openai/v1/
```

Load the `.env` file as shown in [Set up OpenAI](#set-up-openai), then use
`from_azure=True` to send requests through Azure:

<!-- markdownlint-disable MD033 -->

| Stage                               | Create the function with                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Classification](classification.md) | <code class="function-call">[make_openai_classify_fn](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)(<br>&nbsp;&nbsp;&nbsp;&nbsp;...,<br>&nbsp;&nbsp;&nbsp;&nbsp;from_azure=True,<br>)</code>                                                                                                                                                             |
| [Rotation](rotation.md)             | <code class="function-call">[make_openai_classify_fn](../reference/helpers.md#flowde.classify_fns.openai_classify_fn.make_openai_classify_fn)(<br>&nbsp;&nbsp;&nbsp;&nbsp;...,<br>&nbsp;&nbsp;&nbsp;&nbsp;result_structure=[RotationClassification](../reference/data-types.md#flowde.classify_fns.classify_types.RotationClassification),<br>&nbsp;&nbsp;&nbsp;&nbsp;from_azure=True,<br>)</code> |
| [Parsing](parsing.md)               | <code class="function-call">[make_openai_parse_fn](../reference/helpers.md#flowde.parsing_fns.openai_parse.make_openai_parse_fn)(<br>&nbsp;&nbsp;&nbsp;&nbsp;...,<br>&nbsp;&nbsp;&nbsp;&nbsp;from_azure=True,<br>)</code>                                                                                                                                                                          |

<!-- markdownlint-enable MD033 -->

For example, create a classifier:

```python
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn

classify_fn = make_openai_classify_fn(
    input_text="Return 1 if the image is a flowchart, otherwise return 0.",
    model="your-deployment-name",
    effort="medium",
    from_azure=True,
)
```

## Use Gemini

Install the [Gemini extra](../installation.md#gemini-helpers). In the directory
where you will run your script or notebook, create a `.env` file containing:

```text
GOOGLE_GENAI_API_KEY=your-gemini-api-key
```

Load the `.env` file as shown in [Set up OpenAI](#set-up-openai), then create
functions that use the Gemini API to classify, rotate or parse images:

<!-- markdownlint-disable MD033 -->

| Stage                               | Create the function with                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Classification](classification.md) | <code class="function-call">[make_gemini_classify_fn](../reference/helpers.md#flowde.classify_fns.gemini_classify_fn.make_gemini_classify_fn)(...)</code>                                                                                                                                                                                              |
| [Rotation](rotation.md)             | <code class="function-call">[make_gemini_classify_fn](../reference/helpers.md#flowde.classify_fns.gemini_classify_fn.make_gemini_classify_fn)(<br>&nbsp;&nbsp;&nbsp;&nbsp;...,<br>&nbsp;&nbsp;&nbsp;&nbsp;result_structure=[RotationClassification](../reference/data-types.md#flowde.classify_fns.classify_types.RotationClassification),<br>)</code> |
| [Parsing](parsing.md)               | <code class="function-call">[make_gemini_parse_fn](../reference/helpers.md#flowde.parsing_fns.gemini_parse.make_gemini_parse_fn)(...)</code>                                                                                                                                                                                                           |

<!-- markdownlint-enable MD033 -->

For example, create a classifier:

```python
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

classify_fn = make_gemini_classify_fn(
    input_text="Return 1 if the image is a flowchart, otherwise return 0.",
    model="gemini-3.1-flash-lite",
    effort="high",
)
```

## Next step

[Extract images](image-extraction.md), or start with
[classification](classification.md) if you already have PNGs. The
[API reference](../reference/helpers.md) lists the helper parameters.
