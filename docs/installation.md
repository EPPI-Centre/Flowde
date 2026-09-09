# Installation

`flowde` requires Python 3.11 or later.

## Before installation: GPU support {#gpu-support}

<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

!!! note "GPU PaddlePaddle is optional"
    You can skip this step if:

    - you are using CPU-only image extraction; or
    - you are bringing your own image extraction function.

<!-- prettier-ignore-end -->

<!-- markdownlint-disable MD046 -->

The default image extraction method uses PaddleOCR. To use this method with a
GPU, you need to install the appropriate GPU-enabled PaddlePaddle package.

The PaddlePaddle GPU installation depends on your operating system, CUDA
version, and hardware. Follow the
[PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/en/install) to
select the correct installation command.

## Install flowde

### Full installation <span class="heading-note">(recommended)</span> {#full-installation}

For most users, the easiest option is to install `flowde` with all optional
runtime features enabled.

Create and activate a virtual environment with `uv`:

```bash
uv venv
source .venv/bin/activate
```

`flowde` is currently not available on PyPI, so it should be installed directly
from the source repository:

```bash
uv pip install "flowde[all] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

This installs the core package plus all
[specific extras](#install-specific-extras) for:

- PaddleOCR image extraction functions;
- OpenAI-based parsing/classification functions;
- Gemini-based parsing/classification functions;
- benchmarking and evaluation utilities.

### Minimal installation

If you only want the core package, install:

```bash
uv pip install "flowde @ git+https://github.com/EPPI-Centre/Flowde.git"
```

This installs the base dependencies needed by the main package.

Use this option if you are bringing your own classification, parsing, and image
extraction functions and do not need the optional LLM or benchmark dependencies.

### Install specific extras

The following extras install support for specific workflows.

#### PaddleOCR helpers {paddleocr-helpers}

Install this if you want to use the default PaddleOCR based image extraction
function.

```bash
uv pip install "flowde[paddle] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

#### OpenAI helpers

Install this if you want to use the default OpenAI-based parsing or
classification functions:

```bash
uv pip install "flowde[openai] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

#### Gemini helpers

Install this if you want to use the default Gemini-based parsing or
classification functions:

```bash
uv pip install "flowde[gemini] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

#### Benchmarking

Install this if you want to use the benchmark:

```bash
uv pip install "flowde[benchmark] @ git+https://github.com/EPPI-Centre/Flowde.git"
```
