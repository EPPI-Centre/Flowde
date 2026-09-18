# Installation

Flowde requires Python 3.11 or later.

## Full installation {#full-installation}

For CPU extraction and access to all the documented features, install:

```bash
python -m pip install "flowde[all] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

This includes PaddleOCR and CPU PaddlePaddle, OpenAI and Gemini clients, and the
benchmark dependencies. PaddleOCR downloads model files when the extractor
first runs.

The GPU version of PaddlePaddle needs a different installation; follow
[GPU support](#gpu-support) if you want extraction to use an NVIDIA GPU.

## Minimal installation

If you are bringing your own processing functions, install the core package:

```bash
python -m pip install "flowde @ git+https://github.com/EPPI-Centre/Flowde.git"
```

Add only the optional features that you need from the following sections.

## Install specific extras

### PaddleOCR helpers {#paddleocr-helpers}

The `paddle` extra installs PaddleOCR. For CPU extraction, also install the
PaddlePaddle runtime:

```bash
python -m pip install "flowde[paddle] @ git+https://github.com/EPPI-Centre/Flowde.git" paddlepaddle
```

For GPU extraction, use the runtime described under [GPU support](#gpu-support).

### OpenAI helpers

This extra supports both OpenAI and Azure OpenAI:

```bash
python -m pip install "flowde[openai] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

### Gemini helpers

```bash
python -m pip install "flowde[gemini] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

### Benchmarking

```bash
python -m pip install "flowde[benchmark] @ git+https://github.com/EPPI-Centre/Flowde.git"
```

Extras can be combined. For example, install `flowde[openai,benchmark]` if you
already have images and want to parse and evaluate them using OpenAI.

## GPU support {#gpu-support}

PaddleOCR image extraction runs on your computer. An NVIDIA GPU is used only if
your environment has a compatible GPU-enabled PaddlePaddle installation. The
LLM helpers send requests to the selected API provider; a local GPU does not
accelerate those requests.

For a GPU environment, install the Flowde extras you need without the `all`
extra, then follow the official
[PaddlePaddle installation instructions](https://www.paddlepaddle.org.cn/en/install)
to install the GPU runtime for your operating system, Python and CUDA versions.
The `all` extra includes the CPU runtime.

After installing the GPU runtime, check the environment:

```python
import paddle

print(paddle.is_compiled_with_cuda())
print(paddle.device.cuda.device_count())
```

The first value must be `True`, and the second must be at least `1`, before
using
[`make_paddle_layout_extract_fn(device="gpu")`](reference/helpers.md#flowde.extract_fns.paddle_layout_detect_extraction.make_paddle_layout_extract_fn)
.
