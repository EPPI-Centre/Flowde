# CONSORT: from PDFs to parsed diagrams

This recipe uses Gemini 3.1 Pro preview and the prompts that gave us our best
parsing performance on the training dataset. We will add the results here later.

The prompts can also be used with other models. The
[Colab demo](https://colab.research.google.com/github/EPPI-Centre/Flowde/blob/main/notebooks/CONSORT_demo.ipynb)
shows the same approach with the same prompts using Azure OpenAI.

## 1. Set up the package and data

Follow the [full installation](../installation.md#full-installation) and
[Gemini setup](../pipeline/setup-default-funcs.md#use-gemini) instructions.
Copy the code blocks into notebook cells and run the cells in order.

To try the recipe with the public example PDFs, download and extract the
[repository archive](https://github.com/EPPI-Centre/Flowde/archive/refs/heads/main.zip)
and set `data_dir` to `Flowde-main/data/consort-demo`. To use your own PDFs,
change `pdf_dir`.

```python
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

data_dir = Path("Flowde-main/data/consort-demo")
pdf_dir = data_dir / "pdfs"
run_dir = Path("results/consort-example")
model = "gemini-3.1-pro-preview"
effort = "high"
```

## 2. Extract images

```python
from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)
from flowde.extract_imgs import extract_imgs

extraction_dir = run_dir / "extraction"
extract_fn = make_paddle_layout_extract_fn(device="cpu", cpu_threads=1)

extract_imgs(
    pdf_dir=pdf_dir,
    save_dir=extraction_dir,
    extract_fn=extract_fn,
    n_jobs=1,
)
```

## 3. Select CONSORT diagrams

```python
from flowde.classify_fns.classify_types import ConsortClassification
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn
from flowde.classify_imgs import classify_imgs

classification_dir = run_dir / "classification"
classify_fn = make_gemini_classify_fn(
    input_text=(
        "Classify the image as a CONSORT image with label 1.\n"
        "If the image is not a CONSORT image, classify it with label 0."
    ),
    result_structure=ConsortClassification,
    model=model,
    effort=effort,
)

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=extraction_dir,
    save_dir=classification_dir,
    positive_classes={1},
    n_jobs=1,
)

consort_dir = classification_dir / "positive_images"
```

## 4. Correct orientation

```python
from flowde.classify_fns.classify_types import RotationClassification
from flowde.rotate_imgs import INPUT_TEXT as rotation_prompt, rotate_imgs

rotation_dir = run_dir / "rotation"
rotation_fn = make_gemini_classify_fn(
    input_text=rotation_prompt,
    result_structure=RotationClassification,
    model=model,
    effort=effort,
)

angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=consort_dir,
    save_dir=rotation_dir,
    n_jobs=1,
)

rotated_dir = rotation_dir / "rotated_images"
```

## 5. Parse the diagram parts

Parse node text first, then flow, labels and additional text, passing the
previously parsed parts as context.

### Node text

```python
from flowde.parse_imgs import parse_imgs
from flowde.parsing_fns.gemini_parse import make_gemini_parse_fn
from flowde.prompts.consort import consort_nodes_prompt

parsed_dir = run_dir / "parsed"
nodes_fn = make_gemini_parse_fn(
    input_text=consort_nodes_prompt,
    parts_to_parse={"node_text"},
    model=model,
    effort=effort,
)

nodes = parse_imgs(
    parse_fn=nodes_fn,
    img_dir=rotated_dir,
    save_dir=parsed_dir / "node_text",
    n_jobs=1,
)
```

### Flow

```python
from flowde.prompts.consort_flow import consort_flow_prompt

flow_fn = make_gemini_parse_fn(
    input_text=consort_flow_prompt,
    parts_to_parse={"flow"},
    model=model,
    effort=effort,
)

flow = parse_imgs(
    parse_fn=flow_fn,
    img_dir=rotated_dir,
    save_dir=parsed_dir / "flow",
    nodes_dir=parsed_dir / "node_text",
    n_jobs=1,
)
```

### Labels

```python
from flowde.prompts.consort_labels import consort_labels_prompt

labels_fn = make_gemini_parse_fn(
    input_text=consort_labels_prompt,
    parts_to_parse={"labels"},
    model=model,
    effort=effort,
)

node_labels = parse_imgs(
    parse_fn=labels_fn,
    img_dir=rotated_dir,
    save_dir=parsed_dir / "labels",
    nodes_dir=parsed_dir / "node_text",
    flow_dir=parsed_dir / "flow",
    n_jobs=1,
)
```

### Additional text

```python
from flowde.prompts.consort_add_text import consort_add_text_prompt

additional_texts_fn = make_gemini_parse_fn(
    input_text=consort_add_text_prompt,
    parts_to_parse={"additional_texts"},
    model=model,
    effort=effort,
)

additional_texts = parse_imgs(
    parse_fn=additional_texts_fn,
    img_dir=rotated_dir,
    save_dir=parsed_dir / "additional_texts",
    nodes_dir=parsed_dir / "node_text",
    flow_dir=parsed_dir / "flow",
    labels_dir=parsed_dir / "labels",
    n_jobs=1,
)
```

## 6. Combine the results

```python
from flowde.combine_parsed_parts import combine_parsed_parts

combined_dir = parsed_dir / "combined"
combined = combine_parsed_parts(
    nodes_dir=parsed_dir / "node_text",
    flow_dir=parsed_dir / "flow",
    labels_dir=parsed_dir / "labels",
    additional_texts_dir=parsed_dir / "additional_texts",
    save_dir=combined_dir,
)
```
