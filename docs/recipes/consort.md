# CONSORT: from PDFs to benchmark results

This recipe processes the four studies in the public example dataset. It uses
PaddleOCR to extract figures and OpenAI to classify, rotate and parse CONSORT
diagrams. Benchmarking then compares the parsed results with manual annotations.

For a browser-based workshop version, use
[Open demo in Google Colab](https://colab.research.google.com/github/EPPI-Centre/Flowde/blob/main/notebooks/CONSORT_demo.ipynb).
That notebook uses Azure; this recipe uses ordinary OpenAI access.

## 1. Set up the package and data

Follow the [installation guide](../installation.md) and
[OpenAI setup](../pipeline/setup-default-funcs.md#set-up-openai).
The full installation includes the dependencies used below.

Download and extract the
[repository archive](https://github.com/EPPI-Centre/Flowde/archive/refs/heads/main.zip).
The example files are under `Flowde-main/data/consort-demo`. Set `data_dir` to
that directory on your computer:

```python
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

data_dir = Path("Flowde-main/data/consort-demo")
run_dir = Path("results/consort-example")
model = "gpt-5.6-luna"
effort = "medium"
```

The dataset contains PDFs, stored Paddle crops, curated CONSORT reference
images and four-part ground-truth annotations. This recipe extracts new images
and makes live model requests; the stored images are reference material.
Read the [dataset description and article licences](https://github.com/EPPI-Centre/Flowde/blob/main/data/consort-demo/README.md)
when reusing or distributing the example files.

Use a fresh `run_dir` for a first run. To continue an interrupted stage, add
`on_existing="resume"` to that stage's call, keeping its settings unchanged.
See [managing runs](../pipeline/resuming.md) before replacing earlier results.

## 2. Extract images

```python
from flowde.extract_fns.paddle_layout_detect_extraction import (
    make_paddle_layout_extract_fn,
)
from flowde.extract_imgs import extract_imgs

extraction_dir = run_dir / "extraction"
extract_fn = make_paddle_layout_extract_fn(device="cpu", cpu_threads=1)

extract_imgs(
    pdf_dir=data_dir / "pdfs",
    save_dir=extraction_dir,
    extract_fn=extract_fn,
    n_jobs=1,
)

print("Extracted images:", len(list(extraction_dir.glob("*.png"))))
```

The PNGs are saved directly in `extraction_dir`. The first run can take longer
while PaddleOCR downloads and loads its model. For a configured GPU environment,
use `device="gpu"` as described in the
[extraction guide](../pipeline/image-extraction.md).

## 3. Select CONSORT diagrams

```python
from flowde.classify_fns.classify_types import ConsortClassification
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.classify_imgs import classify_imgs

classification_dir = run_dir / "classification"
classify_fn = make_openai_classify_fn(
    input_text=(
        "Return label 1 if the image is a CONSORT participant flow diagram. "
        "Otherwise return label 0."
    ),
    model=model,
    effort=effort,
    result_structure=ConsortClassification,
)

labels = classify_imgs(
    classify_fn=classify_fn,
    img_dir=extraction_dir,
    save_dir=classification_dir,
    positive_classes={1},
    n_jobs=1,
)

consort_dir = classification_dir / "positive_images"
print("Selected images:", len(list(consort_dir.glob("*.png"))))
```

Inspect the selected PNGs before continuing. If no images were selected, review
the extracted images and classification answers. The rotation stage needs at
least one input PNG.

## 4. Correct orientation

```python
from flowde.classify_fns.classify_types import RotationClassification
from flowde.rotate_imgs import rotate_imgs

rotation_dir = run_dir / "rotation"
rotation_fn = make_openai_classify_fn(
    input_text=(
        "Return the clockwise angle needed to orient this image correctly, "
        "so the majority of text reads left to right and top to bottom. "
        "Choose 0, 90, 180 or 270 degrees."
    ),
    model=model,
    effort=effort,
    result_structure=RotationClassification,
)

angles = rotate_imgs(
    classify_fn=rotation_fn,
    img_dir=consort_dir,
    save_dir=rotation_dir,
    n_jobs=1,
)

rotated_dir = rotation_dir / "rotated_images"
```

Every selected image has a corrected copy in `rotated_dir`. The originals in
`consort_dir` are preserved. You can instead rotate all extracted images before
classification; the Colab demo uses that ordering.

## 5. Parse node text

```python
from flowde.parse_imgs import parse_imgs
from flowde.parsing_fns.openai_parse import make_openai_parse_fn
from flowde.prompts.consort import consort_nodes_prompt

parsed_dir = run_dir / "parsed"
nodes_fn = make_openai_parse_fn(
    input_text=consort_nodes_prompt,
    model=model,
    effort=effort,
    parts_to_parse={"node_text"},
)

nodes = parse_imgs(
    parse_fn=nodes_fn,
    img_dir=rotated_dir,
    save_dir=parsed_dir / "node_text",
    n_jobs=1,
)
```

The supplied prompt explains the CONSORT interpretation conventions illustrated
in the [parsing guide](../pipeline/parsing.md#understand-the-flowchart-format).

## 6. Parse labels, flow and additional text

Use the node-text files as context so the later results refer to the same nodes:

```python
from flowde.prompts.consort_add_text import consort_add_text_prompt
from flowde.prompts.consort_flow import consort_flow_prompt
from flowde.prompts.consort_labels import consort_labels_prompt

part_prompts = {
    "labels": consort_labels_prompt,
    "flow": consort_flow_prompt,
    "additional_texts": consort_add_text_prompt,
}

for part, prompt in part_prompts.items():
    part_fn = make_openai_parse_fn(
        input_text=prompt,
        model=model,
        effort=effort,
        parts_to_parse={part},
    )
    results = parse_imgs(
        parse_fn=part_fn,
        img_dir=rotated_dir,
        save_dir=parsed_dir / part,
        nodes_dir=parsed_dir / "node_text",
        n_jobs=1,
    )
    print(f"{part}: saved {len(results)} results")
```

Each part has its own saved run. To resume this loop, use
`on_existing="resume"` for parts whose runs have already started. Parts that
have not started need the default call and a new output directory.

## 7. Combine the results

```python
from flowde.parsing_fns.parsing_types import build_a_partial_flowchart

combined_dir = parsed_dir / "combined"
combined_dir.mkdir(parents=True, exist_ok=True)

for nodes_path in sorted((parsed_dir / "node_text").glob("*.json")):
    diagram = build_a_partial_flowchart(
        nodes_path=nodes_path,
        labels_path=parsed_dir / "labels" / nodes_path.name,
        flow_path=parsed_dir / "flow" / nodes_path.name,
        additional_texts_path=parsed_dir / "additional_texts" / nodes_path.name,
    )
    (combined_dir / nodes_path.name).write_text(
        diagram.model_dump_json(indent=2), encoding="utf-8"
    )
```

The combined files contain the complete parsed diagrams. Keep both these
exports and the separate part runs if you want to inspect intermediate results
or resume processing later. Rerunning this combining loop replaces the
corresponding combined exports without making API requests.

## 8. Compare with the supplied benchmark

First identify which reference diagrams have predictions:

```python
truth_dir = data_dir / "ground-truth"
true_names = {path.stem for path in (truth_dir / "nodes").glob("*.json")}
pred_names = {path.stem for path in combined_dir.glob("*.json")}
matched_names = true_names & pred_names

print(f"Reference diagrams with predictions: {len(matched_names)} / {len(true_names)}")
print("Missing references:", sorted(true_names - pred_names))
print("Other selected figures:", sorted(pred_names - true_names))
```

Extraction versions and model choices can change the selected figures. Check
that matching filenames still refer to the same diagrams as the curated
reference images. Filenames alone cannot establish that two different crops
contain the same diagram.

The supplied ground truth covers the reference CONSORT diagrams. Evaluate only
predictions with a corresponding reference, using a temporary directory for
that subset:

```python
import shutil
from tempfile import TemporaryDirectory

from flowde.benchmarks.parsing.parsing_bench import ParsingBenchmark
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_with_nfc_and_space_normalisation,
)

if matched_names:
    with TemporaryDirectory(prefix="flowde-benchmark-") as temporary:
        prediction_dir = Path(temporary)
        for name in sorted(matched_names):
            shutil.copyfile(
                combined_dir / f"{name}.json",
                prediction_dir / f"{name}.json",
            )

        benchmark = ParsingBenchmark(
            pred_diagrams_dir=prediction_dir,
            distance_fn=levenshtein_with_nfc_and_space_normalisation,
            true_nodes_dir=truth_dir / "nodes",
            true_labels_dir=truth_dir / "labels",
            true_flow_dir=truth_dir / "flow",
            true_additional_texts_dir=truth_dir / "additional_texts",
            expected_num_diagrams=len(true_names),
            allow_missing_pred_diagrams=True,
        )
        print("Node-text cost:", benchmark.total_node_text_cost())
        print("Label cost:", benchmark.total_label_cost())
        print("Additional-text cost:", benchmark.total_additional_text_cost())
        print("Mean flow Jaccard:", benchmark.avg_flow_jaccard())
else:
    print("No reference diagrams have predictions to benchmark.")
```

Text costs count edits after normalisation; lower is better. Flow Jaccard
measures overlap in directed connections; higher is better, with `1.0` meaning
an exact edge match after matching nodes.

`allow_missing_pred_diagrams=True` excludes missing diagrams from these parsing
scores. Report the missing and extra selections printed above alongside the
scores. These parsing scores alone do not measure the complete pipeline's
ability to find every CONSORT diagram.

## 9. Keep or share the results

The run directory contains extracted and corrected images, classifications,
separate parsed parts and combined diagrams. To create an archive:

```python
import shutil

archive_path = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
print(archive_path)
```

The ZIP includes `parsed/combined/` as well as the part outputs and saved run
metadata. To work with your own PDFs, change the input directory and use prompts
appropriate to your flowcharts. Prepare your own annotations if you want to
benchmark those predictions.
