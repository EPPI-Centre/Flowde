"""
Run the demo cells offline with real file processing and simulated model answers.

Installation, downloads and Colab's download button are substituted. Absolute
`/content` paths are redirected to pytest's temporary directory, and batches use
one worker so the model substitutes stay in this process.
"""

import ast
import base64
import json
import shlex
import socket
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, Mock
from zipfile import ZipFile

import pymupdf
import pytest
from IPython.core.inputtransformer2 import TransformerManager
from PIL import Image

from flowde import _batch
from flowde.api_utils import openai_utils
from flowde.classify_fns.classify_types import (
    ConsortClassification,
    RotationClassification,
)
from flowde.extract_fns import paddle_layout_detect_extraction as paddle_extract

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/CONSORT_demo.ipynb"


class ColabPaths(ast.NodeTransformer):
    def __init__(self, content):
        self.content = content

    def visit_Constant(self, node):
        if isinstance(node.value, str) and (
            node.value == "/content" or node.value.startswith("/content/")
        ):
            relative = PurePosixPath(node.value).relative_to("/content")
            node.value = str(self.content.joinpath(*relative.parts))
        return node


@pytest.fixture
def demo_data(tmp_path):
    content = tmp_path / "content"
    data = content / "Flowde-main/data/consort-demo"
    pdfs = data / "pdfs"
    pdfs.mkdir(parents=True)
    with pymupdf.open() as document:
        page = document.new_page(width=100, height=60)
        page.draw_rect((5, 5, 45, 35), color=None, fill=(1, 0, 0))
        page.draw_rect((55, 5, 95, 35), color=None, fill=(0, 0, 1))
        document.save(pdfs / "paper.pdf")

    parts = {
        "nodes": {
            "nodes": [
                {"node_number": 1, "text": "Screened 20"},
                {"node_number": 2, "text": "Included 10"},
            ]
        },
        "labels": {
            "nodes": [
                {"node_number": 1, "labels": ["Recruitment"]},
                {"node_number": 2, "labels": []},
            ]
        },
        "flow": {
            "nodes": [
                {"node_number": 1, "points_to": [2]},
                {"node_number": 2, "points_to": []},
            ]
        },
        "additional_texts": {"additional_texts": ["Trial diagram"]},
    }
    for part, answer in parts.items():
        target = data / "ground-truth" / part / "paper_0.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"options": [answer]}), encoding="utf-8")
    (data / "manifest.json").write_text(
        json.dumps({"papers": [{"consort_images": ["paper_0.png"]}]}),
        encoding="utf-8",
    )
    return content, parts


@pytest.fixture
def offline_notebook(monkeypatch, demo_data, recognises_reference):
    content, parts = demo_data
    monkeypatch.setenv("AZURE_API_BASE", "https://workshop.invalid/v1")
    monkeypatch.setenv("AZURE_API_KEY", "not-a-real-key")
    monkeypatch.setattr("builtins.input", lambda _: "https://workshop.invalid/v1")
    monkeypatch.setattr("getpass.getpass", lambda _: "not-a-real-key")
    monkeypatch.setattr(openai_utils, "load_dotenv", lambda: False)
    monkeypatch.setattr(
        socket.socket,
        "connect",
        Mock(side_effect=AssertionError("Network access is disabled in this test")),
    )
    monkeypatch.setattr(_batch, "effective_n_jobs", lambda _: 1)

    prediction = MagicMock()
    prediction.img = {"res": Image.new("RGB", (100, 60))}
    prediction.__getitem__.return_value = [
        {"label": "image", "coordinate": [5, 5, 45, 35]},
        {"label": "image", "coordinate": [55, 5, 95, 35]},
    ]
    paddle = Mock()
    paddle.predict.return_value = [prediction]
    monkeypatch.setattr(paddle_extract, "LayoutDetection", Mock(return_value=paddle))
    paddle_extract._get_layout_detection_pipeline.cache_clear()

    requests = []

    def model_response(*, model, text_format, background, reasoning, input):  # noqa: A002 - Match the SDK's keyword argument.
        user_content = input[-1]["content"]
        image_url = next(
            p["image_url"] for p in user_content if p["type"] == "input_image"
        )
        with Image.open(BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
            red, _, blue = image.convert("RGB").getpixel(
                (image.width // 2, image.height // 2)
            )
        partial_text = [
            p["text"].split("\n", 1)[1]
            for p in user_content
            if p["type"] == "input_text"
            and p["text"].startswith(
                "The contents of the diagram that has already been parsed:\n"
            )
        ]
        previous = json.loads(partial_text[0]) if partial_text else None
        if text_format is RotationClassification:
            stage = "rotation"
            answer = {"label": 90 if red > blue else 0}
        elif text_format is ConsortClassification:
            stage = "classification"
            answer = {"label": int((red > blue) == recognises_reference)}
        else:
            if "nodes" in text_format.model_fields:
                node_fields = get_args(text_format.model_fields["nodes"].annotation)[
                    0
                ].model_fields
                stage = next(
                    part
                    for field, part in (
                        ("text", "nodes"),
                        ("labels", "labels"),
                        ("points_to", "flow"),
                    )
                    if field in node_fields
                )
            else:
                assert set(text_format.model_fields) == {"additional_texts"}
                stage = "additional_texts"
            answer = parts[stage]
            expected_previous = None
            if stage != "nodes":
                previous_parts = ["nodes"]
                if stage in {"flow", "additional_texts"}:
                    previous_parts.append("labels")
                if stage == "additional_texts":
                    previous_parts.append("flow")
                expected_previous = {
                    "nodes": [
                        {
                            key: value
                            for part in previous_parts
                            for key, value in parts[part]["nodes"][i].items()
                        }
                        for i in range(2)
                    ]
                }
            assert previous == expected_previous, (
                f"Incorrect earlier results supplied to {stage}"
            )
        requests.append(stage)
        return SimpleNamespace(output_text=json.dumps(answer), usage=None)

    client = SimpleNamespace(responses=SimpleNamespace(parse=model_response))
    monkeypatch.setattr(openai_utils, "OpenAI", Mock(return_value=client))

    def system(command):
        assert shlex.split(command)[0] in {"wget", "unzip"}, command
        return 0

    def magic(name, arguments):
        assert name == "pip", f"Unsupported notebook magic: {name}"

    shell = SimpleNamespace(system=system, run_line_magic=magic)
    downloads = Mock()
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=downloads)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {"__name__": "__main__", "get_ipython": lambda: shell}
    yield SimpleNamespace(
        content=content,
        namespace=namespace,
        downloads=downloads,
        requests=requests,
        paddle=paddle,
    )
    paddle_extract._get_layout_detection_pipeline.cache_clear()


def execute_notebook(notebook):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    transformer = TransformerManager()
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        filename = f"{NOTEBOOK}:cell-{index} ({cell.get('id', 'no-id')})"
        try:
            source = transformer.transform_cell("".join(cell["source"]))
            tree = ColabPaths(notebook.content).visit(ast.parse(source, filename))
            exec(compile(tree, filename, "exec"), notebook.namespace)  # noqa: S102 - Execute the checked-in notebook cells with offline substitutes.
        except Exception as error:
            error.add_note(f"Failed while executing {filename}")
            raise


@pytest.mark.parametrize(
    "recognises_reference", [True, False], ids=["matched-reference", "missed-reference"]
)
def test_demo_cells_run_in_order_and_can_be_repeated(
    offline_notebook, recognises_reference
):
    notebook = offline_notebook
    selected_name = "paper_0" if recognises_reference else "paper_1"
    for attempt in range(1, 3):
        execute_notebook(notebook)
        output = notebook.content / "flowde-results"
        assert sorted(p.name for p in (output / "extracted").glob("*.png")) == [
            "paper_0.png",
            "paper_1.png",
        ]
        with (
            Image.open(output / "extracted/paper_0.png") as original,
            Image.open(output / "rotation/rotated_images/paper_0.png") as rotated,
        ):
            assert rotated.size == (original.height, original.width)
        assert [
            p.stem for p in (output / "classification/positive_images").glob("*.png")
        ] == [selected_name]
        complete = notebook.namespace["complete_results"][0].model_dump()
        assert complete == {
            "nodes": [
                {
                    "node_number": 1,
                    "text": "Screened 20",
                    "labels": ["Recruitment"],
                    "points_to": [2],
                },
                {
                    "node_number": 2,
                    "text": "Included 10",
                    "labels": [],
                    "points_to": [],
                },
            ],
            "additional_texts": ["Trial diagram"],
        }
        summary = json.loads((output / "summary.json").read_text())
        assert summary["reference_diagrams"] == 1
        assert summary["recognised_reference_diagrams"] == int(recognises_reference)
        assert summary["missed_diagrams"] == (
            [] if recognises_reference else ["paper_0"]
        )
        assert summary["extra_selections"] == (
            [] if recognises_reference else ["paper_1"]
        )
        if recognises_reference:
            assert summary["node_text_edits"] == 0
            assert summary["mean_flow_jaccard"] == 1
        else:
            assert "node_text_edits" not in summary
            assert "mean_flow_jaccard" not in summary
        assert Counter(notebook.requests) == {
            "rotation": 2 * attempt,
            "classification": 2 * attempt,
            "nodes": attempt,
            "labels": attempt,
            "flow": attempt,
            "additional_texts": attempt,
        }
        assert notebook.paddle.predict.call_count == attempt
        archive = notebook.downloads.call_args.args[0]
        assert Path(archive) == output.with_suffix(".zip")
        with ZipFile(archive) as results:
            assert json.loads(results.read("summary.json")) == summary
            for part in ("node_text", "labels", "flow", "additional_texts"):
                assert f"parsed/{part}/{selected_name}.json" in results.namelist()
    assert notebook.downloads.call_count == 2
