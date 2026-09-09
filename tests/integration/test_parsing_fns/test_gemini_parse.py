import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw

import flowde.parsing_fns.gemini_parse as gemini_parse_fn_module
from flowde.parsing_fns.parsing_types import build_partial_flowchart_schema

pytestmark = pytest.mark.integration


def get_required_env_var(name: str) -> str:
    load_dotenv()
    value = os.getenv(name)

    if value is None:
        raise AssertionError(f"{name} must be set to run this integration test.")

    return value


def create_simple_flowchart_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (500, 350), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((170, 30, 330, 80), outline="black", width=3)
    draw.rectangle((170, 140, 330, 190), outline="black", width=3)
    draw.rectangle((170, 250, 330, 300), outline="black", width=3)

    draw.text((215, 48), "Start", fill="black")
    draw.text((205, 158), "Process", fill="black")
    draw.text((225, 268), "End", fill="black")

    draw.line((250, 80, 250, 140), fill="black", width=3)
    draw.polygon([(250, 140), (242, 128), (258, 128)], fill="black")

    draw.line((250, 190, 250, 250), fill="black", width=3)
    draw.polygon([(250, 250), (242, 238), (258, 238)], fill="black")

    img.save(path)


def create_marker_image(path: Path, marker_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (700, 250), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((30, 30, 670, 220), outline="black", width=4)
    draw.text((80, 105), marker_text, fill="black")

    img.save(path)


def make_partial_flowchart(parts_to_parse: set[str], content: dict[str, Any]):
    Schema = build_partial_flowchart_schema(parts_to_parse=parts_to_parse)
    return Schema.model_validate(content)


@pytest.mark.parametrize(
    ("parts_to_parse", "partial_content"),
    [
        pytest.param(
            {"node_text"},
            {
                "nodes": [
                    {"node_number": 1, "text": "Assessed for eligibility"},
                    {"node_number": 2, "text": "Randomised"},
                ],
            },
            id="node-text",
        ),
        pytest.param(
            {"labels"},
            {
                "nodes": [
                    {"node_number": 1, "labels": ["screening", "eligibility"]},
                    {"node_number": 2, "labels": ["randomisation", "allocation"]},
                ],
            },
            id="labels",
        ),
        pytest.param(
            {"flow"},
            {
                "nodes": [
                    {"node_number": 1, "points_to": [2]},
                    {"node_number": 2, "points_to": []},
                ],
            },
            id="flow",
        ),
        pytest.param(
            {"additional_texts"},
            {
                "additional_texts": [
                    "Figure 1",
                    "Participant flow diagram",
                ],
            },
            id="additional-text",
        ),
        pytest.param(
            {"node_text", "labels"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "labels": ["screening", "eligibility"],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "labels": ["randomisation", "allocation"],
                    },
                ],
            },
            id="node-text-and-labels",
        ),
        pytest.param(
            {"node_text", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "points_to": [],
                    },
                ],
            },
            id="node-text-and-flow",
        ),
        pytest.param(
            {"labels", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "labels": ["screening", "eligibility"],
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "labels": ["randomisation", "allocation"],
                        "points_to": [],
                    },
                ],
            },
            id="labels-and-flow",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "labels": ["screening", "eligibility"],
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "labels": ["randomisation", "allocation"],
                        "points_to": [],
                    },
                ],
            },
            id="node-text-labels-and-flow",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Assessed for eligibility",
                        "labels": ["screening", "eligibility"],
                        "points_to": [2],
                    },
                    {
                        "node_number": 2,
                        "text": "Randomised",
                        "labels": ["randomisation", "allocation"],
                        "points_to": [],
                    },
                ],
                "additional_texts": [
                    "Figure 1",
                    "Participant flow diagram",
                ],
            },
            id="all-parts",
        ),
    ],
)
def test_gemini_parse_fn_receives_and_returns_partial_flowchart_parts(
    tmp_path,
    parts_to_parse,
    partial_content,
):
    get_required_env_var("GOOGLE_API_KEY")

    img_path = tmp_path / "simple_flowchart.png"
    create_simple_flowchart_image(img_path)

    partial_flowchart = make_partial_flowchart(
        parts_to_parse=parts_to_parse,
        content=partial_content,
    )

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text=(
            "You are being given an image and a partial flowchart JSON object. "
            "For this integration test, ignore the image content. "
            "Return exactly the partial flowchart JSON object that has already "
            "been parsed. Do not infer, add, remove, rename, reorder, or modify "
            "any fields or values. Return only JSON matching the response schema."
        ),
        model="gemini-3.5-flash-lite",
        effort="low",
        parts_to_parse=parts_to_parse,
    )

    result = parse_fn(
        img_path=img_path,
        partial_flowchart=partial_flowchart,
    )

    assert result.model_dump() == partial_content


def test_gemini_parse_fn_uses_full_schema_when_parts_to_parse_is_none(tmp_path):
    get_required_env_var("GOOGLE_API_KEY")

    img_path = tmp_path / "simple_flowchart.png"
    create_simple_flowchart_image(img_path)

    partial_content = {
        "nodes": [
            {
                "node_number": 1,
                "text": "Assessed for eligibility",
                "labels": ["screening", "eligibility"],
                "points_to": [2],
            },
            {
                "node_number": 2,
                "text": "Randomised",
                "labels": ["randomisation", "allocation"],
                "points_to": [],
            },
        ],
        "additional_texts": [
            "Figure 1",
            "Participant flow diagram",
        ],
    }

    partial_flowchart = make_partial_flowchart(
        parts_to_parse={"node_text", "labels", "flow", "additional_texts"},
        content=partial_content,
    )

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text=(
            "You are being given an image and a partial flowchart JSON object. "
            "For this integration test, ignore the image content. "
            "Return exactly the partial flowchart JSON object that has already "
            "been parsed. Do not infer, add, remove, rename, reorder, or modify "
            "any fields or values. Return only JSON matching the response schema."
        ),
        model="gemini-3.5-flash-lite",
        effort="low",
        parts_to_parse=None,
    )

    result = parse_fn(
        img_path=img_path,
        partial_flowchart=partial_flowchart,
    )

    assert result.model_dump() == partial_content


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        pytest.param(
            "gemini-3.5-flash-lite",
            "medium",
            id="gemini-3-5-flash-lite-medium",
        ),
        pytest.param(
            "gemini-3-flash-preview",
            "low",
            id="gemini-3-flash-preview-low",
        ),
    ],
)
def test_gemini_parse_fn_receives_input_text_model_and_effort(
    tmp_path,
    model,
    effort,
):
    get_required_env_var("GOOGLE_API_KEY")

    img_path = tmp_path / "simple_flowchart.png"
    create_simple_flowchart_image(img_path)

    input_text = (
        "For this integration test, return the following values exactly in the "
        "additional_texts field, in this exact order: "
        "input_text_marker=GEMINI_INPUT_TEXT_WAS_RECEIVED, "
        f"model_marker={model}, "
        f"effort_marker={effort}. "
        "Ignore the image. Return only JSON matching the response schema."
    )

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text=input_text,
        model=model,
        effort=effort,
        parts_to_parse={"additional_texts"},
    )

    result = parse_fn(img_path=img_path)

    assert result.model_dump() == {
        "additional_texts": [
            "input_text_marker=GEMINI_INPUT_TEXT_WAS_RECEIVED",
            f"model_marker={model}",
            f"effort_marker={effort}",
        ],
    }


def test_gemini_parse_fn_receives_correct_image(tmp_path):
    get_required_env_var("GOOGLE_API_KEY")

    marker_text = "IMAGE_MARKER_ABC123"
    img_path = tmp_path / "marker_image.png"
    create_marker_image(img_path, marker_text)

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text=(
            "Read the marker text shown in the image. "
            "Return it exactly as the only item in additional_texts. "
            "Return only JSON matching the response schema."
        ),
        model="gemini-3.5-flash-lite",
        effort="low",
        parts_to_parse={"additional_texts"},
    )

    result = parse_fn(img_path=img_path)

    assert result.model_dump() == {
        "additional_texts": [marker_text],
    }
