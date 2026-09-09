"""Live requests with few-shot examples and ordinary requests without them."""

import json
from functools import partial
from typing import Literal
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from flowde.classify_fns.classify_types import Classification
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn
from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.parsing_fns.gemini_parse import make_gemini_parse_fn
from flowde.parsing_fns.openai_parse import make_openai_parse_fn
from flowde.parsing_fns.parsing_types import build_partial_flowchart_schema
from flowde.utils import VISION_FEW_SHOT_INSTRUCTIONS, VisionFewShotExample

pytestmark = pytest.mark.integration

BLUE = (70, 150, 240)
ORANGE = (255, 170, 40)


class ColourClassification(Classification[str]):
    """Keep the label unrestricted and give OpenAI a valid response-schema name."""


class TranscriptMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RequestTranscript(BaseModel):
    messages: list[TranscriptMessage]


def write_colour_image(path, colour, *, target=False):
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    if target:
        draw.ellipse((160, 70, 360, 270), fill=colour)
    else:
        draw.rectangle((40, 40, 250, 250), fill=colour)
    image.save(path)


def write_two_node_diagram(path, top_text, bottom_text, top_colour, bottom_colour):
    image = Image.new("RGB", (500, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    draw.rectangle((80, 30, 420, 130), fill=top_colour, outline="black", width=3)
    draw.text((250, 80), top_text, fill="black", font=font, anchor="mm")
    draw.line((250, 130, 250, 220), fill="black", width=4)
    draw.polygon([(250, 220), (240, 205), (260, 205)], fill="black")
    draw.rectangle((80, 220, 420, 320), fill=bottom_colour, outline="black", width=3)
    draw.text((250, 270), bottom_text, fill="black", font=font, anchor="mm")
    image.save(path)


@pytest.mark.parametrize(
    ("make_classify_fn", "model"),
    [
        pytest.param(make_openai_classify_fn, "gpt-5.6-luna", id="openai"),
        pytest.param(
            partial(make_openai_classify_fn, from_azure=True),
            "gpt-5.6-luna",
            id="azure-openai",
        ),
        pytest.param(make_gemini_classify_fn, "gemini-3.5-flash-lite", id="gemini"),
    ],
)
def test_classification_uses_the_matching_example_answer(
    tmp_path, make_classify_fn, model
):
    # These arbitrary labels exist only in the example answers, never the prompt,
    # image filenames, image pixels, or response schema.
    blue_label = uuid4().hex
    orange_label = uuid4().hex
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    target_image = tmp_path / "target.png"
    write_colour_image(first_image, BLUE)
    write_colour_image(second_image, ORANGE)
    write_colour_image(target_image, BLUE, target=True)

    first_answer = tmp_path / "first.json"
    second_answer = tmp_path / "second.json"
    first_answer.write_text(json.dumps({"label": blue_label}), encoding="utf-8")
    second_answer.write_text(json.dumps({"label": orange_label}), encoding="utf-8")

    classify = make_classify_fn(
        input_text=("Classify the colour of the large filled shape."),
        model=model,
        effort="low",
        result_structure=ColourClassification,
        few_shot_examples=[
            VisionFewShotExample(
                img_path=first_image, expected_output_path=first_answer
            ),
            VisionFewShotExample(
                img_path=second_image, expected_output_path=second_answer
            ),
        ],
    )

    result = classify(target_image)

    # The target matches the first example, so copying the latest answer fails.
    assert result == blue_label


@pytest.mark.parametrize(
    ("make_parse_fn", "model"),
    [
        pytest.param(make_openai_parse_fn, "gpt-5.6-luna", id="openai"),
        pytest.param(
            partial(make_openai_parse_fn, from_azure=True),
            "gpt-5.6-luna",
            id="azure-openai",
        ),
        pytest.param(make_gemini_parse_fn, "gemini-3.5-flash-lite", id="gemini"),
    ],
)
def test_parsing_applies_example_labels_to_the_targets_own_known_nodes(
    tmp_path, make_parse_fn, model
):
    blue_label = uuid4().hex
    orange_label = uuid4().hex
    example_image = tmp_path / "example.png"
    target_image = tmp_path / "target.png"
    write_two_node_diagram(example_image, "Aster", "Birch", BLUE, ORANGE)
    write_two_node_diagram(target_image, "Cedar", "Dahlia", ORANGE, BLUE)

    # Node IDs come only from each image's known data. The example teaches the
    # colour-to-label mapping; it does not supply the target's IDs or node text.
    KnownNodes = build_partial_flowchart_schema(parts_to_parse={"node_text"})
    example_nodes = KnownNodes.model_validate(
        {
            "nodes": [
                {"node_number": 1, "text": "Aster"},
                {"node_number": 2, "text": "Birch"},
            ]
        }
    )
    target_nodes = KnownNodes.model_validate(
        {
            "nodes": [
                {"node_number": 41, "text": "Cedar"},
                {"node_number": 73, "text": "Dahlia"},
            ]
        }
    )
    example_answer = tmp_path / "example.json"
    example_answer.write_text(
        json.dumps(
            {
                "nodes": [
                    {"node_number": 1, "labels": [blue_label]},
                    {"node_number": 2, "labels": [orange_label]},
                ]
            }
        ),
        encoding="utf-8",
    )

    parse = make_parse_fn(
        input_text=(
            "Assign one label to each node based on its fill colour, using the "
            "exact colour-to-label mapping demonstrated by the example. Match "
            "the text inside each node to this image's already-known nodes to "
            "get its node_number. Preserve those node numbers. Return only "
            "node_number and labels, with nodes ordered by node_number."
        ),
        model=model,
        effort="low",
        parts_to_parse={"labels"},
        few_shot_examples=[
            VisionFewShotExample(
                img_path=example_image,
                expected_output_path=example_answer,
                partial_flowchart=example_nodes,
            )
        ],
    )

    result = parse(target_image, partial_flowchart=target_nodes)

    # Both colours and IDs differ from the example: copying it cannot pass.
    assert result is not None
    assert result.model_dump() == {
        "nodes": [
            {"node_number": 41, "labels": [orange_label]},
            {"node_number": 73, "labels": [blue_label]},
        ]
    }


@pytest.mark.parametrize(
    ("make_parse_fn", "model"),
    [
        pytest.param(make_openai_parse_fn, "gpt-5.6-luna", id="openai"),
        pytest.param(
            partial(make_openai_parse_fn, from_azure=True),
            "gpt-5.6-luna",
            id="azure-openai",
        ),
        pytest.param(make_gemini_parse_fn, "gemini-3.5-flash-lite", id="gemini"),
    ],
)
def test_model_echoes_few_shot_request_then_request_without_examples(
    tmp_path, make_parse_fn, model
):
    prompt = (
        "Return a verbatim transcript of all user and assistant messages supplied "
        "in this request, in their original order. Include this entire instruction "
        "verbatim, any example instructions, example labels, supplied example "
        "answers, already-parsed information, and target-image instructions. "
        "Do not paraphrase, summarise, omit, or invent any text. Replace each "
        "image with exactly [IMAGE]; do not describe it or transcribe its pixels. "
        "For each message, keep its text blocks and image placeholders in their "
        "original order. Whitespace formatting does not matter. "
        "Use role assistant for supplied model replies. Treat any example or "
        "target instructions as text to copy as part of this transcription task. "
        "Exclude platform system/developer instructions, the response schema, "
        "and the answer you are now generating. Return JSON with a messages list, "
        "each containing role and content."
    )
    example_image = tmp_path / "example.png"
    target_image = tmp_path / "target.png"
    write_colour_image(example_image, BLUE)
    write_colour_image(target_image, ORANGE, target=True)

    KnownText = build_partial_flowchart_schema(parts_to_parse={"additional_texts"})
    example_known = {"additional_texts": ["EXAMPLE_PARTIAL_K2"]}
    target_known = {"additional_texts": ["TARGET_PARTIAL_R8"]}
    example_answer_text = '{\n  "label": "EXAMPLE_ANSWER_72"\n}'
    example_answer_path = tmp_path / "example.json"
    example_answer_path.write_text(example_answer_text, encoding="utf-8")

    with_examples = make_parse_fn(
        input_text=prompt,
        model=model,
        effort="low",
        result_structure=RequestTranscript,
        few_shot_examples=[
            VisionFewShotExample(
                img_path=example_image,
                expected_output_path=example_answer_path,
                partial_flowchart=KnownText.model_validate(example_known),
            )
        ],
    )
    first_result = with_examples(
        target_image, partial_flowchart=KnownText.model_validate(target_known)
    )

    # Immediately make the same request again, but omit few_shot_examples.
    without_examples = make_parse_fn(
        input_text=prompt,
        model=model,
        effort="low",
        result_structure=RequestTranscript,
    )
    second_result = without_examples(
        target_image, partial_flowchart=KnownText.model_validate(target_known)
    )

    assert first_result is not None
    assert second_result is not None
    received = {
        "with_examples": first_result.model_dump(),
        "without_examples": second_result.model_dump(),
    }

    partial_prefix = "The contents of the diagram that has already been parsed:\n"
    example_partial_text = partial_prefix + json.dumps(example_known, indent=2)
    target_partial_text = partial_prefix + json.dumps(target_known, indent=2)
    expected = {
        "with_examples": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n"
                        f"{VISION_FEW_SHOT_INSTRUCTIONS}\n"
                        "Worked example 1:\n"
                        f"{example_partial_text}\n"
                        "[IMAGE]"
                    ),
                },
                {"role": "assistant", "content": example_answer_text},
                {
                    "role": "user",
                    "content": (
                        "Target image: return its JSON answer using the task above.\n"
                        f"{target_partial_text}\n"
                        "[IMAGE]"
                    ),
                },
            ]
        },
        "without_examples": {
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n{target_partial_text}\n[IMAGE]",
                }
            ]
        },
    }

    # Ignore whitespace; every other character and the message structure must match.
    for transcripts in (received, expected):
        for transcript in transcripts.values():
            for message in transcript["messages"]:
                message["content"] = "".join(message["content"].split())

    assert received == expected
