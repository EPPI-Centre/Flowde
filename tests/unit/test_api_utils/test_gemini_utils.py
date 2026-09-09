import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai.types import MediaResolution
from pydantic import BaseModel

import flowde.api_utils.gemini_utils as gemini_utils
from flowde.utils import VisionFewShotExample


class DummyResultStructure(BaseModel):
    answer: str


class FakeResponse:
    text = '{"answer": "yes"}'


class FakeModels:
    def __init__(self, parent_client):
        self.parent_client = parent_client

    def generate_content(self, model, contents, config):
        self.parent_client.generate_content_model = model
        self.parent_client.generate_content_contents = contents
        self.parent_client.generate_content_config = config
        return FakeResponse()


class FakeGenAIClient:
    created_clients = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.models = FakeModels(self)
        FakeGenAIClient.created_clients.append(self)


@pytest.mark.parametrize(
    "thinking_level",
    [
        pytest.param("low", id="low-thinking"),
        pytest.param("medium", id="medium-thinking"),
        pytest.param("high", id="high-thinking"),
    ],
)
def test_send_gemini_request_builds_client_config_and_returns_text(
    monkeypatch,
    thinking_level,
):
    FakeGenAIClient.created_clients = []

    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(gemini_utils.genai, "Client", FakeGenAIClient)

    input_list = ["Parse this image", "fake-image-object"]

    result = gemini_utils.send_gemini_request(
        result_structure=DummyResultStructure,
        input_list=input_list,
        model="gemini-test-model",
        thinking_level=thinking_level,
    )

    assert result == '{"answer": "yes"}'

    client = FakeGenAIClient.created_clients[0]

    assert client.api_key == "fake-api-key"
    assert client.generate_content_model == "gemini-test-model"
    assert client.generate_content_contents == input_list

    config = client.generate_content_config

    assert config.media_resolution == MediaResolution.MEDIA_RESOLUTION_HIGH
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == DummyResultStructure.model_json_schema()
    assert config.thinking_config.thinking_level.value == thinking_level.upper()


class FakeVisionClient:
    created_clients = []

    def __init__(self, api_key):
        self.api_key = api_key
        FakeVisionClient.created_clients.append(self)


class FakeUploadedFile:
    uri = "gs://test-bucket/diagram.png"
    mime_type = "image/png"


def test_gemini_vision_input_list_without_partial_flowchart(monkeypatch):
    FakeVisionClient.created_clients = []

    uploaded = {}

    def fake_create_google_file(img_path, client):
        uploaded["img_path"] = img_path
        uploaded["client"] = client
        return FakeUploadedFile()

    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(gemini_utils.genai, "Client", FakeVisionClient)
    monkeypatch.setattr(gemini_utils, "create_google_file", fake_create_google_file)

    img_path = Path("diagram.png")

    result = gemini_utils.gemini_vision_input_list(
        img_path=img_path,
        input_text="Parse this diagram",
    )

    assert [content.model_dump(exclude_none=True) for content in result] == [
        {
            "role": "user",
            "parts": [
                {"text": "Parse this diagram"},
                {
                    "file_data": {
                        "file_uri": "gs://test-bucket/diagram.png",
                        "mime_type": "image/png",
                    }
                },
            ],
        }
    ]

    client = FakeVisionClient.created_clients[0]

    assert client.api_key == "fake-api-key"
    assert uploaded["img_path"] == img_path
    assert uploaded["client"] is client


class PartialFlowchart(BaseModel):
    nodes: list[str] | None = None
    additional_texts: list[str] | None = None


def test_gemini_vision_input_list_with_partial_flowchart(monkeypatch):
    FakeVisionClient.created_clients = []

    def fake_create_google_file(img_path, client):
        return FakeUploadedFile()

    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(gemini_utils.genai, "Client", FakeVisionClient)
    monkeypatch.setattr(gemini_utils, "create_google_file", fake_create_google_file)

    partial_flowchart = PartialFlowchart(
        nodes=["node 1", "node 2"],
        additional_texts=None,
    )

    result = gemini_utils.gemini_vision_input_list(
        img_path=Path("diagram.png"),
        input_text="Parse this diagram",
        partial_flowchart=partial_flowchart,
    )

    partial_flowchart_text = (
        "The contents of the diagram that has already been parsed:\n"
        "{\n"
        '  "nodes": [\n'
        '    "node 1",\n'
        '    "node 2"\n'
        "  ]\n"
        "}"
    )
    assert [content.model_dump(exclude_none=True) for content in result] == [
        {
            "role": "user",
            "parts": [
                {"text": "Parse this diagram"},
                {"text": partial_flowchart_text},
                {
                    "file_data": {
                        "file_uri": "gs://test-bucket/diagram.png",
                        "mime_type": "image/png",
                    }
                },
            ],
        }
    ]


def test_create_google_file_uploads_file_to_client():
    uploaded = {}

    class FakeFiles:
        def upload(self, file):
            uploaded["file"] = file
            return "uploaded-file-object"

    class FakeClient:
        def __init__(self):
            self.files = FakeFiles()

    file_path = Path("diagram.png")
    client = FakeClient()

    result = gemini_utils.create_google_file(file_path=file_path, client=client)

    assert result == "uploaded-file-object"
    assert uploaded["file"] == file_path


@pytest.fixture
def vision_upload(monkeypatch):
    """Replace the client and uploads; give each image its own reference."""
    monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
    monkeypatch.setattr(gemini_utils.genai, "Client", Mock())
    upload = Mock(
        side_effect=lambda img_path, client: SimpleNamespace(
            uri=f"https://example.test/{img_path.name}",
            mime_type="image/png",
        )
    )
    monkeypatch.setattr(gemini_utils, "create_google_file", upload)
    return upload


@pytest.fixture
def few_shot_examples(tmp_path):
    """Two examples in supplied order; only the first has already-known nodes."""
    # Reverse filename order makes accidental sorting visible in the request.
    first_image = tmp_path / "z_example.png"
    second_image = tmp_path / "a_example.png"
    first_image.write_bytes(b"first image")
    second_image.write_bytes(b"second image")
    first_answer = tmp_path / "z_example.json"
    second_answer = tmp_path / "a_example.json"
    first_answer.write_text('{"label": "éligible"}', encoding="utf-8")
    second_answer.write_text('{"label": "excluded"}', encoding="utf-8")

    return [
        VisionFewShotExample(
            img_path=first_image,
            expected_output_path=first_answer,
            partial_flowchart=PartialFlowchart(nodes=["Example's known node"]),
        ),
        VisionFewShotExample(img_path=second_image, expected_output_path=second_answer),
    ]


@pytest.mark.parametrize(
    "example_kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"few_shot_examples": None}, id="none"),
        pytest.param({"few_shot_examples": []}, id="empty-list"),
    ],
)
@pytest.mark.parametrize(
    "include_partial", [False, True], ids=["without-partial", "with-partial"]
)
def test_no_examples_adds_no_example_messages_or_instructions(
    vision_upload, example_kwargs, include_partial
):
    partial = PartialFlowchart(nodes=["Known node"]) if include_partial else None

    result = gemini_utils.gemini_vision_input_list(
        Path("target.png"),
        "Parse this diagram",
        partial_flowchart=partial,
        **example_kwargs,
    )

    expected_parts = [{"text": "Parse this diagram"}]
    if include_partial:
        expected_parts.append(
            {
                "text": (
                    "The contents of the diagram that has already been parsed:\n"
                    '{\n  "nodes": [\n    "Known node"\n  ]\n}'
                ),
            }
        )
    expected_parts.append(
        {
            "file_data": {
                "file_uri": "https://example.test/target.png",
                "mime_type": "image/png",
            },
        }
    )

    assert [message.model_dump(exclude_none=True) for message in result] == [
        {"role": "user", "parts": expected_parts},
    ]


@pytest.mark.parametrize(
    "include_partial",
    [False, True],
    ids=["without-target-partial", "with-target-partial"],
)
def test_few_shot_vision_input_list_matches_expected_messages(
    vision_upload, few_shot_examples, include_partial
):
    partial = (
        PartialFlowchart(nodes=["Target's known node"]) if include_partial else None
    )

    result = gemini_utils.gemini_vision_input_list(
        Path("target.png"),
        "Extract labels",
        partial_flowchart=partial,
        few_shot_examples=few_shot_examples,
    )

    target_parts = [
        {
            "text": "Target image: return its JSON answer using the task above.",
        }
    ]
    if include_partial:
        target_parts.append(
            {
                "text": (
                    "The contents of the diagram that has already been parsed:\n"
                    '{\n  "nodes": [\n    "Target\'s known node"\n  ]\n}'
                ),
            }
        )
    target_parts.append(
        {
            "file_data": {
                "file_uri": "https://example.test/target.png",
                "mime_type": "image/png",
            },
        }
    )

    expected = [
        {
            "role": "user",
            "parts": [
                {"text": "Extract labels"},
                {
                    "text": (
                        "The following exchanges are worked examples "
                        "of the task above. "
                        "Each example image is followed by its expected JSON answer. "
                        "Any already-parsed information belongs only to the image "
                        "in the same message. Use them as guidance and return an "
                        "answer only for the final target image."
                    )
                },
                {"text": "Worked example 1:"},
                {
                    "text": (
                        "The contents of the diagram that has already been parsed:\n"
                        '{\n  "nodes": [\n    "Example\'s known node"\n  ]\n}'
                    )
                },
                {
                    "file_data": {
                        "file_uri": "https://example.test/z_example.png",
                        "mime_type": "image/png",
                    }
                },
            ],
        },
        {"role": "model", "parts": [{"text": '{\n  "label": "éligible"\n}'}]},
        {
            "role": "user",
            "parts": [
                {"text": "Worked example 2:"},
                {
                    "file_data": {
                        "file_uri": "https://example.test/a_example.png",
                        "mime_type": "image/png",
                    }
                },
            ],
        },
        {"role": "model", "parts": [{"text": '{\n  "label": "excluded"\n}'}]},
        {"role": "user", "parts": target_parts},
    ]

    assert [message.model_dump(exclude_none=True) for message in result] == expected


def test_few_shot_request_raises_instead_of_skipping_an_invalid_example(
    tmp_path, vision_upload
):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"image")
    answer_path = tmp_path / "example.json"
    answer_path.write_text("invalid JSON", encoding="utf-8")
    example = VisionFewShotExample(
        img_path=image_path, expected_output_path=answer_path
    )
    target = tmp_path / "target.png"
    target.write_bytes(b"target image")

    with pytest.raises(json.JSONDecodeError):
        gemini_utils.gemini_vision_input_list(
            target, "Classify this image", few_shot_examples=[example]
        )
