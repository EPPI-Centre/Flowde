import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, TypeAdapter

import flowde.api_utils.openai_utils as openai_utils
from flowde.utils import VisionFewShotExample


class DummyResultStructure(BaseModel):
    answer: str


class FakeOpenAIFileResult:
    id = "fake-file-id"


class FakeOpenAIFiles:
    def __init__(self):
        self.created_file_content = None
        self.created_purpose = None

    def create(self, file, purpose):
        self.created_file_content = file.read()
        self.created_purpose = purpose
        return FakeOpenAIFileResult()


@pytest.fixture
def openai_client_constructor(monkeypatch):
    """Use distinct provider settings and a fresh mock client for each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("AZURE_API_KEY", "fake-azure-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://azure.example/openai/v1/")
    monkeypatch.setattr(openai_utils, "load_dotenv", lambda: None)
    constructor = Mock()
    monkeypatch.setattr(openai_utils, "OpenAI", constructor)
    return constructor


def test_create_openai_vision_file_uploads_file_and_returns_file_id(tmp_path: Path):
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(b"fake image bytes")

    client = Mock(files=FakeOpenAIFiles())

    result = openai_utils.create_openai_vision_file(
        file_path=img_path,
        client=client,
    )

    assert result == "fake-file-id"
    assert client.files.created_file_content == b"fake image bytes"
    assert client.files.created_purpose == "vision"


@pytest.mark.parametrize(
    ("extension", "mime_type"),
    [
        pytest.param(".png", "image/png", id="png"),
        pytest.param(".jpg", "image/jpeg", id="jpg"),
        pytest.param(".jpeg", "image/jpeg", id="jpeg"),
        pytest.param(".webp", "image/webp", id="webp"),
        pytest.param(".gif", "image/gif", id="gif"),
    ],
)
def test_image_data_url_preserves_bytes_and_identifies_the_format(
    tmp_path, extension, mime_type
):
    img_path = tmp_path / f"diagram{extension}"
    img_path.write_bytes(b"\x00\x01\xfe\xff")

    result = openai_utils.image_path_to_data_url(img_path)

    assert result == f"data:{mime_type};base64,AAH+/w=="


@pytest.mark.parametrize(
    "effort",
    [
        pytest.param("low", id="low-effort"),
        pytest.param("medium", id="medium-effort"),
        pytest.param("high", id="high-effort"),
    ],
)
@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_send_openai_request_builds_client_request_and_returns_text(
    openai_client_constructor,
    effort,
    from_azure,
):
    client = openai_client_constructor.return_value
    client.responses.parse.return_value.output_text = '{"answer": "yes"}'

    image_key = "image_url" if from_azure else "file_id"
    input_list = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Parse this image"},
                {
                    "type": "input_image",
                    image_key: (
                        "data:image/png;base64,dGFyZ2V0IGltYWdl"
                        if from_azure
                        else "fake-file-id"
                    ),
                    "detail": "high",
                },
            ],
        }
    ]

    result = openai_utils.send_openai_request(
        result_structure=DummyResultStructure,
        model="gpt-test-model",
        input_list=input_list,
        effort=effort,
        # Leave the backend unspecified in the OpenAI case to check its default too.
        **({"from_azure": True} if from_azure else {}),
    )

    assert result == '{"answer": "yes"}'

    if from_azure:
        openai_client_constructor.assert_called_once_with(
            api_key="fake-azure-key", base_url="https://azure.example/openai/v1/"
        )
    else:
        openai_client_constructor.assert_called_once_with(api_key="fake-openai-key")
    client.responses.parse.assert_called_once_with(
        model="gpt-test-model",
        text_format=DummyResultStructure,
        background=False,
        reasoning={"effort": effort},
        input=input_list,
    )


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_openai_vision_input_list_without_partial_flowchart(
    tmp_path, openai_client_constructor, vision_upload, from_azure
):
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(b"target image")

    result = openai_utils.openai_vision_input_list(
        img_path=img_path,
        input_text="Parse this diagram",
        # Leave the backend unspecified in the OpenAI case to check its default too.
        **({"from_azure": True} if from_azure else {}),
    )

    image_key = "image_url" if from_azure else "file_id"
    expected = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Parse this diagram"},
                {
                    "type": "input_image",
                    image_key: (
                        "data:image/png;base64,dGFyZ2V0IGltYWdl"
                        if from_azure
                        else "file-diagram.png"
                    ),
                    "detail": "high",
                },
            ],
        },
    ]

    assert result == expected

    if from_azure:
        openai_client_constructor.assert_called_once_with(
            api_key="fake-azure-key", base_url="https://azure.example/openai/v1/"
        )
        vision_upload.assert_not_called()
    else:
        openai_client_constructor.assert_called_once_with(api_key="fake-openai-key")
        vision_upload.assert_called_once_with(
            img_path, openai_client_constructor.return_value
        )


class PartialFlowchart(BaseModel):
    nodes: list[str] | None = None
    additional_texts: list[str] | None = None


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_openai_vision_input_list_with_partial_flowchart(
    tmp_path, vision_upload, from_azure
):
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(b"target image")

    partial_flowchart = PartialFlowchart(
        nodes=["node 1", "node 2"],
        additional_texts=None,
    )

    result = openai_utils.openai_vision_input_list(
        img_path=img_path,
        input_text="Parse this diagram",
        partial_flowchart=partial_flowchart,
        from_azure=from_azure,
    )

    image_key = "image_url" if from_azure else "file_id"
    expected = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Parse this diagram"},
                {
                    "type": "input_text",
                    "text": (
                        "The contents of the diagram that has already been parsed:\n"
                        "{\n"
                        '  "nodes": [\n'
                        '    "node 1",\n'
                        '    "node 2"\n'
                        "  ]\n"
                        "}"
                    ),
                },
                {
                    "type": "input_image",
                    image_key: (
                        "data:image/png;base64,dGFyZ2V0IGltYWdl"
                        if from_azure
                        else "file-diagram.png"
                    ),
                    "detail": "high",
                },
            ],
        },
    ]

    assert result == expected
    if from_azure:
        vision_upload.assert_not_called()


@pytest.fixture
def vision_upload(monkeypatch, openai_client_constructor):
    """Replace the client and uploads; give each image its own reference."""
    upload = Mock(side_effect=lambda img_path, client: f"file-{img_path.name}")
    monkeypatch.setattr(openai_utils, "create_openai_vision_file", upload)
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
@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_no_examples_adds_no_example_messages_or_instructions(
    tmp_path, vision_upload, example_kwargs, include_partial, from_azure
):
    target = tmp_path / "target.png"
    target.write_bytes(b"target image")
    partial = PartialFlowchart(nodes=["Known node"]) if include_partial else None

    result = openai_utils.openai_vision_input_list(
        target,
        "Parse this diagram",
        partial_flowchart=partial,
        from_azure=from_azure,
        **example_kwargs,
    )

    image_key = "image_url" if from_azure else "file_id"
    expected_content = [{"type": "input_text", "text": "Parse this diagram"}]
    if include_partial:
        expected_content.append(
            {
                "type": "input_text",
                "text": (
                    "The contents of the diagram that has already been parsed:\n"
                    '{\n  "nodes": [\n    "Known node"\n  ]\n}'
                ),
            }
        )
    expected_content.append(
        {
            "type": "input_image",
            image_key: (
                "data:image/png;base64,dGFyZ2V0IGltYWdl"
                if from_azure
                else "file-target.png"
            ),
            "detail": "high",
        }
    )

    assert result == [{"role": "user", "content": expected_content}]
    if from_azure:
        vision_upload.assert_not_called()


@pytest.mark.parametrize(
    "include_partial",
    [False, True],
    ids=["without-target-partial", "with-target-partial"],
)
@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_few_shot_vision_input_list_matches_expected_messages(
    tmp_path, vision_upload, few_shot_examples, include_partial, from_azure
):
    target = tmp_path / "target.png"
    target.write_bytes(b"target image")
    partial = (
        PartialFlowchart(nodes=["Target's known node"]) if include_partial else None
    )

    result = openai_utils.openai_vision_input_list(
        target,
        "Extract labels",
        partial_flowchart=partial,
        few_shot_examples=few_shot_examples,
        from_azure=from_azure,
    )

    image_key = "image_url" if from_azure else "file_id"
    target_content = [
        {
            "type": "input_text",
            "text": "Target image: return its JSON answer using the task above.",
        }
    ]
    if include_partial:
        target_content.append(
            {
                "type": "input_text",
                "text": (
                    "The contents of the diagram that has already been parsed:\n"
                    '{\n  "nodes": [\n    "Target\'s known node"\n  ]\n}'
                ),
            }
        )
    target_content.append(
        {
            "type": "input_image",
            image_key: (
                "data:image/png;base64,dGFyZ2V0IGltYWdl"
                if from_azure
                else "file-target.png"
            ),
            "detail": "high",
        }
    )

    expected = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract labels"},
                {
                    "type": "input_text",
                    "text": (
                        "The following exchanges are worked examples "
                        "of the task above. "
                        "Each example image is followed by its expected JSON answer. "
                        "Any already-parsed information belongs only to the image "
                        "in the same message. Use them as guidance and return an "
                        "answer only for the final target image."
                    ),
                },
                {"type": "input_text", "text": "Worked example 1:"},
                {
                    "type": "input_text",
                    "text": (
                        "The contents of the diagram that has already been parsed:\n"
                        '{\n  "nodes": [\n    "Example\'s known node"\n  ]\n}'
                    ),
                },
                {
                    "type": "input_image",
                    image_key: (
                        "data:image/png;base64,Zmlyc3QgaW1hZ2U="
                        if from_azure
                        else "file-z_example.png"
                    ),
                    "detail": "high",
                },
            ],
        },
        {"role": "assistant", "content": '{\n  "label": "éligible"\n}'},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Worked example 2:"},
                {
                    "type": "input_image",
                    image_key: (
                        "data:image/png;base64,c2Vjb25kIGltYWdl"
                        if from_azure
                        else "file-a_example.png"
                    ),
                    "detail": "high",
                },
            ],
        },
        {"role": "assistant", "content": '{\n  "label": "excluded"\n}'},
        {"role": "user", "content": target_content},
    ]

    assert result == expected
    if from_azure:
        vision_upload.assert_not_called()


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_few_shot_vision_input_list_is_valid_for_the_openai_sdk(
    tmp_path, vision_upload, few_shot_examples, from_azure
):
    target = tmp_path / "target.png"
    target.write_bytes(b"target image")

    result = openai_utils.openai_vision_input_list(
        target,
        "Extract labels",
        from_azure=from_azure,
        partial_flowchart=PartialFlowchart(nodes=["Target's known node"]),
        few_shot_examples=few_shot_examples,
    )

    TypeAdapter(ResponseInputParam).validate_python(result)


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_few_shot_request_raises_instead_of_skipping_an_invalid_example(
    tmp_path, vision_upload, from_azure
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
        openai_utils.openai_vision_input_list(
            target,
            "Classify this image",
            few_shot_examples=[example],
            from_azure=from_azure,
        )
    if from_azure:
        vision_upload.assert_not_called()
