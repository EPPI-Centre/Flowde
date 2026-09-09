from importlib import import_module
from unittest.mock import Mock

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel

import flowde.parsing_fns.openai_parse as module
from flowde.classify_imgs import classify_imgs
from flowde.parse_imgs import parse_imgs
from flowde.pricing import TokenPrices
from flowde.utils import VisionFewShotExample


class Answer(BaseModel):
    answer: str


@pytest.fixture(
    params=[
        ("openai", False, "classify"),
        ("openai", True, "classify"),
        ("gemini", False, "classify"),
        ("openai", False, "parse"),
        ("openai", True, "parse"),
        ("gemini", False, "parse"),
    ],
    ids=[
        "openai-classify",
        "azure-classify",
        "gemini-classify",
        "openai-parse",
        "azure-parse",
        "gemini-parse",
    ],
)
def provider(request, monkeypatch, tmp_path):
    name, azure, task = request.param
    module_name = (
        f"flowde.classify_fns.{name}_classify_fn"
        if task == "classify"
        else f"flowde.parsing_fns.{name}_parse"
    )
    module = import_module(module_name)
    monkeypatch.setattr(module, f"{name}_vision_input_list", Mock(return_value=[]))
    sender = Mock(
        return_value='{"label": 1}' if task == "classify" else '{"answer": "yes"}'
    )
    monkeypatch.setattr(module, f"send_{name}_request", sender)
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.png").write_text("image")
    options = {"from_azure": azure} if name == "openai" else {}
    if task == "parse":
        options["result_structure"] = Answer
    factory = getattr(module, f"make_{name}_{task}_fn")
    api = classify_imgs if task == "classify" else parse_imgs
    return name, factory, api, options, sender, images


def test_factory_settings_allow_resume_without_custom_metadata(provider, tmp_path):
    _, factory, api, options, sender, images = provider
    first = factory("Read this", "model", **options)
    expected = api(first, images, tmp_path / "output", n_jobs=1)
    sender.reset_mock()
    # Worker count and cost overrides affect execution/reporting, not the answer.
    resumed = factory(
        "Read this",
        "model",
        token_prices=TokenPrices(input=1, output=2),
        few_shot_examples=[],
        **options,
    )

    result = api(resumed, images, tmp_path / "output", on_existing="resume", n_jobs=2)

    assert result == expected
    assert resumed.result_structure is first.result_structure
    sender.assert_not_called()


@pytest.mark.parametrize(
    "provider",
    [("openai", True, "classify"), ("openai", True, "parse")],
    ids=["azure-classify", "azure-parse"],
    indirect=True,
)
def test_azure_resume_allows_a_different_endpoint(provider, tmp_path, monkeypatch):
    _, factory, api, options, sender, images = provider
    monkeypatch.setenv("AZURE_API_BASE", "https://first.example")
    output = tmp_path / "output"
    first = factory("Read this", "model", effort="low", **options)
    expected = api(first, images, output, n_jobs=1)

    monkeypatch.setenv("AZURE_API_BASE", "https://second.example")
    (images / "new-image.png").write_text("new image")
    sender.reset_mock()
    resumed = factory("Read this", "model", effort="low", **options)
    result = api(resumed, images, output, on_existing="resume", n_jobs=1)

    assert result == [expected[0], expected[0]]
    sender.assert_called_once()  # Only the new image needs a model request.


def test_provider_failures_raise_and_leave_the_image_unfinished(provider, tmp_path):
    name, factory, api, options, sender, images = provider
    error = (
        BadRequestError(
            "bad request",
            response=httpx.Response(
                400, request=httpx.Request("POST", "https://unused.example")
            ),
            body={},
        )
        if name == "openai"
        else RuntimeError("bad request")
    )
    sender.side_effect = error
    fn = factory("Read this", "model", **options)
    output = tmp_path / "output"

    with pytest.raises(type(error), match="bad request") as raised:
        api(fn, images, output, n_jobs=1)

    assert raised.value is error
    assert list(output.glob("*.json")) == []
    sender.side_effect = None
    api(fn, images, output, on_existing="resume", n_jobs=1)
    assert sender.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        "prompt",
        "model",
        "effort",
        "schema",
        "example-image",
        "example-answer",
    ],
)
def test_resume_rejects_changes_that_affect_model_answers(
    tmp_path, monkeypatch, change
):
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.png").write_text("image")
    example_image = tmp_path / "example.png"
    example_image.write_text("example image")
    example_answer = tmp_path / "example.json"
    example_answer.write_text('{"answer": "example"}')
    examples = [
        VisionFewShotExample(
            img_path=example_image, expected_output_path=example_answer
        )
    ]
    monkeypatch.setattr(module, "openai_vision_input_list", Mock(return_value=[]))
    sender = Mock(return_value='{"answer": "yes"}')
    monkeypatch.setattr(module, "send_openai_request", sender)
    options = {
        "input_text": "Read this",
        "model": "model",
        "effort": "low",
        "result_structure": Answer,
        "few_shot_examples": examples,
        "from_azure": True,
    }
    output = tmp_path / "output"
    parse_imgs(module.make_openai_parse_fn(**options), images, output, n_jobs=1)
    sender.reset_mock()

    if change == "prompt":
        options["input_text"] = "Different task"
    elif change in {"model", "effort"}:
        options[change] = "different"
    elif change == "schema":

        class DifferentAnswer(BaseModel):
            count: int

        options["result_structure"] = DifferentAnswer
    elif change == "example-image":
        example_image.write_text("changed example")
    elif change == "example-answer":
        example_answer.write_text('{"answer": "different"}')

    with pytest.raises(ValueError, match="settings have changed"):
        parse_imgs(
            module.make_openai_parse_fn(**options),
            images,
            output,
            on_existing="resume",
            n_jobs=1,
        )
    sender.assert_not_called()
