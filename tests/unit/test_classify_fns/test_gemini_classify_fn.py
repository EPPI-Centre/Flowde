from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

import flowde.classify_fns.gemini_classify_fn as gemini_classify_fn
from flowde.classify_fns.classify_types import BinaryClassification


class DummyClassification(BaseModel):
    label: Literal[0, 1]


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(0, id="label-0"),
        pytest.param(1, id="label-1"),
    ],
)
def test_make_gemini_classify_fn_returns_label(monkeypatch, label):
    calls = {}

    def fake_gemini_vision_input_list(img_path, input_text, few_shot_examples=None):
        calls["gemini_vision_input_list_img_path"] = img_path
        calls["gemini_vision_input_list_input_text"] = input_text
        calls["gemini_vision_input_list_few_shot_examples"] = few_shot_examples
        return ["fake-input-list"]

    def fake_send_gemini_request(
        result_structure,
        model,
        input_list,
        thinking_level,
    ):
        calls["send_gemini_request_result_structure"] = result_structure
        calls["send_gemini_request_model"] = model
        calls["send_gemini_request_input_list"] = input_list
        calls["send_gemini_request_thinking_level"] = thinking_level
        return f'{{"label": {label}}}'

    monkeypatch.setattr(
        gemini_classify_fn,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_classify_fn,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    few_shot_examples = [object()]
    classify_fn = gemini_classify_fn.make_gemini_classify_fn(
        input_text="Classify this image",
        model="gemini-test-model",
        result_structure=DummyClassification,
        effort="low",
        few_shot_examples=few_shot_examples,
    )

    img_path = Path("diagram.png")

    result = classify_fn(img_path)

    assert result == label
    assert calls == {
        "gemini_vision_input_list_img_path": img_path,
        "gemini_vision_input_list_input_text": "Classify this image",
        "gemini_vision_input_list_few_shot_examples": few_shot_examples,
        "send_gemini_request_result_structure": DummyClassification,
        "send_gemini_request_model": "gemini-test-model",
        "send_gemini_request_input_list": ["fake-input-list"],
        "send_gemini_request_thinking_level": "low",
    }


@pytest.mark.parametrize(
    "effort",
    [
        pytest.param("low", id="low-effort"),
        pytest.param("medium", id="medium-effort"),
        pytest.param("high", id="high-effort"),
    ],
)
def test_make_gemini_classify_fn_passes_effort_as_thinking_level(
    monkeypatch,
    effort,
):
    calls = {}

    def fake_gemini_vision_input_list(img_path, input_text, few_shot_examples=None):
        return ["fake-input-list"]

    def fake_send_gemini_request(
        result_structure,
        model,
        input_list,
        thinking_level,
    ):
        calls["thinking_level"] = thinking_level
        return '{"label": 1}'

    monkeypatch.setattr(
        gemini_classify_fn,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_classify_fn,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    classify_fn = gemini_classify_fn.make_gemini_classify_fn(
        input_text="Classify this image",
        model="gemini-test-model",
        result_structure=DummyClassification,
        effort=effort,
    )

    classify_fn(Path("diagram.png"))

    assert calls["thinking_level"] == effort


def test_make_gemini_classify_fn_raises_for_invalid_response_json(monkeypatch):
    def fake_gemini_vision_input_list(img_path, input_text, few_shot_examples=None):
        return ["fake-input-list"]

    def fake_send_gemini_request(
        result_structure,
        model,
        input_list,
        thinking_level,
    ):
        return '{"not_label": 1}'

    monkeypatch.setattr(
        gemini_classify_fn,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_classify_fn,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    classify_fn = gemini_classify_fn.make_gemini_classify_fn(
        input_text="Classify this image",
        model="gemini-test-model",
        result_structure=DummyClassification,
    )

    with pytest.raises(ValidationError):
        classify_fn(Path("diagram.png"))


def test_make_gemini_classify_fn_uses_defaults(monkeypatch):
    calls = {}

    def fake_gemini_vision_input_list(img_path, input_text, few_shot_examples=None):
        calls["gemini_vision_input_list_img_path"] = img_path
        calls["gemini_vision_input_list_input_text"] = input_text
        calls["gemini_vision_input_list_few_shot_examples"] = few_shot_examples
        return ["fake-input-list"]

    def fake_send_gemini_request(
        result_structure,
        model,
        input_list,
        thinking_level,
    ):
        calls["send_gemini_request_result_structure"] = result_structure
        calls["send_gemini_request_model"] = model
        calls["send_gemini_request_input_list"] = input_list
        calls["send_gemini_request_thinking_level"] = thinking_level
        return '{"label": 1}'

    monkeypatch.setattr(
        gemini_classify_fn,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_classify_fn,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    classify_fn = gemini_classify_fn.make_gemini_classify_fn(
        input_text="Classify this image",
        model="gemini-test-model",
    )

    img_path = Path("diagram.png")

    result = classify_fn(img_path)

    assert result == 1
    assert calls == {
        "gemini_vision_input_list_img_path": img_path,
        "gemini_vision_input_list_input_text": "Classify this image",
        "gemini_vision_input_list_few_shot_examples": None,
        "send_gemini_request_result_structure": BinaryClassification,
        "send_gemini_request_model": "gemini-test-model",
        "send_gemini_request_input_list": ["fake-input-list"],
        "send_gemini_request_thinking_level": "high",
    }
