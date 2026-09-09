from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

import flowde.classify_fns.openai_classify_fn as openai_classify_fn
from flowde.classify_fns.classify_types import BinaryClassification

pytestmark = pytest.mark.parametrize(
    "from_azure", [False, True], ids=["openai", "azure"]
)


class DummyClassification(BaseModel):
    label: Literal[0, 1]


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(0, id="label-0"),
        pytest.param(1, id="label-1"),
    ],
)
def test_make_openai_classify_fn_returns_label(monkeypatch, label, from_azure):
    calls = {}

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        from_azure=False,
        few_shot_examples=None,
    ):
        calls["openai_vision_input_list_img_path"] = img_path
        calls["openai_vision_input_list_input_text"] = input_text
        calls["openai_vision_input_list_from_azure"] = from_azure
        calls["openai_vision_input_list_few_shot_examples"] = few_shot_examples
        return ["fake-input-list"]

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort,
        from_azure=False,
    ):
        calls["send_openai_request_result_structure"] = result_structure
        calls["send_openai_request_model"] = model
        calls["send_openai_request_input_list"] = input_list
        calls["send_openai_request_effort"] = effort
        calls["send_openai_request_from_azure"] = from_azure
        return f'{{"label": {label}}}'

    monkeypatch.setattr(
        openai_classify_fn,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_classify_fn,
        "send_openai_request",
        fake_send_openai_request,
    )

    few_shot_examples = [object()]
    classify_fn = openai_classify_fn.make_openai_classify_fn(
        input_text="Classify this image",
        model="gpt-test-model",
        result_structure=DummyClassification,
        effort="low",
        few_shot_examples=few_shot_examples,
        from_azure=from_azure,
    )

    img_path = Path("diagram.png")

    result = classify_fn(img_path)

    assert result == label
    assert calls == {
        "openai_vision_input_list_img_path": img_path,
        "openai_vision_input_list_input_text": "Classify this image",
        "openai_vision_input_list_from_azure": from_azure,
        "openai_vision_input_list_few_shot_examples": few_shot_examples,
        "send_openai_request_result_structure": DummyClassification,
        "send_openai_request_model": "gpt-test-model",
        "send_openai_request_input_list": ["fake-input-list"],
        "send_openai_request_effort": "low",
        "send_openai_request_from_azure": from_azure,
    }


@pytest.mark.parametrize(
    "effort",
    [
        pytest.param("low", id="low-effort"),
        pytest.param("medium", id="medium-effort"),
        pytest.param("high", id="high-effort"),
    ],
)
def test_make_openai_classify_fn_passes_effort(
    monkeypatch,
    effort,
    from_azure,
):
    calls = {}

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        from_azure=False,
        few_shot_examples=None,
    ):
        return ["fake-input-list"]

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort,
        from_azure=False,
    ):
        calls["effort"] = effort
        return '{"label": 1}'

    monkeypatch.setattr(
        openai_classify_fn,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_classify_fn,
        "send_openai_request",
        fake_send_openai_request,
    )

    classify_fn = openai_classify_fn.make_openai_classify_fn(
        input_text="Classify this image",
        model="gpt-test-model",
        result_structure=DummyClassification,
        effort=effort,
        from_azure=from_azure,
    )

    result = classify_fn(Path("diagram.png"))

    assert result == 1
    assert calls["effort"] == effort


def test_make_openai_classify_fn_raises_for_invalid_response_json(
    monkeypatch, from_azure
):
    def fake_openai_vision_input_list(
        img_path,
        input_text,
        from_azure=False,
        few_shot_examples=None,
    ):
        return ["fake-input-list"]

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort,
        from_azure=False,
    ):
        return '{"not_label": 1}'

    monkeypatch.setattr(
        openai_classify_fn,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_classify_fn,
        "send_openai_request",
        fake_send_openai_request,
    )

    classify_fn = openai_classify_fn.make_openai_classify_fn(
        input_text="Classify this image",
        model="gpt-test-model",
        result_structure=DummyClassification,
        from_azure=from_azure,
    )

    with pytest.raises(ValidationError):
        classify_fn(Path("diagram.png"))


def test_make_openai_classify_fn_uses_defaults(monkeypatch, from_azure):
    calls = {}

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        from_azure=False,
        few_shot_examples=None,
    ):
        calls["openai_vision_input_list_img_path"] = img_path
        calls["openai_vision_input_list_input_text"] = input_text
        calls["openai_vision_input_list_from_azure"] = from_azure
        calls["openai_vision_input_list_few_shot_examples"] = few_shot_examples
        return ["fake-input-list"]

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort,
        from_azure=False,
    ):
        calls["send_openai_request_result_structure"] = result_structure
        calls["send_openai_request_model"] = model
        calls["send_openai_request_input_list"] = input_list
        calls["send_openai_request_effort"] = effort
        calls["send_openai_request_from_azure"] = from_azure
        return '{"label": 1}'

    monkeypatch.setattr(
        openai_classify_fn,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_classify_fn,
        "send_openai_request",
        fake_send_openai_request,
    )

    classify_fn = openai_classify_fn.make_openai_classify_fn(
        input_text="Classify this image",
        model="openai-test-model",
        # Leave the backend unspecified in the OpenAI case to check its default too.
        **({"from_azure": True} if from_azure else {}),
    )

    img_path = Path("diagram.png")

    result = classify_fn(img_path)

    assert result == 1
    assert calls == {
        "openai_vision_input_list_img_path": img_path,
        "openai_vision_input_list_input_text": "Classify this image",
        "openai_vision_input_list_from_azure": from_azure,
        "openai_vision_input_list_few_shot_examples": None,
        "send_openai_request_result_structure": BinaryClassification,
        "send_openai_request_model": "openai-test-model",
        "send_openai_request_input_list": ["fake-input-list"],
        "send_openai_request_effort": "high",
        "send_openai_request_from_azure": from_azure,
    }
