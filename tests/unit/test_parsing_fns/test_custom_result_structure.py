from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from flowde.parsing_fns import gemini_parse, openai_parse
from flowde.parsing_fns.parsing_types import build_partial_flowchart_schema

pytestmark = pytest.mark.parametrize(
    ("parse_module", "make_parse_fn"),
    [
        pytest.param(openai_parse, openai_parse.make_openai_parse_fn, id="openai"),
        pytest.param(
            openai_parse,
            partial(openai_parse.make_openai_parse_fn, from_azure=True),
            id="azure",
        ),
        pytest.param(gemini_parse, gemini_parse.make_gemini_parse_fn, id="gemini"),
    ],
)


class StudySummary(BaseModel):
    title: str
    participant_count: int = Field(ge=0)


@pytest.fixture
def build_vision_input(monkeypatch, parse_module):
    """Replace image preparation so these tests need no files or uploads."""
    builder = Mock(return_value=["prepared vision input"])
    if parse_module is openai_parse:
        monkeypatch.setattr(parse_module, "openai_vision_input_list", builder)
    else:
        monkeypatch.setattr(parse_module, "gemini_vision_input_list", builder)
    return builder


@pytest.fixture
def send_request(monkeypatch, parse_module):
    """Return a JSON answer without calling either provider's API."""
    sender = Mock(return_value='{"title": "Example study", "participant_count": 12}')
    if parse_module is openai_parse:
        monkeypatch.setattr(parse_module, "send_openai_request", sender)
    else:
        monkeypatch.setattr(parse_module, "send_gemini_request", sender)
    return sender


@pytest.mark.parametrize(
    "include_partial", [False, True], ids=["without-context", "with-flowchart-context"]
)
def test_custom_schema_is_used_for_the_request_and_returned_result(
    make_parse_fn, build_vision_input, send_request, include_partial
):
    partial_flowchart = None
    if include_partial:
        NodeText = build_partial_flowchart_schema({"node_text"})
        partial_flowchart = NodeText(
            nodes=[{"node_number": 1, "text": "12 participants"}]
        )

    parse_fn = make_parse_fn(
        input_text="Summarise this study.",
        model="test-model",
        result_structure=StudySummary,
    )

    result = parse_fn(Path("diagram.png"), partial_flowchart=partial_flowchart)

    assert type(result) is StudySummary
    assert result.model_dump() == {"title": "Example study", "participant_count": 12}
    send_request.assert_called_once()
    assert send_request.call_args.kwargs["result_structure"] is StudySummary
    build_vision_input.assert_called_once()
    assert build_vision_input.call_args.kwargs["partial_flowchart"] is partial_flowchart


@pytest.mark.parametrize(
    "response_text",
    [
        pytest.param('{"title": "Example study"}', id="missing-required-field"),
        pytest.param(
            '{"title": "Example study", "participant_count": "many"}',
            id="invalid-field-type",
        ),
        pytest.param(
            '{"title": "Example study", "participant_count": -1}',
            id="violates-custom-constraint",
        ),
    ],
)
def test_response_must_satisfy_the_custom_schema(
    make_parse_fn, build_vision_input, send_request, response_text
):
    send_request.return_value = response_text
    parse_fn = make_parse_fn(
        input_text="Summarise this study.",
        model="test-model",
        result_structure=StudySummary,
    )

    with pytest.raises(ValidationError) as raised:
        parse_fn(Path("diagram.png"))

    assert raised.value.errors()[0]["loc"] == ("participant_count",)


@pytest.mark.parametrize(
    "parts_to_parse",
    [
        pytest.param({"node_text"}, id="nonempty-parts"),
        pytest.param(set(), id="empty-parts"),
    ],
)
def test_custom_schema_and_parsing_parts_are_rejected_when_creating_the_parser(
    make_parse_fn, build_vision_input, send_request, parts_to_parse
):
    with pytest.raises(ValueError, match="Cannot specify both"):
        make_parse_fn(
            input_text="Summarise this study.",
            model="test-model",
            result_structure=StudySummary,
            parts_to_parse=parts_to_parse,
        )

    build_vision_input.assert_not_called()
    send_request.assert_not_called()
