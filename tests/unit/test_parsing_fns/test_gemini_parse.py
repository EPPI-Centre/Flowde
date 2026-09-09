from pathlib import Path

import pytest

import flowde.parsing_fns.gemini_parse as gemini_parse_fn_module


class DummyPartialFlowchart:
    pass


@pytest.mark.parametrize(
    ("parts_to_parse", "response_text", "expected_result"),
    [
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": 1, "text": "Node 1"}]}',
            {"nodes": [{"node_number": 1, "text": "Node 1"}]},
            id="node-text",
        ),
        pytest.param(
            {"labels"},
            '{"nodes": [{"node_number": 1, "labels": ["included", "screened"]}]}',
            {"nodes": [{"node_number": 1, "labels": ["included", "screened"]}]},
            id="labels",
        ),
        pytest.param(
            {"flow"},
            '{"nodes": [{"node_number": 1, "points_to": [2, 3]}]}',
            {"nodes": [{"node_number": 1, "points_to": [2, 3]}]},
            id="flow",
        ),
        pytest.param(
            {"additional_texts"},
            '{"additional_texts": ["Figure 1", "Trial flow"]}',
            {"additional_texts": ["Figure 1", "Trial flow"]},
            id="additional-text",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            (
                '{"nodes": ['
                '{"node_number": 1, "text": "Node 1", '
                '"labels": ["included", "screened"], "points_to": [2]}'
                '], "additional_texts": ["Figure 1", "Trial flow"]}'
            ),
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    }
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="all-parts",
        ),
        pytest.param(
            None,
            (
                '{"nodes": ['
                '{"node_number": 1, "text": "Node 1", '
                '"labels": ["included", "screened"], "points_to": [2]}'
                '], "additional_texts": ["Figure 1", "Trial flow"]}'
            ),
            {
                "nodes": [
                    {
                        "node_number": 1,
                        "text": "Node 1",
                        "labels": ["included", "screened"],
                        "points_to": [2],
                    }
                ],
                "additional_texts": ["Figure 1", "Trial flow"],
            },
            id="default-all-parts",
        ),
    ],
)
def test_make_gemini_parse_fn_returns_validated_response(
    tmp_path,
    monkeypatch,
    parts_to_parse,
    response_text,
    expected_result,
):
    img_path = tmp_path / "diagram_1.png"
    partial_flowchart = DummyPartialFlowchart()

    input_list = ["input text", "partial flowchart text", "uploaded image"]

    calls = {
        "gemini_vision_input_list": [],
        "send_gemini_request": [],
    }

    def fake_gemini_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        few_shot_examples=None,
    ):
        calls["gemini_vision_input_list"].append(
            {
                "img_path": img_path,
                "input_text": input_text,
                "partial_flowchart": partial_flowchart,
                "few_shot_examples": few_shot_examples,
            }
        )
        return input_list

    def fake_send_gemini_request(
        result_structure,
        input_list,
        model,
        thinking_level="high",
    ):
        calls["send_gemini_request"].append(
            {
                "result_structure": result_structure,
                "input_list": input_list,
                "model": model,
                "thinking_level": thinking_level,
            }
        )
        return response_text

    monkeypatch.setattr(
        gemini_parse_fn_module,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_parse_fn_module,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    few_shot_examples = [object()]
    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text="Parse this diagram",
        model="gemini-test-model",
        effort="medium",
        parts_to_parse=parts_to_parse,
        few_shot_examples=few_shot_examples,
    )

    result = parse_fn(
        img_path=img_path,
        partial_flowchart=partial_flowchart,
    )

    assert result.model_dump() == expected_result

    assert calls["gemini_vision_input_list"] == [
        {
            "img_path": img_path,
            "input_text": "Parse this diagram",
            "partial_flowchart": partial_flowchart,
            "few_shot_examples": few_shot_examples,
        }
    ]

    assert len(calls["send_gemini_request"]) == 1
    assert calls["send_gemini_request"][0]["input_list"] == input_list
    assert calls["send_gemini_request"][0]["model"] == "gemini-test-model"
    assert calls["send_gemini_request"][0]["thinking_level"] == "medium"

    result_structure = calls["send_gemini_request"][0]["result_structure"]
    assert result_structure is result.__class__


def test_make_gemini_parse_fn_passes_none_partial_flowchart_by_default(
    tmp_path,
    monkeypatch,
):
    img_path = tmp_path / "diagram_1.png"
    input_list = ["input text", "uploaded image"]
    response_text = '{"nodes": [{"node_number": 1, "text": "Node 1"}]}'

    calls = {
        "gemini_vision_input_list": [],
        "send_gemini_request": [],
    }

    def fake_gemini_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        few_shot_examples=None,
    ):
        calls["gemini_vision_input_list"].append(
            {
                "img_path": img_path,
                "input_text": input_text,
                "partial_flowchart": partial_flowchart,
                "few_shot_examples": few_shot_examples,
            }
        )
        return input_list

    def fake_send_gemini_request(
        result_structure,
        input_list,
        model,
        thinking_level="high",
    ):
        calls["send_gemini_request"].append(
            {
                "result_structure": result_structure,
                "input_list": input_list,
                "model": model,
                "thinking_level": thinking_level,
            }
        )
        return response_text

    monkeypatch.setattr(
        gemini_parse_fn_module,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_parse_fn_module,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text="Parse node text",
        model="gemini-test-model",
        parts_to_parse={"node_text"},
    )

    result = parse_fn(img_path=img_path)

    assert result.model_dump() == {
        "nodes": [
            {"node_number": 1, "text": "Node 1"},
        ],
    }

    assert calls["gemini_vision_input_list"] == [
        {
            "img_path": img_path,
            "input_text": "Parse node text",
            "partial_flowchart": None,
            "few_shot_examples": None,
        }
    ]

    assert calls["send_gemini_request"][0]["thinking_level"] == "high"


@pytest.mark.parametrize(
    ("parts_to_parse", "response_text"),
    [
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": 1, "labels": ["wrong field"]}]}',
            id="node-text-response-has-labels-instead-of-text",
        ),
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": 1, "points_to": [2]}]}',
            id="node-text-response-has-flow-instead-of-text",
        ),
        pytest.param(
            {"node_text"},
            '{"additional_texts": ["wrong top-level field"]}',
            id="node-text-response-has-additional-texts-instead-of-nodes",
        ),
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["extra"]}]}',
            id="node-text-response-has-extra-labels",
        ),
        pytest.param(
            {"labels"},
            '{"nodes": [{"node_number": 1, "text": "wrong field"}]}',
            id="labels-response-has-text-instead-of-labels",
        ),
        pytest.param(
            {"labels"},
            '{"nodes": [{"node_number": 1, "points_to": [2]}]}',
            id="labels-response-has-flow-instead-of-labels",
        ),
        pytest.param(
            {"labels"},
            '{"additional_texts": ["wrong top-level field"]}',
            id="labels-response-has-additional-texts-instead-of-nodes",
        ),
        pytest.param(
            {"labels"},
            '{"nodes": [{"node_number": 1, "labels": ["screening"], "text": "extra"}]}',
            id="labels-response-has-extra-text",
        ),
        pytest.param(
            {"flow"},
            '{"nodes": [{"node_number": 1, "text": "wrong field"}]}',
            id="flow-response-has-text-instead-of-points-to",
        ),
        pytest.param(
            {"flow"},
            '{"nodes": [{"node_number": 1, "labels": ["wrong field"]}]}',
            id="flow-response-has-labels-instead-of-points-to",
        ),
        pytest.param(
            {"flow"},
            '{"additional_texts": ["wrong top-level field"]}',
            id="flow-response-has-additional-texts-instead-of-nodes",
        ),
        pytest.param(
            {"flow"},
            '{"nodes": [{"node_number": 1, "points_to": [2], "text": "extra"}]}',
            id="flow-response-has-extra-text",
        ),
        pytest.param(
            {"additional_texts"},
            '{"nodes": [{"node_number": 1, "text": "wrong field"}]}',
            id="additional-text-response-has-nodes-instead-of-additional-texts",
        ),
        pytest.param(
            {"additional_texts"},
            '{"additional_texts": ["Figure 1"], "nodes": [{"node_number": 1, "text": "extra"}]}',
            id="additional-text-response-has-extra-nodes",
        ),
        pytest.param(
            {"node_text", "labels"},
            '{"nodes": [{"node_number": 1, "text": "Node 1"}]}',
            id="node-text-labels-response-missing-labels",
        ),
        pytest.param(
            {"node_text", "labels"},
            '{"nodes": [{"node_number": 1, "labels": ["screening"]}]}',
            id="node-text-labels-response-missing-text",
        ),
        pytest.param(
            {"node_text", "labels"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["screening"], "points_to": [2]}]}',
            id="node-text-labels-response-has-extra-flow",
        ),
        pytest.param(
            {"node_text", "flow"},
            '{"nodes": [{"node_number": 1, "text": "Node 1"}]}',
            id="node-text-flow-response-missing-flow",
        ),
        pytest.param(
            {"node_text", "flow"},
            '{"nodes": [{"node_number": 1, "points_to": [2]}]}',
            id="node-text-flow-response-missing-text",
        ),
        pytest.param(
            {"node_text", "flow"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "points_to": [2], "labels": ["extra"]}]}',
            id="node-text-flow-response-has-extra-labels",
        ),
        pytest.param(
            {"labels", "flow"},
            '{"nodes": [{"node_number": 1, "labels": ["screening"]}]}',
            id="labels-flow-response-missing-flow",
        ),
        pytest.param(
            {"labels", "flow"},
            '{"nodes": [{"node_number": 1, "points_to": [2]}]}',
            id="labels-flow-response-missing-labels",
        ),
        pytest.param(
            {"labels", "flow"},
            '{"nodes": [{"node_number": 1, "labels": ["screening"], "points_to": [2], "text": "extra"}]}',
            id="labels-flow-response-has-extra-text",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["screening"]}]}',
            id="node-text-labels-flow-response-missing-flow",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "points_to": [2]}]}',
            id="node-text-labels-flow-response-missing-labels",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            '{"nodes": [{"node_number": 1, "labels": ["screening"], "points_to": [2]}]}',
            id="node-text-labels-flow-response-missing-text",
        ),
        pytest.param(
            {"node_text", "labels", "flow"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["screening"], "points_to": [2]}], "additional_texts": ["extra"]}',
            id="node-text-labels-flow-response-has-extra-additional-texts",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["screening"], "points_to": [2]}]}',
            id="all-parts-response-missing-additional-texts",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            '{"additional_texts": ["Figure 1"]}',
            id="all-parts-response-missing-nodes",
        ),
        pytest.param(
            {"node_text", "labels", "flow", "additional_texts"},
            '{"nodes": [{"node_number": 1, "text": "Node 1", "labels": ["screening"]}], "additional_texts": ["Figure 1"]}',
            id="all-parts-response-missing-flow",
        ),
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": "one", "text": "Node 1"}]}',
            id="node-number-wrong-type",
        ),
        pytest.param(
            {"node_text"},
            '{"nodes": [{"node_number": 1, "text": 123}]}',
            id="text-wrong-type",
        ),
        pytest.param(
            {"labels"},
            '{"nodes": [{"node_number": 1, "labels": "screening"}]}',
            id="labels-wrong-type",
        ),
        pytest.param(
            {"flow"},
            '{"nodes": [{"node_number": 1, "points_to": "2"}]}',
            id="points-to-wrong-type",
        ),
        pytest.param(
            {"additional_texts"},
            '{"additional_texts": "Figure 1"}',
            id="additional-texts-wrong-type",
        ),
        pytest.param(
            {"node_text"},
            '{"nodes": []',
            id="invalid-json",
        ),
    ],
)
def test_make_gemini_parse_fn_raises_if_response_does_not_match_schema(
    tmp_path,
    monkeypatch,
    parts_to_parse,
    response_text,
):
    img_path = tmp_path / "diagram_1.png"

    def fake_gemini_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        few_shot_examples=None,
    ):
        return ["input text", "uploaded image"]

    def fake_send_gemini_request(
        result_structure,
        input_list,
        model,
        thinking_level="high",
    ):
        return response_text

    monkeypatch.setattr(
        gemini_parse_fn_module,
        "gemini_vision_input_list",
        fake_gemini_vision_input_list,
    )
    monkeypatch.setattr(
        gemini_parse_fn_module,
        "send_gemini_request",
        fake_send_gemini_request,
    )

    parse_fn = gemini_parse_fn_module.make_gemini_parse_fn(
        input_text="Parse flowchart",
        model="gemini-test-model",
        parts_to_parse=parts_to_parse,
    )

    with pytest.raises(ValueError):
        parse_fn(img_path=img_path)
