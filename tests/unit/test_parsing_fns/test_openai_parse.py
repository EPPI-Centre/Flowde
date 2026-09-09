from pathlib import Path

import pytest

import flowde.parsing_fns.openai_parse as openai_parse_fn_module

pytestmark = pytest.mark.parametrize(
    "from_azure", [False, True], ids=["openai", "azure"]
)


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
def test_make_openai_parse_fn_returns_validated_response(
    tmp_path,
    monkeypatch,
    parts_to_parse,
    response_text,
    expected_result,
    from_azure,
):
    img_path = tmp_path / "diagram_1.png"
    partial_flowchart = DummyPartialFlowchart()

    input_list = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Parse this diagram"},
                {"type": "input_image", "file_id": "file-123", "detail": "high"},
            ],
        }
    ]

    calls = {
        "openai_vision_input_list": [],
        "send_openai_request": [],
    }

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        from_azure=False,
        few_shot_examples=None,
    ):
        calls["openai_vision_input_list"].append(
            {
                "img_path": img_path,
                "input_text": input_text,
                "partial_flowchart": partial_flowchart,
                "from_azure": from_azure,
                "few_shot_examples": few_shot_examples,
            }
        )
        return input_list

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort="high",
        from_azure=False,
    ):
        calls["send_openai_request"].append(
            {
                "result_structure": result_structure,
                "model": model,
                "input_list": input_list,
                "effort": effort,
                "from_azure": from_azure,
            }
        )
        return response_text

    monkeypatch.setattr(
        openai_parse_fn_module,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_parse_fn_module,
        "send_openai_request",
        fake_send_openai_request,
    )

    few_shot_examples = [object()]
    parse_fn = openai_parse_fn_module.make_openai_parse_fn(
        input_text="Parse this diagram",
        model="openai-test-model",
        effort="medium",
        parts_to_parse=parts_to_parse,
        few_shot_examples=few_shot_examples,
        from_azure=from_azure,
    )

    result = parse_fn(
        img_path=img_path,
        partial_flowchart=partial_flowchart,
    )

    assert result.model_dump() == expected_result

    assert calls["openai_vision_input_list"] == [
        {
            "img_path": img_path,
            "input_text": "Parse this diagram",
            "partial_flowchart": partial_flowchart,
            "from_azure": from_azure,
            "few_shot_examples": few_shot_examples,
        }
    ]

    assert len(calls["send_openai_request"]) == 1
    assert calls["send_openai_request"][0]["result_structure"] is result.__class__
    assert calls["send_openai_request"][0]["model"] == "openai-test-model"
    assert calls["send_openai_request"][0]["input_list"] == input_list
    assert calls["send_openai_request"][0]["effort"] == "medium"
    assert calls["send_openai_request"][0]["from_azure"] is from_azure


def test_make_openai_parse_fn_passes_none_partial_flowchart_by_default(
    tmp_path,
    monkeypatch,
    from_azure,
):
    img_path = tmp_path / "diagram_1.png"
    input_list = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Parse node text"},
                {"type": "input_image", "file_id": "file-123", "detail": "high"},
            ],
        }
    ]
    response_text = '{"nodes": [{"node_number": 1, "text": "Node 1"}]}'

    calls = {
        "openai_vision_input_list": [],
        "send_openai_request": [],
    }

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        from_azure=False,
        few_shot_examples=None,
    ):
        calls["openai_vision_input_list"].append(
            {
                "img_path": img_path,
                "input_text": input_text,
                "partial_flowchart": partial_flowchart,
                "from_azure": from_azure,
                "few_shot_examples": few_shot_examples,
            }
        )
        return input_list

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort="high",
        from_azure=False,
    ):
        calls["send_openai_request"].append(
            {
                "result_structure": result_structure,
                "model": model,
                "input_list": input_list,
                "effort": effort,
                "from_azure": from_azure,
            }
        )
        return response_text

    monkeypatch.setattr(
        openai_parse_fn_module,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_parse_fn_module,
        "send_openai_request",
        fake_send_openai_request,
    )

    parse_fn = openai_parse_fn_module.make_openai_parse_fn(
        input_text="Parse node text",
        model="openai-test-model",
        parts_to_parse={"node_text"},
        # Leave the backend unspecified in the OpenAI case to check its default too.
        **({"from_azure": True} if from_azure else {}),
    )

    result = parse_fn(img_path=img_path)

    assert result.model_dump() == {
        "nodes": [
            {"node_number": 1, "text": "Node 1"},
        ],
    }

    assert calls["openai_vision_input_list"] == [
        {
            "img_path": img_path,
            "input_text": "Parse node text",
            "partial_flowchart": None,
            "from_azure": from_azure,
            "few_shot_examples": None,
        }
    ]

    assert calls["send_openai_request"][0]["model"] == "openai-test-model"
    assert calls["send_openai_request"][0]["input_list"] == input_list
    assert calls["send_openai_request"][0]["effort"] == "high"
    assert calls["send_openai_request"][0]["from_azure"] is from_azure
    assert calls["send_openai_request"][0]["result_structure"] is result.__class__


# TODO: This needs to test all things that can go wrong and then need to do it for the gemini_parse_fn as well
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
def test_make_openai_parse_fn_raises_if_response_does_not_match_schema(
    tmp_path,
    monkeypatch,
    parts_to_parse,
    response_text,
    from_azure,
):
    img_path = tmp_path / "diagram_1.png"

    def fake_openai_vision_input_list(
        img_path,
        input_text,
        partial_flowchart=None,
        from_azure=False,
        few_shot_examples=None,
    ):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": input_text},
                    {"type": "input_image", "file_id": "file-123", "detail": "high"},
                ],
            }
        ]

    def fake_send_openai_request(
        result_structure,
        model,
        input_list,
        effort="high",
        from_azure=False,
    ):
        return response_text

    monkeypatch.setattr(
        openai_parse_fn_module,
        "openai_vision_input_list",
        fake_openai_vision_input_list,
    )
    monkeypatch.setattr(
        openai_parse_fn_module,
        "send_openai_request",
        fake_send_openai_request,
    )

    parse_fn = openai_parse_fn_module.make_openai_parse_fn(
        input_text="Parse flowchart",
        model="openai-test-model",
        parts_to_parse=parts_to_parse,
        from_azure=from_azure,
    )

    with pytest.raises(ValueError):
        parse_fn(img_path=img_path)
