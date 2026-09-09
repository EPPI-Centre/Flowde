from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from flowde.api_utils import gemini_utils, openai_utils
from flowde.classify_fns import gemini_classify_fn, openai_classify_fn
from flowde.parsing_fns import gemini_parse, openai_parse
from flowde.pricing import GEMINI_PRICES_PER_1M, OPENAI_PRICES_PER_1M, TokenPrices
from flowde.usage import RequestUsage, usage_scope


@pytest.fixture
def openai_response():
    return SimpleNamespace(
        output_text='{"answer": "yes"}',
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            input_tokens_details=SimpleNamespace(
                cached_tokens=400, cache_write_tokens=100
            ),
            output_tokens_details=SimpleNamespace(reasoning_tokens=150),
        ),
    )


@pytest.fixture
def gemini_response():
    return SimpleNamespace(
        text='{"answer": "yes"}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=100,
            thoughts_token_count=100,
            cached_content_token_count=400,
            total_token_count=1200,
        ),
    )


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_openai_prices_cache_reads_writes_and_reasoning_without_double_counting(
    openai_response,
    from_azure,
):
    result = openai_utils.openai_request_usage(
        openai_response, "gpt-5.6-luna", from_azure=from_azure
    )

    # 500 ordinary input + 400 cache reads + 100 cache writes; output already
    # includes the 150 reasoning tokens. Rates: .20/.02/.25/1.20 per million.
    assert result.total_tokens == 1200
    assert result.output_tokens == 200
    assert result.cost == pytest.approx(0.000373)
    assert result.provider == ("Azure OpenAI" if from_azure else "OpenAI")


def test_gemini_prices_cached_input_and_adds_thinking_to_output_once(gemini_response):
    result = gemini_utils.gemini_request_usage(gemini_response, "gemini-3.5-flash")

    # 600 ordinary input at $1.50, 400 cached at $.15, 200 output at $9 per million.
    assert result.total_tokens == 1200
    assert result.output_tokens == 200
    assert result.cost == pytest.approx(0.00276)


@pytest.mark.parametrize(
    ("prices", "input_tokens", "expected"),
    [
        pytest.param(
            GEMINI_PRICES_PER_1M["gemini-3.1-pro-preview"],
            200_000,
            0.412,
            id="gemini-at-boundary",
        ),
        pytest.param(
            GEMINI_PRICES_PER_1M["gemini-3.1-pro-preview"],
            200_001,
            0.818004,
            id="gemini-above-boundary",
        ),
        pytest.param(
            OPENAI_PRICES_PER_1M["gpt-5.6-luna"],
            272_000,
            0.0556,
            id="openai-at-boundary",
        ),
        pytest.param(
            OPENAI_PRICES_PER_1M["gpt-5.6-luna"],
            272_001,
            0.1106004,
            id="openai-above-boundary",
        ),
        pytest.param(
            GEMINI_PRICES_PER_1M["gemini-3.5-flash-lite"],
            200_001,
            0.0625003,
            id="flat-rate-model-stays-flat",
        ),
    ],
)
def test_long_context_rates_apply_to_the_whole_request_only_above_the_boundary(
    prices, input_tokens, expected
):
    assert prices.estimate(input_tokens, 1000) == pytest.approx(expected)


@pytest.mark.parametrize("provider", ["openai", "azure", "gemini"])
def test_unknown_models_still_report_tokens(provider, openai_response, gemini_response):
    if provider in {"openai", "azure"}:
        result = openai_utils.openai_request_usage(
            openai_response, "unknown-model", from_azure=provider == "azure"
        )
    else:
        result = gemini_utils.gemini_request_usage(gemini_response, "unknown-model")

    assert result.model == "unknown-model"
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.total_tokens == 1200
    assert result.cost is None


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize(
    "usage", [None, SimpleNamespace()], ids=["no-metadata", "empty-metadata"]
)
def test_missing_usage_is_unavailable_instead_of_zero(provider, usage):
    if provider == "openai":
        result = openai_utils.openai_request_usage(
            SimpleNamespace(usage=usage), "gpt-5.6-luna"
        )
    else:
        result = gemini_utils.gemini_request_usage(
            SimpleNamespace(usage_metadata=usage), "gemini-3.5-flash"
        )

    assert result.total_tokens is None
    assert result.cost is None


@pytest.mark.parametrize(
    ("provider", "model", "expected_cost"),
    [
        pytest.param("OpenAI", "gpt-5.6-luna", 0.0015, id="openai-known-model"),
        pytest.param("OpenAI", "unknown-model", 0.0015, id="openai-unknown-model"),
        pytest.param("Azure OpenAI", "gpt-5.6-luna", 0.0015, id="azure-known-model"),
        pytest.param(
            "Azure OpenAI", "my-deployment", 0.0015, id="azure-custom-deployment"
        ),
        pytest.param("Gemini", "gemini-3.5-flash", 0.0014, id="gemini-known-model"),
        pytest.param("Gemini", "unknown-model", 0.0014, id="gemini-unknown-model"),
    ],
)
def test_explicit_prices_override_defaults_and_price_unknown_models(
    openai_response,
    gemini_response,
    provider,
    model,
    expected_cost,
):
    prices = TokenPrices(input=1, cached_input=0.5, cache_write=2, output=3)
    if provider == "Gemini":
        priced = gemini_utils.gemini_request_usage(
            gemini_response, model, token_prices=prices
        )
    else:
        priced = openai_utils.openai_request_usage(
            openai_response,
            model,
            from_azure=provider == "Azure OpenAI",
            token_prices=prices,
        )

    # OpenAI: (500*1 + 400*0.5 + 100*2 + 200*3) / 1M = $0.0015.
    # Gemini: (600*1 + 400*0.5 + 200*3) / 1M = $0.0014 (no cache writes).
    assert priced.total_tokens == 1200
    assert priced.provider == provider
    assert priced.model == model
    assert priced.cost == pytest.approx(expected_cost)


class Answer(BaseModel):
    answer: str


@pytest.fixture(params=["openai", "azure", "gemini"])
def send_request(request, monkeypatch, openai_response, gemini_response):
    """Replace only the network client; keep request and accounting code real."""
    provider = request.param
    if provider == "gemini":
        monkeypatch.setattr(gemini_utils, "load_dotenv", lambda: None)
        client = Mock()
        client.models.generate_content.return_value = gemini_response
        monkeypatch.setattr(gemini_utils.genai, "Client", Mock(return_value=client))

        def send(print_cost, *, token_prices=None):
            return gemini_utils.send_gemini_request(
                Answer,
                [],
                "gemini-3.5-flash",
                print_cost=print_cost,
                token_prices=token_prices,
            )
    else:
        monkeypatch.setattr(openai_utils, "load_dotenv", lambda: None)
        client = Mock()
        client.responses.parse.return_value = openai_response
        monkeypatch.setattr(openai_utils, "OpenAI", Mock(return_value=client))

        def send(print_cost, *, token_prices=None):
            return openai_utils.send_openai_request(
                Answer,
                "gpt-5.6-luna",
                [],
                from_azure=provider == "azure",
                print_cost=print_cost,
                token_prices=token_prices,
            )

    return send


@pytest.mark.parametrize(
    ("send_request", "provider", "model", "expected_cost"),
    [
        pytest.param("openai", "OpenAI", "gpt-5.6-luna", 0.0015, id="openai"),
        pytest.param("azure", "Azure OpenAI", "gpt-5.6-luna", 0.0015, id="azure"),
        pytest.param("gemini", "Gemini", "gemini-3.5-flash", 0.0014, id="gemini"),
    ],
    indirect=["send_request"],
)
def test_request_uses_custom_prices_for_reported_usage(
    send_request, provider, model, expected_cost
):
    prices = TokenPrices(input=1, cached_input=0.5, cache_write=2, output=3)
    reports = []

    with usage_scope(reports.append):
        result = send_request(print_cost=False, token_prices=prices)

    assert result == '{"answer": "yes"}'
    assert len(reports) == 1
    assert reports[0].provider == provider
    assert reports[0].model == model
    assert reports[0].total_tokens == 1200
    assert reports[0].cost == pytest.approx(expected_cost)


@pytest.mark.parametrize("print_cost", [False, True])
def test_request_sends_usage_to_active_collector_without_printing(
    send_request, print_cost, capsys
):
    reports = []
    with usage_scope(reports.append):
        result = send_request(print_cost)

    assert result == '{"answer": "yes"}'
    assert len(reports) == 1
    assert isinstance(reports[0], RequestUsage)
    assert reports[0].total_tokens == 1200
    assert reports[0].cost is not None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("print_cost", [False, True])
def test_standalone_requests_only_print_when_requested(
    send_request, print_cost, capsys
):
    assert send_request(print_cost) == '{"answer": "yes"}'

    output = capsys.readouterr().out
    if print_cost:
        assert "request cost:" in output
        assert "total_tokens=1200" in output
    else:
        assert output == ""


@pytest.mark.parametrize(
    ("factory", "provider", "from_azure", "task"),
    [
        (openai_classify_fn.make_openai_classify_fn, "openai", False, "classify"),
        (openai_classify_fn.make_openai_classify_fn, "openai", True, "classify"),
        (gemini_classify_fn.make_gemini_classify_fn, "gemini", False, "classify"),
        (openai_parse.make_openai_parse_fn, "openai", False, "parse"),
        (openai_parse.make_openai_parse_fn, "openai", True, "parse"),
        (gemini_parse.make_gemini_parse_fn, "gemini", False, "parse"),
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
def test_factories_forward_custom_prices_without_changing_results(
    monkeypatch, factory, provider, from_azure, task
):
    prices = TokenPrices(input=1, output=4, cached_input=0.1)
    answer = '{"label": 1}' if task == "classify" else '{"answer": "yes"}'
    sender = Mock(return_value=answer)
    # Patch the helpers in the module where this factory imports them.
    monkeypatch.setattr(
        f"{factory.__module__}.{provider}_vision_input_list", Mock(return_value=[])
    )
    monkeypatch.setattr(f"{factory.__module__}.send_{provider}_request", sender)
    options = {"from_azure": from_azure} if provider == "openai" else {}
    if task == "parse":
        options["result_structure"] = Answer

    fn = factory("Read this image", "my-model", token_prices=prices, **options)
    result = fn(img_path="unused.png")

    assert result == (1 if task == "classify" else Answer(answer="yes"))
    assert sender.call_args.kwargs["token_prices"] is prices


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_uncached_requests_without_optional_breakdowns_still_have_usage(provider):
    if provider == "openai":
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200)
        )
        result = openai_utils.openai_request_usage(response, "gpt-5.6-luna")
        expected_cost = 0.00044
    else:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1000, candidates_token_count=200
            )
        )
        result = gemini_utils.gemini_request_usage(response, "gemini-3.5-flash")
        expected_cost = 0.0033

    assert result.total_tokens == 1200
    assert result.cost == pytest.approx(expected_cost)
