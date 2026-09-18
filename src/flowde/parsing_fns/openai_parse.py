from pathlib import Path
from typing import cast

from pydantic import BaseModel

from flowde._run_settings import model_function
from flowde.api_utils.openai_utils import openai_vision_input_list, send_openai_request
from flowde.parsing_fns.parsing_types import (
    ParseType,
    ParsingFunction,
)
from flowde.parsing_fns.parsing_utils import get_result_structure
from flowde.pricing import TokenPrices
from flowde.utils import VisionFewShotExample


def make_openai_parse_fn(
    input_text: str,
    model: str,
    effort: str = "high",
    parts_to_parse: set[ParseType] | None = None,
    result_structure: type[BaseModel] | None = None,
    few_shot_examples: list[VisionFewShotExample] | None = None,
    from_azure: bool = False,
    *,
    token_prices: TokenPrices | None = None,
) -> ParsingFunction:
    """
    Create an image parser using the OpenAI or Azure OpenAI Responses API.

    Parameters
    ----------
    input_text : str
        Instructions sent with each target image. The prompt describes what to
        extract and how to interpret the diagram. When using earlier node data
        as context, the prompt can ask the model to preserve those node numbers.
    model : str
        OpenAI model ID, or Azure deployment name when `from_azure=True`.
        The selected model must support image inputs, structured outputs and
        the requested reasoning effort.
    effort : str, optional
        Value sent as the Responses API's `reasoning.effort`. Defaults to
        `"high"`. Supported values depend on the selected model; the factory
        does not check model compatibility before returning the parser.
    parts_to_parse : set[ParseType] | None, optional
        Parts to include in a generated flowchart response schema. You can
        request one or several of the following parts:

        - `"node_text"`: `nodes` with `node_number` and `text`.
        - `"labels"`: `nodes` with `node_number` and `labels`.
        - `"flow"`: `nodes` with `node_number` and `points_to`.
        - `"additional_texts"`: a top-level `additional_texts` list.

        Node-based parts share one `nodes` list when requested together.
        Unselected fields are omitted, and the generated schema rejects extra
        fields. Defaults to `None`: all four parts are requested unless a
        custom `result_structure` is supplied. Cannot be supplied together
        with `result_structure`.
    result_structure : type[BaseModel] | None, optional
        Custom Pydantic response class. Defaults to `None`, which generates
        the schema from `parts_to_parse`. A custom class replaces the standard
        flowchart schema and must be compatible with OpenAI's
        [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
        The returned parser validates responses against this class and returns
        an instance of the class. Cannot be supplied with `parts_to_parse`.
    few_shot_examples : list[VisionFewShotExample] | None, optional
        Worked examples included before the target image in every request,
        in list order. Each
        [`VisionFewShotExample`][flowde.utils.VisionFewShotExample] supplies an
        image, an `expected_output_path` containing the expected JSON response,
        and optionally a `partial_flowchart` containing context for that example.
        The expected JSON should match the selected parts or custom schema,
        without a benchmark ground truth's `options` wrapper. The expected
        JSON is not validated against the response schema before sending.
        Defaults to `None`, meaning no worked examples.
    from_azure : bool, optional
        Whether to use Azure OpenAI. Defaults to `False`, which reads
        `OPENAI_API_KEY`. `True` reads `AZURE_API_KEY` and `AZURE_API_BASE`.
        The Azure base URL should end in `/openai/v1/`, without `responses`.
        Environment settings are also loaded from `.env` when the returned
        parser is called.
    token_prices : TokenPrices | None, optional
        Optional [`TokenPrices`][flowde.pricing.TokenPrices] override for usage
        cost estimates, in USD per million tokens. Defaults to `None`, which
        looks up `model` in Flowde's bundled OpenAI price table, including for
        Azure. If prices or required usage figures are unavailable, the cost
        estimate is unavailable. An Azure deployment name that differs from
        the model ID may need an explicit override. This affects estimates,
        not provider billing or the parsing request.

    Returns
    -------
    ParsingFunction
        Callable accepting `img_path: Path` and optional
        `partial_flowchart: BaseModel | None`, defaulting to `None`. The callable
        returns a validated instance of its `result_structure` class. That class
        is either the supplied custom class or the generated flowchart schema.
        The callable also exposes `run_settings` for Flowde's saved-run checks.

    Raises
    ------
    ValueError
        When both `parts_to_parse` and `result_structure` are supplied. This is
        checked when creating the parser, before any request is made.
    pydantic.ValidationError
        When the returned parser is called and the model's response is not
        valid JSON matching the chosen response schema.
    OSError
        When the returned parser cannot read a target image, example image or
        expected-response file.
    json.JSONDecodeError
        When an example's expected-response file is read during parsing and
        does not contain valid JSON.
    openai.OpenAIError
        When calling the returned parser fails because of credentials,
        connectivity or an API error, including rejected request settings.

    Notes
    -----
    Creating the parser makes no API request and does not check credentials.
    Calling the parser sends the prompt, target image, optional partial context
    and any worked examples to the provider. Ordinary OpenAI requests upload
    image files; Azure requests include image bytes as base64 data URLs.

    The target's `partial_flowchart` is serialised as JSON text alongside the
    image. Context supplies information to the model; context fields are not
    automatically merged into the response. The parser's output schema chooses
    which fields to request. The generated flowchart schema does not check that
    returned node numbers match the supplied context or that connections refer
    to existing nodes.

    The parser returns a Pydantic model without saving a JSON file. You can pass
    the parser to [`parse_imgs()`][flowde.parse_imgs.parse_imgs] to process a
    directory and save the results. Flowde checks the declared provider, prompt,
    model, effort, response schema and worked examples when resuming a run,
    including the contents of example image and expected-response files.

    Examples
    --------
    Create a parser for node text without making a model request:

    ```python
    from flowde.parsing_fns.openai_parse import make_openai_parse_fn

    parse_fn = make_openai_parse_fn(
        input_text=(
            "Extract each node's text. Use consecutive node numbers starting at 1."
        ),
        model="gpt-5.6-luna",
        effort="medium",
        parts_to_parse={"node_text"},
    )
    ```

    """
    result_structure = get_result_structure(
        parts_to_parse=parts_to_parse,
        result_structure=result_structure,
    )

    def openai_parse(
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> BaseModel:
        input_list = openai_vision_input_list(
            img_path=img_path,
            input_text=input_text,
            partial_flowchart=partial_flowchart,
            from_azure=from_azure,
            few_shot_examples=few_shot_examples,
        )

        response_text = send_openai_request(
            **({"token_prices": token_prices} if token_prices is not None else {}),
            result_structure=result_structure,
            model=model,
            input_list=input_list,
            effort=effort,
            from_azure=from_azure,
        )

        return result_structure.model_validate_json(response_text)

    return cast(
        "ParsingFunction",
        model_function(
            openai_parse,
            result_structure=result_structure,
            provider=("Azure OpenAI" if from_azure else "OpenAI"),
            input_text=input_text,
            model=model,
            effort=effort,
            few_shot_examples=few_shot_examples,
        ),
    )
