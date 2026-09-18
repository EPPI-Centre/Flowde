from pathlib import Path

from flowde._run_settings import model_function
from flowde.api_utils.openai_utils import openai_vision_input_list, send_openai_request
from flowde.classify_fns.classify_types import (
    BinaryClassification,
    Classification,
    ClassificationFunction,
    LabelType,
)
from flowde.pricing import TokenPrices
from flowde.utils import VisionFewShotExample


def make_openai_classify_fn(
    input_text: str,
    model: str,
    result_structure: type[Classification[LabelType]] = BinaryClassification,
    effort: str = "high",
    few_shot_examples: list[VisionFewShotExample] | None = None,
    from_azure: bool = False,
    *,
    token_prices: TokenPrices | None = None,
) -> ClassificationFunction[LabelType]:
    """
    Create an image classifier using the OpenAI or Azure OpenAI Responses API.

    Parameters
    ----------
    input_text : str
        Instructions sent with each target image. The prompt defines what the
        labels mean, such as `1` for a flowchart and `0` for another image.
    model : str
        OpenAI model ID, or Azure deployment name when `from_azure=True`.
        The selected model must support image inputs, structured outputs and
        the requested reasoning effort.
    result_structure : type[Classification[LabelType]], optional
        Pydantic response class containing a `label` field. Defaults to
        [`BinaryClassification`](data-types.md#flowde.classify_fns.classify_types.BinaryClassification),
        whose labels are the integers `0` and `1`. You can supply a different
        classification schema for other labels, or
        [`RotationClassification`](data-types.md#flowde.classify_fns.classify_types.RotationClassification)
        for labels of `0`, `90`, `180` or `270`. For rotation, the prompt should
        request the clockwise correction needed to make the image upright.
        Custom schemas must be compatible with OpenAI's
        [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
    effort : str, optional
        Value sent as the Responses API's `reasoning.effort`. Defaults to
        `"high"`. Supported values depend on the selected model; the factory
        does not check model compatibility before returning the classifier.
    few_shot_examples : list[VisionFewShotExample] | None, optional
        Worked examples included before the target image in every request,
        in list order. Each
        [`VisionFewShotExample`][flowde.utils.VisionFewShotExample] supplies an
        image and an `expected_output_path` pointing to its expected JSON
        response, such as `{"label": 1}`. The expected response should follow
        `result_structure`; the expected JSON is not validated against that
        schema before sending. Defaults to `None`, meaning no worked examples.
    from_azure : bool, optional
        Whether to use Azure OpenAI. Defaults to `False`, which reads
        `OPENAI_API_KEY`. `True` reads `AZURE_API_KEY` and `AZURE_API_BASE`.
        The Azure base URL should end in `/openai/v1/`, without `responses`.
        Environment settings are also loaded from `.env` when the returned
        classifier is called.
    token_prices : TokenPrices | None, optional
        Optional [`TokenPrices`][flowde.pricing.TokenPrices] override for usage
        cost estimates, in USD per million tokens. Defaults to `None`, which
        looks up `model` in Flowde's bundled OpenAI price table, including for
        Azure. If prices or required usage figures are unavailable, the cost
        estimate is unavailable. An Azure deployment name that differs from
        the model ID may need an explicit override. This affects estimates,
        not provider billing or the classification request.

    Returns
    -------
    ClassificationFunction[LabelType]
        Callable accepting `img_path: Path` and returning the validated `label`
        value, rather than the complete Pydantic response. With the default
        schema, the return value is integer `0` or `1`. The callable exposes
        `result_structure` and `run_settings` for Flowde's saved-run checks.

    Raises
    ------
    pydantic.ValidationError
        When the returned classifier is called and the model's response is
        not valid JSON matching `result_structure`.
    OSError
        When the returned classifier cannot read a target image, example image
        or expected-response file.
    json.JSONDecodeError
        When an example's expected-response file is read during classification
        and does not contain valid JSON.
    openai.OpenAIError
        When calling the returned classifier fails because of credentials,
        connectivity or an API error, including rejected request settings.

    Notes
    -----
    Creating the classifier makes no API request and does not check credentials.
    Calling the classifier sends the prompt, target image and any worked examples
    to the provider, then validates the response. Ordinary OpenAI requests upload
    image files; Azure requests include image bytes as base64 data URLs.

    The classifier does not save labels or copy images. You can pass the
    classifier to [`classify_imgs()`][flowde.classify_imgs.classify_imgs] for a
    saved classification run, or to
    [`rotate_imgs()`][flowde.rotate_imgs.rotate_imgs] with a rotation schema and
    prompt. Flowde checks the declared provider, prompt, model, effort, response
    schema and worked examples when resuming a run, including the contents of
    example image and expected-response files.

    Examples
    --------
    Create a binary classifier without making a model request:

    ```python
    from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn

    classify_fn = make_openai_classify_fn(
        input_text="Return 1 if the image is a flowchart, otherwise return 0.",
        model="gpt-5.6-luna",
        effort="medium",
    )
    ```

    """

    def openai_classify(img_path: Path) -> LabelType:
        input_list = openai_vision_input_list(
            img_path,
            input_text,
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

        return result_structure.model_validate_json(response_text).label

    return model_function(
        openai_classify,
        result_structure=result_structure,
        provider=("Azure OpenAI" if from_azure else "OpenAI"),
        input_text=input_text,
        model=model,
        effort=effort,
        few_shot_examples=few_shot_examples,
    )
