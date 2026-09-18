from pathlib import Path

from flowde._run_settings import model_function
from flowde.api_utils.gemini_utils import gemini_vision_input_list, send_gemini_request
from flowde.classify_fns.classify_types import (
    BinaryClassification,
    Classification,
    ClassificationFunction,
    LabelType,
)
from flowde.pricing import TokenPrices
from flowde.utils import VisionFewShotExample


def make_gemini_classify_fn(
    input_text: str,
    model: str,
    result_structure: type[Classification[LabelType]] = BinaryClassification,
    effort: str = "high",
    few_shot_examples: list[VisionFewShotExample] | None = None,
    *,
    token_prices: TokenPrices | None = None,
) -> ClassificationFunction[LabelType]:
    """
    Create an image classifier using Gemini with a structured JSON response.

    Parameters
    ----------
    input_text : str
        Instructions sent with each target image. The prompt defines what the
        labels mean, such as `1` for a flowchart and `0` for another image.
    model : str
        Gemini model ID. The selected model must support image inputs,
        structured outputs and the requested thinking level.
    result_structure : type[Classification[LabelType]], optional
        Pydantic response class containing a `label` field. Defaults to
        [`BinaryClassification`](data-types.md#flowde.classify_fns.classify_types.BinaryClassification),
        whose labels are the integers `0` and `1`. You can supply a different
        classification schema for other labels, or
        [`RotationClassification`](data-types.md#flowde.classify_fns.classify_types.RotationClassification)
        for labels of `0`, `90`, `180` or `270`. For rotation, the prompt should
        request the clockwise correction needed to make the image upright.
        The class's JSON schema is sent to Gemini as `response_json_schema`
        and must be supported by the selected model.
    effort : str, optional
        Value passed to Gemini's `ThinkingConfig` as `thinking_level`. Defaults
        to `"high"`. Supported values depend on the selected model; the factory
        does not check model compatibility before returning the classifier.
    few_shot_examples : list[VisionFewShotExample] | None, optional
        Worked examples included before the target image in every request,
        in list order. Each
        [`VisionFewShotExample`][flowde.utils.VisionFewShotExample] supplies an
        image and an `expected_output_path` pointing to its expected JSON
        response, such as `{"label": 1}`. The expected response should follow
        `result_structure`; the expected JSON is not validated against that
        schema before sending. Defaults to `None`, meaning no worked examples.
    token_prices : TokenPrices | None, optional
        Optional [`TokenPrices`][flowde.pricing.TokenPrices] override for usage
        cost estimates, in USD per million tokens. Defaults to `None`, which
        looks up `model` in Flowde's bundled Gemini price table. If prices or
        required usage figures are unavailable, the cost estimate is unavailable.
        This affects estimates, not provider billing or the classification
        request.

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
        not valid JSON matching `result_structure`, or no response text is
        available. Invalid SDK configuration can also raise this error.
    OSError
        When the returned classifier cannot read a target image, example image
        or expected-response file.
    json.JSONDecodeError
        When an example's expected-response file is read during classification
        and does not contain valid JSON.
    google.genai.errors.APIError
        When calling the returned classifier encounters an API error, such as
        rejected credentials, a model name or request settings.

    Notes
    -----
    Creating the classifier makes no API request and does not check credentials.
    When the classifier is called, Flowde loads environment settings from `.env`
    and reads `GOOGLE_GENAI_API_KEY`. The request helper uploads the target and
    example images, sends the prompt and any worked examples to Gemini, then
    validates the response. SDK setup, file access and connection errors
    propagate to the caller.

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
    from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

    classify_fn = make_gemini_classify_fn(
        input_text="Return 1 if the image is a flowchart, otherwise return 0.",
        model="gemini-3.1-flash-lite",
        effort="medium",
    )
    ```

    """

    def gemini_classify(img_path: Path) -> LabelType:
        input_list = gemini_vision_input_list(
            img_path,
            input_text,
            few_shot_examples=few_shot_examples,
        )

        response_text = send_gemini_request(
            **({"token_prices": token_prices} if token_prices is not None else {}),
            result_structure=result_structure,
            model=model,
            input_list=input_list,
            thinking_level=effort,
        )

        return result_structure.model_validate_json(response_text).label

    return model_function(
        gemini_classify,
        result_structure=result_structure,
        provider="Gemini",
        input_text=input_text,
        model=model,
        effort=effort,
        few_shot_examples=few_shot_examples,
    )
