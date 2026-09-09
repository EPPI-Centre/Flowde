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
