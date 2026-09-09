from pathlib import Path
from typing import cast

from pydantic import BaseModel

from flowde._run_settings import model_function
from flowde.api_utils.gemini_utils import gemini_vision_input_list, send_gemini_request
from flowde.parsing_fns.parsing_types import (
    ParseType,
    ParsingFunction,
)
from flowde.parsing_fns.parsing_utils import get_result_structure
from flowde.pricing import TokenPrices
from flowde.utils import VisionFewShotExample


def make_gemini_parse_fn(
    input_text: str,
    model: str,
    effort: str = "high",
    parts_to_parse: set[ParseType] | None = None,
    result_structure: type[BaseModel] | None = None,
    few_shot_examples: list[VisionFewShotExample] | None = None,
    *,
    token_prices: TokenPrices | None = None,
) -> ParsingFunction:
    result_structure = get_result_structure(
        parts_to_parse=parts_to_parse,
        result_structure=result_structure,
    )

    def gemini_parse(
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> BaseModel:
        input_list = gemini_vision_input_list(
            img_path=img_path,
            input_text=input_text,
            partial_flowchart=partial_flowchart,
            few_shot_examples=few_shot_examples,
        )

        response_text = send_gemini_request(
            **({"token_prices": token_prices} if token_prices is not None else {}),
            result_structure=result_structure,
            model=model,
            input_list=input_list,
            thinking_level=effort,
        )

        return result_structure.model_validate_json(response_text)

    return cast(
        "ParsingFunction",
        model_function(
            gemini_parse,
            result_structure=result_structure,
            provider="Gemini",
            input_text=input_text,
            model=model,
            effort=effort,
            few_shot_examples=few_shot_examples,
        ),
    )
