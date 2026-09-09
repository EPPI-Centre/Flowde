# pragma: exclude file
# TODO: Remove this coverage exclusion when adding Claude tests.

from pathlib import Path
from typing import cast

from pydantic import BaseModel

from flowde._run_settings import model_function
from flowde.api_utils.claude_utils import (
    ClaudeEffort,
    claude_vision_input_list,
    send_claude_request,
)
from flowde.parsing_fns.parsing_types import (
    ParseType,
    ParsingFunction,
)
from flowde.parsing_fns.parsing_utils import get_result_structure
from flowde.utils import VisionFewShotExample


def make_claude_parse_fn(
    input_text: str,
    model: str,
    effort: ClaudeEffort = "high",
    max_tokens: int = 16_384,
    parts_to_parse: set[ParseType] | None = None,
    result_structure: type[BaseModel] | None = None,
    few_shot_examples: list[VisionFewShotExample] | None = None,
) -> ParsingFunction:
    result_structure = get_result_structure(
        parts_to_parse=parts_to_parse,
        result_structure=result_structure,
    )

    def claude_parse(
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> BaseModel:
        input_list = claude_vision_input_list(
            img_path=img_path,
            input_text=input_text,
            partial_flowchart=partial_flowchart,
            few_shot_examples=few_shot_examples,
        )

        response_text = send_claude_request(
            result_structure=result_structure,
            model=model,
            input_list=input_list,
            effort=effort,
            max_tokens=max_tokens,
        )

        return result_structure.model_validate_json(response_text)

    return cast(
        "ParsingFunction",
        model_function(
            claude_parse,
            result_structure=result_structure,
            provider="Claude",
            input_text=input_text,
            model=model,
            effort=effort,
            few_shot_examples=few_shot_examples,
            max_tokens=max_tokens,
        ),
    )
