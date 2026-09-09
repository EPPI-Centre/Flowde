import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    File,
    GenerateContentConfig,
    MediaResolution,
    Part,
    ThinkingConfig,
)
from pydantic import BaseModel

from flowde.exponential_backoff import retry_with_exponential_backoff
from flowde.pricing import GEMINI_PRICES_PER_1M, TokenPrices, token_count
from flowde.usage import (
    RequestUsage,
    print_request_usage,
    report_usage,
    usage_reporting_active,
)
from flowde.utils import VISION_FEW_SHOT_INSTRUCTIONS, VisionFewShotExample


@retry_with_exponential_backoff(errors=(genai.errors.APIError,))
def create_google_file(file_path: Path, client: genai.Client) -> File:
    return client.files.upload(file=file_path)


def _partial_flowchart_text(partial_flowchart: BaseModel) -> str:
    return "The contents of the diagram that has already been parsed:\n" + json.dumps(
        partial_flowchart.model_dump(exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )


def gemini_request_usage(
    response: object,
    model: str,
    *,
    token_prices: TokenPrices | None = None,
) -> RequestUsage:
    usage = getattr(response, "usage_metadata", None)
    input_tokens = token_count(usage, "prompt_token_count")
    candidates = token_count(usage, "candidates_token_count")
    thoughts = token_count(usage, "thoughts_token_count") or 0
    tool_tokens = token_count(usage, "tool_use_prompt_token_count") or 0
    output_tokens = None if candidates is None else candidates + thoughts
    total_tokens = token_count(usage, "total_token_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens + tool_tokens
    cached_tokens = token_count(usage, "cached_content_token_count") or 0
    prices = token_prices or GEMINI_PRICES_PER_1M.get(model)
    # Tool-use prompt tokens are additional input, already included in total usage.
    charged_input = None if input_tokens is None else input_tokens + tool_tokens
    cost = (
        None
        if prices is None
        else prices.estimate(
            charged_input,
            output_tokens,
            cached_tokens=cached_tokens,
        )
    )
    return RequestUsage(
        provider="Gemini",
        model=model,
        input_tokens=charged_input,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )


@retry_with_exponential_backoff(errors=(genai.errors.APIError,))
def send_gemini_request(
    result_structure: type[BaseModel],
    input_list: list[Content],
    model: str,
    thinking_level: str = "high",
    print_cost: bool = False,
    *,
    token_prices: TokenPrices | None = None,
) -> str | None:
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

    config = GenerateContentConfig(
        media_resolution=MediaResolution.MEDIA_RESOLUTION_HIGH,
        thinking_config=ThinkingConfig(thinking_level=thinking_level),
        response_mime_type="application/json",
        response_json_schema=result_structure.model_json_schema(),
    )

    response = client.models.generate_content(
        model=model,
        contents=input_list,
        config=config,
    )

    if usage_reporting_active() or print_cost:
        usage = gemini_request_usage(response, model, token_prices=token_prices)
        if usage_reporting_active():
            report_usage(usage)
        else:
            print_request_usage(usage)

    return response.text


def gemini_vision_input_list(
    img_path: Path,
    input_text: str,
    partial_flowchart: BaseModel | None = None,
    few_shot_examples: list[VisionFewShotExample] | None = None,
) -> list[Content]:
    """Build worked example exchanges followed by the target, with instructions once."""
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

    def make_user_parts(
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> list[Part]:
        img = create_google_file(img_path, client)

        parts = []

        if partial_flowchart is not None:
            parts.append(
                Part(
                    text=_partial_flowchart_text(
                        partial_flowchart=partial_flowchart,
                    )
                )
            )

        parts.append(
            Part.from_uri(
                file_uri=img.uri,
                mime_type=img.mime_type,
            )
        )

        return parts

    contents: list[Content] = []

    for index, example in enumerate(few_shot_examples or [], start=1):
        contents.append(
            Content(
                role="user",
                parts=[
                    Part(text=f"Worked example {index}:"),
                    *make_user_parts(
                        img_path=example.img_path,
                        partial_flowchart=example.partial_flowchart,
                    ),
                ],
            )
        )

        contents.append(
            Content(
                role="model",
                parts=[
                    Part(text=example.expected_output_text()),
                ],
            )
        )

    target_parts = make_user_parts(
        img_path=img_path, partial_flowchart=partial_flowchart
    )
    if few_shot_examples:
        target_parts.insert(
            0,
            Part(text="Target image: return its JSON answer using the task above."),
        )
    contents.append(Content(role="user", parts=target_parts))

    # Keep the caller's prompt separate from the built-in example instructions.
    instructions = [Part(text=input_text)]
    if few_shot_examples:
        instructions.append(Part(text=VISION_FEW_SHOT_INSTRUCTIONS))
    contents[0].parts = instructions + (contents[0].parts or [])

    return contents
