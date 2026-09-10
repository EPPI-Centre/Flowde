import base64
import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import InternalServerError, OpenAI
from pydantic import BaseModel

from flowde.exponential_backoff import retry_with_exponential_backoff
from flowde.pricing import OPENAI_PRICES_PER_1M, TokenPrices, token_count
from flowde.usage import (
    RequestUsage,
    print_request_usage,
    report_usage,
    usage_reporting_active,
)
from flowde.utils import VISION_FEW_SHOT_INSTRUCTIONS, VisionFewShotExample

# TODO: should we really be using the upload file in deployment?


@retry_with_exponential_backoff(errors=(InternalServerError,))
def create_openai_vision_file(file_path: Path, client: OpenAI) -> str:
    """
    Upload a file to OpenAI and return the file ID.

    Parameters
    ----------
    file_path : Path
        Path to local file to be uploaded.
    client : OpenAI
        OpenAI client instance.

    Returns
    -------
    str
        File ID of the uploaded file.

    """
    with file_path.open("rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def image_path_to_data_url(img_path: Path) -> str:
    # Keep supported image types consistent across operating systems.
    image_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = image_types.get(img_path.suffix.lower())
    if mime_type is None:
        mime_type = mimetypes.guess_type(img_path)[0] or "image/png"

    with img_path.open("rb") as file_content:
        encoded = base64.b64encode(file_content.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def openai_request_usage(
    response: object,
    model: str,
    *,
    from_azure: bool = False,
    token_prices: TokenPrices | None = None,
) -> RequestUsage:
    usage = getattr(response, "usage", None)
    input_tokens = token_count(usage, "input_tokens")
    output_tokens = token_count(usage, "output_tokens")
    total_tokens = token_count(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    details = getattr(usage, "input_tokens_details", None)
    cached_tokens = token_count(details, "cached_tokens") or 0
    cache_write_tokens = token_count(details, "cache_write_tokens") or 0
    # Assume the same default token prices for OpenAI and Azure OpenAI.
    prices = token_prices or OPENAI_PRICES_PER_1M.get(model)
    cost = (
        None
        if prices is None
        else prices.estimate(
            input_tokens,
            output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    )
    return RequestUsage(
        provider="Azure OpenAI" if from_azure else "OpenAI",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )


@retry_with_exponential_backoff(errors=(InternalServerError,))
def send_openai_request(
    result_structure: type[BaseModel],
    model: str,
    input_list: list,
    effort: str = "high",
    from_azure: bool = False,
    print_cost: bool = False,
    *,
    token_prices: TokenPrices | None = None,
) -> str:
    load_dotenv()

    if from_azure:
        client = OpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            base_url=os.getenv("AZURE_API_BASE"),
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.parse(
        model=model,
        text_format=result_structure,
        background=False,
        reasoning={"effort": effort},
        input=input_list,
    )

    if usage_reporting_active() or print_cost:
        usage = openai_request_usage(
            response,
            model,
            from_azure=from_azure,
            token_prices=token_prices,
        )
        if usage_reporting_active():
            report_usage(usage)
        else:
            print_request_usage(usage)

    return response.output_text


def openai_vision_input_list(
    img_path: Path,
    input_text: str,
    partial_flowchart: BaseModel | None = None,
    from_azure: bool = False,
    few_shot_examples: list[VisionFewShotExample] | None = None,
) -> list[dict]:
    """Build worked example exchanges followed by the target, with instructions once."""
    load_dotenv()

    if from_azure:
        client = OpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            base_url=os.getenv("AZURE_API_BASE"),
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def make_user_content(
        img_path: Path,
        partial_flowchart: BaseModel | None = None,
    ) -> list[dict]:
        content_list = []

        if partial_flowchart is not None:
            content_list.append(
                {
                    "type": "input_text",
                    "text": (
                        "The contents of the diagram that has already been parsed:\n"
                        + json.dumps(
                            partial_flowchart.model_dump(exclude_none=True),
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                }
            )

        if from_azure:
            content_list.append(
                {
                    "type": "input_image",
                    "image_url": image_path_to_data_url(img_path),
                    "detail": "high",
                }
            )
        else:
            file_id = create_openai_vision_file(img_path, client)
            content_list.append(
                {
                    "type": "input_image",
                    "file_id": file_id,
                    "detail": "high",
                }
            )

        return content_list

    input_list: list[dict] = []

    for index, example in enumerate(few_shot_examples or [], start=1):
        input_list.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Worked example {index}:"},
                    *make_user_content(
                        img_path=example.img_path,
                        partial_flowchart=example.partial_flowchart,
                    ),
                ],
            }
        )

        input_list.append(
            {
                "role": "assistant",
                "content": example.expected_output_text(),
            }
        )

    target_content = make_user_content(
        img_path=img_path, partial_flowchart=partial_flowchart
    )
    if few_shot_examples:
        target_content.insert(
            0,
            {
                "type": "input_text",
                "text": "Target image: return its JSON answer using the task above.",
            },
        )
    input_list.append({"role": "user", "content": target_content})

    # Keep the caller's prompt separate from the built-in example instructions.
    instructions = [{"type": "input_text", "text": input_text}]
    if few_shot_examples:
        instructions.append(
            {"type": "input_text", "text": VISION_FEW_SHOT_INSTRUCTIONS}
        )
    input_list[0]["content"] = instructions + input_list[0]["content"]

    return input_list
