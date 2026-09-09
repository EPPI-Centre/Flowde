# pragma: exclude file
# TODO: Remove this coverage exclusion when adding Claude tests.

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

from anthropic import AnthropicFoundry, APIError
from dotenv import load_dotenv
from pydantic import BaseModel

from flowde.exponential_backoff import retry_with_exponential_backoff
from flowde.utils import VisionFewShotExample

ClaudeEffort = Literal["low", "medium", "high"]

CLAUDE_PRICES_PER_1M = {
    "claude-sonnet-4-5": {
        "input": 3.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
        "output": 15.00,
    },
    "claude-opus-4-5": {
        "input": 5.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
        "output": 25.00,
    },
    "claude-opus-4-8": {
        "input": 5.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
        "output": 25.00,
    },
}


def _get_claude_client() -> AnthropicFoundry:
    load_dotenv()

    return AnthropicFoundry(
        api_key=os.getenv("ANTHROPIC_FOUNDRY_API_KEY"),
        base_url=os.getenv("ANTHROPIC_FOUNDRY_BASE_URL"),
    )


def _partial_flowchart_text(partial_flowchart: BaseModel) -> str:
    return "The contents of the diagram that has already been parsed:\n" + json.dumps(
        partial_flowchart.model_dump(exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )


def _image_content_block(img_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(img_path.name)

    if mime_type is None:
        msg = f"Could not determine MIME type for image: {img_path}"
        raise ValueError(msg)

    encoded_image = base64.standard_b64encode(img_path.read_bytes()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": encoded_image,
        },
    }


def print_claude_estimated_cost(response: object, model: str) -> None:
    prices = CLAUDE_PRICES_PER_1M.get(model)
    usage = getattr(response, "usage", None)

    if usage is None:
        print("Estimated Claude request cost: unavailable; no usage metadata.")
        return

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

    if prices is None:
        print(
            "Estimated Claude request cost: unavailable; no prices configured for "
            f"{model!r}. "
            f"({input_tokens=}, {cache_read_tokens=}, "
            f"{cache_write_tokens=}, {output_tokens=})"
        )
        return

    input_cost = input_tokens / 1_000_000 * prices["input"]
    cache_read_cost = cache_read_tokens / 1_000_000 * prices["cache_read"]
    cache_write_cost = cache_write_tokens / 1_000_000 * prices["cache_write"]
    output_cost = output_tokens / 1_000_000 * prices["output"]

    estimated_cost = input_cost + cache_read_cost + cache_write_cost + output_cost

    print(
        f"Estimated Claude request cost: ${estimated_cost:.6f} "
        f"({input_cost=:.6f}, "
        f"{cache_read_cost=:.6f}, "
        f"{cache_write_cost=:.6f}, "
        f"{output_cost=:.6f}, "
        f"{input_tokens=}, "
        f"{cache_read_tokens=}, "
        f"{cache_write_tokens=}, "
        f"{output_tokens=})"
    )


@retry_with_exponential_backoff(errors=(APIError,))
def send_claude_request(
    result_structure: type[BaseModel],
    model: str,
    input_list: list[dict[str, Any]],
    effort: ClaudeEffort = "high",
    max_tokens: int = 16_384,
) -> str:
    client = _get_claude_client()

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=input_list,
        output_config={"effort": effort},
        output_format=result_structure,
    )

    print_claude_estimated_cost(response=response, model=model)

    return response.content[0].text


def claude_vision_input_list(
    img_path: Path,
    input_text: str,
    partial_flowchart: BaseModel | None = None,
    few_shot_examples: list[VisionFewShotExample] | None = None,
) -> list[dict[str, Any]]:
    def make_user_content(
        image_path: Path,
        partial: BaseModel | None = None,
        cache_prompt: bool = False,
    ) -> list[dict[str, Any]]:
        prompt_block: dict[str, Any] = {
            "type": "text",
            "text": input_text,
        }

        if cache_prompt:
            prompt_block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

        content: list[dict[str, Any]] = [prompt_block]

        if partial is not None:
            content.append(
                {
                    "type": "text",
                    "text": _partial_flowchart_text(partial),
                }
            )

        content.append(_image_content_block(image_path))

        return content

    input_list: list[dict[str, Any]] = []

    for example in few_shot_examples or []:
        input_list.append(
            {
                "role": "user",
                "content": make_user_content(
                    image_path=example.img_path,
                    partial=example.partial_flowchart,
                ),
            }
        )

        input_list.append(
            {
                "role": "assistant",
                "content": example.expected_output_text(),
            }
        )

    input_list.append(
        {
            "role": "user",
            "content": make_user_content(
                image_path=img_path,
                partial=partial_flowchart,
                cache_prompt=True,
            ),
        }
    )

    return input_list
