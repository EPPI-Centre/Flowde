import os
from pathlib import Path
from typing import Literal

import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pydantic import BaseModel

import flowde.classify_fns.openai_classify_fn as openai_classify_fn_module


def check_openai_credentials(from_azure: bool) -> None:
    load_dotenv()
    names = ("AZURE_API_KEY", "AZURE_API_BASE") if from_azure else ("OPENAI_API_KEY",)
    for name in names:
        if not os.getenv(name):
            raise AssertionError(f"{name} must be set to run this integration test.")


def create_simple_flowchart_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (500, 350), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((170, 30, 330, 80), outline="black", width=3)
    draw.rectangle((170, 140, 330, 190), outline="black", width=3)
    draw.rectangle((170, 250, 330, 300), outline="black", width=3)

    draw.text((215, 48), "Start", fill="black")
    draw.text((205, 158), "Process", fill="black")
    draw.text((225, 268), "End", fill="black")

    draw.line((250, 80, 250, 140), fill="black", width=3)
    draw.polygon([(250, 140), (242, 128), (258, 128)], fill="black")

    draw.line((250, 190, 250, 250), fill="black", width=3)
    draw.polygon([(250, 250), (242, 238), (258, 238)], fill="black")

    img.save(path)


def create_non_flowchart_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (500, 350), "white")
    draw = ImageDraw.Draw(img)

    draw.text(
        (50, 150),
        "This is a plain text image.\nIt is not a flowchart.",
        fill="black",
    )

    img.save(path)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"]),
]


def test_openai_classify_fn_classifies_simple_flowchart_image(tmp_path, from_azure):
    check_openai_credentials(from_azure)

    img_path = tmp_path / "simple_flowchart.png"
    create_simple_flowchart_image(img_path)

    classify_fn = openai_classify_fn_module.make_openai_classify_fn(
        input_text=(
            "Classify whether this image is a flowchart. "
            "Return JSON matching the schema. "
            "Use label 1 if it is a flowchart and label 0 if it is not."
        ),
        model="gpt-5.6-luna",
        effort="low",
        from_azure=from_azure,
    )

    result = classify_fn(img_path)

    assert result == 1


def test_openai_classify_fn_classifies_non_flowchart_image(tmp_path, from_azure):
    check_openai_credentials(from_azure)

    img_path = tmp_path / "plain_text.png"
    create_non_flowchart_image(img_path)

    classify_fn = openai_classify_fn_module.make_openai_classify_fn(
        input_text=(
            "Classify whether this image is a flowchart. "
            "Return JSON matching the schema. "
            "Use label 1 if it is a flowchart and label 0 if it is not."
        ),
        model="gpt-5.6-luna",
        effort="low",
        from_azure=from_azure,
    )

    result = classify_fn(img_path)

    assert result == 0


class ResultStructureW2(BaseModel):
    label: Literal[0, 1, 2]


def test_openai_classify_fn_accepts_different_structure(tmp_path, from_azure):
    check_openai_credentials(from_azure)

    img_path = tmp_path / "plain_text.png"
    create_non_flowchart_image(img_path)

    classify_fn = openai_classify_fn_module.make_openai_classify_fn(
        input_text="Return 2",
        model="gpt-5.6-luna",
        result_structure=ResultStructureW2,
        effort="low",
        from_azure=from_azure,
    )

    result = classify_fn(img_path)

    assert result == 2
