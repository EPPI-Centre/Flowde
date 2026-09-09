import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

import flowde.parse_imgs as parse_imgs_module
from flowde import model_function
from flowde.usage import RequestUsage, report_usage


class ParsedFlowchart(BaseModel):
    img_name: str
    partial_flowchart: dict | None = None


class PartialFlowchart(BaseModel):
    source: str
    nodes: list[dict]


def create_test_file(path: Path, content: str = "fake content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_img_without_partial(img_path: Path) -> ParsedFlowchart:
    return ParsedFlowchart(img_name=img_path.name)


def parse_img_with_partial(
    img_path: Path, partial_flowchart: PartialFlowchart
) -> ParsedFlowchart:
    return ParsedFlowchart(
        img_name=img_path.name,
        partial_flowchart=partial_flowchart.model_dump(),
    )


# Custom parsers declare the class used to restore saved results.
parse_img_without_partial = model_function(
    parse_img_without_partial, result_structure=ParsedFlowchart, parser="filename"
)
parse_img_with_partial = model_function(
    parse_img_with_partial,
    result_structure=ParsedFlowchart,
    parser="filename-and-partial",
)


def parse_should_not_be_called(**kwargs):
    raise AssertionError("parse_fn should not be called")


def test_parse_imgs_from_paths_names_outputs_and_resumes_in_input_order(tmp_path):
    img_paths = [
        tmp_path / "second-set" / "diagram.v2.png",
        tmp_path / "first-set" / "a.jpg",
    ]
    for img_path in img_paths:
        create_test_file(img_path)
    save_dir = tmp_path / "nested" / "parsed"
    parser = model_function(
        Mock(wraps=parse_img_without_partial),
        result_structure=ParsedFlowchart,
        parser="filename",
    )

    results = parse_imgs_module.parse_imgs_from_paths(
        parser, img_paths, save_dir, n_jobs=1
    )

    assert [result.img_name for result in results] == ["diagram.v2.png", "a.jpg"]
    assert {
        path.name: json.loads(path.read_text())["img_name"]
        for path in save_dir.glob("*.json")
    } == {"diagram.v2.json": "diagram.v2.png", "a.json": "a.jpg"}
    parser.reset_mock()

    restored = parse_imgs_module.parse_imgs_from_paths(
        parser, img_paths, save_dir, n_jobs=1, on_existing="resume"
    )

    assert restored == results
    parser.assert_not_called()


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_parse_imgs_from_paths_writes_parse_results_without_partial_flowcharts(
    tmp_path, n_jobs
):
    img_paths = [
        tmp_path / "imgs" / "diagram_1.png",
        tmp_path / "imgs" / "diagram_2.png",
    ]
    save_dir = tmp_path / "parsed"

    for img_path in img_paths:
        create_test_file(img_path)

    result = parse_imgs_module.parse_imgs_from_paths(
        parse_fn=parse_img_without_partial,
        img_paths=img_paths,
        save_dir=save_dir,
        n_jobs=n_jobs,
    )

    assert [response.model_dump() for response in result] == [
        {
            "img_name": "diagram_1.png",
            "partial_flowchart": None,
        },
        {
            "img_name": "diagram_2.png",
            "partial_flowchart": None,
        },
    ]

    assert json.loads((save_dir / "diagram_1.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_1.png",
        "partial_flowchart": None,
    }
    assert json.loads((save_dir / "diagram_2.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_2.png",
        "partial_flowchart": None,
    }


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize(
    (
        "include_labels",
        "include_additional_texts",
        "include_flow",
    ),
    [
        pytest.param(False, False, False, id="nodes-only"),
        pytest.param(True, False, False, id="nodes-and-labels"),
        pytest.param(False, True, False, id="nodes-and-additional-texts"),
        pytest.param(False, False, True, id="nodes-and-flow"),
        pytest.param(True, True, False, id="nodes-labels-and-additional-texts"),
        pytest.param(True, False, True, id="nodes-labels-and-flow"),
        pytest.param(False, True, True, id="nodes-additional-texts-and-flow"),
        pytest.param(True, True, True, id="nodes-labels-additional-texts-and-flow"),
    ],
)
def test_parse_imgs_from_paths_writes_parse_results_with_partial_flowcharts(
    tmp_path,
    monkeypatch,
    n_jobs,
    include_labels,
    include_additional_texts,
    include_flow,
):
    img_paths = [
        tmp_path / "imgs" / "diagram_1.png",
        tmp_path / "imgs" / "diagram_2.png",
    ]
    save_dir = tmp_path / "parsed"
    nodes_paths = [
        tmp_path / "nodes" / "diagram_1.json",
        tmp_path / "nodes" / "diagram_2.json",
    ]
    labels_paths = (
        [
            tmp_path / "labels" / "diagram_1.json",
            tmp_path / "labels" / "diagram_2.json",
        ]
        if include_labels
        else None
    )
    additional_texts_paths = (
        [
            tmp_path / "additional_texts" / "diagram_1.json",
            tmp_path / "additional_texts" / "diagram_2.json",
        ]
        if include_additional_texts
        else None
    )
    flow_paths = (
        [
            tmp_path / "flow" / "diagram_1.json",
            tmp_path / "flow" / "diagram_2.json",
        ]
        if include_flow
        else None
    )

    for img_path in img_paths:
        create_test_file(img_path)

    partial_flowcharts = [
        PartialFlowchart(
            source="diagram_1",
            nodes=[{"node_number": 1, "text": "Node 1"}],
        ),
        PartialFlowchart(
            source="diagram_2",
            nodes=[{"node_number": 1, "text": "Node 2"}],
        ),
    ]

    calls = []

    def fake_build_partial_flowcharts(
        nodes_paths,
        labels_paths=None,
        additional_texts_paths=None,
        flow_paths=None,
    ):
        calls.append(
            {
                "nodes_paths": nodes_paths,
                "labels_paths": labels_paths,
                "additional_texts_paths": additional_texts_paths,
                "flow_paths": flow_paths,
            }
        )
        return partial_flowcharts

    monkeypatch.setattr(
        parse_imgs_module,
        "build_partial_flowcharts",
        fake_build_partial_flowcharts,
    )

    result = parse_imgs_module.parse_imgs_from_paths(
        parse_fn=parse_img_with_partial,
        img_paths=img_paths,
        save_dir=save_dir,
        nodes_paths=nodes_paths,
        labels_paths=labels_paths,
        additional_texts_paths=additional_texts_paths,
        flow_paths=flow_paths,
        n_jobs=n_jobs,
    )

    assert [response.model_dump() for response in result] == [
        {
            "img_name": "diagram_1.png",
            "partial_flowchart": {
                "source": "diagram_1",
                "nodes": [{"node_number": 1, "text": "Node 1"}],
            },
        },
        {
            "img_name": "diagram_2.png",
            "partial_flowchart": {
                "source": "diagram_2",
                "nodes": [{"node_number": 1, "text": "Node 2"}],
            },
        },
    ]

    assert calls == [
        {
            "nodes_paths": nodes_paths,
            "labels_paths": labels_paths,
            "additional_texts_paths": additional_texts_paths,
            "flow_paths": flow_paths,
        }
    ]

    assert json.loads((save_dir / "diagram_1.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_1.png",
        "partial_flowchart": {
            "source": "diagram_1",
            "nodes": [{"node_number": 1, "text": "Node 1"}],
        },
    }
    assert json.loads((save_dir / "diagram_2.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_2.png",
        "partial_flowchart": {
            "source": "diagram_2",
            "nodes": [{"node_number": 1, "text": "Node 2"}],
        },
    }


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize(
    (
        "img_filenames",
        "nodes_filenames",
        "labels_filenames",
        "additional_texts_filenames",
        "flow_filenames",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            ["diagram_1.png", "diagram_2.png"],
            ["diagram_1.json"],
            None,
            None,
            None,
            [
                "All provided path lists must have the same length.",
            ],
            id="nodes-paths-wrong-length",
        ),
        pytest.param(
            ["diagram_1.png", "diagram_2.png"],
            ["diagram_1.json", "diagram_2.json"],
            ["diagram_1.json"],
            None,
            None,
            [
                "All provided path lists must have the same length.",
            ],
            id="labels-paths-wrong-length",
        ),
        pytest.param(
            ["diagram_1.png", "diagram_2.png"],
            ["diagram_1.json", "diagram_2.json"],
            None,
            ["diagram_1.json"],
            None,
            [
                "All provided path lists must have the same length.",
            ],
            id="additional-texts-paths-wrong-length",
        ),
        pytest.param(
            ["diagram_1.png", "diagram_2.png"],
            ["diagram_1.json", "diagram_2.json"],
            None,
            None,
            ["diagram_1.json"],
            [
                "All provided path lists must have the same length.",
            ],
            id="flow-paths-wrong-length",
        ),
        pytest.param(
            ["diagram_1.png"],
            ["different_stem.json"],
            None,
            None,
            None,
            [
                "Mismatched stems at index 0",
                "diagram_1",
                "different_stem",
            ],
            id="nodes-path-stem-mismatch",
        ),
        pytest.param(
            ["diagram_1.png"],
            ["diagram_1.json"],
            ["different_stem.json"],
            None,
            None,
            [
                "Mismatched stems at index 0",
                "diagram_1",
                "different_stem",
            ],
            id="labels-path-stem-mismatch",
        ),
        pytest.param(
            ["diagram_1.png"],
            ["diagram_1.json"],
            None,
            ["different_stem.json"],
            None,
            [
                "Mismatched stems at index 0",
                "diagram_1",
                "different_stem",
            ],
            id="additional-texts-path-stem-mismatch",
        ),
        pytest.param(
            ["diagram_1.png"],
            ["diagram_1.json"],
            None,
            None,
            ["different_stem.json"],
            [
                "Mismatched stems at index 0",
                "diagram_1",
                "different_stem",
            ],
            id="flow-path-stem-mismatch",
        ),
        pytest.param(
            ["diagram_1.png"],
            None,
            ["diagram_1.json"],
            None,
            None,
            [
                "nodes_paths must be provided if any of labels_paths, "
                "additional_texts_paths, or flow_paths are provided, since node numbers "
                "are needed to join the different parts together."
            ],
            id="labels-paths-without-nodes-paths",
        ),
        pytest.param(
            ["diagram_1.png"],
            None,
            None,
            ["diagram_1.json"],
            None,
            [
                "nodes_paths must be provided if any of labels_paths, "
                "additional_texts_paths, or flow_paths are provided, since node numbers "
                "are needed to join the different parts together."
            ],
            id="additional-texts-paths-without-nodes-paths",
        ),
        pytest.param(
            ["diagram_1.png"],
            None,
            None,
            None,
            ["diagram_1.json"],
            [
                "nodes_paths must be provided if any of labels_paths, "
                "additional_texts_paths, or flow_paths are provided, since node numbers "
                "are needed to join the different parts together."
            ],
            id="flow-paths-without-nodes-paths",
        ),
        pytest.param(
            ["diagram_1.png"],
            None,
            ["diagram_1.json"],
            ["diagram_1.json"],
            ["diagram_1.json"],
            [
                "nodes_paths must be provided if any of labels_paths, "
                "additional_texts_paths, or flow_paths are provided, since node numbers "
                "are needed to join the different parts together."
            ],
            id="all-part-paths-without-nodes-paths",
        ),
    ],
)
def test_parse_imgs_from_paths_validation_errors_do_not_write_outputs(
    tmp_path,
    n_jobs,
    img_filenames,
    nodes_filenames,
    labels_filenames,
    additional_texts_filenames,
    flow_filenames,
    expected_msg_parts,
):
    img_paths = [tmp_path / "imgs" / filename for filename in img_filenames]
    save_dir = tmp_path / "parsed"
    nodes_paths = (
        None
        if nodes_filenames is None
        else [tmp_path / "nodes" / filename for filename in nodes_filenames]
    )
    labels_paths = (
        None
        if labels_filenames is None
        else [tmp_path / "labels" / filename for filename in labels_filenames]
    )
    additional_texts_paths = (
        None
        if additional_texts_filenames is None
        else [
            tmp_path / "additional_texts" / filename
            for filename in additional_texts_filenames
        ]
    )
    flow_paths = (
        None
        if flow_filenames is None
        else [tmp_path / "flow" / filename for filename in flow_filenames]
    )

    for path in img_paths:
        create_test_file(path)

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs_from_paths(
            parse_fn=parse_should_not_be_called,
            img_paths=img_paths,
            save_dir=save_dir,
            nodes_paths=nodes_paths,
            labels_paths=labels_paths,
            additional_texts_paths=additional_texts_paths,
            flow_paths=flow_paths,
            n_jobs=n_jobs,
        )

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg

    assert not save_dir.exists()


@pytest.mark.parametrize("show_usage", [False, True], ids=["hidden", "visible"])
def test_parse_imgs_controls_usage_display(tmp_path, capsys, show_usage):
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    create_test_file(img_dir / "diagram.png")

    def parse_with_usage(img_path):
        report_usage(RequestUsage(provider="test", model="a", cost=2, total_tokens=300))
        return parse_img_without_partial(img_path)

    parse_with_usage = model_function(
        parse_with_usage, result_structure=ParsedFlowchart, parser="with-usage"
    )

    result = parse_imgs_module.parse_imgs(
        parse_fn=parse_with_usage,
        img_dir=img_dir,
        save_dir=save_dir,
        n_jobs=1,
        show_usage=show_usage,
    )

    assert result == [ParsedFlowchart(img_name="diagram.png")]
    output = capsys.readouterr()
    text = output.out + output.err
    assert "1/1" in output.err
    assert ("Est. cost:" in text) == show_usage
    assert ("Tokens:" in text) == show_usage


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_parse_imgs_writes_parse_results_without_partial_flowcharts(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir / "diagram_2.png")
    create_test_file(img_dir / "diagram_1.png")

    result = parse_imgs_module.parse_imgs(
        parse_fn=parse_img_without_partial,
        img_dir=img_dir,
        save_dir=save_dir,
        n_jobs=n_jobs,
    )

    assert [response.model_dump() for response in result] == [
        {
            "img_name": "diagram_1.png",
            "partial_flowchart": None,
        },
        {
            "img_name": "diagram_2.png",
            "partial_flowchart": None,
        },
    ]

    assert json.loads((save_dir / "diagram_1.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_1.png",
        "partial_flowchart": None,
    }
    assert json.loads((save_dir / "diagram_2.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_2.png",
        "partial_flowchart": None,
    }

    expected_paths = {
        img_dir,
        img_dir / "diagram_1.png",
        img_dir / "diagram_2.png",
        save_dir,
        save_dir / "diagram_1.json",
        save_dir / "diagram_2.json",
    }

    assert {
        path for path in tmp_path.rglob("*") if ".flowde" not in path.parts
    } == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_parse_imgs_ignores_non_png_files(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir / "diagram_1.png")
    create_test_file(img_dir / "not_an_image.txt")
    create_test_file(img_dir / "fake_jpg.jpg")

    result = parse_imgs_module.parse_imgs(
        parse_fn=parse_img_without_partial,
        img_dir=img_dir,
        save_dir=save_dir,
        n_jobs=n_jobs,
    )

    assert [response.model_dump() for response in result] == [
        {
            "img_name": "diagram_1.png",
            "partial_flowchart": None,
        },
    ]

    assert json.loads((save_dir / "diagram_1.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_1.png",
        "partial_flowchart": None,
    }

    expected_paths = {
        img_dir,
        img_dir / "diagram_1.png",
        img_dir / "not_an_image.txt",
        img_dir / "fake_jpg.jpg",
        save_dir,
        save_dir / "diagram_1.json",
    }

    assert {
        path for path in tmp_path.rglob("*") if ".flowde" not in path.parts
    } == expected_paths


@pytest.mark.parametrize(
    ("img_extensions", "non_matching_filename", "expected_extensions"),
    [
        pytest.param(None, "diagram.jpg", "png", id="default-png"),
        pytest.param(
            {"jpg", "jpeg"},
            "diagram.png",
            "jpeg, jpg",
            id="specified-extensions",
        ),
    ],
)
def test_parse_imgs_raises_if_no_images_match_extensions(
    tmp_path: Path,
    img_extensions: set[str] | None,
    non_matching_filename: str,
    expected_extensions: str,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir / "not_an_image.txt")
    create_test_file(img_dir / non_matching_filename)

    expected_message = (
        f"No image files matching extensions [{expected_extensions}] were found "
        f"in directory {img_dir}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            img_extensions=img_extensions,
        )

    assert not save_dir.exists()


@pytest.mark.parametrize(
    "range_indices",
    [
        pytest.param(None, id="whole-directory"),
        pytest.param((0, 1), id="slice-does-not-hide-duplicate"),
    ],
)
def test_parse_imgs_rejects_duplicate_image_stems(tmp_path, range_indices):
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    create_test_file(img_dir / "diagram.jpg")
    create_test_file(img_dir / "diagram.png")

    with pytest.raises(ValueError, match="Duplicate file stem 'diagram'") as raised:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            img_extensions={"jpg", "png"},
            range_indices=range_indices,
            n_jobs=1,
        )

    assert "diagram.jpg" in str(raised.value)
    assert "diagram.png" in str(raised.value)
    assert not save_dir.exists()


def test_parse_imgs_from_paths_rejects_duplicate_image_stems(tmp_path):
    img_paths = [tmp_path / "diagram.jpg", tmp_path / "diagram.png"]
    for img_path in img_paths:
        create_test_file(img_path)

    save_path = tmp_path / "parsed" / "diagram.json"
    original_result = '{"img_name": "previous.png", "partial_flowchart": null}'
    create_test_file(save_path, original_result)

    with pytest.raises(ValueError, match="Duplicate file stem 'diagram'"):
        parse_imgs_module.parse_imgs_from_paths(
            parse_fn=parse_should_not_be_called,
            img_paths=img_paths,
            save_dir=save_path.parent,
            n_jobs=1,
        )

    assert save_path.read_text(encoding="utf-8") == original_result


def test_parse_imgs_writes_distinct_results_for_mixed_image_formats(tmp_path):
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    create_test_file(img_dir / "diagram_2.png")
    create_test_file(img_dir / "diagram_1.jpg")

    results = parse_imgs_module.parse_imgs(
        parse_fn=parse_img_without_partial,
        img_dir=img_dir,
        save_dir=save_dir,
        img_extensions={"jpg", "png"},
        n_jobs=1,
    )

    assert [result.img_name for result in results] == ["diagram_1.jpg", "diagram_2.png"]
    assert json.loads((save_dir / "diagram_1.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_1.jpg",
        "partial_flowchart": None,
    }
    assert json.loads((save_dir / "diagram_2.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram_2.png",
        "partial_flowchart": None,
    }


def test_parse_imgs_ignores_duplicate_stems_in_unselected_formats(tmp_path):
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    create_test_file(img_dir / "diagram.png")
    create_test_file(img_dir / "diagram.jpg")

    results = parse_imgs_module.parse_imgs(
        parse_fn=parse_img_without_partial,
        img_dir=img_dir,
        save_dir=save_dir,
        n_jobs=1,
    )

    assert [result.img_name for result in results] == ["diagram.png"]
    assert json.loads((save_dir / "diagram.json").read_text(encoding="utf-8")) == {
        "img_name": "diagram.png",
        "partial_flowchart": None,
    }


def test_parse_imgs_raises_if_img_dir_does_not_exist(tmp_path: Path) -> None:
    img_dir = tmp_path / "missing_imgs"
    save_dir = tmp_path / "parsed"

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
        )

    assert str(exc_info.value) == (
        f"Image directory {img_dir} does not exist or is not a directory."
    )
    assert not save_dir.exists()
    assert set(tmp_path.rglob("*")) == set()


def test_parse_imgs_raises_if_img_dir_is_file(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs.png"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir)

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
        )

    assert str(exc_info.value) == (
        f"Image directory {img_dir} does not exist or is not a directory."
    )
    assert not save_dir.exists()

    expected_paths = {
        img_dir,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("show_usage", [False, True])
def test_parse_imgs_passes_sorted_paths_to_parse_imgs_from_paths(
    show_usage: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    nodes_dir = tmp_path / "nodes"
    labels_dir = tmp_path / "labels"
    additional_texts_dir = tmp_path / "additional_texts"
    flow_dir = tmp_path / "flow"

    for stem in ["diagram_2", "diagram_1", "diagram_3"]:
        create_test_file(img_dir / f"{stem}.png")
        create_test_file(nodes_dir / f"{stem}.json")
        create_test_file(labels_dir / f"{stem}.json")
        create_test_file(additional_texts_dir / f"{stem}.json")
        create_test_file(flow_dir / f"{stem}.json")

    captured_kwargs = {}

    def fake_parse_imgs_from_paths(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            ParsedFlowchart(img_name="diagram_1.png"),
            ParsedFlowchart(img_name="diagram_2.png"),
            ParsedFlowchart(img_name="diagram_3.png"),
        ]

    monkeypatch.setattr(
        parse_imgs_module,
        "parse_imgs_from_paths",
        fake_parse_imgs_from_paths,
    )

    result = parse_imgs_module.parse_imgs(
        parse_fn=parse_should_not_be_called,
        img_dir=img_dir,
        save_dir=save_dir,
        nodes_dir=nodes_dir,
        labels_dir=labels_dir,
        additional_texts_dir=additional_texts_dir,
        flow_dir=flow_dir,
        n_jobs=3,
        show_usage=show_usage,
    )

    assert [response.model_dump() for response in result] == [
        {"img_name": "diagram_1.png", "partial_flowchart": None},
        {"img_name": "diagram_2.png", "partial_flowchart": None},
        {"img_name": "diagram_3.png", "partial_flowchart": None},
    ]

    assert captured_kwargs == {
        "parse_fn": parse_should_not_be_called,
        "img_paths": [
            img_dir / "diagram_1.png",
            img_dir / "diagram_2.png",
            img_dir / "diagram_3.png",
        ],
        "save_dir": save_dir,
        "nodes_paths": [
            nodes_dir / "diagram_1.json",
            nodes_dir / "diagram_2.json",
            nodes_dir / "diagram_3.json",
        ],
        "labels_paths": [
            labels_dir / "diagram_1.json",
            labels_dir / "diagram_2.json",
            labels_dir / "diagram_3.json",
        ],
        "additional_texts_paths": [
            additional_texts_dir / "diagram_1.json",
            additional_texts_dir / "diagram_2.json",
            additional_texts_dir / "diagram_3.json",
        ],
        "flow_paths": [
            flow_dir / "diagram_1.json",
            flow_dir / "diagram_2.json",
            flow_dir / "diagram_3.json",
        ],
        "n_jobs": 3,
        "show_usage": show_usage,
        "on_existing": "error",
    }

    expected_paths = {
        img_dir,
        img_dir / "diagram_1.png",
        img_dir / "diagram_2.png",
        img_dir / "diagram_3.png",
        nodes_dir,
        nodes_dir / "diagram_1.json",
        nodes_dir / "diagram_2.json",
        nodes_dir / "diagram_3.json",
        labels_dir,
        labels_dir / "diagram_1.json",
        labels_dir / "diagram_2.json",
        labels_dir / "diagram_3.json",
        additional_texts_dir,
        additional_texts_dir / "diagram_1.json",
        additional_texts_dir / "diagram_2.json",
        additional_texts_dir / "diagram_3.json",
        flow_dir,
        flow_dir / "diagram_1.json",
        flow_dir / "diagram_2.json",
        flow_dir / "diagram_3.json",
    }

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize(
    "range_indices",
    [
        pytest.param((0, 4), id="full-range"),
        pytest.param((0, 2), id="start-slice"),
        pytest.param((1, 3), id="middle-slice"),
        pytest.param((2, 4), id="end-slice"),
        pytest.param((1, 4), id="from-middle-to-end"),
        pytest.param((0, 1), id="single-first"),
        pytest.param((3, 4), id="single-last"),
        pytest.param((2, 2), id="empty-slice"),
    ],
)
@pytest.mark.parametrize(
    (
        "include_nodes",
        "include_labels",
        "include_additional_texts",
        "include_flow",
    ),
    [
        pytest.param(False, False, False, False, id="no-partials"),
        pytest.param(True, False, False, False, id="nodes-only"),
        pytest.param(True, True, False, False, id="nodes-labels"),
        pytest.param(True, False, True, False, id="nodes-additional-texts"),
        pytest.param(True, False, False, True, id="nodes-flow"),
        pytest.param(True, True, True, False, id="nodes-labels-additional-texts"),
        pytest.param(True, True, False, True, id="nodes-labels-flow"),
        pytest.param(True, False, True, True, id="nodes-additional-texts-flow"),
        pytest.param(True, True, True, True, id="all-partials"),
    ],
)
def test_parse_imgs_passes_range_indices_to_parse_imgs_from_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    range_indices: tuple[int, int],
    include_nodes: bool,
    include_labels: bool,
    include_additional_texts: bool,
    include_flow: bool,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"
    nodes_dir = tmp_path / "nodes" if include_nodes else None
    labels_dir = tmp_path / "labels" if include_labels else None
    additional_texts_dir = (
        tmp_path / "additional_texts" if include_additional_texts else None
    )
    flow_dir = tmp_path / "flow" if include_flow else None

    stems = ["diagram_1", "diagram_2", "diagram_3", "diagram_4"]

    for stem in stems:
        create_test_file(img_dir / f"{stem}.png")

        if nodes_dir is not None:
            create_test_file(nodes_dir / f"{stem}.json")

        if labels_dir is not None:
            create_test_file(labels_dir / f"{stem}.json")

        if additional_texts_dir is not None:
            create_test_file(additional_texts_dir / f"{stem}.json")

        if flow_dir is not None:
            create_test_file(flow_dir / f"{stem}.json")

    expected_stems = stems[range_indices[0] : range_indices[1]]

    captured_kwargs = {}

    def fake_parse_imgs_from_paths(**kwargs):
        captured_kwargs.update(kwargs)
        return [ParsedFlowchart(img_name=f"{stem}.png") for stem in expected_stems]

    monkeypatch.setattr(
        parse_imgs_module,
        "parse_imgs_from_paths",
        fake_parse_imgs_from_paths,
    )

    result = parse_imgs_module.parse_imgs(
        parse_fn=parse_should_not_be_called,
        img_dir=img_dir,
        save_dir=save_dir,
        nodes_dir=nodes_dir,
        labels_dir=labels_dir,
        additional_texts_dir=additional_texts_dir,
        flow_dir=flow_dir,
        range_indices=range_indices,
        n_jobs=3,
    )

    assert [response.model_dump() for response in result] == [
        {
            "img_name": f"{stem}.png",
            "partial_flowchart": None,
        }
        for stem in expected_stems
    ]

    assert captured_kwargs == {
        "parse_fn": parse_should_not_be_called,
        "img_paths": [img_dir / f"{stem}.png" for stem in expected_stems],
        "save_dir": save_dir,
        "nodes_paths": (
            [nodes_dir / f"{stem}.json" for stem in expected_stems]
            if nodes_dir is not None
            else None
        ),
        "labels_paths": (
            [labels_dir / f"{stem}.json" for stem in expected_stems]
            if labels_dir is not None
            else None
        ),
        "additional_texts_paths": (
            [additional_texts_dir / f"{stem}.json" for stem in expected_stems]
            if additional_texts_dir is not None
            else None
        ),
        "flow_paths": (
            [flow_dir / f"{stem}.json" for stem in expected_stems]
            if flow_dir is not None
            else None
        ),
        "n_jobs": 3,
        "show_usage": True,
        "on_existing": "error",
    }

    expected_paths = {
        img_dir,
        *[img_dir / f"{stem}.png" for stem in stems],
    }

    if nodes_dir is not None:
        expected_paths.update(
            {
                nodes_dir,
                *[nodes_dir / f"{stem}.json" for stem in stems],
            }
        )

    if labels_dir is not None:
        expected_paths.update(
            {
                labels_dir,
                *[labels_dir / f"{stem}.json" for stem in stems],
            }
        )

    if additional_texts_dir is not None:
        expected_paths.update(
            {
                additional_texts_dir,
                *[additional_texts_dir / f"{stem}.json" for stem in stems],
            }
        )

    if flow_dir is not None:
        expected_paths.update(
            {
                flow_dir,
                *[flow_dir / f"{stem}.json" for stem in stems],
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize(
    (
        "nodes_dir_name",
        "labels_dir_name",
        "additional_texts_dir_name",
        "flow_dir_name",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            "nodes",
            "labels",
            None,
            None,
            ["All provided path lists must have the same length."],
            id="labels-dir-wrong-length",
        ),
        pytest.param(
            "nodes",
            None,
            "additional_texts",
            None,
            ["All provided path lists must have the same length."],
            id="additional-texts-dir-wrong-length",
        ),
        pytest.param(
            "nodes",
            None,
            None,
            "flow",
            ["All provided path lists must have the same length."],
            id="flow-dir-wrong-length",
        ),
    ],
)
def test_parse_imgs_raises_if_partial_dirs_have_wrong_number_of_json_files(
    tmp_path: Path,
    nodes_dir_name: str | None,
    labels_dir_name: str | None,
    additional_texts_dir_name: str | None,
    flow_dir_name: str | None,
    expected_msg_parts: list[str],
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir / "diagram_1.png")
    create_test_file(img_dir / "diagram_2.png")

    nodes_dir = tmp_path / nodes_dir_name if nodes_dir_name is not None else None
    labels_dir = tmp_path / labels_dir_name if labels_dir_name is not None else None
    additional_texts_dir = (
        tmp_path / additional_texts_dir_name
        if additional_texts_dir_name is not None
        else None
    )
    flow_dir = tmp_path / flow_dir_name if flow_dir_name is not None else None

    if nodes_dir is not None:
        create_test_file(nodes_dir / "diagram_1.json")
        create_test_file(nodes_dir / "diagram_2.json")

    if labels_dir is not None:
        create_test_file(labels_dir / "diagram_1.json")

    if additional_texts_dir is not None:
        create_test_file(additional_texts_dir / "diagram_1.json")

    if flow_dir is not None:
        create_test_file(flow_dir / "diagram_1.json")

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            nodes_dir=nodes_dir,
            labels_dir=labels_dir,
            additional_texts_dir=additional_texts_dir,
            flow_dir=flow_dir,
        )

    msg = str(exc_info.value)
    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg

    assert not save_dir.exists()


@pytest.mark.parametrize(
    (
        "img_stem",
        "nodes_stem",
        "labels_stem",
        "additional_texts_stem",
        "flow_stem",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            "diagram_1",
            "different_stem",
            None,
            None,
            None,
            ["Mismatched stems at index 0", "diagram_1", "different_stem"],
            id="nodes-stem-mismatch",
        ),
        pytest.param(
            "diagram_1",
            "diagram_1",
            "different_stem",
            None,
            None,
            ["Mismatched stems at index 0", "diagram_1", "different_stem"],
            id="labels-stem-mismatch",
        ),
        pytest.param(
            "diagram_1",
            "diagram_1",
            None,
            "different_stem",
            None,
            ["Mismatched stems at index 0", "diagram_1", "different_stem"],
            id="additional-texts-stem-mismatch",
        ),
        pytest.param(
            "diagram_1",
            "diagram_1",
            None,
            None,
            "different_stem",
            ["Mismatched stems at index 0", "diagram_1", "different_stem"],
            id="flow-stem-mismatch",
        ),
        pytest.param(
            "different_stem",
            "diagram_1",
            "diagram_1",
            "diagram_1",
            "diagram_1",
            ["Mismatched stems at index 0", "different_stem", "diagram_1"],
            id="img-stem-mismatch",
        ),
        pytest.param(
            "diagram_1",
            "diagram_1",
            "different_label_stem",
            "different_additional_texts_stem",
            "different_flow_stem",
            ["Mismatched stems at index 0", "diagram_1", "different_label_stem"],
            id="multiple-partial-stem-mismatches",
        ),
    ],
)
def test_parse_imgs_raises_if_partial_dirs_have_mismatched_json_stems(
    tmp_path: Path,
    img_stem: str,
    nodes_stem: str | None,
    labels_stem: str | None,
    additional_texts_stem: str | None,
    flow_stem: str | None,
    expected_msg_parts: list[str],
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    nodes_dir = tmp_path / "nodes" if nodes_stem is not None else None
    labels_dir = tmp_path / "labels" if labels_stem is not None else None
    additional_texts_dir = (
        tmp_path / "additional_texts" if additional_texts_stem is not None else None
    )
    flow_dir = tmp_path / "flow" if flow_stem is not None else None

    create_test_file(img_dir / f"{img_stem}.png")

    if nodes_dir is not None and nodes_stem is not None:
        create_test_file(nodes_dir / f"{nodes_stem}.json")

    if labels_dir is not None and labels_stem is not None:
        create_test_file(labels_dir / f"{labels_stem}.json")

    if additional_texts_dir is not None and additional_texts_stem is not None:
        create_test_file(additional_texts_dir / f"{additional_texts_stem}.json")

    if flow_dir is not None and flow_stem is not None:
        create_test_file(flow_dir / f"{flow_stem}.json")

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            nodes_dir=nodes_dir,
            labels_dir=labels_dir,
            additional_texts_dir=additional_texts_dir,
            flow_dir=flow_dir,
        )

    msg = str(exc_info.value)
    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg

    assert not save_dir.exists()

    expected_paths = {
        img_dir,
        img_dir / f"{img_stem}.png",
    }

    if nodes_dir is not None and nodes_stem is not None:
        expected_paths.update(
            {
                nodes_dir,
                nodes_dir / f"{nodes_stem}.json",
            }
        )

    if labels_dir is not None and labels_stem is not None:
        expected_paths.update(
            {
                labels_dir,
                labels_dir / f"{labels_stem}.json",
            }
        )

    if additional_texts_dir is not None and additional_texts_stem is not None:
        expected_paths.update(
            {
                additional_texts_dir,
                additional_texts_dir / f"{additional_texts_stem}.json",
            }
        )

    if flow_dir is not None and flow_stem is not None:
        expected_paths.update(
            {
                flow_dir,
                flow_dir / f"{flow_stem}.json",
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize(
    (
        "labels_dir_given",
        "additional_texts_dir_given",
        "flow_dir_given",
    ),
    [
        pytest.param(True, False, False, id="labels-without-nodes"),
        pytest.param(False, True, False, id="additional-texts-without-nodes"),
        pytest.param(False, False, True, id="flow-without-nodes"),
        pytest.param(True, True, True, id="all-parts-without-nodes"),
    ],
)
def test_parse_imgs_raises_if_partial_dirs_are_given_without_nodes_dir(
    tmp_path: Path,
    labels_dir_given: bool,
    additional_texts_dir_given: bool,
    flow_dir_given: bool,
) -> None:
    img_dir = tmp_path / "imgs"
    save_dir = tmp_path / "parsed"

    create_test_file(img_dir / "diagram_1.png")

    labels_dir = tmp_path / "labels" if labels_dir_given else None
    additional_texts_dir = (
        tmp_path / "additional_texts" if additional_texts_dir_given else None
    )
    flow_dir = tmp_path / "flow" if flow_dir_given else None

    if labels_dir is not None:
        create_test_file(labels_dir / "diagram_1.json")

    if additional_texts_dir is not None:
        create_test_file(additional_texts_dir / "diagram_1.json")

    if flow_dir is not None:
        create_test_file(flow_dir / "diagram_1.json")

    with pytest.raises(ValueError) as exc_info:
        parse_imgs_module.parse_imgs(
            parse_fn=parse_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            nodes_dir=None,
            labels_dir=labels_dir,
            additional_texts_dir=additional_texts_dir,
            flow_dir=flow_dir,
        )

    assert str(exc_info.value) == (
        "nodes_paths must be provided if any of labels_paths, "
        "additional_texts_paths, or flow_paths are provided, since node "
        "numbers are needed to join the different parts together."
    )

    assert not save_dir.exists()
