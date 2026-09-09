import json
from pathlib import Path

import pytest

from flowde.utils import (
    VisionFewShotExample,
    _run_parallel_fn_with_better_errors,
    _run_parallel_fn_with_kwargs_better_errors,
    apply_fn_parallel_on_dict_of_lists,
    apply_fn_parallel_on_list,
    parse_bool,
    validate_matching_stems,
    validate_same_path_lengths,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("true", True, id="lowercase-true"),
        pytest.param("false", False, id="lowercase-false"),
        pytest.param("TrUe", True, id="mixed-case-true"),
        pytest.param("FaLsE", False, id="mixed-case-false"),
    ],
)
def test_parse_bool_accepts_true_and_false_ignoring_case(value, expected):
    assert parse_bool(value) is expected


@pytest.mark.parametrize("value", ["yes", ""], ids=["unsupported-word", "empty-string"])
def test_parse_bool_rejects_unsupported_text(value):
    with pytest.raises(ValueError, match="expected true or false"):
        parse_bool(value)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"label": 1}, id="classification"),
        pytest.param(
            {"nodes": [{"node_number": 1, "text": "Éligible — 日本語"}]},
            id="nodes-only-with-unicode",
        ),
        pytest.param({"nodes": [{"node_number": 1, "points_to": [2]}]}, id="flow-only"),
        pytest.param({"custom": [True, None, {"value": 2}]}, id="custom-json-fields"),
    ],
)
def test_few_shot_example_preserves_the_supplied_answer(tmp_path, answer):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"image")
    answer_path = tmp_path / "example.json"
    answer_path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
    example = VisionFewShotExample(
        img_path=image_path, expected_output_path=answer_path
    )

    assert json.loads(example.expected_output_text()) == answer


@pytest.mark.parametrize(
    ("file_problem", "expected_error"),
    [
        pytest.param("invalid-json", json.JSONDecodeError, id="invalid-json"),
        pytest.param("missing-file", FileNotFoundError, id="file-removed-after-setup"),
    ],
)
def test_few_shot_example_raises_if_its_answer_cannot_be_read(
    tmp_path, file_problem, expected_error
):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"image")
    answer_path = tmp_path / "example.json"
    answer_path.write_text('{"label": 1}', encoding="utf-8")
    example = VisionFewShotExample(
        img_path=image_path, expected_output_path=answer_path
    )
    if file_problem == "invalid-json":
        answer_path.write_text("not JSON", encoding="utf-8")
    else:
        answer_path.unlink()

    with pytest.raises(expected_error):
        example.expected_output_text()


def square(x: int) -> int:
    return x * x


def as_string(x: object) -> str:
    return str(x)


def get_key(d: dict[str, int]) -> int:
    return d["x"]


def raise_on_two(x: int) -> int:
    if x == 2:
        msg = "bad value"
        raise ValueError(msg)
    return x


def test_run_parallel_fn_with_better_errors_returns_result() -> None:
    def add_one(x: int) -> int:
        return x + 1

    result = _run_parallel_fn_with_better_errors(add_one, 1)

    assert result == 2


def test_run_parallel_fn_with_better_errors_wraps_exception() -> None:
    def fail(x: int) -> int:
        msg = "bad item"
        raise ValueError(msg)

    with pytest.raises(RuntimeError) as exc_info:
        _run_parallel_fn_with_better_errors(fail, 123)

    msg = str(exc_info.value)

    assert "Error while processing item 123" in msg
    assert "Original exception: ValueError: bad item" in msg
    assert "Traceback:" in msg


@pytest.mark.parametrize(
    ("fn", "items", "expected"),
    [
        (square, [1, 2, 3, 4], [1, 4, 9, 16]),
        (as_string, [1, "a", None], ["1", "a", "None"]),
        (get_key, [{"x": 1}, {"x": 2}], [1, 2]),
        (square, [], []),
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_apply_fn_parallel_on_list(fn, items, expected, n_jobs):
    result = apply_fn_parallel_on_list(
        fn=fn,
        items=items,
        msg="Testing apply_fn_parallel_on_list",
        n_jobs=n_jobs,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("n_jobs", "expected_error"), [(1, ValueError), (2, RuntimeError)]
)
def test_apply_fn_parallel_on_list_raises(n_jobs, expected_error):
    with pytest.raises(expected_error, match="bad value"):
        apply_fn_parallel_on_list(
            fn=raise_on_two,
            items=[1, 2, 3],
            msg="Testing errors",
            n_jobs=n_jobs,
        )


def add(**kwargs: int) -> int:
    return sum(kwargs.values())


def format_name(first: str, middle: str, last: str) -> str:
    return f"{first} {middle} {last}"


def maybe_divide(a: int, b: int) -> float:
    return a / b


def always_raises_with_kwargs(a: int, b: int) -> int:
    msg = f"bad values: {a}, {b}"
    raise ValueError(msg)


def test_run_parallel_fn_with_kwargs_better_errors_returns_result() -> None:
    def add(x: int, y: int) -> int:
        return x + y

    result = _run_parallel_fn_with_kwargs_better_errors(
        add,
        {"x": 1, "y": 2},
    )

    assert result == 3


def test_run_parallel_fn_with_kwargs_better_errors_wraps_exception() -> None:
    def fail(x: int, y: int) -> int:
        msg = "bad kwargs"
        raise ValueError(msg)

    kwargs = {"x": 1, "y": 2}

    with pytest.raises(RuntimeError) as exc_info:
        _run_parallel_fn_with_kwargs_better_errors(fail, kwargs)

    msg = str(exc_info.value)

    assert "Error while processing kwargs {'x': 1, 'y': 2}" in msg
    assert "Original exception: ValueError: bad kwargs" in msg
    assert "Traceback:" in msg


@pytest.mark.parametrize(
    ("fn", "items_by_param", "expected"),
    [
        (
            add,
            {"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]},
            [111, 222, 333],
        ),
        (
            format_name,
            {
                "first": ["John", "Jane"],
                "middle": ["Alex", "Dom"],
                "last": ["Smith", "Jones"],
            },
            ["John Alex Smith", "Jane Dom Jones"],
        ),
        (
            maybe_divide,
            {"a": [10, 20, 30], "b": [2, 4, 5]},
            [5.0, 5.0, 6.0],
        ),
        (
            add,
            {"a": [], "b": []},
            [],
        ),
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_apply_fn_parallel_on_dict_of_lists(fn, items_by_param, expected, n_jobs):
    result = apply_fn_parallel_on_dict_of_lists(
        fn=fn,
        items_by_param=items_by_param,
        msg="Testing apply_fn_parallel_on_dict_of_lists",
        n_jobs=n_jobs,
    )

    assert result == expected


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_apply_fn_parallel_on_dict_of_lists_raises_for_empty_items_by_param(n_jobs):
    with pytest.raises(ValueError, match="items_by_param cannot be empty."):
        apply_fn_parallel_on_dict_of_lists(
            fn=add,
            items_by_param={},
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize(
    "items_by_param",
    [
        {"a": [1, 2], "b": [10]},
        {"a": [1], "b": [10, 20]},
        {"a": [], "b": [10]},
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_apply_fn_parallel_on_dict_of_lists_raises_for_unequal_lengths(
    items_by_param, n_jobs
):
    with pytest.raises(
        ValueError, match="All parameter lists must have the same length."
    ):
        apply_fn_parallel_on_dict_of_lists(
            fn=add,
            items_by_param=items_by_param,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize(
    ("n_jobs", "expected_error"), [(1, ValueError), (2, RuntimeError)]
)
def test_apply_fn_parallel_on_dict_of_lists_raises_for_fn_errors(
    n_jobs, expected_error
):
    items_by_param = {"a": [1, 2, 3], "b": [0, 1, 2]}
    with pytest.raises(expected_error, match="bad values:"):
        apply_fn_parallel_on_dict_of_lists(
            fn=always_raises_with_kwargs,
            items_by_param=items_by_param,
            msg="Testing errors",
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize(
    "path_lists",
    [
        (
            [Path("img_1.png"), Path("img_2.png")],
            [Path("save_1.json"), Path("save_2.json")],
            None,
            None,
            None,
            None,
        ),
        (
            [Path("img_1.png"), Path("img_2.png")],
            [Path("save_1.json"), Path("save_2.json")],
            [Path("nodes_1.json"), Path("nodes_2.json")],
            None,
            None,
            None,
        ),
        (
            [Path("img_1.png"), Path("img_2.png")],
            [Path("save_1.json"), Path("save_2.json")],
            [Path("nodes_1.json"), Path("nodes_2.json")],
            [Path("labels_1.json"), Path("labels_2.json")],
            [Path("texts_1.json"), Path("texts_2.json")],
            None,
        ),
        (
            [Path("img_1.png"), Path("img_2.png")],
            [Path("save_1.json"), Path("save_2.json")],
            [Path("nodes_1.json"), Path("nodes_2.json")],
            [Path("labels_1.json"), Path("labels_2.json")],
            [Path("texts_1.json"), Path("texts_2.json")],
            [Path("flow_1.json"), Path("flow_2.json")],
        ),
        (
            [],
            [],
            None,
            None,
            None,
            None,
        ),
        (
            [],
            [],
            [],
            [],
            [],
            [],
        ),
        (
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ],
)
def test_validate_same_path_lengths_passes_for_matching_lengths(path_lists):
    validate_same_path_lengths(*path_lists)


@pytest.mark.parametrize(
    ("path_lists", "expected_msg"),
    [
        (
            (
                [Path("img_1.png"), Path("img_2.png")],
                [Path("save_1.json")],
                None,
                None,
                None,
                None,
            ),
            "All provided path lists must have the same length. ",
        ),
        (
            (
                [Path("img_1.png"), Path("img_2.png")],
                [Path("save_1.json"), Path("save_2.json")],
                [Path("nodes_1.json")],
                None,
                None,
                None,
            ),
            "All provided path lists must have the same length. ",
        ),
        (
            (
                [Path("img_1.png")],
                [Path("save_1.json")],
                None,
                [Path("labels_1.json"), Path("labels_2.json")],
                None,
                None,
            ),
            "All provided path lists must have the same length. ",
        ),
        (
            (
                [Path("img_1.png"), Path("img_2.png")],
                [Path("save_1.json"), Path("save_2.json")],
                None,
                None,
                [Path("texts_1.json")],
                None,
            ),
            "All provided path lists must have the same length. ",
        ),
        (
            (
                [Path("img_1.png"), Path("img_2.png")],
                [Path("save_1.json"), Path("save_2.json")],
                None,
                None,
                None,
                [Path("flow_1.json")],
            ),
            "All provided path lists must have the same length. ",
        ),
        (
            (
                [Path("img_1.png"), Path("img_2.png"), Path("img_3.png")],
                [Path("save_1.json")],
                [Path("nodes_1.json"), Path("nodes_2.json")],
                [Path("labels_1.json"), Path("labels_2.json"), Path("labels_3.json")],
                [],
                [Path("flow_1.json"), Path("flow_2.json"), Path("flow_3.json")],
            ),
            "All provided path lists must have the same length. ",
        ),
    ],
)
def test_validate_same_path_lengths_raises_for_mismatched_lengths(
    path_lists,
    expected_msg,
):
    with pytest.raises(ValueError) as exc_info:
        validate_same_path_lengths(*path_lists)

    assert expected_msg in str(exc_info.value)


@pytest.mark.parametrize(
    "path_lists",
    [
        (
            None,
            None,
        ),
        ([Path("a.png"), Path("b.png")],),
        (
            [Path("a.png"), Path("b.png")],
            None,
        ),
        (
            [Path("a.png"), Path("b.png")],
            [Path("a.json"), Path("b.json")],
        ),
        (
            [Path("nested/images/a.png"), Path("nested/images/b.png")],
            [Path("nested/nodes/a.json"), Path("nested/nodes/b.json")],
            [Path("nested/labels/a.json"), Path("nested/labels/b.json")],
        ),
        (
            [],
            [],
            None,
        ),
    ],
)
def test_validate_matching_stems_passes_for_matching_stems(path_lists):
    validate_matching_stems(*path_lists)


@pytest.mark.parametrize(
    ("path_lists", "expected_msg_parts"),
    [
        (
            (
                [Path("a.png"), Path("b.png")],
                [Path("a.json")],
            ),
            [
                "All provided path lists must have the same length.",
                "List 0 has length 2, list 1 has length 1.",
            ],
        ),
        (
            (
                [Path("a.png")],
                [Path("a.json"), Path("b.json")],
            ),
            [
                "All provided path lists must have the same length.",
                "List 0 has length 1, list 1 has length 2.",
            ],
        ),
        (
            (
                [Path("a.png"), Path("b.png")],
                None,
                [Path("a.json")],
            ),
            [
                "All provided path lists must have the same length.",
                "List 0 has length 2, list 1 has length 1.",
            ],
        ),
    ],
)
def test_validate_matching_stems_raises_for_mismatched_lengths(
    path_lists,
    expected_msg_parts,
):
    with pytest.raises(ValueError) as exc_info:
        validate_matching_stems(*path_lists)

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg


@pytest.mark.parametrize(
    ("path_lists", "expected_msg"),
    [
        (
            (
                [Path("a.png"), Path("b.png")],
                [Path("a.json"), Path("c.json")],
            ),
            "Mismatched stems at index 1: ['b', 'c']",
        ),
        (
            (
                [Path("a.png")],
                [Path("b.json")],
            ),
            "Mismatched stems at index 0: ['a', 'b']",
        ),
        (
            (
                [Path("a.png"), Path("b.png")],
                [Path("a.json"), Path("b.json")],
                [Path("a.txt"), Path("c.txt")],
            ),
            "Mismatched stems at index 1: ['b', 'b', 'c']",
        ),
        (
            (
                [Path("nested/images/diagram_1.png")],
                [Path("nested/nodes/diagram_2.json")],
            ),
            "Mismatched stems at index 0: ['diagram_1', 'diagram_2']",
        ),
    ],
)
def test_validate_matching_stems_raises_for_mismatched_stems(
    path_lists,
    expected_msg,
):
    with pytest.raises(ValueError) as exc_info:
        validate_matching_stems(*path_lists)

    assert str(exc_info.value) == expected_msg
