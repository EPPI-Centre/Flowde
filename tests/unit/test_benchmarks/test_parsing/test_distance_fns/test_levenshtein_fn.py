import pytest

from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_fn,
    levenshtein_with_nfc_and_space_normalisation,
)


@pytest.mark.parametrize(
    ("true_text", "pred_text", "expected"),
    [
        pytest.param(
            "abc",
            "abc",
            0,
            id="identical-strings",
        ),
        pytest.param(
            "abc",
            "abd",
            1,
            id="single-substitution",
        ),
        pytest.param(
            "abc",
            "abcd",
            1,
            id="single-insertion",
        ),
        pytest.param(
            "abcd",
            "abc",
            1,
            id="single-deletion",
        ),
        pytest.param(
            "kitten",
            "sitting",
            3,
            id="classic-levenshtein-example",
        ),
        pytest.param(
            "",
            "abc",
            3,
            id="empty-true-string",
        ),
        pytest.param(
            "abc",
            "",
            3,
            id="empty-pred-string",
        ),
        pytest.param(
            None,
            "abc",
            3,
            id="true-text-none-treated-as-empty-string",
        ),
        pytest.param(
            "abc",
            None,
            3,
            id="pred-text-none-treated-as-empty-string",
        ),
        pytest.param(
            None,
            "",
            0,
            id="true-text-none-and-pred-empty-string",
        ),
        pytest.param(
            "",
            None,
            0,
            id="true-empty-string-and-pred-text-none",
        ),
    ],
)
def test_levenshtein_fn_returns_expected_distance(true_text, pred_text, expected):
    assert levenshtein_fn(true_text=true_text, pred_text=pred_text) == expected


def test_levenshtein_fn_raises_if_both_texts_are_none():
    with pytest.raises(ValueError) as exc_info:
        levenshtein_fn(true_text=None, pred_text=None)

    assert "Both true_text and pred_text cannot be None." in str(exc_info.value)


@pytest.mark.parametrize(
    ("true_text", "pred_text", "expected"),
    [
        pytest.param(
            "abc",
            "abc",
            0,
            id="identical-strings",
        ),
        pytest.param(
            "abc",
            "abd",
            1,
            id="single-substitution",
        ),
        pytest.param(
            "abc",
            "abcd",
            1,
            id="single-insertion",
        ),
        pytest.param(
            "abcd",
            "abc",
            1,
            id="single-deletion",
        ),
        pytest.param(
            "kitten",
            "sitting",
            3,
            id="classic-levenshtein-example",
        ),
        pytest.param(
            None,
            "abc",
            3,
            id="true-text-none-treated-as-empty-string",
        ),
        pytest.param(
            "abc",
            None,
            3,
            id="pred-text-none-treated-as-empty-string",
        ),
        pytest.param(
            None,
            "",
            0,
            id="true-text-none-and-pred-empty-string",
        ),
        pytest.param(
            "",
            None,
            0,
            id="true-empty-string-and-pred-text-none",
        ),
        pytest.param(
            "hello    world",
            "hello world",
            0,
            id="multiple-spaces-normalised",
        ),
        pytest.param(
            "hello  world   again",
            "hello world again",
            0,
            id="several-space-runs-normalised",
        ),
        pytest.param(
            "hello\tworld",
            "hello world",
            0,
            id="tab-normalised-to-space",
        ),
        pytest.param(
            "hello\t\tworld",
            "hello world",
            0,
            id="multiple-tabs-normalised-to-one-space",
        ),
        pytest.param(
            "hello\t  \tworld",
            "hello world",
            0,
            id="mixed-tabs-and-spaces-normalised",
        ),
        pytest.param(
            "hello\r\nworld",
            "hello\nworld",
            0,
            id="windows-crlf-newline-normalised",
        ),
        pytest.param(
            "hello\rworld",
            "hello\nworld",
            0,
            id="old-mac-cr-newline-normalised",
        ),
        pytest.param(
            "hello\n\nworld",
            "hello\nworld",
            0,
            id="multiple-newlines-normalised",
        ),
        pytest.param(
            "hello\n\n\nworld",
            "hello\nworld",
            0,
            id="many-newlines-normalised",
        ),
        pytest.param(
            "hello   \n   world",
            "hello\nworld",
            0,
            id="spaces-around-newline-removed",
        ),
        pytest.param(
            "hello\t\n\tworld",
            "hello\nworld",
            0,
            id="tabs-around-newline-removed",
        ),
        pytest.param(
            " hello world ",
            "hello world",
            0,
            id="leading-and-trailing-spaces-stripped",
        ),
        pytest.param(
            "\n\nhello\n\nworld\n\n",
            "hello\nworld",
            0,
            id="leading-trailing-newlines-stripped-and-inner-newlines-collapsed",
        ),
        pytest.param(
            "e\u0301",
            "é",
            0,
            id="nfc-normalises-combining-acute",
        ),
        pytest.param(
            "Cafe\u0301   Society",
            "Café Society",
            0,
            id="nfc-and-space-normalisation-together",
        ),
        pytest.param(
            "Allocated  \n\n\t n  =  50",
            "Allocated\nn = 50",
            0,
            id="flowchart-like-text-normalised",
        ),
        pytest.param(
            "Allocated\nn = 50",
            "Allocated n = 50",
            1,
            id="single-newline-preserved-not-flattened-to-space",
        ),
        pytest.param(
            "A\nB  C\n\nD\tE",
            "A\nB C\nD E",
            0,
            id="preserves-single-newlines-while-normalising-spaces",
        ),
        pytest.param(
            "A\nB",
            "A B",
            1,
            id="single-newline-still-counts-as-different-from-space",
        ),
    ],
)
def test_levenshtein_with_nfc_and_space_normalisation_returns_expected_distance(
    true_text,
    pred_text,
    expected,
):
    assert (
        levenshtein_with_nfc_and_space_normalisation(
            true_text=true_text,
            pred_text=pred_text,
        )
        == expected
    )


def test_levenshtein_with_nfc_and_space_normalisation_raises_if_both_texts_are_none():
    with pytest.raises(ValueError) as exc_info:
        levenshtein_with_nfc_and_space_normalisation(
            true_text=None,
            pred_text=None,
        )

    assert "Both true_text and pred_text cannot be None." in str(exc_info.value)
