import pytest

from flowde.benchmarks.parsing.text_distance_fns.normalise_text import (
    nfc_and_space_normalise,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "hello world",
            "hello world",
            id="unchanged-basic-text",
        ),
        pytest.param(
            "hello    world",
            "hello world",
            id="collapses-multiple-spaces-to-one-space",
        ),
        pytest.param(
            "hello  world   again",
            "hello world again",
            id="collapses-several-space-runs",
        ),
        pytest.param(
            " hello world ",
            "hello world",
            id="strips-leading-and-trailing-spaces",
        ),
        pytest.param(
            "\thello\tworld\t",
            "hello world",
            id="tabs-treated-as-spaces",
        ),
        pytest.param(
            "hello\t\tworld",
            "hello world",
            id="multiple-tabs-collapse-to-one-space",
        ),
        pytest.param(
            "hello\t  \tworld",
            "hello world",
            id="mixed-tabs-and-spaces-collapse-to-one-space",
        ),
        pytest.param(
            "hello\nworld",
            "hello\nworld",
            id="preserves-single-newline",
        ),
        pytest.param(
            "hello\n\nworld",
            "hello\nworld",
            id="collapses-multiple-newlines-to-one-newline",
        ),
        pytest.param(
            "hello\n\n\nworld",
            "hello\nworld",
            id="collapses-many-newlines-to-one-newline",
        ),
        pytest.param(
            "hello \n world",
            "hello\nworld",
            id="removes-single-spaces-around-newline",
        ),
        pytest.param(
            "hello    \n    world",
            "hello\nworld",
            id="removes-multiple-spaces-around-newline",
        ),
        pytest.param(
            "hello\t\n\tworld",
            "hello\nworld",
            id="removes-tabs-around-newline-after-tab-normalisation",
        ),
        pytest.param(
            "hello\r\nworld",
            "hello\nworld",
            id="normalises-windows-crlf-newline",
        ),
        pytest.param(
            "hello\rworld",
            "hello\nworld",
            id="normalises-old-mac-cr-newline",
        ),
        pytest.param(
            "hello\r\n\r\nworld",
            "hello\nworld",
            id="normalises-and-collapses-multiple-crlf-newlines",
        ),
        pytest.param(
            "hello\r\rworld",
            "hello\nworld",
            id="normalises-and-collapses-multiple-cr-newlines",
        ),
        pytest.param(
            " hello\t \r\n \tworld  ",
            "hello\nworld",
            id="normalises-mixed-tabs-spaces-crlf-and-strip",
        ),
        pytest.param(
            "Allocated  \n\n\t n  =  50",
            "Allocated\nn = 50",
            id="flowchart-like-text-with-extra-spaces-tabs-and-newlines",
        ),
        pytest.param(
            "e\u0301",
            "é",
            id="nfc-composes-e-plus-combining-acute",
        ),
        pytest.param(
            "Cafe\u0301  Society",
            "Café Society",
            id="nfc-composes-combining-accent-and-collapses-spaces",
        ),
        pytest.param(
            "é",
            "é",
            id="nfc-leaves-already-composed-character",
        ),
        pytest.param(
            "A\nB  C\n\nD\tE",
            "A\nB C\nD E",
            id="preserves-single-newlines-while-normalising-spaces",
        ),
        pytest.param(
            "\n\nhello\n\nworld\n\n",
            "hello\nworld",
            id="strips-leading-and-trailing-newlines-and-collapses-inner-newlines",
        ),
        pytest.param(
            "hello \t \n \t world",
            "hello\nworld",
            id="mixed-whitespace-around-newline-is-removed",
        ),
    ],
)
def test_nfc_and_space_normalise_returns_expected_text(text, expected):
    assert nfc_and_space_normalise(text) == expected
