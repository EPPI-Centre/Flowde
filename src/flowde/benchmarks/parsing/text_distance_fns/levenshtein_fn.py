from Levenshtein import distance

from flowde.benchmarks.parsing.text_distance_fns.normalise_text import (
    nfc_and_space_normalise,
    numbers_only_normalise,
)


def levenshtein_fn(
    true_text: str | None,
    pred_text: str | None,
) -> int:
    if true_text is None and pred_text is None:
        msg = "Both true_text and pred_text cannot be None."
        raise ValueError(msg)

    if true_text is None:
        true_text = ""
    if pred_text is None:
        pred_text = ""

    return distance(true_text, pred_text)


def levenshtein_with_text_normalisation(
    true_text: str | None,
    pred_text: str | None,
) -> int:
    """
    Count character edits after normalising both input strings.

    This is the default text distance used by `ParsingBenchmark`. Normalisation
    reduces differences caused by formatting and alternative character forms
    before Levenshtein distance counts insertions, deletions and substitutions.

    Parameters
    ----------
    true_text : str | None
        Ground-truth text. `None` represents unmatched predicted text and is
        treated as an empty string.
    pred_text : str | None
        Predicted text. `None` represents missing predicted text and is treated
        as an empty string.

    Returns
    -------
    int
        Minimum number of single-character edits between the normalised
        strings. `0` means the normalised strings are identical. The result is
        an edit count, not a score scaled between zero and one.

    Raises
    ------
    ValueError
        If both `true_text` and `pred_text` are `None`.

    Notes
    -----
    Both strings receive the same normalisation:

    - Apply Unicode NFC so equivalent composed and decomposed characters match.
    - Standardise line endings, convert tabs to spaces, collapse repeated
      ordinary spaces and newlines, and strip surrounding whitespace.
    - Standardise supported bullet, dash, quotation-mark and caret characters.
      A line beginning with `o` or `O` followed by whitespace becomes a bullet.
    - Standardise spacing around punctuation, brackets, equals signs, plus
      signs and hyphens between word characters, including digits.
    - Collapse repeated HTML line-break tags to one `<br>` tag.
    - Convert superscript ordinal suffixes following digits, such as `1ˢᵗ`
      to `1st`.

    Normalisation does not lowercase the text or remove accents. Single
    newlines remain distinct from spaces unless a punctuation-spacing rule
    removes the newline. HTML break tags are not converted to newlines.

    Examples
    --------
    >>> levenshtein_with_text_normalisation("n=10", "n = 10")
    0
    >>> levenshtein_with_text_normalisation("n = 10", "n = 11")
    1
    >>> levenshtein_with_text_normalisation(None, " ABC ")
    3

    """
    if true_text is None and pred_text is None:
        msg = "Both true_text and pred_text cannot be None."
        raise ValueError(msg)

    if true_text is None:
        true_text = ""
    if pred_text is None:
        pred_text = ""

    true_text = nfc_and_space_normalise(true_text)
    pred_text = nfc_and_space_normalise(pred_text)

    return levenshtein_fn(true_text, pred_text)


def number_only_levenshtein(true_text: str | None, pred_text: str | None) -> int:
    if true_text is None and pred_text is None:
        msg = "Both true_text and pred_text cannot be None."
        raise ValueError(msg)

    if true_text is None:
        true_text = ""
    if pred_text is None:
        pred_text = ""

    true_numbers = numbers_only_normalise(true_text)
    pred_numbers = numbers_only_normalise(pred_text)

    return distance(true_numbers, pred_numbers)
