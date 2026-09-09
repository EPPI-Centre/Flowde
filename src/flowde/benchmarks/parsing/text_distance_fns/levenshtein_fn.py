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


def levenshtein_with_nfc_and_space_normalisation(
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
