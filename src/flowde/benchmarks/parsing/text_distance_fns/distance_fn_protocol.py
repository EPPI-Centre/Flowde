from typing import Protocol


class DistanceFnProtocol(Protocol):
    """
    Describe a callable that calculates a cost between true and predicted text.

    Pass a function with this interface as `distance_fn` to
    [`ParsingBenchmark`][flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark]
    to compare node text, labels and additional text. Lower costs must indicate
    closer matches because the benchmark minimises text costs when matching
    predictions to ground truth.

    Notes
    -----
    This protocol describes the callable interface; it does not calculate
    costs or validate returned values. A function or callable object can
    satisfy the interface without inheriting from `DistanceFnProtocol`.

    """

    def __call__(
        self,
        *,
        true_text: str | None,
        pred_text: str | None,
    ) -> int | float:
        """
        Calculate the comparison cost for one pair of text values.

        Parameters
        ----------
        true_text : str | None
            Ground-truth text. The benchmark supplies this argument by keyword.
            `None` represents predicted text with no ground-truth match.
        pred_text : str | None
            Predicted text. The benchmark supplies this argument by keyword.
            `None` represents ground-truth text with no prediction.

        Returns
        -------
        int | float
            Finite, non-negative comparison cost. Lower values must indicate
            closer text matches. Return a cost rather than a similarity score
            where higher values indicate closer matches.

        Notes
        -----
        The function must handle either argument being `None` and assign a
        cost to the unmatched text. Flowde's built-in distance functions treat
        `None` as an empty string. A custom function can choose another penalty
        for unmatched text. The benchmark does not compare two `None` values.

        """
        ...
