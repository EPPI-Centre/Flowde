from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from flowde.benchmarks.parsing.check_parsing_bench_data import (
    N_DIAGRAMS_IN_BENCHMARK,
    run_all_benchmark_true_data_checks,
    validate_pred_diagram_sources,
)
from flowde.benchmarks.parsing.load_diagrams import (
    load_pred_diagrams_from_sources,
    load_true_diagram_options,
)
from flowde.benchmarks.parsing.match_diagrams import (
    match_additional_texts,
    match_diagrams,
    match_node_labels,
    match_nodes,
)
from flowde.benchmarks.parsing.parsing_bench_types import (
    DiagramMatch,
    FlowScores,
    NodeMatches,
    PredDiagramStructure,
    TextListMatches,
)
from flowde.benchmarks.parsing.text_distance_fns.distance_fn_protocol import (
    DistanceFnProtocol,
)
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_with_text_normalisation,
)


@dataclass(frozen=True)
class ParsingBenchmarkCapabilities:
    """Benchmark operations supported by a predicted diagram structure."""

    node_matches: bool
    node_matches_with_flow: bool
    node_and_label_matches: bool
    additional_text_matches: bool
    flow_scores: bool
    diagram_matches: bool

    @classmethod
    def from_pred_structure(
        cls,
        structure: PredDiagramStructure,
    ) -> "ParsingBenchmarkCapabilities":
        return cls(
            node_matches=structure.node_text,
            node_matches_with_flow=structure.node_text and structure.flow,
            node_and_label_matches=structure.node_text and structure.labels,
            additional_text_matches=(
                structure.node_text and structure.additional_texts
            ),
            flow_scores=structure.node_text and structure.flow,
            diagram_matches=(
                structure.node_text
                and structure.labels
                and structure.flow
                and structure.additional_texts
            ),
        )

    @property
    def available_methods(self) -> tuple[str, ...]:
        methods: list[str] = []
        if self.node_matches:
            methods.extend(("node_matches", "total_node_text_cost"))
        if self.node_matches_with_flow:
            methods.append("node_matches_with_flow")
        if self.node_and_label_matches:
            methods.extend(("node_and_label_matches", "total_label_cost"))
        if self.additional_text_matches:
            methods.extend(("additional_text_matches", "total_additional_text_cost"))
        if self.flow_scores:
            methods.extend(
                (
                    "all_flow_scores",
                    "all_flow_jaccard_scores",
                    "avg_flow_jaccard",
                )
            )
        if self.diagram_matches:
            methods.append("diagram_matches")
        return tuple(methods)


# TODO: Turn these into cached properties and make it so that none of the
# attributes can be changed after innit. One of them breaks when you do this so
# be careful
class ParsingBenchmark:
    """
    Evaluate parsed flowcharts against accepted ground-truth interpretations.

    Construction loads and validates saved JSON files. Scoring methods match
    predicted nodes to ground-truth nodes and compare node text, labels,
    directed connections and additional text. Benchmarking runs locally,
    without opening images, changing source files or making model requests.

    Parameters
    ----------
    pred_diagrams_dir : Path | Sequence[Path]
        One directory containing prediction JSONs, or a sequence of directories
        containing separately parsed parts. Reads top-level `*.json` files;
        subdirectories are not searched. Each filename stem identifies a
        flowchart, and each file contains one prediction object rather than
        a ground-truth `options` list.

        Predictions must include node text. Labels, flow and additional text
        are optional. All JSONs within a prediction directory must contain
        the same parts. Separate prediction directories must contain matching
        filename stems and non-overlapping parts. Node-based parts must use
        the same node numbers for each flowchart. The benchmark joins those
        parts in memory; combined JSON files are not required.
    distance_fn : DistanceFnProtocol, optional
        Function accepting `true_text` and `pred_text` as keyword arguments
        and returning a finite, non-negative `int` or `float`. Either argument
        may be `None` to represent unmatched text; the supplied functions
        treat `None` as an empty string. Lower costs mean closer text matches.

        Defaults to
        [`levenshtein_with_text_normalisation()`][flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn.levenshtein_with_text_normalisation],
        which normalises Unicode, whitespace and selected punctuation before
        counting character edits. You can supply a different function to
        change the text comparison used for matching and scoring.
    allow_missing_pred_diagrams : bool, optional
        Defaults to `False`, requiring predictions for every ground-truth
        flowchart. With `True`, evaluate only flowcharts that have predictions.
        Missing predictions receive no penalty and do not contribute to
        averages. Predictions without corresponding ground truth remain
        invalid.
    true_nodes_dir : Path, optional
        Directory of ground-truth node-text JSONs. Each file contains an
        `options` list of objects with `nodes`; each node supplies
        `node_number` and `text`.
    true_labels_dir : Path, optional
        Directory of ground-truth label JSONs. Each file contains an `options`
        list of objects with `nodes`; each node supplies `node_number` and a
        `labels` list of strings, which may be empty.
    true_additional_texts_dir : Path, optional
        Directory of ground-truth additional-text JSONs. Each file contains
        an `options` list of objects with an `additional_texts` list of
        strings, which may be empty.
    true_flow_dir : Path, optional
        Directory of ground-truth flow JSONs. Each file contains an `options`
        list of objects with `nodes`; each node supplies `node_number` and a
        `points_to` list of destination node numbers.
    expected_num_diagrams : int, optional
        Number of flowcharts in the complete ground-truth dataset, before
        filtering missing predictions. Counts filename stems, not accepted
        interpretations. Defaults to `346`, the original research dataset's
        size. Supply the count for your own dataset.

    Attributes
    ----------
    pred_diagrams : list[Diagram]
        Loaded predictions, including any parts joined from separate
        directories, in sorted filename-stem order.
    true_diagrams_options_list : list[DiagramOptions]
        Accepted ground-truth interpretations for each evaluated flowchart,
        in the same filename-stem order as `pred_diagrams`.
    pred_structure : PredDiagramStructure
        Which parts the predictions contain: node text, labels, flow and
        additional text.
    capabilities : ParsingBenchmarkCapabilities
        Which scoring operations the predicted parts support. The
        `available_methods` tuple lists the supported method names.

    Raises
    ------
    OSError
        If a discovered JSON file cannot be opened.
    TypeError
        If a prediction or ground-truth field has an invalid type.
    ValueError
        If JSON is malformed, prediction directories are empty, file counts
        or stems disagree, Ground-Truth Options do not align, or a node or
        connection fails validation. Also raised for missing predicted node
        text, overlapping predicted parts, or inconsistent predicted parts
        or node numbers across their source files.

    Notes
    -----
    Supply all four ground-truth directories for your dataset, even when the
    predictions contain only some parts. The default ground-truth paths refer
    to the original research dataset, which is not included in a package
    installation. The four directories must contain matching filename stems.
    For each flowchart, the four `options` lists must have the same length;
    entries at the same index form one Ground-Truth Option and must agree on
    node numbers across node-based parts.

    Scoring methods select Node Matches by minimising node-text cost. Among
    ties, the benchmark maximises flow similarity, then uses label cost to
    resolve remaining ties. Missing predicted parts are skipped. Ground-Truth
    Option selection follows the same order, then uses additional-text cost
    to resolve remaining ties; a complete tie selects the first option.
    All scoring methods use the available predicted parts for matching.
    Prediction and ground-truth node numbers do not need to match.

    Text-cost methods sum matched and unmatched text costs. Flow-score methods
    compare directed connections after matching nodes. Results follow sorted
    filename-stem order, and `avg_flow_jaccard()` gives each evaluated
    flowchart equal weight. Methods requiring an unparsed part raise
    `ValueError`. A scoring method raises `RuntimeError` if the node matcher
    cannot establish an optimal solution; an invalid text-distance value
    raises `ValueError` during scoring.

    Examples
    --------
    Using existing predictions and a ground-truth dataset of one flowchart:

    >>> from pathlib import Path
    >>> from flowde.benchmarks.parsing.parsing_bench import ParsingBenchmark
    >>> truth_dir = Path("data/ground-truth")
    >>> benchmark = ParsingBenchmark(
    ...     pred_diagrams_dir=Path("results/predictions"),
    ...     true_nodes_dir=truth_dir / "nodes",
    ...     true_labels_dir=truth_dir / "labels",
    ...     true_flow_dir=truth_dir / "flow",
    ...     true_additional_texts_dir=truth_dir / "additional_texts",
    ...     expected_num_diagrams=1,
    ... )
    >>> print(benchmark.total_node_text_cost())
    >>> print(benchmark.avg_flow_jaccard())

    """

    def __init__(
        self,
        pred_diagrams_dir: Path | Sequence[Path],
        distance_fn: DistanceFnProtocol = levenshtein_with_text_normalisation,
        allow_missing_pred_diagrams: bool = False,
        true_nodes_dir: Path = Path(
            "./experiment_data/parsing/training-smoking-cessation/"
            "ground-truth-nodes-texts-numbers-row-major-column/"
        ),
        true_labels_dir: Path = Path(
            "./experiment_data/parsing/training-smoking-cessation/"
            "ground-truth-labels-attempt-2/"
        ),
        true_additional_texts_dir: Path = Path(
            "./experiment_data/parsing/training-smoking-cessation/"
            "ground-truth-additional-texts/"
        ),
        true_flow_dir: Path = Path(
            "./experiment_data/parsing/training-smoking-cessation/ground-truth-flow/"
        ),
        expected_num_diagrams: int = N_DIAGRAMS_IN_BENCHMARK,
    ) -> None:
        pred_sources = validate_pred_diagram_sources(
            pred_diagrams_dirs=pred_diagrams_dir
        )

        # TODO: Ideally would move these checks to using the loaded diagrams
        # rather than the paths as that is what we are actually using.
        run_all_benchmark_true_data_checks(
            pred_sources=pred_sources,
            true_nodes_dir=true_nodes_dir,
            true_labels_dir=true_labels_dir,
            true_additional_texts_dir=true_additional_texts_dir,
            true_flow_dir=true_flow_dir,
            allow_missing_pred_diagrams=allow_missing_pred_diagrams,
            expected_num_diagrams=expected_num_diagrams,
        )

        pred_diagrams = load_pred_diagrams_from_sources(pred_sources)
        pred_structure = pred_sources.structure

        true_diagrams_options_list = load_true_diagram_options(
            true_nodes_dir=true_nodes_dir,
            true_labels_dir=true_labels_dir,
            true_additional_texts_dir=true_additional_texts_dir,
            true_flow_dir=true_flow_dir,
        )

        if allow_missing_pred_diagrams:
            pred_diagrams_img_codes = {pd.parent_img_code for pd in pred_diagrams}
            true_diagrams_options_list = [
                td
                for td in true_diagrams_options_list
                if td.parent_img_code in pred_diagrams_img_codes
            ]

        pred_diagrams.sort(key=lambda pd: pd.parent_img_code)
        true_diagrams_options_list.sort(key=lambda tdo: tdo.parent_img_code)

        self.pred_diagrams = pred_diagrams
        self.true_diagrams_options_list = true_diagrams_options_list
        self.allow_missing_pred_diagrams = allow_missing_pred_diagrams
        self.distance_fn = distance_fn
        self.pred_sources = pred_sources
        self.pred_structure = pred_structure
        self.capabilities = ParsingBenchmarkCapabilities.from_pred_structure(
            pred_structure
        )

        # Compatibility aliases for existing callers. Capability decisions now
        # come from one validated structure rather than repeated trial checks.
        self.match_nodes_valid = self.capabilities.node_matches
        self.match_node_and_labels_valid = self.capabilities.node_and_label_matches
        self.match_additional_texts_valid = self.capabilities.additional_text_matches
        self.match_diagrams_valid = self.capabilities.diagram_matches
        self.flow_scores_valid = self.capabilities.flow_scores

    def _require_capability(
        self,
        *,
        available: bool,
        operation: str,
        required_parts: frozenset[str],
    ) -> None:
        if available:
            return

        missing_parts = sorted(required_parts - self.pred_structure.parts)
        msg = (
            f"{operation} is unavailable for predicted diagrams containing "
            f"{self.pred_structure.describe()}. Missing required components: "
            f"{', '.join(missing_parts)}."
        )
        raise ValueError(msg)

    def diagram_matches(self) -> list[DiagramMatch]:
        """
        Match all four parsed parts for each evaluated flowchart.

        Requires predicted node text, labels, flow and additional text. The
        benchmark selects one Ground-Truth Option per prediction and uses
        that option for the node, label, additional-text and flow comparisons.

        Returns
        -------
        list[DiagramMatch]
            One complete comparison per flowchart, in sorted filename-stem
            order. Each result contains the prediction, the selected
            ground-truth diagram, Node Matches with label matches, and
            additional-text matches. Text totals and flow scores are
            available as properties on each result.

        Raises
        ------
        ValueError
            If any required predicted part is missing or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        self._require_capability(
            available=self.capabilities.diagram_matches,
            operation="Diagram matching",
            required_parts=frozenset(
                {"node_text", "labels", "flow", "additional_texts"}
            ),
        )

        return match_diagrams(
            true_diagrams_options_list=self.true_diagrams_options_list,
            pred_diagrams=self.pred_diagrams,
            distance_fn=self.distance_fn,
        )

    def node_matches(
        self,
        range_indices: tuple[int | None, int | None] | None = None,
    ) -> list[NodeMatches]:
        """
        Match predicted nodes to ground-truth nodes for each flowchart.

        Requires predicted node text. Matching minimises node-text cost,
        uses flow similarity to resolve ties when flow was parsed, and then
        uses label cost when labels were parsed. Ground-Truth Option selection
        also uses additional-text cost to resolve remaining ties when that
        part was parsed. A complete tie selects the first Ground-Truth Option.

        Parameters
        ----------
        range_indices : tuple[int | None, int | None] | None, optional
            A `(start, stop)` slice of the evaluated flowcharts in sorted
            filename-stem order. `start` is included and `stop` is excluded.
            Either bound may be `None`; negative indices follow Python
            slicing rules. Defaults to `None`, which matches all evaluated
            flowcharts. An empty slice returns an empty list.

        Returns
        -------
        list[NodeMatches]
            One collection per selected flowchart, in sorted filename-stem
            order. Each collection identifies the selected Ground-Truth
            Option and records paired nodes, unmatched ground-truth nodes and
            unmatched predicted nodes, together with their node-text costs.
            Node numbers identify nodes within each diagram; matching does
            not require equal predicted and ground-truth node numbers.

        Raises
        ------
        ValueError
            If predicted node text is missing or a text-distance value fails
            validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        Notes
        -----
        The method uses every available predicted part for matching, even
        though the returned result focuses on nodes. Label matches may be
        populated while resolving ties; use
        [`node_and_label_matches()`][flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_and_label_matches]
        when you need label matches for every node.

        """
        self._require_capability(
            available=self.capabilities.node_matches,
            operation="Node matching",
            required_parts=frozenset({"node_text"}),
        )

        start, stop = range_indices if range_indices is not None else (None, None)
        true_diagram_options = self.true_diagrams_options_list[start:stop]
        pred_diagrams = self.pred_diagrams[start:stop]
        diagram_pairs = zip(true_diagram_options, pred_diagrams, strict=True)

        return [
            match_nodes(
                true_diagram_options=tdo,
                pred_diagram=pd,
                distance_fn=self.distance_fn,
            )
            for tdo, pd in tqdm(
                diagram_pairs,
                total=len(pred_diagrams),
                desc="Matching nodes",
                unit="diagram",
            )
        ]

    def node_matches_with_flow(self) -> list[NodeMatches]:
        """
        Return Node Matches after requiring predicted flow as well as text.

        Uses the same matching policy as
        [`node_matches()`][flowde.benchmarks.parsing.parsing_bench.ParsingBenchmark.node_matches].
        The additional flow requirement ensures that each returned collection
        can calculate its `flow_score` property.

        Returns
        -------
        list[NodeMatches]
            One collection per evaluated flowchart, in sorted filename-stem
            order, including the selected Ground-Truth Option and unmatched
            nodes. Flow compares directed connections after applying the
            Node Matches.

        Raises
        ------
        ValueError
            If predicted node text or flow is missing, or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        self._require_capability(
            available=self.capabilities.node_matches_with_flow,
            operation="Node matching with flow",
            required_parts=frozenset({"node_text", "flow"}),
        )

        return self.node_matches()

    def node_and_label_matches(self) -> list[NodeMatches]:
        """
        Match nodes, then match the labels belonging to each Node Match.

        Requires predicted node text and labels. The benchmark selects nodes
        and a Ground-Truth Option using all available predicted parts, then
        pairs label strings within each Node Match to minimise label cost.
        Labels belonging to unmatched nodes count as unmatched labels.

        Returns
        -------
        list[NodeMatches]
            One collection per evaluated flowchart, in sorted filename-stem
            order. Every `NodeMatch.label_matches` contains a `TextListMatches`
            object, including an empty collection when both nodes have no
            labels. `total_label_error_cost` sums the collection's label costs.

        Raises
        ------
        ValueError
            If predicted node text or labels are missing, or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        self._require_capability(
            available=self.capabilities.node_and_label_matches,
            operation="Node and label matching",
            required_parts=frozenset({"node_text", "labels"}),
        )

        node_matches_list = self.node_matches()

        return [
            match_node_labels(node_matches=nms, distance_fn=self.distance_fn)
            for nms in node_matches_list
        ]

    def additional_text_matches(self) -> list[TextListMatches]:
        """
        Match additional-text strings for each evaluated flowchart.

        Requires predicted node text and additional text. The benchmark first
        selects a Ground-Truth Option using all available predicted parts,
        then pairs that option's additional text with the prediction to
        minimise additional-text cost. Strings are matched by cost, not by
        their positions in the input lists.

        Returns
        -------
        list[TextListMatches]
            One collection per flowchart, in sorted filename-stem order.
            Each collection contains paired and unmatched strings with their
            original list indices and costs. Individual records also identify
            the flowchart and selected Ground-Truth Option. If both diagrams
            have empty additional-text lists, the collection has no records
            and `total_cost` is zero.

        Raises
        ------
        ValueError
            If predicted node text or additional text is missing, or a
            text-distance value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        self._require_capability(
            available=self.capabilities.additional_text_matches,
            operation="Additional text matching",
            required_parts=frozenset({"node_text", "additional_texts"}),
        )

        additional_text_matches_list = []
        diagram_pairs = zip(
            self.true_diagrams_options_list,
            self.pred_diagrams,
            strict=True,
        )
        for true_diagrams_options, pred_diagram in tqdm(
            diagram_pairs,
            total=len(self.pred_diagrams),
            desc="Matching additional texts",
            unit="diagram",
        ):
            node_matches = match_nodes(
                true_diagram_options=true_diagrams_options,
                pred_diagram=pred_diagram,
                distance_fn=self.distance_fn,
            )

            true_diagram = true_diagrams_options.options[
                node_matches.true_diagram_option_idx
            ]

            additional_text_matches_list.append(
                match_additional_texts(
                    true_diagram=true_diagram,
                    pred_diagram=pred_diagram,
                    distance_fn=self.distance_fn,
                )
            )

        return additional_text_matches_list

    def total_node_text_cost(self) -> float:
        """
        Sum node-text costs across all evaluated flowcharts.

        Returns
        -------
        float
            Total cost of paired and unmatched node text, using `distance_fn`.
            With the default distance function, the total counts character
            edits after text normalisation. Lower is better; the total is not
            an average or percentage. Requires predicted node text.

        Raises
        ------
        ValueError
            If predicted node text is missing or a text-distance value fails
            validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        node_matches_list = self.node_matches()
        return sum(nm.total_node_text_cost for nm in node_matches_list)

    def total_label_cost(self) -> float:
        """
        Sum label costs across all evaluated flowcharts.

        Returns
        -------
        float
            Total cost of paired and unmatched label strings, including
            labels on unmatched nodes, using `distance_fn`. Lower is better;
            the total is not an average or percentage. Requires predicted
            node text and labels. Node text selects the Node Matches but
            does not contribute to this label-only total.

        Raises
        ------
        ValueError
            If predicted node text or labels are missing, or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        node_and_label_matches_list = self.node_and_label_matches()
        return sum(nm.total_label_error_cost for nm in node_and_label_matches_list)

    def total_additional_text_cost(self) -> float:
        """
        Sum additional-text costs across all evaluated flowcharts.

        Returns
        -------
        float
            Total cost of paired and unmatched additional-text strings, using
            `distance_fn`. Lower is better; the total is not an average or
            percentage. Requires predicted node text and additional text.
            Node text helps select the Ground-Truth Option but does not
            contribute to this additional-text-only total.

        Raises
        ------
        ValueError
            If predicted node text or additional text is missing, or a
            text-distance value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        additional_text_matches_list = self.additional_text_matches()
        return sum(matches.total_cost for matches in additional_text_matches_list)

    def all_flow_scores(self) -> list[FlowScores]:
        """
        Score the predicted directed connections in each flowchart.

        Requires predicted node text and flow. The benchmark matches nodes
        before comparing connections, so predicted node numbers may differ
        from ground-truth node numbers. Connection direction matters:
        `1 -> 2` and `2 -> 1` represent different connections.

        Returns
        -------
        list[FlowScores]
            One score object per evaluated flowchart, in sorted filename-stem
            order. Each object contains correct, extra and missing connection
            counts, precision, recall, F1, Jaccard similarity, and the sets of
            extra and missing connections. Reported connections use matched
            ground-truth node numbers; unmatched predicted endpoints receive
            generated identifiers starting at `10000`.

        Raises
        ------
        ValueError
            If predicted node text or flow is missing, or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        self._require_capability(
            available=self.capabilities.flow_scores,
            operation="Flow scoring",
            required_parts=frozenset({"node_text", "flow"}),
        )

        return [nms.flow_score for nms in self.node_matches_with_flow()]

    def all_flow_jaccard_scores(self) -> list[float]:
        """
        Return one flow Jaccard similarity score per evaluated flowchart.

        Returns
        -------
        list[float]
            Scores in sorted filename-stem order, calculated after matching
            nodes. Each score is `TP / (TP + FP + FN)`, between `0.0` and
            `1.0`; higher is better. A score of `1.0` means the directed
            connection sets agree. Two empty connection sets score `1.0`.
            Requires predicted node text and flow.

        Raises
        ------
        ValueError
            If predicted node text or flow is missing, or a text-distance
            value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        return [fs.jaccard for fs in self.all_flow_scores()]

    def avg_flow_jaccard(self) -> float:
        """
        Average the flow Jaccard scores of the evaluated flowcharts.

        Returns
        -------
        float
            Arithmetic mean of the individual flow Jaccard scores, between
            `0.0` and `1.0`; higher is better. Each flowchart has equal weight,
            regardless of its number of connections. Missing predictions
            excluded with `allow_missing_pred_diagrams=True` do not contribute
            to the mean. Requires predicted node text and flow.

        Raises
        ------
        ValueError
            If predicted node text or flow is missing, there are no evaluated
            flowcharts, or a text-distance value fails validation.
        RuntimeError
            If the node matcher cannot establish an optimal solution.

        """
        scores = self.all_flow_jaccard_scores()
        if len(scores) == 0:
            msg = (
                "Cannot calculate average flow Jaccard because there are no "
                "flow scores."
            )
            raise ValueError(msg)

        return sum(scores) / len(scores)
