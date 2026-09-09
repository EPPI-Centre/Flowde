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
    A validated parsing benchmark with explicit prediction capabilities.

    ``pred_diagrams_dir`` may be one directory of combined predictions or a
    sequence of directories containing separately parsed, non-overlapping
    components. Component structure is inferred and the files are joined by
    diagram stem and predicted node number before benchmarking.
    """

    def __init__(
        self,
        pred_diagrams_dir: Path | Sequence[Path],
        distance_fn: DistanceFnProtocol,
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
        self._require_capability(
            available=self.capabilities.node_matches_with_flow,
            operation="Node matching with flow",
            required_parts=frozenset({"node_text", "flow"}),
        )

        return self.node_matches()

    def node_and_label_matches(self) -> list[NodeMatches]:
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
        node_matches_list = self.node_matches()
        return sum(nm.total_node_text_cost for nm in node_matches_list)

    def total_label_cost(self) -> float:
        node_and_label_matches_list = self.node_and_label_matches()
        return sum(nm.total_label_error_cost for nm in node_and_label_matches_list)

    def total_additional_text_cost(self) -> float:
        additional_text_matches_list = self.additional_text_matches()
        return sum(matches.total_cost for matches in additional_text_matches_list)

    def all_flow_scores(self) -> list[FlowScores]:
        self._require_capability(
            available=self.capabilities.flow_scores,
            operation="Flow scoring",
            required_parts=frozenset({"node_text", "flow"}),
        )

        return [nms.flow_score for nms in self.node_matches_with_flow()]

    def all_flow_jaccard_scores(self) -> list[float]:
        return [fs.jaccard for fs in self.all_flow_scores()]

    def avg_flow_jaccard(self) -> float:
        scores = self.all_flow_jaccard_scores()
        if len(scores) == 0:
            msg = (
                "Cannot calculate average flow Jaccard because there are no "
                "flow scores."
            )
            raise ValueError(msg)

        return sum(scores) / len(scores)
