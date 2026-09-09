from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from flowde.benchmarks.parsing.check_parsing_bench_data import (
    N_DIAGRAMS_IN_BENCHMARK,
)
from flowde.benchmarks.parsing.parsing_bench import ParsingBenchmark
from flowde.benchmarks.parsing.text_distance_fns.levenshtein_fn import (
    levenshtein_with_nfc_and_space_normalisation,
)
from flowde.utils import parse_bool

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "training-smoking-cessation"
PRED_DIAGRAMS_DIR = DATASET_DIR / "parsing" / "pred" / "template_parse_all_results"
TRUE_NODES_DIR = DATASET_DIR / "parsing" / "ground-truth" / "nodes"
TRUE_LABELS_DIR = DATASET_DIR / "parsing" / "ground-truth" / "labels"
TRUE_ADDITIONAL_TEXTS_DIR = (
    DATASET_DIR / "parsing" / "ground-truth" / "additional_texts"
)
TRUE_FLOW_DIR = DATASET_DIR / "parsing" / "ground-truth" / "flow"
ALLOW_MISSING_PRED_DIAGRAMS = False
EXPECTED_NUM_DIAGRAMS = N_DIAGRAMS_IN_BENCHMARK


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Benchmark parsed flowchart diagrams.")
    parser.add_argument(
        "--pred-diagrams-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Prediction directory. Repeat this argument to join separately parsed "
            "components (default: "
            f"{PRED_DIAGRAMS_DIR})."
        ),
    )
    parser.add_argument(
        "--true-nodes-dir",
        type=Path,
        default=TRUE_NODES_DIR,
        help=f"Ground-truth nodes directory (default: {TRUE_NODES_DIR}).",
    )
    parser.add_argument(
        "--true-labels-dir",
        type=Path,
        default=TRUE_LABELS_DIR,
        help=f"Ground-truth labels directory (default: {TRUE_LABELS_DIR}).",
    )
    parser.add_argument(
        "--true-additional-texts-dir",
        type=Path,
        default=TRUE_ADDITIONAL_TEXTS_DIR,
        help=(
            "Ground-truth additional-text directory "
            f"(default: {TRUE_ADDITIONAL_TEXTS_DIR})."
        ),
    )
    parser.add_argument(
        "--true-flow-dir",
        type=Path,
        default=TRUE_FLOW_DIR,
        help=f"Ground-truth flow directory (default: {TRUE_FLOW_DIR}).",
    )
    parser.add_argument(
        "--allow-missing-pred-diagrams",
        type=parse_bool,
        default=ALLOW_MISSING_PRED_DIAGRAMS,
        help=(
            "Benchmark only predictions whose image codes are present "
            f"(default: {ALLOW_MISSING_PRED_DIAGRAMS})."
        ),
    )
    parser.add_argument(
        "--expected-num-diagrams",
        type=int,
        default=EXPECTED_NUM_DIAGRAMS,
        help=(
            "Expected number of ground-truth diagrams "
            f"(default: {EXPECTED_NUM_DIAGRAMS})."
        ),
    )
    args = parser.parse_args(argv)
    pred_diagrams_dirs = args.pred_diagrams_dir or [PRED_DIAGRAMS_DIR]

    benchmark = ParsingBenchmark(
        pred_diagrams_dir=pred_diagrams_dirs,
        distance_fn=levenshtein_with_nfc_and_space_normalisation,
        allow_missing_pred_diagrams=args.allow_missing_pred_diagrams,
        true_nodes_dir=args.true_nodes_dir,
        true_labels_dir=args.true_labels_dir,
        true_additional_texts_dir=args.true_additional_texts_dir,
        true_flow_dir=args.true_flow_dir,
        expected_num_diagrams=args.expected_num_diagrams,
    )

    diagram_matches = benchmark.diagram_matches()
    print(f"Matched diagrams: {len(diagram_matches)}")
    print(f"Total node text cost: {benchmark.total_node_text_cost()}")
    print(f"Total label cost: {benchmark.total_label_cost()}")
    print(f"Total additional-text cost: {benchmark.total_additional_text_cost()}")
    print(f"Average flow Jaccard: {benchmark.avg_flow_jaccard():.3f}")


if __name__ == "__main__":
    main()
