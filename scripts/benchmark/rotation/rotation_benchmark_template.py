from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from flowde.benchmarks.rotation.rotation_benchmark import RotationBenchmark

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "training-smoking-cessation"
TRUE_JSON_PATH = DATASET_DIR / "rotation" / "true_rotation_classification.json"
PRED_JSON_PATH = DATASET_DIR / "rotation" / "pred_gemini_3.1_rotation.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Benchmark image rotation classifications.")
    parser.add_argument(
        "--true-path",
        type=Path,
        default=TRUE_JSON_PATH,
        help=f"Ground-truth JSON path (default: {TRUE_JSON_PATH}).",
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=PRED_JSON_PATH,
        help=f"Predictions JSON path (default: {PRED_JSON_PATH}).",
    )
    args = parser.parse_args(argv)

    benchmark = RotationBenchmark(
        true_path=args.true_path,
        pred_path=args.pred_path,
    )

    print(f"Accuracy: {benchmark.accuracy:.3f}")
    for incorrect_prediction in benchmark.incorrect_predictions:
        print(incorrect_prediction.img_path)
        print(f"True: {incorrect_prediction.true}")
        print(f"Pred: {incorrect_prediction.pred}")


if __name__ == "__main__":
    main()
