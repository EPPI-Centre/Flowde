from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from flowde.classify_fns.classify_types import ConsortClassification
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

# from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.classify_imgs import classify_imgs

INPUT_TEXT = """
Classify the image as CONSORT image with label 1.
If the image is not a CONSORT image, classify it with label 0.
"""

MODEL = "gemini-3.1-flash-lite-preview"
# MODEL = "gpt-5.4-mini"
THINKING_LEVEL = "low"

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "training-smoking-cessation"
IMG_DIR = DATASET_DIR / "extraction" / "paddle_layout_detect_p3"
SAVE_DIR = DATASET_DIR / "classification" / "template_example"
START_INDEX = None
STOP_INDEX = None
N_JOBS = 1

RESULT_STRUCTURE = ConsortClassification


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Classify extracted flowchart images.")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model name (default: {MODEL}).",
    )
    parser.add_argument(
        "--thinking-level",
        default=THINKING_LEVEL,
        help=f"Model thinking level (default: {THINKING_LEVEL}).",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=IMG_DIR,
        help=f"Directory containing PNG images (default: {IMG_DIR}).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help=f"Dedicated classification output directory (default: {SAVE_DIR}).",
    )
    parser.add_argument(
        "--copy-positive-images",
        action="store_true",
        help="Copy images labelled 1 into the run's positive_images directory.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help=(
            "Inclusive first sorted-image index; without a stop, run to the end "
            f"(default: {START_INDEX})."
        ),
    )
    parser.add_argument(
        "--stop-index",
        type=int,
        default=STOP_INDEX,
        help=(
            "Exclusive last sorted-image index; omit to run through the end "
            f"(default: {STOP_INDEX})."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=f"Number of parallel model calls (default: {N_JOBS}).",
    )
    parser.add_argument(
        "--on-existing",
        choices=("error", "resume", "overwrite"),
        default="error",
        help="How to handle an output directory containing previous work.",
    )
    args = parser.parse_args(argv)

    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1.")

    classify_fn = make_gemini_classify_fn(
        input_text=INPUT_TEXT,
        model=args.model,
        result_structure=RESULT_STRUCTURE,
        effort=args.thinking_level,
    )

    # classify_fn = make_openai_classify_fn(
    #     input_text=INPUT_TEXT,
    #     model=args.model,
    #     result_structure=RESULT_STRUCTURE,
    #     effort=args.thinking_level,
    # )

    classify_imgs(
        classify_fn=classify_fn,
        img_dir=args.img_dir,
        save_dir=args.save_dir,
        positive_classes={1} if args.copy_positive_images else None,
        range_indices=(args.start_index, args.stop_index),
        n_jobs=args.n_jobs,
        on_existing=args.on_existing,
    )


if __name__ == "__main__":
    main()
