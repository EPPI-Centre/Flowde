from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from flowde.classify_fns.classify_types import RotationClassification
from flowde.classify_fns.gemini_classify_fn import make_gemini_classify_fn

# from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.rotate_imgs import INPUT_TEXT, rotate_imgs
from flowde.utils import parse_bool

MODEL = "gemini-3.1-flash-lite-preview"
THINKING_LEVEL = "low"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "training-smoking-cessation"
IMG_DIR = DATASET_DIR / "extraction" / "paddle_layout_detect_p3"
JSON_PATH = DATASET_DIR / "rotation" / "pred_gemini_3.1_rotation.json"
SAVE_DIR = None
SAVE_IN_PLACE = False
N_JOBS = 1


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Classify and rotate flowchart images.")
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
        "--json-path",
        type=Path,
        default=JSON_PATH,
        help=f"Path for the JSON results (default: {JSON_PATH}).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help="Optional directory in which to save rotated images.",
    )
    parser.add_argument(
        "--save-in-place",
        type=parse_bool,
        default=SAVE_IN_PLACE,
        help=(
            "Rotate non-zero-angle images in the input directory "
            f"(default: {SAVE_IN_PLACE})."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=f"Number of parallel model calls (default: {N_JOBS}).",
    )
    args = parser.parse_args(argv)

    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1.")

    classify_fn = make_gemini_classify_fn(
        input_text=INPUT_TEXT,
        model=args.model,
        result_structure=RotationClassification,
        effort=args.thinking_level,
    )

    # classify_fn = make_openai_classify_fn(
    #     input_text=INPUT_TEXT,
    #     model=args.model,
    #     result_structure=RotationClassification,
    #     effort=args.thinking_level,
    # )

    rotate_imgs(
        classify_fn=classify_fn,
        img_dir=args.img_dir,
        save_dir=args.save_dir,
        json_path=args.json_path,
        save_in_place=args.save_in_place,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
