from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from flowde.parse_imgs import parse_imgs
from flowde.parsing_fns.gemini_parse import make_gemini_parse_fn

# from flowde.parsing_fns.openai_parse import make_openai_parse_fn

MODEL = "gemini-3.1-flash-lite-preview"
THINKING_LEVEL = "low"
PARTS_TO_PARSE = {"node_text"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "training-smoking-cessation"
EXTRACTED_IMGS_METHOD = "true_consort_pp_layout_detection_p3"
DIR_NAME_TO_SAVE_RESULTS = "template_parse_just_nodes_results"
IMG_DIR = DATASET_DIR / "extraction" / EXTRACTED_IMGS_METHOD
PRED_DIR = DATASET_DIR / "parsing" / "pred"
SAVE_DIR = PRED_DIR / DIR_NAME_TO_SAVE_RESULTS
START_INDEX = None
STOP_INDEX = None
N_JOBS = 1


INPUT_TEXT = """
I am sending you an image of a particpant flow diagram.
I want you to parse just the nodes.
Please use the following descriptions for parsing these:

NODE:
- A node is any piece of text that follows the flow of the particpant diagram.
- It may be in a shape such as a box, and it may not.
- Usually has an arrow point to it or away from it, but not always.
- Usually, descriptive words that appear to apply to multiple nodes, are
themselves not nodes.
For example, often, in the margin or in the middle of the diagram there will be
labels such as 'Enrollment', 'Follow-up', 'Analysis' amongst others. These are
not nodes and should not be parsed as such, but rather as labels.
Typically, anything that describes the number of particpants at a particular
stage of the flow, is a node.
- Sometimes, you will see text on arrows, between two boxes. You need to make a
judgement as to wether this is a node or a label.
If it is essential to the participant flow such as it listing numbers of
participants lost to follow-up, or being analsyed, etc, then it is definitely a
node an not a label.
If it appear to apply to multiple nodes, then it is likely a label. You will
have to make a judgement based on what is in front of you.
- Our ultimate goal is to capture the flow of the participants in our own data
structure.
In order to effectively do this, sometimes we need to duplicate nodes.
For example, if there are two branches of the flow diagram, say one for control
and one for the intervention, sometimes those two branches will both point to
the same node, that describes something that happened, and then separate back
into their own branches again. Visually, when looking at the image, it's clear
to see which branch is which, but when we use this data structure to capture
the flow, it isn't. We need to be able to distinguish between the two branches.
The easiest way to this, it to treat the joint up node as a label if possible,
provided that it's only descriptive of the following two nodes and doesn't
describe the participant flow. That way the two branches remain separate in the
data structure and it's obvious which nodes belong to which branch.
However, if it is necessary to keep the node as a node because it captures the
flow, then we need to duplicate it. That way we include one node for each
branch, and the branches remain separate in the data structure.
It is only needed to duplicate a node if two or more branches join and again
diverge back into branches, if they join and no longer separate into branches,
we do not need to duplicate.
- Nodes that describe an intervention or something that happend to the
particpants should usually stay as a node, and not a label.

Important Instructions:
- Parse all supscripts and subscripts as such and in unicode.
- Correctly parse all newlines as such.
- Text should not overlap between labels, nodes and additional text. If a chunk
of text is a node, it cannot also be a label or additional text and vice versa.
You are parsing the diagram as it appears.
- Node numbers should start at 1 and be incremented by 1 for each new node.

"""


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Parse node text from extracted images.")
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
        help=f"Directory for parsed JSON (default: {SAVE_DIR}).",
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

    parse_fn = make_gemini_parse_fn(
        input_text=INPUT_TEXT,
        model=args.model,
        effort=args.thinking_level,
        parts_to_parse=PARTS_TO_PARSE,
    )

    # parse_fn = make_openai_parse_fn(
    #     input_text=INPUT_TEXT,
    #     model=args.model,
    #     effort=args.thinking_level,
    #     parts_to_parse=PARTS_TO_PARSE,
    # )

    parse_imgs(
        parse_fn=parse_fn,
        img_dir=args.img_dir,
        save_dir=args.save_dir,
        range_indices=(args.start_index, args.stop_index),
        n_jobs=args.n_jobs,
        on_existing=args.on_existing,
    )


if __name__ == "__main__":
    main()
