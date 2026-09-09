from pathlib import Path

from flowde.classify_fns.openai_classify_fn import make_openai_classify_fn
from flowde.classify_imgs import classify_imgs

REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = (
    REPO_ROOT / "data/training-smoking-cessation/extraction/paddle_layout_detect_p3"
)

if __name__ == "__main__":
    classify_fn = make_openai_classify_fn(
        input_text=(
            "Return label 1 if this image is a CONSORT participant flow diagram. "
            "Otherwise return label 0."
        ),
        model="gpt-5.6-luna",
        effort="low",
    )

    labels = classify_imgs(
        classify_fn=classify_fn,
        img_dir=IMG_DIR,
        save_dir=REPO_ROOT / "example_runs" / "classification_10_images",
        range_indices=(0, 10),
        n_jobs=2,
    )
    print(labels)
