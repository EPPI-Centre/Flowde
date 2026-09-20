import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from flowde import _run_state, model_function
from flowde.rotate_imgs import rotate_imgs, rotate_imgs_from_paths


def create_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (2, 3))
    image.putdata([1, 2, 3, 4, 5, 6])
    image.save(path)
    return path


def classifier():
    return model_function(lambda path: int(path.stem), version=1)


def labels(output):
    return json.loads((output / "rotations.json").read_text())


@pytest.mark.parametrize("max_concurrent_jobs", [1, 2])
def test_explicit_paths_preserve_order_and_always_save_copies(
    tmp_path, max_concurrent_jobs
):
    sources = [
        create_image(tmp_path / "inputs" / f"{a}.png") for a in (270, 0, 90, 180)
    ]
    originals = [p.read_bytes() for p in sources]
    output = tmp_path / "output"

    result = rotate_imgs_from_paths(
        classifier(), sources, output, max_concurrent_jobs=max_concurrent_jobs
    )

    assert result == [270, 0, 90, 180]
    assert {p.name for p in output.iterdir()} == {
        "rotations.json",
        "rotated_images",
        ".flowde",
    }
    assert sorted(p.name for p in (output / "rotated_images").iterdir()) == [
        "0.png",
        "180.png",
        "270.png",
        "90.png",
    ]
    assert labels(output) == [
        {"img_path": str(p), "label": int(p.stem)} for p in sorted(sources)
    ]
    assert [p.read_bytes() for p in sources] == originals


@pytest.mark.parametrize(
    ("angle", "size", "pixels"),
    [
        (0, (2, 3), [1, 2, 3, 4, 5, 6]),
        (90, (3, 2), [5, 3, 1, 6, 4, 2]),
        (180, (2, 3), [6, 5, 4, 3, 2, 1]),
        (270, (3, 2), [2, 4, 6, 1, 3, 5]),
    ],
)
def test_clockwise_correction_and_resume_leave_originals_unchanged(
    tmp_path, angle, size, pixels, calls
):
    source = create_image(tmp_path / "source.png")
    original = source.read_bytes()
    output = tmp_path / "output"

    def classify(path):
        calls.append(path.name)
        return angle

    fn = model_function(classify, version=1)
    assert rotate_imgs_from_paths(fn, [source], output, max_concurrent_jobs=1) == [
        angle
    ]
    assert list(calls) == [source.name]
    del calls[:]

    assert rotate_imgs_from_paths(
        fn, [source], output, max_concurrent_jobs=1, on_existing="resume"
    ) == [angle]

    assert list(calls) == []
    assert source.read_bytes() == original
    with Image.open(output / "rotated_images" / source.name) as corrected:
        assert corrected.size == size
        assert [
            corrected.getpixel((x, y)) for y in range(size[1]) for x in range(size[0])
        ] == pixels


@pytest.mark.parametrize("max_concurrent_jobs", [1, 2])
def test_directory_sorts_top_level_pngs_and_resumes_overlapping_slices(
    tmp_path, max_concurrent_jobs
):
    inputs = tmp_path / "inputs"
    for angle in (90, 0, 270, 180):
        create_image(inputs / f"{angle}.png")
    create_image(inputs / "nested" / "unexpected.png")
    (inputs / "ignore.txt").write_text("not an image")
    output = tmp_path / "output"

    assert rotate_imgs(
        classifier(),
        inputs,
        output,
        range_indices=(0, 2),
        max_concurrent_jobs=max_concurrent_jobs,
    ) == [0, 180]
    assert rotate_imgs(
        classifier(),
        inputs,
        output,
        range_indices=(1, None),
        max_concurrent_jobs=max_concurrent_jobs,
        on_existing="resume",
    ) == [180, 270, 90]
    assert [entry["label"] for entry in labels(output)] == [0, 180, 270, 90]
    assert len(list((output / "rotated_images").glob("*.png"))) == 4


@pytest.mark.parametrize("invalid", [45, -90, 360, "90", None])
def test_invalid_angle_preserves_previous_copy_and_stops_new_images(
    tmp_path, invalid, calls
):
    inputs = tmp_path / "inputs"
    for name in ("a", "b", "c"):
        create_image(inputs / f"{name}.png")

    def process(path):
        calls.append(path.stem)
        return invalid if path.stem == "b" else 90

    output = tmp_path / "output"
    with pytest.raises((ValueError, TypeError)):
        rotate_imgs(model_function(process), inputs, output, max_concurrent_jobs=1)

    assert list(calls) == ["a", "b"]
    assert labels(output) == [{"img_path": str(inputs / "a.png"), "label": 90}]
    assert sorted(p.name for p in (output / "rotated_images").iterdir()) == ["a.png"]


@pytest.mark.parametrize(
    "stage", ["encode-image", "replace-image", "record-completion"]
)
def test_failed_image_publication_resumes_without_repeating_or_rotating_twice(
    tmp_path, monkeypatch, stage, calls
):
    source = create_image(tmp_path / "source.png")
    original = source.read_bytes()
    output = tmp_path / "output"

    def classify(path):
        calls.append(path.name)
        return 90

    fn = model_function(classify, version=1)
    save_image = Image.Image.save
    replace = Path.replace
    save_state = _run_state.RunState.save

    def broken_image_save(image, *args, **kwargs):
        msg = "image encoding failed"
        raise OSError(msg)

    def broken_replace(path, target):
        if Path(target).suffix == ".png":
            msg = "image replacement failed"
            raise OSError(msg)
        return replace(path, target)

    def broken_completion(state):
        if any(record["completed"] for record in state.items.values()):
            msg = "completion save failed"
            raise OSError(msg)
        save_state(state)

    if stage == "encode-image":
        monkeypatch.setattr(Image.Image, "save", broken_image_save)
    elif stage == "replace-image":
        monkeypatch.setattr(Path, "replace", broken_replace)
    else:
        monkeypatch.setattr(_run_state.RunState, "save", broken_completion)

    with pytest.raises(OSError, match="failed"):
        rotate_imgs_from_paths(fn, [source], output, max_concurrent_jobs=1)
    assert list(calls) == [source.name]
    del calls[:]
    state = json.loads((output / ".flowde" / "run.state").read_text())
    record = state["items"][str(source.resolve())]
    assert record["has_result"]
    assert not record["completed"]
    assert record["result"] == 90
    corrected_path = output / "rotated_images" / source.name
    assert corrected_path.exists() == (stage == "record-completion")
    assert not list(output.rglob(".flowde-*"))

    monkeypatch.setattr(Image.Image, "save", save_image)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(_run_state.RunState, "save", save_state)
    assert rotate_imgs_from_paths(
        fn, [source], output, max_concurrent_jobs=1, on_existing="resume"
    ) == [90]

    assert list(calls) == []
    assert source.read_bytes() == original
    with Image.open(corrected_path) as corrected:
        assert corrected.size == (3, 2)
        assert [corrected.getpixel((x, y)) for y in range(2) for x in range(3)] == [
            5,
            3,
            1,
            6,
            4,
            2,
        ]


def test_missing_copy_is_recreated_but_modified_copy_is_rejected(tmp_path, calls):
    source = create_image(tmp_path / "source.png")
    output = tmp_path / "output"

    def classify(path):
        calls.append(path.name)
        return 90

    fn = model_function(classify)
    rotate_imgs_from_paths(fn, [source], output, max_concurrent_jobs=1)
    corrected = output / "rotated_images" / source.name
    original_copy = corrected.read_bytes()
    corrected.unlink()
    assert list(calls) == [source.name]
    del calls[:]

    rotate_imgs_from_paths(
        fn, [source], output, max_concurrent_jobs=1, on_existing="resume"
    )
    assert corrected.read_bytes() == original_copy
    assert list(calls) == []

    corrected.write_bytes(b"user edited copy")
    with pytest.raises(ValueError, match="Saved output has been modified"):
        rotate_imgs_from_paths(
            fn, [source], output, max_concurrent_jobs=1, on_existing="resume"
        )
    assert corrected.read_bytes() == b"user edited copy"
    assert list(calls) == []


@pytest.mark.parametrize("on_existing", ["error", "overwrite"])
def test_can_start_in_empty_directory(tmp_path, on_existing):
    source = create_image(tmp_path / "0.png")
    output = tmp_path / "output"
    output.mkdir()
    assert rotate_imgs_from_paths(
        classifier(), [source], output, max_concurrent_jobs=1, on_existing=on_existing
    ) == [0]


@pytest.mark.parametrize(
    "extra", ["notes.json", "notes.txt", "rotated_images/notes.txt"]
)
def test_overwrite_preserves_unrecorded_files(tmp_path, extra):
    source = create_image(tmp_path / "0.png")
    output = tmp_path / "output"
    rotate_imgs_from_paths(classifier(), [source], output, max_concurrent_jobs=1)
    (output / extra).write_text("keep this")
    before = {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="unexpected file or directory"):
        rotate_imgs_from_paths(
            classifier(),
            [source],
            output,
            max_concurrent_jobs=1,
            on_existing="overwrite",
        )

    assert {str(p): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before


def test_source_changed_during_model_request_is_not_rotated(tmp_path):
    source = create_image(tmp_path / "source.png")
    output = tmp_path / "output"

    def change_source(path):
        Image.new("L", (4, 5)).save(path)
        return 90

    with pytest.raises(ValueError, match="Input image changed before rotating"):
        rotate_imgs_from_paths(
            model_function(change_source), [source], output, max_concurrent_jobs=1
        )
    assert not (output / "rotated_images" / source.name).exists()


@pytest.mark.parametrize("case", ["missing", "file", "empty", "nested-only"])
def test_invalid_input_directory_makes_no_requests(tmp_path, case):
    inputs = tmp_path / "inputs"
    if case == "file":
        inputs.write_text("not a directory")
    elif case in ("empty", "nested-only"):
        inputs.mkdir()
        if case == "nested-only":
            create_image(inputs / "nested" / "0.png")
    fn = model_function(Mock(spec=[], return_value=0))
    with pytest.raises(ValueError, match=r"Image directory|No PNG image files"):
        rotate_imgs(fn, inputs, tmp_path / "output", max_concurrent_jobs=1)
    fn.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_empty_list_and_duplicate_stems_make_no_requests(tmp_path):
    fn = model_function(Mock(spec=[], return_value=0))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="cannot be empty"):
        rotate_imgs_from_paths(fn, [], output, max_concurrent_jobs=1)
    sources = [create_image(tmp_path / folder / "same.png") for folder in ("a", "b")]
    with pytest.raises(ValueError, match=r"[Dd]uplicate"):
        rotate_imgs_from_paths(fn, sources, output, max_concurrent_jobs=1)
    fn.assert_not_called()
    assert not output.exists()
