import json
from pathlib import Path

import pytest
from PIL import Image

import flowde.rotate_imgs as rotate_imgs_module


def create_test_img(path: Path, size: tuple[int, int] = (10, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def create_test_file(path: Path, content: str = "not an image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def classify_angle_from_filename(img_path: Path) -> int:
    return {
        "angle_0.png": 0,
        "angle_90.png": 90,
        "angle_180.png": 180,
        "angle_270.png": 270,
        "invalid_45.png": 45,
    }[img_path.name]


def classify_should_not_be_called(img_path: Path) -> int:
    raise AssertionError("classify_fn should not be called")


def save_paths_and_save_in_place_by_save_mode(
    save_mode: str,
    img_paths: list[Path],
    tmp_path: Path,
) -> tuple[list[Path] | None, bool]:
    save_paths = (
        [tmp_path / "outputs" / img_path.name for img_path in img_paths]
        if save_mode == "save_paths"
        else None
    )

    save_in_place = save_mode == "save_in_place"

    return save_paths, save_in_place


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
@pytest.mark.parametrize(
    "save_mode",
    ["save_paths", "save_in_place", "no_save"],
    ids=["save-paths", "save-in-place", "no-save"],
)
def test_rotate_imgs_from_paths_returns_classification_responses(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
    save_mode: str,
) -> None:
    img_paths = [
        tmp_path / "angle_0.png",
        tmp_path / "angle_90.png",
        tmp_path / "angle_180.png",
        tmp_path / "angle_270.png",
    ]

    for img_path in img_paths:
        create_test_img(img_path)

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None
    save_paths, save_in_place = save_paths_and_save_in_place_by_save_mode(
        save_mode=save_mode,
        img_paths=img_paths,
        tmp_path=tmp_path,
    )

    responses = rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=classify_angle_from_filename,
        img_paths=img_paths,
        json_path=json_path,
        save_paths=save_paths,
        save_in_place=save_in_place,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90, 180, 270]

    expected_paths = {
        tmp_path / "angle_0.png",
        tmp_path / "angle_90.png",
        tmp_path / "angle_180.png",
        tmp_path / "angle_270.png",
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    if save_paths is not None:
        expected_paths.update(
            {
                tmp_path / "outputs",
                tmp_path / "outputs" / "angle_0.png",
                tmp_path / "outputs" / "angle_90.png",
                tmp_path / "outputs" / "angle_180.png",
                tmp_path / "outputs" / "angle_270.png",
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize(
    "save_mode",
    ["save_paths", "save_in_place", "no_save"],
    ids=["save-paths", "save-in-place", "no-save"],
)
def test_rotate_imgs_from_paths_changes_json(
    tmp_path: Path,
    n_jobs: int,
    save_mode: str,
) -> None:
    img_0 = tmp_path / "angle_0.png"
    img_90 = tmp_path / "angle_90.png"
    img_paths = [img_0, img_90]

    json_path = tmp_path / "results" / "rotations.json"

    for img_path in img_paths:
        create_test_img(img_path)

    save_paths, save_in_place = save_paths_and_save_in_place_by_save_mode(
        save_mode=save_mode,
        img_paths=img_paths,
        tmp_path=tmp_path,
    )

    responses = rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=classify_angle_from_filename,
        img_paths=img_paths,
        json_path=json_path,
        save_paths=save_paths,
        save_in_place=save_in_place,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90]
    assert json_path.exists()
    assert json.loads(json_path.read_text("utf-8")) == [
        {"img_path": str(img_0), "label": 0},
        {"img_path": str(img_90), "label": 90},
    ]

    expected_paths = {
        img_0,
        img_90,
        json_path.parent,
        json_path,
    }

    if save_paths is not None:
        expected_paths.update(
            {
                tmp_path / "outputs",
                tmp_path / "outputs" / "angle_0.png",
                tmp_path / "outputs" / "angle_90.png",
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
def test_rotate_imgs_from_paths_save_paths_saves_rotated_images(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
) -> None:
    img_paths = [
        tmp_path / "inputs" / "angle_0.png",
        tmp_path / "inputs" / "angle_90.png",
        tmp_path / "inputs" / "angle_180.png",
        tmp_path / "inputs" / "angle_270.png",
    ]
    save_paths = [
        tmp_path / "outputs" / "angle_0.png",
        tmp_path / "outputs" / "angle_90.png",
        tmp_path / "outputs" / "angle_180.png",
        tmp_path / "outputs" / "angle_270.png",
    ]

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None

    for img_path in img_paths:
        create_test_img(img_path, size=(10, 20))

    responses = rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=classify_angle_from_filename,
        img_paths=img_paths,
        json_path=json_path,
        save_paths=save_paths,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90, 180, 270]

    expected_sizes = [
        (10, 20),
        (20, 10),
        (10, 20),
        (20, 10),
    ]

    for img_path, save_path, expected_size in zip(
        img_paths,
        save_paths,
        expected_sizes,
        strict=True,
    ):
        assert save_path.exists()

        with Image.open(save_path) as rotated_img:
            assert rotated_img.size == expected_size

        with Image.open(img_path) as original_img:
            assert original_img.size == (10, 20)

    expected_paths = {
        tmp_path / "inputs",
        tmp_path / "inputs" / "angle_0.png",
        tmp_path / "inputs" / "angle_90.png",
        tmp_path / "inputs" / "angle_180.png",
        tmp_path / "inputs" / "angle_270.png",
        tmp_path / "outputs",
        tmp_path / "outputs" / "angle_0.png",
        tmp_path / "outputs" / "angle_90.png",
        tmp_path / "outputs" / "angle_180.png",
        tmp_path / "outputs" / "angle_270.png",
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
def test_rotate_imgs_from_paths_save_in_place_rotates_original_images(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
) -> None:
    img_paths = [
        tmp_path / "angle_0.png",
        tmp_path / "angle_90.png",
        tmp_path / "angle_180.png",
        tmp_path / "angle_270.png",
    ]

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None

    for img_path in img_paths:
        create_test_img(img_path, size=(10, 20))

    responses = rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=classify_angle_from_filename,
        img_paths=img_paths,
        json_path=json_path,
        save_paths=None,
        save_in_place=True,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90, 180, 270]

    expected_sizes = [
        (10, 20),
        (20, 10),
        (10, 20),
        (20, 10),
    ]

    for img_path, expected_size in zip(img_paths, expected_sizes, strict=True):
        with Image.open(img_path) as img:
            assert img.size == expected_size

    expected_paths = {
        tmp_path / "angle_0.png",
        tmp_path / "angle_90.png",
        tmp_path / "angle_180.png",
        tmp_path / "angle_270.png",
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
def test_rotate_imgs_from_paths_does_not_modify_images_if_no_save_option_given(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
) -> None:
    img_path = tmp_path / "angle_90.png"
    create_test_img(img_path, size=(10, 20))

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None

    responses = rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=classify_angle_from_filename,
        img_paths=[img_path],
        json_path=json_path,
        save_paths=None,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [90]

    with Image.open(img_path) as img:
        assert img.size == (10, 20)

    assert not (tmp_path / "outputs").exists()

    expected_paths = {
        img_path,
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize(
    (
        "img_filenames",
        "save_filenames",
        "save_in_place",
        "classify_fn",
        "expected_msg_parts",
    ),
    [
        pytest.param(
            ["angle_90.png"],
            ["rotated.png"],
            True,
            classify_should_not_be_called,
            ["Cannot specify save_paths if save_in_place is True."],
            id="save-in-place-and-save-paths",
        ),
        pytest.param(
            ["angle_0.png", "angle_90.png"],
            ["angle_0.png"],
            False,
            classify_should_not_be_called,
            ["Length of save_paths must match length of img_paths."],
            id="mismatched-save-paths",
        ),
        pytest.param(
            [],
            [],
            False,
            classify_should_not_be_called,
            ["img_paths cannot be empty."],
            id="empty-img-paths",
        ),
        pytest.param(
            ["invalid_45.png"],
            ["invalid_45.png"],
            False,
            classify_angle_from_filename,
            [
                "Invalid angle 45 for image",
                "Angle must be one of the following: 0, 90, 180, or 270 degrees.",
            ],
            id="invalid-angle-save-paths",
        ),
        pytest.param(
            ["invalid_45.png"],
            None,
            True,
            classify_angle_from_filename,
            [
                "Invalid angle 45 for image",
                "Angle must be one of the following: 0, 90, 180, or 270 degrees.",
            ],
            id="invalid-angle-save-in-place",
        ),
        pytest.param(
            ["invalid_45.png"],
            None,
            False,
            classify_angle_from_filename,
            [
                "Invalid angle 45 for image",
                "Angle must be one of the following: 0, 90, 180, or 270 degrees.",
            ],
            id="invalid-angle-no-save",
        ),
    ],
)
def test_rotate_imgs_from_paths_validation_errors_and_unmodified_outputs(
    tmp_path: Path,
    n_jobs: int,
    img_filenames: list[str],
    save_filenames: list[str] | None,
    save_in_place: bool,
    classify_fn,
    expected_msg_parts: list[str],
) -> None:
    img_paths = [tmp_path / "inputs" / filename for filename in img_filenames]
    save_paths = (
        None
        if save_filenames is None
        else [tmp_path / "outputs" / filename for filename in save_filenames]
    )
    json_path = tmp_path / "results" / "rotations.json"

    for img_path in img_paths:
        create_test_img(img_path, size=(10, 20))

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs_from_paths(
            classify_fn=classify_fn,
            img_paths=img_paths,
            json_path=json_path,
            save_paths=save_paths,
            save_in_place=save_in_place,
            n_jobs=n_jobs,
        )

    msg = str(exc_info.value)

    for expected_msg_part in expected_msg_parts:
        assert expected_msg_part in msg

    assert not json_path.exists()
    assert not json_path.parent.exists()

    if save_paths is not None:
        for save_path in save_paths:
            assert not save_path.exists()
        assert not (tmp_path / "outputs").exists()

    for img_path in img_paths:
        with Image.open(img_path) as img:
            assert img.size == (10, 20)

    expected_paths = set()

    if img_paths:
        expected_paths.add(tmp_path / "inputs")
        expected_paths.update(img_paths)

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_rotate_imgs_returns_classification_responses_from_img_dir(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "inputs"

    create_test_img(img_dir / "angle_270.png")
    create_test_img(img_dir / "angle_0.png")
    create_test_img(img_dir / "angle_90.png")
    create_test_img(img_dir / "angle_180.png")

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=None,
        json_path=None,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [0, 180, 270, 90]


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_rotate_imgs_saves_json_from_img_dir(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "inputs"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_img(img_dir / "angle_90.png")
    create_test_img(img_dir / "angle_0.png")

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=None,
        json_path=json_path,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90]
    assert json_path.exists()
    assert json.loads(json_path.read_text("utf-8")) == [
        {"img_path": str(img_dir / "angle_0.png"), "label": 0},
        {"img_path": str(img_dir / "angle_90.png"), "label": 90},
    ]

    expected_paths = {
        img_dir,
        img_dir / "angle_0.png",
        img_dir / "angle_90.png",
        json_path.parent,
        json_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
def test_rotate_imgs_save_dir_saves_rotated_images_from_img_dir(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
) -> None:
    img_dir = tmp_path / "inputs"
    save_dir = tmp_path / "outputs"

    create_test_img(img_dir / "angle_90.png", size=(10, 20))
    create_test_img(img_dir / "angle_0.png", size=(10, 20))
    create_test_img(img_dir / "angle_270.png", size=(10, 20))

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=save_dir,
        json_path=json_path,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [0, 270, 90]

    output_paths = sorted(save_dir.glob("*.png"))
    assert output_paths == [
        save_dir / "angle_0.png",
        save_dir / "angle_270.png",
        save_dir / "angle_90.png",
    ]

    expected_sizes_by_filename = {
        "angle_0.png": (10, 20),
        "angle_270.png": (20, 10),
        "angle_90.png": (20, 10),
    }

    for output_path in output_paths:
        with Image.open(output_path) as img:
            assert img.size == expected_sizes_by_filename[output_path.name]

    for input_path in sorted(img_dir.glob("*.png")):
        with Image.open(input_path) as img:
            assert img.size == (10, 20)

    expected_paths = {
        img_dir,
        img_dir / "angle_0.png",
        img_dir / "angle_270.png",
        img_dir / "angle_90.png",
        save_dir,
        save_dir / "angle_0.png",
        save_dir / "angle_270.png",
        save_dir / "angle_90.png",
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "no-json"])
def test_rotate_imgs_save_in_place_rotates_images_from_img_dir(
    tmp_path: Path,
    n_jobs: int,
    json_mode: bool,
) -> None:
    img_dir = tmp_path / "inputs"

    create_test_img(img_dir / "angle_90.png", size=(10, 20))
    create_test_img(img_dir / "angle_0.png", size=(10, 20))
    create_test_img(img_dir / "angle_270.png", size=(10, 20))

    json_path = tmp_path / "results" / "rotations.json" if json_mode else None

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=None,
        json_path=json_path,
        save_in_place=True,
        n_jobs=n_jobs,
    )

    assert responses == [0, 270, 90]

    expected_sizes_by_filename = {
        "angle_0.png": (10, 20),
        "angle_270.png": (20, 10),
        "angle_90.png": (20, 10),
    }

    for img_path in sorted(img_dir.glob("*.png")):
        with Image.open(img_path) as img:
            assert img.size == expected_sizes_by_filename[img_path.name]

    expected_paths = {
        img_dir,
        img_dir / "angle_0.png",
        img_dir / "angle_270.png",
        img_dir / "angle_90.png",
    }

    if json_path is not None:
        expected_paths.update(
            {
                json_path.parent,
                json_path,
            }
        )

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_rotate_imgs_does_not_modify_images_if_no_save_option_given(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "inputs"

    create_test_img(img_dir / "angle_90.png", size=(10, 20))
    create_test_img(img_dir / "angle_0.png", size=(10, 20))

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=None,
        json_path=None,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [0, 90]

    for img_path in sorted(img_dir.glob("*.png")):
        with Image.open(img_path) as img:
            assert img.size == (10, 20)

    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "results").exists()

    expected_paths = {
        img_dir,
        img_dir / "angle_0.png",
        img_dir / "angle_90.png",
    }

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_rotate_imgs_ignores_non_png_files(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    img_dir = tmp_path / "inputs"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_img(img_dir / "angle_90.png", size=(10, 20))
    create_test_file(img_dir / "not_an_image.txt")
    create_test_file(img_dir / "fake_jpg.jpg")

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_angle_from_filename,
        img_dir=img_dir,
        save_dir=None,
        json_path=json_path,
        save_in_place=False,
        n_jobs=n_jobs,
    )

    assert responses == [90]

    assert json.loads(json_path.read_text("utf-8")) == [
        {"img_path": str(img_dir / "angle_90.png"), "label": 90},
    ]

    expected_paths = {
        img_dir,
        img_dir / "angle_90.png",
        img_dir / "not_an_image.txt",
        img_dir / "fake_jpg.jpg",
        json_path.parent,
        json_path,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_rotate_imgs_raises_if_img_dir_does_not_exist(tmp_path: Path) -> None:
    img_dir = tmp_path / "missing_inputs"
    json_path = tmp_path / "results" / "rotations.json"

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs(
            classify_fn=classify_should_not_be_called,
            img_dir=img_dir,
            save_dir=None,
            json_path=json_path,
            save_in_place=False,
        )

    assert str(exc_info.value) == (
        f"Image directory {img_dir} does not exist or is not a directory."
    )

    assert not json_path.exists()
    assert not json_path.parent.exists()

    assert set(tmp_path.rglob("*")) == set()


def test_rotate_imgs_raises_if_img_dir_is_not_a_directory(tmp_path: Path) -> None:
    img_dir = tmp_path / "inputs.png"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_file(img_dir, content="not a directory")

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs(
            classify_fn=classify_should_not_be_called,
            img_dir=img_dir,
            save_dir=None,
            json_path=json_path,
            save_in_place=False,
        )

    assert str(exc_info.value) == (
        f"Image directory {img_dir} does not exist or is not a directory."
    )

    assert not json_path.exists()
    assert not json_path.parent.exists()

    expected_paths = {
        img_dir,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_rotate_imgs_raises_if_no_pngs_found(tmp_path: Path) -> None:
    img_dir = tmp_path / "inputs"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_file(img_dir / "not_an_image.txt")
    create_test_file(img_dir / "fake_jpg.jpg")

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs(
            classify_fn=classify_should_not_be_called,
            img_dir=img_dir,
            save_dir=None,
            json_path=json_path,
            save_in_place=False,
        )

    assert str(exc_info.value) == f"No PNG image files found in directory {img_dir}."

    assert not json_path.exists()
    assert not json_path.parent.exists()

    expected_paths = {
        img_dir,
        img_dir / "not_an_image.txt",
        img_dir / "fake_jpg.jpg",
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_rotate_imgs_raises_if_img_dir_empty(tmp_path: Path) -> None:
    img_dir = tmp_path / "inputs"
    img_dir.mkdir(parents=True)

    json_path = tmp_path / "results" / "rotations.json"

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs(
            classify_fn=classify_should_not_be_called,
            img_dir=img_dir,
            save_dir=None,
            json_path=json_path,
            save_in_place=False,
        )

    assert str(exc_info.value) == f"No PNG image files found in directory {img_dir}."

    assert not json_path.exists()
    assert not json_path.parent.exists()

    expected_paths = {
        img_dir,
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_rotate_imgs_raises_if_save_dir_and_save_in_place_are_both_given(
    tmp_path: Path,
) -> None:
    img_dir = tmp_path / "inputs"
    save_dir = tmp_path / "outputs"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_img(img_dir / "angle_90.png")

    with pytest.raises(ValueError) as exc_info:
        rotate_imgs_module.rotate_imgs(
            classify_fn=classify_should_not_be_called,
            img_dir=img_dir,
            save_dir=save_dir,
            json_path=json_path,
            save_in_place=True,
        )

    assert str(exc_info.value) == "Cannot specify save_dir if save_in_place is True."

    assert not save_dir.exists()
    assert not json_path.exists()
    assert not json_path.parent.exists()

    with Image.open(img_dir / "angle_90.png") as img:
        assert img.size == (10, 20)

    expected_paths = {
        img_dir,
        img_dir / "angle_90.png",
    }

    assert set(tmp_path.rglob("*")) == expected_paths


def test_rotate_imgs_passes_sorted_paths_to_rotate_imgs_from_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img_dir = tmp_path / "inputs"
    save_dir = tmp_path / "outputs"
    json_path = tmp_path / "results" / "rotations.json"

    create_test_img(img_dir / "angle_90.png")
    create_test_img(img_dir / "angle_0.png")
    create_test_img(img_dir / "angle_270.png")

    captured_kwargs = {}

    def fake_rotate_imgs_from_paths(**kwargs):
        captured_kwargs.update(kwargs)
        return [0, 270, 90]

    monkeypatch.setattr(
        rotate_imgs_module,
        "rotate_imgs_from_paths",
        fake_rotate_imgs_from_paths,
    )

    responses = rotate_imgs_module.rotate_imgs(
        classify_fn=classify_should_not_be_called,
        img_dir=img_dir,
        save_dir=save_dir,
        json_path=json_path,
        save_in_place=False,
        n_jobs=3,
    )

    expected_img_paths = [
        img_dir / "angle_0.png",
        img_dir / "angle_270.png",
        img_dir / "angle_90.png",
    ]

    assert responses == [0, 270, 90]

    assert captured_kwargs == {
        "classify_fn": classify_should_not_be_called,
        "img_paths": expected_img_paths,
        "json_path": json_path,
        "save_paths": [
            save_dir / "angle_0.png",
            save_dir / "angle_270.png",
            save_dir / "angle_90.png",
        ],
        "save_in_place": False,
        "n_jobs": 3,
    }

    expected_paths = {
        img_dir,
        img_dir / "angle_0.png",
        img_dir / "angle_270.png",
        img_dir / "angle_90.png",
    }

    assert set(tmp_path.rglob("*")) == expected_paths


@pytest.mark.parametrize(
    ("angle", "expected"), [(90, [5, 3, 1, 6, 4, 2]), (270, [2, 4, 6, 1, 3, 5])]
)
@pytest.mark.parametrize("save_in_place", [False, True])
def test_rotation_applies_clockwise_correction(
    tmp_path, angle, expected, save_in_place
):
    source = tmp_path / "sideways.png"
    destination = tmp_path / "upright.png"
    original = Image.new("L", (2, 3))
    original.putdata([1, 2, 3, 4, 5, 6])
    original.save(source)

    rotate_imgs_module.rotate_imgs_from_paths(
        classify_fn=lambda _: angle,
        img_paths=[source],
        save_in_place=save_in_place,
        save_paths=None if save_in_place else [destination],
        n_jobs=1,
    )

    with Image.open(source if save_in_place else destination) as corrected:
        assert corrected.size == (3, 2)
        assert [
            corrected.getpixel((x, y)) for y in range(2) for x in range(3)
        ] == expected
