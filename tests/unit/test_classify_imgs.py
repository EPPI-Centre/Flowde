import json
from unittest.mock import Mock

import pytest

from flowde import model_function
from flowde.classify_imgs import classify_imgs, classify_imgs_from_paths
from flowde.usage import RequestUsage, report_usage


@pytest.fixture
def images(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for name in ("b.png", "a.png", "c.png"):
        (directory / name).write_text(name)
    (directory / "ignored.jpg").write_text("ignored")
    return directory


@pytest.fixture
def classifier():
    def classify(img_path):
        return {"a.png": 1, "b.png": 2, "c.png": 0}[img_path.name]

    return model_function(classify, labels="filename")


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize(
    "positive_classes", [None, {1, 2}], ids=["labels-only", "copy-positives"]
)
def test_classification_saves_ordered_labels_and_selected_images(
    images, classifier, tmp_path, n_jobs, positive_classes
):
    output = tmp_path / "output"

    result = classify_imgs(
        classifier, images, output, positive_classes=positive_classes, n_jobs=n_jobs
    )

    assert result == [1, 2, 0]
    assert json.loads((output / "classifications.json").read_text()) == [
        {"img_path": str(images / "a.png"), "label": 1},
        {"img_path": str(images / "b.png"), "label": 2},
        {"img_path": str(images / "c.png"), "label": 0},
    ]
    assert (output / ".flowde" / "run.state").is_file()
    assert list(output.rglob("*.json")) == [output / "classifications.json"]
    if positive_classes is None:
        assert not (output / "positive_images").exists()
    else:
        assert sorted(path.name for path in (output / "positive_images").iterdir()) == [
            "a.png",
            "b.png",
        ]
        for name in ("a.png", "b.png"):
            assert (output / "positive_images" / name).read_bytes() == (
                images / name
            ).read_bytes()


def test_explicit_paths_preserve_the_supplied_return_order(
    images, classifier, tmp_path
):
    result = classify_imgs_from_paths(
        classifier, [images / "c.png", images / "a.png"], tmp_path / "output", n_jobs=1
    )
    assert result == [0, 1]


@pytest.mark.parametrize(
    "label", [False, "relevant", 2], ids=["boolean", "string", "integer"]
)
def test_custom_label_types_survive_resume(images, tmp_path, label):
    classify = model_function(Mock(return_value=label), label=label)
    output = tmp_path / "output"
    options = {
        "positive_classes": {label},
        "n_jobs": 1,
    }
    # Mock would otherwise invent an attribute that ordinary functions don't have.
    classify.result_structure = None
    classify_imgs(classify, images, output, **options)
    classify.reset_mock()

    result = classify_imgs(classify, images, output, on_existing="resume", **options)

    assert result == [label] * 3
    assert all(type(value) is type(label) for value in result)
    classify.assert_not_called()
    assert len(list((output / "positive_images").glob("*.png"))) == 3


@pytest.mark.parametrize(
    "ranges",
    [((None, 1), (1, None)), ((0, 2), (1, 3))],
    ids=["disjoint-slices", "overlapping-slices"],
)
def test_resumed_slices_keep_earlier_json_entries(images, classifier, tmp_path, ranges):
    output = tmp_path / "output"
    classify_imgs(classifier, images, output, range_indices=ranges[0], n_jobs=1)

    result = classify_imgs(
        classifier,
        images,
        output,
        range_indices=ranges[1],
        on_existing="resume",
        n_jobs=1,
    )

    assert result == [1, 2, 0][slice(*ranges[1])]
    assert [
        entry["label"]
        for entry in json.loads((output / "classifications.json").read_text())
    ] == [1, 2, 0]


@pytest.mark.parametrize("show_usage", [False, True], ids=["hidden", "visible"])
def test_classify_imgs_controls_usage_display(images, tmp_path, capsys, show_usage):
    def classify(img_path):
        report_usage(RequestUsage(provider="test", model="a", cost=2, total_tokens=300))
        return 1

    classify = model_function(classify)
    result = classify_imgs(
        classify,
        images,
        tmp_path / "output",
        n_jobs=1,
        show_usage=show_usage,
    )

    assert result == [1, 1, 1]
    output = capsys.readouterr()
    text = output.out + output.err
    assert "3/3" in output.err
    assert ("Est. cost:" in text) == show_usage
    assert ("Tokens:" in text) == show_usage


@pytest.mark.parametrize("case", ["missing", "file", "empty", "no-pngs"])
def test_invalid_image_directories_do_not_create_outputs(tmp_path, classifier, case):
    images = tmp_path / "images"
    if case == "file":
        images.touch()
    elif case in {"empty", "no-pngs"}:
        images.mkdir()
        if case == "no-pngs":
            (images / "other.jpg").touch()
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=r"directory|PNG"):
        classify_imgs(classifier, images, output)

    assert not output.exists()


def test_empty_slice_does_not_start_a_run(images, classifier, tmp_path):
    with pytest.raises(ValueError, match="img_paths cannot be empty"):
        classify_imgs(classifier, images, tmp_path / "output", range_indices=(2, 2))
    assert not (tmp_path / "output").exists()
