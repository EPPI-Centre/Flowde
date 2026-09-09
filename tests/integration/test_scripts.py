import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
FIXTURE_IMAGE = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "fixtures"
    / "paddle_expected"
    / "Bush_2018_0.png"
)
GROUND_TRUTH_FIXTURE_ROOT = (
    REPO_ROOT / "tests" / "integration" / "fixtures" / "ground_truth" / "parsing"
)
FIXTURE_NAME = FIXTURE_IMAGE.name
GEMINI_TEST_MODEL = "gemini-3.5-flash-lite"
OPENAI_TEST_MODEL = "gpt-5.4-mini"
TEST_THINKING_LEVEL = "low"


def run_script(
    script_path: Path,
    args: Sequence[str],
    *,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    temp_repo = tmp_path / "repo"
    shutil.copytree(SCRIPTS_ROOT, temp_repo / "scripts")
    temp_script_path = temp_repo / script_path.relative_to(REPO_ROOT)

    resolved_tmp_path = tmp_path.resolve()
    for arg in args:
        arg_path = Path(arg)
        if arg_path.is_absolute() and not arg_path.resolve().is_relative_to(
            resolved_tmp_path
        ):
            msg = f"Script argument path is outside the temporary directory: {arg}"
            raise ValueError(msg)

    return subprocess.run(  # noqa: S603
        [sys.executable, str(temp_script_path), *args],
        cwd=temp_repo,
        check=True,
        capture_output=True,
        text=True,
    )


def make_fixture_image_dir(tmp_path: Path) -> Path:
    if not FIXTURE_IMAGE.is_file():
        msg = f"Test fixture does not exist: {FIXTURE_IMAGE}"
        raise AssertionError(msg)

    if FIXTURE_IMAGE.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/v1"
    ):
        msg = (
            "The image fixture is still a Git LFS pointer. Run "
            "`git lfs checkout tests/integration/fixtures/paddle_expected/"
            "Bush_2018_0.png` before running script integration tests."
        )
        raise AssertionError(msg)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    shutil.copy2(FIXTURE_IMAGE, image_dir / FIXTURE_NAME)
    return image_dir


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def load_ground_truth_component(component_name: str) -> dict:
    path = (
        GROUND_TRUTH_FIXTURE_ROOT / component_name / f"{Path(FIXTURE_NAME).stem}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def make_temporary_ground_truth_dirs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    temporary_ground_truth_root = tmp_path / "ground-truth"
    temporary_nodes_dir = temporary_ground_truth_root / "nodes"
    temporary_labels_dir = temporary_ground_truth_root / "labels"
    temporary_additional_texts_dir = temporary_ground_truth_root / "additional_texts"
    temporary_flow_dir = temporary_ground_truth_root / "flow"

    for component_name, temporary_component_dir in [
        ("nodes", temporary_nodes_dir),
        ("labels", temporary_labels_dir),
        ("additional_texts", temporary_additional_texts_dir),
        ("flow", temporary_flow_dir),
    ]:
        fixture_ground_truth_path = (
            GROUND_TRUTH_FIXTURE_ROOT
            / component_name
            / f"{Path(FIXTURE_NAME).stem}.json"
        )
        temporary_component_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            fixture_ground_truth_path,
            temporary_component_dir / fixture_ground_truth_path.name,
        )

    return (
        temporary_nodes_dir,
        temporary_labels_dir,
        temporary_additional_texts_dir,
        temporary_flow_dir,
    )


def make_partial_dirs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    partial_root = tmp_path / "partial"
    nodes_dir = partial_root / "nodes"
    labels_dir = partial_root / "labels"
    additional_texts_dir = partial_root / "additional_texts"
    flow_dir = partial_root / "flow"

    nodes = load_ground_truth_component("nodes")["options"][0]["nodes"]
    labels = load_ground_truth_component("labels")["options"][0]["nodes"]
    additional_texts = load_ground_truth_component("additional_texts")["options"][0]
    flow = load_ground_truth_component("flow")["options"][0]["nodes"]

    write_json(nodes_dir / f"{Path(FIXTURE_NAME).stem}.json", {"nodes": nodes})
    write_json(labels_dir / f"{Path(FIXTURE_NAME).stem}.json", {"nodes": labels})
    write_json(
        additional_texts_dir / f"{Path(FIXTURE_NAME).stem}.json",
        additional_texts,
    )
    write_json(flow_dir / f"{Path(FIXTURE_NAME).stem}.json", {"nodes": flow})

    return nodes_dir, labels_dir, additional_texts_dir, flow_dir


def make_predicted_diagram() -> dict:
    nodes = load_ground_truth_component("nodes")["options"][0]["nodes"]
    labels = load_ground_truth_component("labels")["options"][0]["nodes"]
    flow = load_ground_truth_component("flow")["options"][0]["nodes"]
    additional_texts = load_ground_truth_component("additional_texts")["options"][0][
        "additional_texts"
    ]

    return {
        "nodes": [
            {
                "node_number": node["node_number"],
                "text": node["text"],
                "labels": label["labels"],
                "points_to": flow_node["points_to"],
            }
            for node, label, flow_node in zip(nodes, labels, flow, strict=True)
        ],
        "additional_texts": additional_texts,
    }


def load_one_result(output_dir: Path) -> dict:
    result_path = output_dir / f"{Path(FIXTURE_NAME).stem}.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def test_classification_benchmark_script_runs(
    tmp_path: Path,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"
    classification = [{"img_path": "Bush_2018_0.png", "label": 1}]
    write_json(true_path, classification)
    write_json(pred_path, classification)

    result = run_script(
        SCRIPTS_ROOT
        / "benchmark"
        / "classification"
        / "consort_classification_benchmark.py",
        ["--true-path", str(true_path), "--pred-path", str(pred_path)],
        tmp_path=tmp_path,
    )

    assert "Accuracy: 1.000" in result.stdout


def test_rotation_benchmark_script_runs(
    tmp_path: Path,
) -> None:
    true_path = tmp_path / "true.json"
    pred_path = tmp_path / "pred.json"
    classification = [{"img_path": "Bush_2018_0.png", "label": 0}]
    write_json(true_path, classification)
    write_json(pred_path, classification)

    result = run_script(
        SCRIPTS_ROOT / "benchmark" / "rotation" / "rotation_benchmark_template.py",
        ["--true-path", str(true_path), "--pred-path", str(pred_path)],
        tmp_path=tmp_path,
    )

    assert "Accuracy: 1.000" in result.stdout


def test_parse_benchmark_script_runs(
    tmp_path: Path,
) -> None:
    true_nodes_dir, true_labels_dir, true_additional_texts_dir, true_flow_dir = (
        make_temporary_ground_truth_dirs(tmp_path)
    )
    pred_dir = tmp_path / "predictions"
    write_json(
        pred_dir / f"{Path(FIXTURE_NAME).stem}.json",
        make_predicted_diagram(),
    )

    result = run_script(
        SCRIPTS_ROOT / "benchmark" / "parsing" / "parse_all_bench_template.py",
        [
            "--pred-diagrams-dir",
            str(pred_dir),
            "--true-nodes-dir",
            str(true_nodes_dir),
            "--true-labels-dir",
            str(true_labels_dir),
            "--true-additional-texts-dir",
            str(true_additional_texts_dir),
            "--true-flow-dir",
            str(true_flow_dir),
            "--allow-missing-pred-diagrams",
            "false",
            "--expected-num-diagrams",
            "1",
        ],
        tmp_path=tmp_path,
    )

    assert "Matched diagrams: 1" in result.stdout


def test_classification_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    result_path = tmp_path / "classification" / "classifications.json"

    run_script(
        SCRIPTS_ROOT / "inference" / "classification" / "classify_imgs_template.py",
        [
            "--model",
            GEMINI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_path.parent),
            "--copy-positive-images",
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["img_path"] == str(image_dir / FIXTURE_NAME)
    assert result[0]["label"] == 1


def test_rotation_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    result_path = tmp_path / "rotation" / "results.json"
    rotated_image_dir = tmp_path / "rotation" / "rotated_images"

    run_script(
        SCRIPTS_ROOT / "inference" / "rotation" / "rotate_template.py",
        [
            "--model",
            GEMINI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--json-path",
            str(result_path),
            "--save-dir",
            str(rotated_image_dir),
            "--save-in-place",
            "false",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["img_path"] == str(image_dir / FIXTURE_NAME)
    assert result[0]["label"] == 0


def test_parse_nodes_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    result_dir = tmp_path / "nodes"

    run_script(
        SCRIPTS_ROOT / "inference" / "parsing" / "parse_nodes_template.py",
        [
            "--model",
            GEMINI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_dir),
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = load_one_result(result_dir)
    assert result["nodes"]


def test_parse_labels_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    nodes_dir, _, _, _ = make_partial_dirs(tmp_path)
    result_dir = tmp_path / "labels"

    run_script(
        SCRIPTS_ROOT / "inference" / "parsing" / "parse_labels_template.py",
        [
            "--model",
            OPENAI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_dir),
            "--nodes-dir",
            str(nodes_dir),
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = load_one_result(result_dir)
    assert result["nodes"]
    assert any(node["labels"] for node in result["nodes"])


def test_parse_additional_text_script_runs(
    tmp_path: Path,
) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    nodes_dir, labels_dir, _, _ = make_partial_dirs(tmp_path)
    result_dir = tmp_path / "additional_texts"

    run_script(
        SCRIPTS_ROOT / "inference" / "parsing" / "parse_additional_text_template.py",
        [
            "--model",
            GEMINI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_dir),
            "--nodes-dir",
            str(nodes_dir),
            "--labels-dir",
            str(labels_dir),
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = load_one_result(result_dir)
    assert isinstance(result["additional_texts"], list)


def test_parse_flow_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    nodes_dir, labels_dir, additional_texts_dir, _ = make_partial_dirs(tmp_path)
    result_dir = tmp_path / "flow"

    run_script(
        SCRIPTS_ROOT / "inference" / "parsing" / "parse_flow_template.py",
        [
            "--model",
            GEMINI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_dir),
            "--nodes-dir",
            str(nodes_dir),
            "--labels-dir",
            str(labels_dir),
            "--additional-texts-dir",
            str(additional_texts_dir),
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = load_one_result(result_dir)
    assert result["nodes"]
    assert any(node["points_to"] for node in result["nodes"])


def test_parse_all_script_runs(tmp_path: Path) -> None:
    image_dir = make_fixture_image_dir(tmp_path)
    result_dir = tmp_path / "all"

    run_script(
        SCRIPTS_ROOT / "inference" / "parsing" / "parse_all_template.py",
        [
            "--model",
            OPENAI_TEST_MODEL,
            "--thinking-level",
            TEST_THINKING_LEVEL,
            "--img-dir",
            str(image_dir),
            "--save-dir",
            str(result_dir),
            "--start-index",
            "0",
            "--stop-index",
            "1",
            "--n-jobs",
            "1",
        ],
        tmp_path=tmp_path,
    )

    result = load_one_result(result_dir)
    assert result["nodes"]
    assert any(node["labels"] for node in result["nodes"])
    assert any(node["points_to"] for node in result["nodes"])
    assert isinstance(result["additional_texts"], list)
