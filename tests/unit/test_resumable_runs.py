import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from google.genai.errors import ClientError
from openai import BadRequestError
from pydantic import BaseModel, ConfigDict

from flowde import _batch, _run_state, model_function
from flowde.classify_imgs import classify_imgs
from flowde.parse_imgs import parse_imgs
from flowde.usage import RequestUsage, report_usage


class Answer(BaseModel):
    name: str


@pytest.fixture(params=["classification", "parsing"])
def task(request):
    return request.param


@pytest.fixture
def images(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for name in ("a", "b", "c"):
        (directory / f"{name}.png").write_text(name)
    return directory


def make_function(task, action=None):
    def process(img_path):
        if action is not None:
            action(img_path)
        return img_path.stem if task == "classification" else Answer(name=img_path.stem)

    return model_function(
        process, result_structure=Answer if task == "parsing" else None, version=1
    )


def run(task, fn, images, output, **options):
    api = classify_imgs if task == "classification" else parse_imgs
    return api(fn, images, output, **{"n_jobs": 1, **options})


def names(task, results):
    return results if task == "classification" else [result.name for result in results]


def saved_names(task, output):
    if task == "classification":
        return [
            entry["label"]
            for entry in json.loads((output / "classifications.json").read_text())
        ]
    return sorted(
        json.loads(path.read_text())["name"] for path in output.glob("*.json")
    )


def test_failure_preserves_successes_and_resume_skips_them(task, images, tmp_path):
    output = tmp_path / "output"
    calls = []

    def fail_on_b(path):
        calls.append(path.stem)
        if path.stem == "b":
            # The first result must already be saved while the batch is still running.
            assert saved_names(task, output) == ["a"]
            msg = "quota exhausted"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="quota exhausted"):
        run(task, make_function(task, fail_on_b), images, output)

    assert calls == ["a", "b"]
    assert saved_names(task, output) == ["a"]
    state = json.loads((output / ".flowde" / "run.state").read_text())
    assert (
        state["items"][str((images / "b.png").resolve())]["error"]["message"]
        == "quota exhausted"
    )

    calls.clear()
    result = run(
        task,
        make_function(task, lambda path: calls.append(path.stem)),
        images,
        output,
        on_existing="resume",
    )

    assert calls == ["b", "c"]
    assert names(task, result) == ["a", "b", "c"]
    assert saved_names(task, output) == ["a", "b", "c"]


def wait_for(condition):
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            msg = "Other worker did not reach the expected stage"
            raise AssertionError(msg)
        time.sleep(0.01)


@pytest.mark.parametrize("provider", ["custom", "openai", "gemini"])
def test_parallel_failure_saves_the_other_active_image_and_starts_no_more(
    task, images, tmp_path, provider
):
    output = tmp_path / "output"
    started = tmp_path / "b-started"
    state_path = output / ".flowde" / "run.state"
    failed_key = str((images / "a.png").resolve())

    def process(path):
        if path.stem == "a":
            wait_for(started.exists)
            msg = "first image failed"
            if provider == "openai":
                raise BadRequestError(
                    msg,
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "https://unused.example")
                    ),
                    body={},
                )
            if provider == "gemini":
                raise ClientError(400, {"error": {"message": msg}})
            raise RuntimeError(msg)
        if path.stem == "b":
            started.touch()
            # Finish only after the parent has observed the other worker's error.
            wait_for(
                lambda: (
                    json.loads(state_path.read_text())["items"][failed_key]["error"]
                    is not None
                )
            )
        if path.stem == "c":
            (tmp_path / "unexpected-third-request").touch()

    error_type = {
        "custom": RuntimeError,
        "openai": BadRequestError,
        "gemini": ClientError,
    }[provider]
    with pytest.raises(error_type, match="first image failed"):
        run(task, make_function(task, process), images, output, n_jobs=2)

    assert saved_names(task, output) == ["b"]
    assert not (tmp_path / "unexpected-third-request").exists()
    calls = []
    result = run(
        task,
        make_function(task, lambda path: calls.append(path.stem)),
        images,
        output,
        on_existing="resume",
    )
    assert calls == ["a", "c"]
    assert names(task, result) == ["a", "b", "c"]


def test_existing_work_requires_an_explicit_choice(task, images, tmp_path):
    output = tmp_path / "output"
    fn = make_function(task)
    run(task, fn, images, output)
    original = (output / ".flowde" / "run.state").read_bytes()

    with pytest.raises(FileExistsError, match=r"resume.*overwrite"):
        run(task, fn, images, output)
    assert (output / ".flowde" / "run.state").read_bytes() == original

    fn = model_function(fn, version=2)
    with pytest.raises(ValueError, match="settings have changed"):
        run(task, fn, images, output, on_existing="resume")
    result = run(task, fn, images, output, on_existing="overwrite")
    assert names(task, result) == ["a", "b", "c"]


def test_overwrite_removes_old_outputs_when_the_input_images_change(
    task, images, tmp_path
):
    output = tmp_path / "output"
    options = {"positive_classes": {"a", "b", "c"}} if task == "classification" else {}
    run(task, make_function(task), images, output, **options)
    # Missing or edited recorded outputs should not prevent explicit overwrite.
    old_output = output / "positive_images" / "c.png" if options else output / "c.json"
    old_output.unlink()
    old_json = output / ("classifications.json" if options else "a.json")
    old_json.write_text("edited output")

    new_images = tmp_path / "new-images"
    new_images.mkdir()
    (new_images / "d.png").write_text("d")
    result = run(task, make_function(task), new_images, output, on_existing="overwrite")

    assert names(task, result) == ["d"]
    assert saved_names(task, output) == ["d"]
    expected_json = "classifications.json" if task == "classification" else "d.json"
    assert {path.name for path in output.iterdir()} == {".flowde", expected_json}
    state = json.loads((output / ".flowde" / "run.state").read_text())
    assert set(state["items"]) == {str((new_images / "d.png").resolve())}


def directory_contents(directory):
    """Snapshot the directory to check that a refused overwrite leaves it intact."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in directory.rglob("*")
    }


@pytest.mark.parametrize(
    "state_contents",
    [
        pytest.param(None, id="missing-state"),
        pytest.param("{", id="broken-json"),
        pytest.param("{}", id="unrelated-json"),
        pytest.param({"version": 999}, id="unsupported-version"),
    ],
)
def test_overwrite_requires_a_valid_run_record(images, tmp_path, state_contents):
    output = tmp_path / "output"
    run("parsing", make_function("parsing"), images, output)
    state_path = output / ".flowde" / "run.state"
    if state_contents is None:
        state_path.unlink()
    elif isinstance(state_contents, dict):
        state = json.loads(state_path.read_text())
        state.update(state_contents)
        state_path.write_text(json.dumps(state))
    else:
        state_path.write_text(state_contents)
    before = directory_contents(output)
    calls = Mock()

    with pytest.raises(
        ValueError, match=r"a valid Flowde \.flowde/run\.state is required"
    ):
        run(
            "parsing",
            make_function("parsing", calls),
            images,
            output,
            on_existing="overwrite",
        )

    calls.assert_not_called()
    assert directory_contents(output) == before


@pytest.mark.parametrize(
    "unrecorded_path",
    [
        pytest.param("unrecorded.json", id="unrecorded-json"),
        pytest.param("positive_images/notes.txt", id="file-among-image-copies"),
        pytest.param(".flowde/notes.txt", id="file-among-metadata"),
        pytest.param("unrelated-directory", id="empty-directory"),
    ],
)
def test_overwrite_refuses_unrecorded_contents_before_deleting_anything(
    images, tmp_path, unrecorded_path
):
    output = tmp_path / "output"
    run(
        "classification",
        make_function("classification"),
        images,
        output,
        positive_classes={"a", "b", "c"},
    )
    extra = output / unrecorded_path
    if extra.suffix:
        extra.write_text("User's own file")
    else:
        extra.mkdir()
    before = directory_contents(output)
    calls = Mock()

    with pytest.raises(ValueError, match="unexpected file or directory") as raised:
        run(
            "classification",
            make_function("classification", calls),
            images,
            output,
            on_existing="overwrite",
        )

    assert str(Path(unrecorded_path)) in str(raised.value)
    calls.assert_not_called()
    assert directory_contents(output) == before


@pytest.mark.parametrize(
    "absolute", [False, True], ids=["parent-path", "absolute-path"]
)
def test_overwrite_rejects_recorded_paths_outside_the_run(images, tmp_path, absolute):
    output = tmp_path / "output"
    run("parsing", make_function("parsing"), images, output)
    outside = tmp_path / "outside.json"
    outside.write_text("Unrelated work")
    unsafe_path = str(outside) if absolute else "../outside.json"
    state_path = output / ".flowde" / "run.state"
    state = json.loads(state_path.read_text())
    state["items"][str((images / "a.png").resolve())]["output"] = unsafe_path
    state["artifacts"][unsafe_path] = state["artifacts"].pop("a.json")
    state_path.write_text(json.dumps(state))
    before = directory_contents(output)
    calls = Mock()

    with pytest.raises(
        ValueError, match=r"a valid Flowde \.flowde/run\.state is required"
    ):
        run(
            "parsing",
            make_function("parsing", calls),
            images,
            output,
            on_existing="overwrite",
        )

    calls.assert_not_called()
    assert directory_contents(output) == before
    assert outside.read_text() == "Unrelated work"


def test_overwrite_refuses_a_recorded_output_replaced_by_a_symlink(images, tmp_path):
    output = tmp_path / "output"
    run("parsing", make_function("parsing"), images, output)
    outside = tmp_path / "outside.json"
    outside.write_text("Unrelated work")
    saved_json = output / "b.json"
    saved_json.unlink()
    saved_json.symlink_to(outside)
    before = directory_contents(output)
    calls = Mock()

    with pytest.raises(ValueError, match=r"unexpected file or directory b\.json"):
        run(
            "parsing",
            make_function("parsing", calls),
            images,
            output,
            on_existing="overwrite",
        )

    calls.assert_not_called()
    assert saved_json.is_symlink()
    assert directory_contents(output) == before
    assert outside.read_text() == "Unrelated work"


def test_resume_requires_a_saved_run_even_in_an_existing_directory(
    task, images, tmp_path
):
    output = tmp_path / "empty"
    output.mkdir()
    with pytest.raises(ValueError, match="No saved run"):
        run(task, make_function(task), images, output, on_existing="resume")


def test_completed_input_changes_are_rejected_before_new_requests(
    task, images, tmp_path
):
    output = tmp_path / "output"
    run(task, make_function(task), images, output)
    (images / "a.png").write_text("changed image")
    calls = Mock()

    with pytest.raises(ValueError, match="Previously processed image"):
        run(task, make_function(task, calls), images, output, on_existing="resume")
    calls.assert_not_called()


def test_resume_rejects_a_different_image_with_the_same_output_name(
    task, images, tmp_path
):
    output = tmp_path / "output"
    run(task, make_function(task), images, output)
    other_images = tmp_path / "other-images"
    other_images.mkdir()
    (other_images / "a.png").write_text("a different image")
    calls = Mock()

    with pytest.raises(ValueError, match="already associated with a different image"):
        run(
            task,
            make_function(task, calls),
            other_images,
            output,
            on_existing="resume",
        )

    calls.assert_not_called()
    assert saved_names(task, output) == ["a", "b", "c"]


def test_failed_inputs_can_be_fixed_and_new_images_can_be_added(task, images, tmp_path):
    output = tmp_path / "output"

    def check_image(path):
        if path.read_text() == "b":
            msg = "broken image"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="broken image"):
        run(task, make_function(task, check_image), images, output)
    (images / "b.png").write_text("repaired")
    (images / "d.png").write_text("new")

    result = run(
        task, make_function(task, check_image), images, output, on_existing="resume"
    )
    assert names(task, result) == ["a", "b", "c", "d"]


def test_usage_from_failed_attempts_is_restored_without_recounting_saved_answers(
    task, images, tmp_path, monkeypatch
):
    output = tmp_path / "output"

    def charged_request(path):
        report_usage(
            RequestUsage(provider="test", model="model", total_tokens=100, cost=1)
        )
        if path.stem == "b":
            msg = "response failed validation"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="failed validation"):
        run(task, make_function(task, charged_request), images, output)
    initial_totals = []
    real_display = _batch.UsageProgress

    def display(*args, **kwargs):
        totals = kwargs["totals"]
        initial_totals.append(
            (
                len(totals.finished_items),
                sum(totals.tokens.values()),
                sum(totals.costs.values()),
            )
        )
        return real_display(*args, **kwargs)

    monkeypatch.setattr(_batch, "UsageProgress", display)

    def successful_request(path):
        report_usage(
            RequestUsage(provider="test", model="model", total_tokens=100, cost=1)
        )

    run(
        task,
        make_function(task, successful_request),
        images,
        output,
        on_existing="resume",
        n_jobs=2,
    )

    assert initial_totals == [(1, 200, 2)]
    state = json.loads((output / ".flowde" / "run.state").read_text())
    usages = [usage for item in state["items"].values() for usage in item["usage"]]
    assert sum(usage["total_tokens"] for usage in usages) == 400
    assert sum(usage["cost"] for usage in usages) == 4


def test_none_is_a_failure_and_never_saved_as_a_success(task, images, tmp_path):
    fn = make_function(task)
    empty = Mock(return_value=None)
    empty.run_settings = fn.run_settings
    empty.result_structure = getattr(fn, "result_structure", None)
    output = tmp_path / "output"

    with pytest.raises(TypeError, match="None is a failure"):
        run(task, empty, images, output)

    empty.assert_called_once()
    state = json.loads((output / ".flowde" / "run.state").read_text())
    assert not any(item["completed"] for item in state["items"].values())
    assert list(output.glob("*.json")) == []


def test_copy_failure_does_not_repeat_a_successful_classification(
    images, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    calls = []
    fn = make_function("classification", lambda path: calls.append(path.stem))
    write = _run_state.atomic_write

    def broken_copy(path, data):
        if path.parent.name == "positive_images":
            msg = "copy failed"
            raise OSError(msg)
        write(path, data)

    monkeypatch.setattr(_run_state, "atomic_write", broken_copy)
    with pytest.raises(OSError, match="copy failed"):
        classify_imgs(fn, images, output, positive_classes={"a"}, n_jobs=1)
    assert calls == ["a"]
    monkeypatch.setattr(_run_state, "atomic_write", write)

    result = classify_imgs(
        fn, images, output, positive_classes={"a"}, n_jobs=1, on_existing="resume"
    )

    assert calls == ["a", "b", "c"]
    assert result == ["a", "b", "c"]
    assert (output / "positive_images" / "a.png").read_bytes() == (
        images / "a.png"
    ).read_bytes()


def test_first_interrupt_saves_current_image_and_stops_before_the_next(
    task, images, tmp_path
):
    output = tmp_path / "output"
    calls = []

    def interrupt(path):
        calls.append(path.stem)
        # Invoke the installed signal handler without interrupting the pytest runner.
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    with pytest.raises(KeyboardInterrupt, match="Stopped after saving"):
        run(task, make_function(task, interrupt), images, output)
    assert calls == ["a"]
    assert saved_names(task, output) == ["a"]
    assert names(
        task, run(task, make_function(task), images, output, on_existing="resume")
    ) == ["a", "b", "c"]


def test_second_interrupt_escapes_the_current_function(task, images, tmp_path):
    output = tmp_path / "output"

    def interrupt_twice(path):
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        msg = "Forced interruption did not stop the function"
        raise AssertionError(msg)

    with pytest.raises(KeyboardInterrupt, match="Forced stop"):
        run(task, make_function(task, interrupt_twice), images, output)
    assert names(
        task, run(task, make_function(task), images, output, on_existing="resume")
    ) == ["a", "b", "c"]


def test_overwrite_cannot_delete_source_images(task, images):
    with pytest.raises(ValueError, match="contains an input file"):
        run(task, make_function(task), images, images, on_existing="overwrite")
    assert (images / "a.png").read_text() == "a"


def test_a_second_writer_cannot_open_the_same_run(tmp_path):
    output = tmp_path / "output"
    with (
        _run_state.output_lock(output),
        pytest.raises(RuntimeError, match="Another run"),
        _run_state.output_lock(output),
    ):
        pytest.fail("The second writer acquired the lock")


def test_changed_partial_flowchart_is_rejected_before_resuming(images, tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    for image in images.glob("*.png"):
        (nodes / f"{image.stem}.json").write_text(
            '{"nodes": [{"node_number": 1, "text": "Known node"}]}'
        )
    calls = []

    def parse(img_path, partial_flowchart=None):
        calls.append(img_path.name)
        return Answer(name=partial_flowchart.nodes[0].text)

    parse = model_function(parse, result_structure=Answer)
    output = tmp_path / "output"
    parse_imgs(parse, images, output, nodes_dir=nodes, n_jobs=1)
    calls.clear()
    (nodes / "a.json").write_text(
        '{"nodes": [{"node_number": 1, "text": "Different known node"}]}'
    )

    with pytest.raises(ValueError, match="partial flowchart has changed"):
        parse_imgs(
            parse,
            images,
            output,
            nodes_dir=nodes,
            on_existing="resume",
            n_jobs=1,
        )
    assert calls == []


def test_failed_atomic_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "run.state"
    path.write_text("previous complete record")

    def fail_replace(source, destination):
        msg = "disk failure"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        _run_state.atomic_write(path, b"replacement record")
    assert path.read_text() == "previous complete record"
    assert list(tmp_path.iterdir()) == [path]


def test_modified_saved_outputs_are_not_silently_replaced(task, images, tmp_path):
    output = tmp_path / "output"
    fn = make_function(task)
    run(task, fn, images, output)
    artifact = next(output.glob("*.json"))
    artifact.write_text("user edited result")

    with pytest.raises(ValueError, match="Saved output has been modified"):
        run(task, fn, images, output, on_existing="resume")
    assert artifact.read_text() == "user edited result"


def test_custom_parser_restores_its_declared_type_and_json_fields(images, tmp_path):
    class DatedAnswer(BaseModel):
        model_config = ConfigDict(strict=True)

        processed_at: datetime

    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    parser = model_function(
        Mock(return_value=DatedAnswer(processed_at=timestamp)),
        result_structure=DatedAnswer,
        version=1,
    )
    output = tmp_path / "output"
    first = parse_imgs(parser, images, output, n_jobs=1)
    parser.reset_mock()

    restored = parse_imgs(parser, images, output, n_jobs=1, on_existing="resume")

    assert restored == first
    assert all(isinstance(result, DatedAnswer) for result in restored)
    assert all(result.processed_at == timestamp for result in restored)
    parser.assert_not_called()


def test_custom_functions_must_declare_settings_before_a_run(task, images, tmp_path):
    fn = make_function(task)
    del fn.run_settings
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="must declare their settings"):
        run(task, fn, images, output)

    assert not output.exists()


def test_custom_parsers_must_declare_the_result_class(images, tmp_path):
    fn = make_function("parsing")
    del fn.result_structure
    output = tmp_path / "output"

    with pytest.raises(TypeError, match=r"parse_fn\.result_structure"):
        parse_imgs(fn, images, output, n_jobs=1)

    assert not output.exists()


def test_overwrite_cannot_delete_examples_referenced_in_settings(images, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    example = output / "example.png"
    example.write_text("example image")
    fn = model_function(make_function("classification"), example=example)

    with pytest.raises(ValueError, match="contains an input file"):
        classify_imgs(fn, images, output, on_existing="overwrite", n_jobs=1)

    assert example.read_text() == "example image"
