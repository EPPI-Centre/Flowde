"""Storage costs and recovery guarantees of the per-input run records."""

import json
from unittest.mock import Mock

import pytest

from flowde import _run_state
from flowde._batch import run_batch
from flowde._run_settings import file_digest
from flowde._run_state import RunState, output_lock
from flowde.usage import RequestUsage


def make_run(tmp_path, count=2):
    output = tmp_path / "output"
    inputs = [
        {
            "key": str(tmp_path / f"image_{index:04}.png"),
            "path": str(tmp_path / f"image_{index:04}.png"),
            "outputs": [f"image_{index:04}.json"],
            "input": {"file_digest": "a" * 64},
        }
        for index in range(count)
    ]
    with output_lock(output):
        state = RunState(output, "parsing", {}, "error")
        state.prepare(inputs)
    return state, inputs


def contents(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_progress_writes_only_the_affected_input_record(tmp_path, monkeypatch):
    state, inputs = make_run(tmp_path)
    assert {path.name for path in state.input_records_path.iterdir()} == {
        "image_0000.state",
        "image_0001.state",
    }
    key = inputs[0]["key"]
    before = contents(state.root)
    writes = []
    original = _run_state.atomic_write

    def record_write(path, content):
        writes.append(path)
        original(path, content)

    monkeypatch.setattr(_run_state, "atomic_write", record_write)
    state.add_usage(key, RequestUsage(model="test", provider="test", total_tokens=3))
    state.failure(key, RuntimeError("first attempt failed"))
    state.result(key, {"text": "complete answer"})
    state.publish(key)

    changed = {state.record_path(key), state.root / inputs[0]["outputs"][0]}
    assert set(writes) == changed
    after = contents(state.root)
    for name, content in before.items():
        if state.root / name not in changed:
            assert after[name] == content
    restored = RunState(state.root, "parsing", {}, "resume")
    assert restored.input_records[key]["completed"] is True
    assert restored.input_records[key]["error"] is None
    assert restored.input_records[key]["result"] == {"text": "complete answer"}
    assert restored.input_records[key]["outputs"] == ["image_0000.json"]
    assert "output" not in restored.input_records[key]
    assert restored.totals().tokens == {0: 3}


@pytest.mark.parametrize("resume", [False, True], ids=["new-run", "resume"])
@pytest.mark.parametrize("second_name", ["diagram.jpg", "DIAGRAM.png"])
@pytest.mark.parametrize("distinct_resolved_names", [False, True])
def test_stem_collisions_are_rejected_before_processing_or_writing(
    tmp_path, resume, second_name, distinct_resolved_names
):
    output = tmp_path / "output"
    inputs = [
        {
            "key": str(path),
            "path": str(path),
            "outputs": [],
            "input": {"file_digest": "a" * 64},
        }
        for path in (
            tmp_path / "first" / "diagram.png",
            tmp_path / "second" / second_name,
        )
    ]
    if distinct_resolved_names:
        for index, item in enumerate(inputs):
            item["key"] = str(tmp_path / "resolved" / f"source_{index}.png")
    with output_lock(output):
        if resume:
            state = RunState(
                output, "classification", {"positive_classes": None}, "error"
            )
            state.prepare(inputs[:1])
            state.result(inputs[0]["key"], "positive")
            state.publish(inputs[0]["key"])
    before = contents(output)
    fn = Mock()
    with pytest.raises(ValueError, match="filename stems must be unique ignoring case"):
        run_batch(
            fn,
            inputs[1:] if resume else inputs,
            output,
            kind="classification",
            settings={"positive_classes": None},
            encode=str,
            decode=str,
            n_jobs=1,
            show_usage=False,
            on_existing="resume" if resume else "error",
        )
    fn.assert_not_called()
    assert contents(output) == before


def test_doubling_parsed_inputs_approximately_doubles_state_bytes_written(
    tmp_path, monkeypatch
):
    original = _run_state.atomic_write
    written = []

    def record_write(path, content):
        if ".flowde" in path.parts:
            written.append(len(content))
        original(path, content)

    monkeypatch.setattr(_run_state, "atomic_write", record_write)

    def run(count):
        written.clear()
        state, inputs = make_run(tmp_path / str(count), count)
        for item in inputs:
            key = item["key"]
            state.add_usage(
                key, RequestUsage(model="test", provider="test", total_tokens=3)
            )
            state.result(key, {"text": "x" * 8192})
            state.publish(key)
        return sum(written)

    small = run(10)
    large = run(20)
    assert 1.9 < large / small < 2.1


@pytest.mark.parametrize("mode", ["resume", "overwrite"])
@pytest.mark.parametrize("target", ["metadata", "item", "shared"])
@pytest.mark.parametrize("damage", ["missing", "invalid-json", "invalid-utf8", "shape"])
def test_damaged_internal_records_stop_before_processing_or_changing_files(
    tmp_path, mode, target, damage
):
    state, inputs = make_run(tmp_path)
    key = inputs[0]["key"]
    state.result(key, {"text": "saved answer"})
    state.publish(key)
    path = {
        "metadata": state.metadata_path,
        "item": state.record_path(key),
        "shared": state.shared_path,
    }[target]
    if damage == "missing":
        path.unlink()
    elif damage == "invalid-json":
        path.write_bytes(b"{")
    elif damage == "invalid-utf8":
        path.write_bytes(b"\xff")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if target == "metadata":
            del data["input_paths"]
        elif target == "item":
            del data["result"]
        else:
            data["unrelated.json"] = "a" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
    before = contents(state.root)
    fn = Mock()
    with pytest.raises(
        ValueError, match="Missing or invalid saved run record"
    ) as error:
        run_batch(
            fn,
            inputs[1:],  # Validate even records outside the selected inputs.
            state.root,
            kind="parsing",
            settings={},
            encode=lambda x: x,
            decode=lambda x: x,
            n_jobs=1,
            show_usage=False,
            on_existing=mode,
        )
    assert str(path) in str(error.value)
    fn.assert_not_called()
    assert contents(state.root) == before


@pytest.mark.parametrize("mode", ["resume", "overwrite"])
def test_legacy_run_is_rejected_without_changing_files(tmp_path, mode):
    state, inputs = make_run(tmp_path)
    key = inputs[0]["key"]
    state.result(key, {"text": "saved answer"})
    state.publish(key)
    legacy = {**state.data, "version": 1}
    state.metadata_path.write_text(json.dumps(legacy), encoding="utf-8")
    for path in state.input_records_path.iterdir():
        path.unlink()
    state.input_records_path.rmdir()
    state.shared_path.unlink()
    before = contents(state.root)
    with pytest.raises(ValueError, match="Old runs cannot be resumed or overwritten"):
        RunState(state.root, "parsing", {}, mode)
    assert contents(state.root) == before


@pytest.mark.parametrize("mode", ["resume", "overwrite"])
@pytest.mark.parametrize(
    "outputs",
    [None, "image_0000.json", [42], [], ["image_0000.json", "image_0000.json"]],
    ids=["missing", "not-a-list", "not-a-filename", "missing-output", "duplicate"],
)
def test_invalid_output_inventory_is_rejected_without_changing_files(
    tmp_path, mode, outputs
):
    state, inputs = make_run(tmp_path)
    key = inputs[0]["key"]
    state.result(key, {"text": "saved answer"})
    state.publish(key)
    path = state.record_path(key)
    record = json.loads(path.read_text(encoding="utf-8"))
    if outputs is None:
        del record["outputs"]
    else:
        record["outputs"] = outputs
    path.write_text(json.dumps(record), encoding="utf-8")
    before = contents(state.root)

    with pytest.raises(ValueError, match="Missing or invalid saved run record"):
        RunState(state.root, "parsing", {}, mode)

    assert contents(state.root) == before


@pytest.mark.parametrize("interruption", ["before-output", "before-completion"])
def test_saved_answer_recovers_without_repeating_parsing(
    tmp_path, monkeypatch, interruption
):
    state, inputs = make_run(tmp_path, 1)
    key = inputs[0]["key"]
    result = {"text": "résumé: complete answer"}
    state.result(key, result)
    original = _run_state.atomic_write

    def interrupt_write(path, content):
        stop_at = (
            state.root / inputs[0]["outputs"][0]
            if interruption == "before-output"
            else state.record_path(key)
        )
        if path == stop_at:
            raise KeyboardInterrupt
        original(path, content)

    monkeypatch.setattr(_run_state, "atomic_write", interrupt_write)
    with pytest.raises(KeyboardInterrupt):
        state.publish(key)
    monkeypatch.setattr(_run_state, "atomic_write", original)
    fn = Mock()
    assert run_batch(
        fn,
        inputs,
        state.root,
        kind="parsing",
        settings={},
        encode=lambda x: x,
        decode=lambda x: x,
        n_jobs=1,
        show_usage=False,
        on_existing="resume",
    ) == [result]
    fn.assert_not_called()
    assert (
        json.loads((state.root / inputs[0]["outputs"][0]).read_text(encoding="utf-8"))
        == result
    )
    restored = RunState(state.root, "parsing", {}, "resume")
    assert restored.input_records[key]["completed"] is True


def test_missing_public_json_is_recreated_from_completed_record(tmp_path):
    state, inputs = make_run(tmp_path, 1)
    key = inputs[0]["key"]
    result = {"text": "saved answer"}
    state.result(key, result)
    state.publish(key)
    public = state.root / inputs[0]["outputs"][0]
    public.unlink()
    fn = Mock()
    assert run_batch(
        fn,
        inputs,
        state.root,
        kind="parsing",
        settings={},
        encode=lambda x: x,
        decode=lambda x: x,
        n_jobs=1,
        show_usage=False,
        on_existing="resume",
    ) == [result]
    fn.assert_not_called()
    assert json.loads(public.read_text(encoding="utf-8")) == result


def test_saved_positive_classification_can_copy_the_image_without_reclassifying(
    tmp_path,
):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image bytes")
    inputs = [
        {
            "key": str(image),
            "path": str(image),
            "outputs": [],
            "input": {"file_digest": file_digest(image)},
        }
    ]
    settings = {"positive_classes": ["positive"]}
    output = tmp_path / "output"
    with output_lock(output):
        state = RunState(output, "classification", settings, "error")
        state.prepare(inputs)
        state.result(str(image), "positive")
    # Simulate stopping after saving the answer, before publishing any output.
    assert not (output / "positive_images" / image.name).exists()
    fn = Mock()

    assert run_batch(
        fn,
        inputs,
        output,
        kind="classification",
        settings=settings,
        encode=str,
        decode=str,
        n_jobs=1,
        show_usage=False,
        on_existing="resume",
    ) == ["positive"]

    fn.assert_not_called()
    assert (output / "positive_images" / image.name).read_bytes() == image.read_bytes()
