import json
import os
import pickle
import signal
import time
from threading import Lock, current_thread, main_thread

import pytest
from joblib.externals.loky.process_executor import TerminatedWorkerError

from flowde import _batch, _run_state, model_function
from flowde.classify_imgs import classify_imgs
from flowde.exponential_backoff import retry_with_exponential_backoff
from flowde.usage import RequestUsage, report_usage


@pytest.fixture
def images(tmp_path):
    directory = tmp_path / "images"
    directory.mkdir()
    for name in "abcde":
        (directory / f"{name}.png").write_bytes(b"custom classifier fixture")
    return directory


def wait_for(condition):
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(
                "The other worker or caller did not reach the expected stage"
            )
        time.sleep(0.01)


def test_jobs_overlap_in_one_child_and_save_on_the_calling_thread(
    images, tmp_path, monkeypatch, calls
):
    save_threads = []
    original = _run_state.RunState.result

    def save(state, key, value):
        save_threads.append(current_thread())
        original(state, key, value)

    monkeypatch.setattr(_run_state.RunState, "result", save)

    def classify(path):
        calls.append(("start", os.getpid(), current_thread().ident))
        (tmp_path / f"started-{path.stem}").touch()
        if path.stem in "abc":
            wait_for(
                lambda: all((tmp_path / f"started-{name}").exists() for name in "abc")
            )
        report_usage(RequestUsage(provider="test", model="fake", total_tokens=10))
        calls.append(("finish", os.getpid(), current_thread().ident))
        return path.stem

    results = classify_imgs(
        model_function(classify), images, tmp_path / "out", max_concurrent_jobs=3
    )
    assert results == list("abcde")
    active = peak = 0
    for event, _, _ in calls:
        active += 1 if event == "start" else -1
        peak = max(peak, active)
    assert peak == 3
    assert active == 0
    assert len({thread for _, _, thread in calls}) == 3
    assert len({pid for _, pid, _ in calls}) == 1
    assert os.getpid() not in {pid for _, pid, _ in calls}
    assert save_threads == [main_thread()] * 5


def test_one_job_at_a_time_still_uses_a_copy_of_captured_state(images, tmp_path):
    visited = []

    def classify(path):
        assert os.getpid() != parent_pid
        visited.append(path.stem)
        return len(visited)

    parent_pid = os.getpid()
    assert classify_imgs(
        model_function(classify), images, tmp_path / "out", max_concurrent_jobs=1
    ) == [1, 2, 3, 4, 5]
    assert visited == []


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("stage", ["upload", "retry"])
def test_stop_finishes_active_jobs_including_later_requests_and_retries(
    images, tmp_path, monkeypatch, workers, stage, calls
):
    output = tmp_path / "out"
    release = tmp_path / "release"
    receive = _batch.UsageProgress.receive

    def stop(display, index, usage):
        receive(display, index, usage)
        if usage is not None:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            release.touch()

    monkeypatch.setattr(_batch.UsageProgress, "receive", stop)

    def interrupt():
        report_usage(RequestUsage(provider="test", model="fake", total_tokens=1))
        wait_for(release.exists)

    def classify(path):
        calls.append((path.stem, "upload"))
        (tmp_path / f"started-{path.stem}").touch()
        wait_for(
            lambda: all(
                (tmp_path / f"started-{name}").exists() for name in "ab"[:workers]
            )
        )
        if path.stem == "a" and stage == "upload":
            interrupt()
        if path.stem != "a":
            wait_for(release.exists)
        attempts = 0

        @retry_with_exponential_backoff(errors=(ConnectionError,), initial_delay=0.01)
        def model():
            nonlocal attempts
            attempts += 1
            calls.append((path.stem, "model"))
            if attempts == 1:
                if path.stem == "a" and stage == "retry":
                    interrupt()
                raise ConnectionError("temporary model failure")
            return path.stem

        return model()

    with pytest.raises(KeyboardInterrupt, match="Stopped after saving"):
        classify_imgs(
            model_function(classify), images, output, max_concurrent_jobs=workers
        )
    active_names = list("ab"[:workers])
    assert sorted({name for name, _ in calls}) == active_names
    for name in active_names:
        assert [step for image, step in calls if image == name] == [
            "upload",
            "model",
            "model",
        ]
    saved = json.loads((output / "classifications.json").read_text())
    assert [entry["label"] for entry in saved] == active_names
    monkeypatch.setattr(_batch.UsageProgress, "receive", receive)
    del calls[:]

    def resume(path):
        calls.append(path.stem)
        return path.stem

    assert classify_imgs(
        model_function(resume),
        images,
        output,
        max_concurrent_jobs=2,
        on_existing="resume",
    ) == list("abcde")
    assert sorted(calls) == list("abcde"[workers:])


@pytest.mark.parametrize("workers", [1, 2])
def test_unexpected_worker_exit_raises_and_saved_work_can_resume(
    images, tmp_path, calls, workers
):
    output = tmp_path / "out"
    third_started = tmp_path / "c-started"

    def crash(path):
        calls.append(path.stem)
        if path.stem == "b":
            if workers == 2:
                # C starts after A is saved and its concurrency slot is freed.
                wait_for(third_started.exists)
            os._exit(17)
        if path.stem == "c":
            third_started.touch()
            # Keep C active until B kills their shared worker process.
            wait_for((tmp_path / "release-c").exists)
        return path.stem

    with pytest.raises(TerminatedWorkerError):
        classify_imgs(
            model_function(crash), images, output, max_concurrent_jobs=workers
        )

    assert sorted(calls) == list("ab" if workers == 1 else "abc")
    saved = json.loads((output / "classifications.json").read_text())
    assert [entry["label"] for entry in saved] == ["a"]
    del calls[:]

    def resume(path):
        calls.append(path.stem)
        return path.stem

    assert classify_imgs(
        model_function(resume),
        images,
        output,
        max_concurrent_jobs=workers,
        on_existing="resume",
    ) == list("abcde")
    assert sorted(calls) == list("bcde")


@pytest.mark.parametrize("unserializable", ["function", "result"])
def test_serialization_failure_raises_without_falling_back_to_caller(
    images, tmp_path, unserializable
):
    parent_pid = os.getpid()
    lock = Lock() if unserializable == "function" else None

    def classify(path):
        assert os.getpid() != parent_pid
        return lock if lock is not None else Lock()

    with pytest.raises((TypeError, pickle.PicklingError), match="pickle"):
        classify_imgs(
            model_function(classify), images, tmp_path / "out", max_concurrent_jobs=1
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_concurrency_makes_no_calls_or_output(images, tmp_path, value):
    output = tmp_path / "out"

    def classify(path):
        raise AssertionError(
            "An invalid concurrency must be rejected before calling the function"
        )

    with pytest.raises(ValueError, match="max_concurrent_jobs"):
        classify_imgs(
            model_function(classify), images, output, max_concurrent_jobs=value
        )
    assert not output.exists()
