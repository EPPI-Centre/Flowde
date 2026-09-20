"""Interrupt real scripts and notebook kernels running dummy image jobs."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def wait_for(condition):
    deadline = time.monotonic() + 30
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(
                "The caller or image worker did not reach the expected stage"
            )
        time.sleep(0.02)


def scenario(directory, workers, force):
    return dedent(f"""
        import json
        import os
        from pathlib import Path
        from time import monotonic, sleep
        from pydantic import BaseModel
        from flowde import model_function
        from flowde.parse_imgs import parse_imgs
        from flowde.usage import RequestUsage, report_usage

        root = Path({str(directory)!r})
        images = root / "images"
        images.mkdir()
        for name in "abcd":
            (images / f"{{name}}.png").write_bytes(b"dummy image")
        output = root / "output"
        caller_pid = os.getpid()
        caller_list = []

        class Answer(BaseModel):
            name: str
            pid: int

        def parse(img_path):
            caller_list.append(img_path.stem)
            (root / f"started-{{img_path.stem}}").write_text(str(os.getpid()))
            report_usage(RequestUsage(provider="dummy", model="test", total_tokens=10))
            if img_path.stem != "a":
                deadline = monotonic() + 30
                while not (root / "release").exists():
                    if monotonic() > deadline:
                        raise TimeoutError("The test did not release the active jobs")
                    sleep(0.01)
            return Answer(name=img_path.stem, pid=os.getpid())

        try:
            parse_imgs(model_function(parse, result_structure=Answer), images, output,
                       max_concurrent_jobs={workers})
            raise AssertionError("The run did not stop")
        except KeyboardInterrupt as error:
            assert ("Forced stop" if {force!r} else "Stopped after saving") in str(error)

        assert os.getpid() == caller_pid
        assert caller_list == []
        saved = sorted(path.stem for path in output.glob("*.json"))
        assert saved == (list("a") if {force!r} else list("abc"[:{workers} + 1]))

        state = json.loads((output / ".flowde" / "run.state").read_text())
        saved_usage = [usage for item in state["items"].values() for usage in item["usage"]]
        tokens = sum(item["total_tokens"] for item in saved_usage)
        if {force!r}:
            assert 10 <= tokens <= 10 * ({workers} + 1)
        else:
            assert tokens == 10 * ({workers} + 1)

        def resume(img_path):
            (root / f"resumed-{{img_path.stem}}").touch()
            return Answer(name=img_path.stem, pid=os.getpid())

        results = parse_imgs(model_function(resume, result_structure=Answer), images, output,
                             max_concurrent_jobs={workers}, on_existing="resume")
        assert [result.name for result in results] == list("abcd")
        assert all(isinstance(result, Answer) for result in results)
        assert all(result.pid != caller_pid for result in results)
        assert sorted(path.name for path in root.glob("resumed-*")) == [
            f"resumed-{{name}}" for name in "abcd" if name not in saved
        ]
        state = json.loads((output / ".flowde" / "run.state").read_text())
        usage = [usage for item in state["items"].values() for usage in item["usage"]]
        assert usage == saved_usage
        (root / "passed").touch()
    """)


def active_jobs_started(directory, workers):
    return (directory / "output" / "a.json").exists() and all(
        (directory / f"started-{name}").exists() for name in "bc"[:workers]
    )


def run_script_interrupt_and_resume(
    tmp_path, workers, force, send_interrupt, *, creationflags=0
):
    script = tmp_path / "run.py"
    script.write_text(scenario(tmp_path, workers, force))
    log = tmp_path / "script.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            wait_for(lambda: active_jobs_started(tmp_path, workers))
            send_interrupt(process)
            wait_for(lambda: "Stopping after active" in log.read_text())
            if force:
                send_interrupt(process)
            else:
                (tmp_path / "release").touch()
            assert process.wait(timeout=30) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert (tmp_path / "passed").exists(), log.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="Uses POSIX SIGINT delivery")
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("force", [False, True], ids=["graceful", "forced"])
def test_script_interrupt_and_resume(tmp_path, workers, force):
    run_script_interrupt_and_resume(
        tmp_path, workers, force, lambda process: process.send_signal(signal.SIGINT)
    )


def send_windows_ctrl_c(process):
    # CTRL_C_EVENT broadcasts to a console, so attach only this helper to the
    # script's private console. Never detach pytest from its own console.
    sender = dedent("""
        import ctypes
        import signal
        import sys

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.FreeConsole()  # The helper may already have no console.
        if not kernel32.AttachConsole(int(sys.argv[1])):
            raise ctypes.WinError(ctypes.get_last_error())
        # Attaching resets console handlers; ignore Ctrl+C in this helper only.
        if not kernel32.SetConsoleCtrlHandler(None, True):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.GenerateConsoleCtrlEvent(signal.CTRL_C_EVENT, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    """)
    subprocess.run(
        [sys.executable, "-c", sender, str(process.pid)], check=True, timeout=10
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Uses Windows console Ctrl+C")
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("force", [False, True], ids=["graceful", "forced"])
def test_windows_script_interrupt_and_resume(tmp_path, workers, force):
    run_script_interrupt_and_resume(
        tmp_path,
        workers,
        force,
        send_windows_ctrl_c,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


@pytest.fixture
def notebook(tmp_path):
    from jupyter_client import KernelManager

    manager = KernelManager(connection_file=str(tmp_path / "kernel.json"))
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    manager.start_kernel(
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    )
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        yield manager, client
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("force", [False, True], ids=["graceful", "forced"])
def test_notebook_interrupt_and_resume(notebook, tmp_path, workers, force):
    manager, client = notebook
    request = client.execute(scenario(tmp_path, workers, force))
    messages = []
    wait_for(lambda: active_jobs_started(tmp_path, workers))
    manager.interrupt_kernel()
    stopping = False
    while True:
        message = client.get_iopub_msg(timeout=30)
        if message["parent_header"].get("msg_id") != request:
            continue
        messages.append(message)
        if not stopping and "Stopping after active" in message["content"].get(
            "text", ""
        ):
            stopping = True
            if force:
                manager.interrupt_kernel()
            else:
                (tmp_path / "release").touch()
        if (
            message["msg_type"] == "status"
            and message["content"]["execution_state"] == "idle"
        ):
            break
    assert stopping
    assert not [m["content"] for m in messages if m["msg_type"] == "error"]
    assert (tmp_path / "passed").exists()
    # A new cell must still see the pre-interrupt kernel and its objects.
    request = client.execute(
        "assert os.getpid() == caller_pid; assert caller_list == []"
    )
    while True:
        reply = client.get_shell_msg(timeout=30)
        if reply["parent_header"].get("msg_id") == request:
            assert reply["content"]["status"] == "ok", reply["content"]
            break
