"""Inspect a real terminal's rendered screen using simulated requests; no paid APIs."""

import codecs
import errno
import os
import select
import signal
import subprocess
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from textwrap import dedent
from time import monotonic

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def terminal(tmp_path):
    """Run Python in a pseudo-terminal and yield its changing screen contents."""
    pty = import_module("pty")
    termios = import_module("termios")
    pyte = import_module("pyte")
    repo_root = Path(__file__).resolve().parents[2]

    @contextmanager
    def run(code, width=120):
        master, slave = pty.openpty()
        termios.tcsetwinsize(slave, (40, width))
        screen = pyte.Screen(width, 40)
        stream = pyte.Stream(screen)
        decoder = codecs.getincrementaldecoder("utf-8")()

        def screens():
            deadline = monotonic() + 30
            while True:
                ready, _, _ = select.select([master], [], [], 0.1)
                if not ready:
                    assert monotonic() < deadline, "\n".join(screen.display)
                    continue
                try:
                    data = os.read(master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break  # Linux signals the closed terminal this way.
                if not data:
                    break
                stream.feed(decoder.decode(data))
                # Keep internal blank rows so accidental downward movement fails.
                yield (
                    "\n".join(line.rstrip() for line in screen.display)
                    .rstrip()
                    .splitlines()
                )
            assert process.wait(timeout=10) == 0, "\n".join(screen.display)

        try:
            process = subprocess.Popen(  # noqa: S603 - Executes only test-defined code.
                [sys.executable, "-u", "-c", dedent(code)],
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=slave,
                cwd=tmp_path,
                env={
                    **os.environ,
                    "PYTHONPATH": str(repo_root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                    "TERM": "xterm-256color",
                },
                start_new_session=True,
            )
        finally:
            os.close(slave)
        try:
            yield screens()
        finally:
            # Also stop worker processes if a screen assertion fails mid-run.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            os.close(master)

    return run


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("width", [80, 120], ids=["80-columns", "120-columns"])
def test_terminal_updates_in_place_after_each_request_before_the_image_returns(
    terminal, tmp_path, n_jobs, width
):
    releases = tmp_path / "displayed"
    releases.mkdir()
    code = f"""
        from pathlib import Path
        from time import monotonic, sleep
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            for request in (1, 2):
                report_usage(RequestUsage(
                    provider="demo", model="a", cost=request,
                    total_tokens=request * 100,
                ))
                deadline = monotonic() + 10
                while not (Path({str(releases)!r}) / str(request)).exists():
                    if monotonic() > deadline:
                        raise TimeoutError("Usage was not displayed during the request")
                    sleep(0.01)
            return image

        print("Output before the run")
        result = apply_fn_parallel_on_list(
            classify, ["image"], n_jobs={n_jobs}, msg="Classifying images"
        )
        print(result)
    """

    with terminal(code, width) as screens:
        for request, (cost, tokens) in enumerate([(1, 100), (3, 300)], start=1):
            cost_line = (
                f"Est. cost: ${cost:.6f} total | ${cost:.6f} avg/image | "
                f"${cost:.6f} max/image"
            )
            token_line = (
                f"Tokens: {tokens} total | {tokens:.1f} avg/image | {tokens} max/image"
            )
            for lines in screens:
                if token_line in lines:
                    # Both reports must update the same rows before completion.
                    assert len(lines) == 4
                    assert lines[0] == "Output before the run"
                    assert "Classifying images" in lines[1]
                    assert "0/1" in lines[1]
                    assert lines[2:] == [cost_line, token_line]
                    (releases / str(request)).touch()
                    break
            else:
                pytest.fail(f"No live terminal update for request {request}")
        final = list(screens)[-1]

    assert len(final) == 5
    assert final[0] == "Output before the run"
    assert "1/1" in final[1]
    assert final[2:] == [cost_line, token_line, "['image']"]


def test_terminal_updates_usage_after_each_completed_image(terminal, tmp_path):
    code = f"""
        from pathlib import Path
        from time import monotonic, sleep
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            # The test allows each image to run after checking the previous display.
            deadline = monotonic() + 10
            while not (Path({str(tmp_path)!r}) / f"allow-{{image}}").exists():
                if monotonic() > deadline:
                    raise TimeoutError("The test did not allow the next image")
                sleep(0.01)
            report_usage(RequestUsage(
                provider="demo", model="a", cost=image, total_tokens=image * 100,
            ))
            # Allow normal tqdm redraw timing before marking the image complete.
            sleep(0.2)
            return image

        result = apply_fn_parallel_on_list(
            classify, [1, 2], n_jobs=1, msg="Classifying images"
        )
        assert result == [1, 2]
    """

    with terminal(code) as screens:
        for lines in screens:
            if not lines:
                continue

            # Initially there are no usage figures. Allow the first image to run.
            if "0/2" in lines[0] and not (tmp_path / "allow-1").exists():
                assert len(lines) == 1
                (tmp_path / "allow-1").touch()

            # Wait for the full token line: terminal output can arrive in chunks.
            elif (
                "1/2" in lines[0]
                and len(lines) == 3
                and lines[-1].endswith("max/image")
                and not (tmp_path / "allow-2").exists()
            ):
                assert lines[1:] == [
                    (
                        "Est. cost: $1.000000 total | $1.000000 avg/image | "
                        "$1.000000 max/image"
                    ),
                    "Tokens: 100 total | 100.0 avg/image | 100 max/image",
                ]
                (tmp_path / "allow-2").touch()

    assert (tmp_path / "allow-2").exists(), "No display for the first completed image"
    assert "2/2" in lines[0]
    assert lines[1:] == [
        "Est. cost: $3.000000 total | $1.500000 avg/image | $2.000000 max/image",
        "Tokens: 300 total | 150.0 avg/image | 200 max/image",
    ]


@pytest.mark.parametrize("show_usage", [False, True], ids=["hidden", "no-reports"])
def test_terminal_keeps_one_bar_when_usage_is_hidden_or_unavailable(
    terminal, show_usage
):
    code = f"""
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            if not {show_usage!r}:
                report_usage(RequestUsage(
                    provider="demo", model="a", cost=1, total_tokens=100,
                ))
            return image

        print("Output before the run")
        result = apply_fn_parallel_on_list(
            classify, [1, 2], n_jobs=2, msg="Classifying images",
            show_usage={show_usage!r},
        )
        print(result)
    """
    with terminal(code) as screens:
        for lines in screens:
            assert "Est. cost:" not in "\n".join(lines)
            assert "Tokens:" not in "\n".join(lines)
            assert sum("Classifying images" in line for line in lines) <= 1

    assert len(lines) == 3
    assert lines[0] == "Output before the run"
    assert "2/2" in lines[1]
    assert lines[2] == "[1, 2]"


def test_terminal_keeps_usage_after_failure_and_starts_a_fresh_bar_next_run(terminal):
    code = """
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            report_usage(RequestUsage(
                provider="demo", model="a", cost=2, total_tokens=300,
            ))
            raise ValueError("invalid answer")

        print("First run")
        try:
            apply_fn_parallel_on_list(classify, [1], n_jobs=2, msg="Classifying images")
        except RuntimeError:
            pass
        else:
            raise AssertionError("The existing error must still propagate")
        print("Second run")
        result = apply_fn_parallel_on_list(
            str, [42], n_jobs=2, msg="Classifying images"
        )
        print(result)
    """
    with terminal(code) as screens:
        final = list(screens)[-1]

    assert len(final) == 7
    assert final[0] == "First run"
    assert "0/1" in final[1]
    assert final[2:5] == [
        "Est. cost: $2.000000 total | $2.000000 avg/image | $2.000000 max/image",
        "Tokens: 300 total | 300.0 avg/image | 300 max/image",
        "Second run",
    ]
    assert "1/1" in final[5]
    assert final[6] == "['42']"
