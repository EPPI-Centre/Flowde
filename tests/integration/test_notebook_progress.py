"""Exercise real notebook output messages with simulated requests; no paid APIs."""

import os
import sys
from html import escape
from importlib import import_module
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def notebook_kernel(tmp_path_factory):
    import_module("ipykernel")
    jupyter_client = import_module("jupyter_client")
    directory = tmp_path_factory.mktemp("notebook-kernel")
    repo_root = Path(__file__).resolve().parents[2]
    manager = jupyter_client.KernelManager(
        connection_file=str(directory / "kernel.json")
    )
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    manager.start_kernel(
        cwd=directory, env={**os.environ, "PYTHONPATH": str(repo_root / "src")}
    )
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        yield client
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


def execute_cell(kernel, code):
    """Yield live output, including updates sent before the cell has finished."""
    request = kernel.execute(dedent(code))
    while True:
        message = kernel.get_iopub_msg(timeout=30)
        if message["parent_header"].get("msg_id") != request:
            continue
        if (
            message["msg_type"] == "status"
            and message["content"]["execution_state"] == "idle"
        ):
            return
        yield message


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("use_kwargs", [False, True], ids=["list", "keyword-lists"])
def test_notebook_updates_one_output_before_the_image_function_returns(
    notebook_kernel, tmp_path, n_jobs, use_kwargs
):
    release = tmp_path / "usage-was-displayed"
    messages = []
    code = f"""
        from pathlib import Path
        from time import monotonic, sleep
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import (
            apply_fn_parallel_on_list, apply_fn_parallel_on_dict_of_lists,
        )

        def classify(image):
            report_usage(RequestUsage(
                provider="demo", model="a", cost=1, total_tokens=100,
            ))
            deadline = monotonic() + 10
            while not Path({str(release)!r}).exists():
                if monotonic() > deadline:
                    raise TimeoutError("Usage was not displayed during the request")
                sleep(0.01)
            report_usage(RequestUsage(
                provider="demo", model="a", cost=2, total_tokens=200,
            ))
            return image

        print("Output before the run")
        if {use_kwargs!r}:
            result = apply_fn_parallel_on_dict_of_lists(
                classify, {{"image": ["first", "second"]}},
                n_jobs={n_jobs}, msg="Classifying images",
            )
        else:
            result = apply_fn_parallel_on_list(
                classify, ["first", "second"], n_jobs={n_jobs}, msg="Classifying images"
            )
        print(result)
    """

    for message in execute_cell(notebook_kernel, code):
        messages.append(message)
        if message["msg_type"] == "update_display_data":
            text = message["content"]["data"]["text/plain"]
            if "Tokens:" in text and not release.exists():
                # Neither image can finish until the client receives usage.
                assert "0/2" in text
                release.touch()

    assert not [m for m in messages if m["msg_type"] == "error"]
    displays = [m["content"] for m in messages if m["msg_type"] == "display_data"]
    updates = [m["content"] for m in messages if m["msg_type"] == "update_display_data"]
    assert len(displays) == 1
    assert release.exists(), "Usage should arrive before the callable returns"
    assert {u["transient"]["display_id"] for u in updates} == {
        displays[0]["transient"]["display_id"]
    }
    final = updates[-1]["data"]["text/plain"]
    assert "2/2" in final
    assert final.splitlines()[1:] == [
        "Est. cost: $6.000000 total | $3.000000 avg/image | $3.000000 max/image",
        "Tokens: 600 total | 300.0 avg/image | 300 max/image",
    ]
    for line in final.splitlines():
        assert escape(line) in updates[-1]["data"]["text/html"]
    # Updating the bar must not clear unrelated output or print extra bar lines.
    assert not [m for m in messages if m["msg_type"] == "clear_output"]
    assert "".join(
        m["content"]["text"] for m in messages if m["msg_type"] == "stream"
    ) == ("Output before the run\n['first', 'second']\n")


def test_notebook_updates_usage_after_each_completed_image(notebook_kernel, tmp_path):
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
            return image

        result = apply_fn_parallel_on_list(
            classify, [1, 2], n_jobs=1, msg="Classifying images"
        )
        assert result == [1, 2]
    """

    for message in execute_cell(notebook_kernel, code):
        assert message["msg_type"] != "error", message["content"]
        if message["msg_type"] not in {"display_data", "update_display_data"}:
            continue
        lines = message["content"]["data"]["text/plain"].splitlines()

        # Initially there are no usage figures. Allow the first image to run.
        if message["msg_type"] == "display_data":
            assert "0/2" in lines[0]
            assert len(lines) == 1
            (tmp_path / "allow-1").touch()

        # Check the completed first image before allowing the second to run.
        elif "1/2" in lines[0] and not (tmp_path / "allow-2").exists():
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


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("show_usage", [False, True], ids=["hidden", "no-reports"])
def test_notebook_keeps_a_single_bar_when_usage_is_hidden_or_unavailable(
    notebook_kernel, n_jobs, show_usage
):
    messages = list(
        execute_cell(
            notebook_kernel,
            f"""
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            if not {show_usage!r}:
                report_usage(RequestUsage(
                    provider="demo", model="a", cost=1, total_tokens=100,
                ))
            return image

        result = apply_fn_parallel_on_list(
            classify, [1, 2], n_jobs={n_jobs}, show_usage={show_usage!r},
        )
        assert result == [1, 2]
    """,
        )
    )

    assert not [
        m for m in messages if m["msg_type"] in {"error", "stream", "clear_output"}
    ]
    displays = [m for m in messages if m["msg_type"] == "display_data"]
    updates = [m for m in messages if m["msg_type"] == "update_display_data"]
    assert len(displays) == 1
    assert "2/2" in updates[-1]["content"]["data"]["text/plain"]
    for message in [*displays, *updates]:
        assert len(message["content"]["data"]["text/plain"].splitlines()) == 1


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_notebook_keeps_usage_after_failure_and_starts_a_new_display_next_run(
    notebook_kernel, n_jobs
):
    messages = list(
        execute_cell(
            notebook_kernel,
            f"""
        from flowde.usage import RequestUsage, report_usage
        from flowde.utils import apply_fn_parallel_on_list

        def classify(image):
            report_usage(RequestUsage(
                provider="demo", model="a", cost=2, total_tokens=300,
            ))
            raise ValueError("invalid answer")

        try:
            apply_fn_parallel_on_list(classify, [1], n_jobs={n_jobs})
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("The existing error must still propagate")
        assert apply_fn_parallel_on_list(str, [42], n_jobs={n_jobs}) == ["42"]
    """,
        )
    )

    assert not [
        m for m in messages if m["msg_type"] in {"error", "stream", "clear_output"}
    ]
    displays = [m["content"] for m in messages if m["msg_type"] == "display_data"]
    updates = [m["content"] for m in messages if m["msg_type"] == "update_display_data"]
    assert len(displays) == 2
    first_id, second_id = (d["transient"]["display_id"] for d in displays)
    assert first_id != second_id
    failed = [u for u in updates if u["transient"]["display_id"] == first_id][-1]
    text = failed["data"]["text/plain"]
    assert "0/1" in text
    assert "Est. cost: $2.000000 total" in text
    assert "Tokens: 300 total" in text
    subsequent = [u for u in updates if u["transient"]["display_id"] == second_id]
    assert "1/1" in subsequent[-1]["data"]["text/plain"]
    assert all(len(u["data"]["text/plain"].splitlines()) == 1 for u in subsequent)


def test_notebook_html_escapes_angle_brackets_in_description_and_unpriced_model_name(
    notebook_kernel,
):
    messages = list(
        execute_cell(
            notebook_kernel,
            """
        from flowde._progress import UsageProgress
        from flowde.usage import RequestUsage

        progress = UsageProgress(1, "Classifying <images>", show_usage=True)
        progress.receive(0, RequestUsage(
            provider="demo", model="<my-deployment>", total_tokens=100,
        ))
        progress.receive(0, None)
        progress.close()
    """,
        )
    )

    assert not [m for m in messages if m["msg_type"] == "error"]
    updates = [m for m in messages if m["msg_type"] == "update_display_data"]
    html = updates[-1]["content"]["data"]["text/html"]
    assert "Classifying &lt;images&gt;" in html
    assert "unpriced: demo/&lt;my-deployment&gt;" in html
    assert "Tokens: 100 total" in html
    assert "<my-deployment>" not in html
