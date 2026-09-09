import sys
from multiprocessing.managers import SyncManager
from types import SimpleNamespace

import pytest
from joblib.externals.loky.backend.context import get_context

from flowde._progress import UsageProgress
from flowde.usage import RequestUsage, UsageTotals, report_usage, usage_scope
from flowde.utils import apply_fn_parallel_on_dict_of_lists, apply_fn_parallel_on_list


@pytest.mark.parametrize(
    "shell", [None, SimpleNamespace()], ids=["no-shell", "terminal-ipython"]
)
def test_importing_ipython_without_a_notebook_keeps_terminal_output(
    monkeypatch, capsys, shell
):
    monkeypatch.setitem(
        sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: shell)
    )
    progress = UsageProgress(1, "Classifying images", show_usage=True)
    progress.receive(
        0, RequestUsage(provider="test", model="a", cost=1, total_tokens=100)
    )
    progress.receive(0, None)
    progress.close()

    output = capsys.readouterr()
    assert "1/1" in output.err
    assert output.out.splitlines() == [
        "Est. cost: $1.000000 total | $1.000000 avg/image | $1.000000 max/image",
        "Tokens: 100 total | 100.0 avg/image | 100 max/image",
    ]


def test_average_and_maximum_group_all_calls_for_the_same_image():
    totals = UsageTotals()
    totals.add(0, RequestUsage(provider="test", model="a", cost=1, total_tokens=1000))
    totals.add(0, RequestUsage(provider="test", model="b", cost=2, total_tokens=2000))
    totals.add(1, RequestUsage(provider="test", model="a", cost=4, total_tokens=500))

    assert totals.lines() == (
        "Est. cost: $7.000000 total | $3.500000 avg/image | $4.000000 max/image",
        "Tokens: 3,500 total | 1,750.0 avg/image | 3,000 max/image",
    )


def test_unpriced_requests_keep_tokens_and_do_not_count_as_zero_cost():
    totals = UsageTotals()
    totals.add(
        0, RequestUsage(provider="test", model="priced", cost=2, total_tokens=100)
    )
    totals.add(1, RequestUsage(provider="test", model="new", total_tokens=300))

    assert totals.lines() == (
        (
            "Est. cost: $2.000000 known total | $2.000000 avg/image | "
            "$2.000000 max/image | unpriced: test/new"
        ),
        "Tokens: 400 total | 200.0 avg/image | 300 max/image",
    )


def test_absent_usage_is_distinguished_from_reported_zero_usage():
    totals = UsageTotals()
    totals.add(0, RequestUsage(provider="test", model="empty", cost=0, total_tokens=0))
    totals.add(1, RequestUsage(provider="test", model="missing"))
    totals.finished_items.update({0, 1, 2})

    assert totals.lines() == (
        (
            "Est. cost: $0.000000 known total | $0.000000 avg/image | "
            "$0.000000 max/image | unpriced: test/missing | 1 images without usage"
        ),
        (
            "Tokens: 0 known total | 0.0 avg/image | 0 max/image | "
            "usage unavailable: test/missing | 1 images without usage"
        ),
    )


def test_reporting_scope_restores_the_previous_collector_after_an_error():
    outer, inner = [], []
    usage = RequestUsage(provider="test", model="a", total_tokens=10)

    def fail():
        with usage_scope(inner.append):
            report_usage(usage)
            message = "invalid answer"
            raise ValueError(message)

    with usage_scope(outer.append):
        with pytest.raises(ValueError, match="invalid answer"):
            fail()
        report_usage(usage)
    report_usage(usage)

    assert outer == [usage]
    assert inner == [usage]


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("use_kwargs", [False, True], ids=["list", "keyword-lists"])
def test_progress_reports_usage_per_image_and_preserves_results(
    capsys, n_jobs, use_kwargs
):
    def classify(image):
        if image == "first":
            report_usage(
                RequestUsage(provider="test", model="a", cost=1, total_tokens=1000)
            )
            report_usage(
                RequestUsage(provider="test", model="a", cost=2, total_tokens=2000)
            )
        else:
            report_usage(
                RequestUsage(provider="test", model="b", cost=4, total_tokens=500)
            )
        return image

    if use_kwargs:
        result = apply_fn_parallel_on_dict_of_lists(
            classify, {"image": ["first", "second"]}, n_jobs=n_jobs
        )
    else:
        result = apply_fn_parallel_on_list(classify, ["first", "second"], n_jobs=n_jobs)

    assert result == ["first", "second"]
    assert capsys.readouterr().out.splitlines() == [
        "Est. cost: $7.000000 total | $3.500000 avg/image | $4.000000 max/image",
        "Tokens: 3,500 total | 1,750.0 avg/image | 3,000 max/image",
    ]


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_custom_functions_without_usage_need_no_extra_arguments_or_return_values(
    capsys, n_jobs
):
    def classify(image):
        return image > 1

    assert apply_fn_parallel_on_list(classify, [1, 2], n_jobs=n_jobs) == [False, True]
    output = capsys.readouterr()
    assert "Est. cost:" not in output.out + output.err
    assert "Tokens:" not in output.out + output.err


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_usage_reaches_the_display_before_the_image_function_returns(
    monkeypatch, n_jobs
):
    with SyncManager(ctx=get_context()) as manager:
        displayed = manager.Event()
        receive = UsageProgress.receive

        def observe(self, item, usage):
            receive(self, item, usage)
            if usage is not None:
                displayed.set()

        monkeypatch.setattr(UsageProgress, "receive", observe)

        def classify(image):
            report_usage(RequestUsage(provider="test", model="a", total_tokens=123))
            assert displayed.wait(timeout=10), (
                "Usage must arrive before the image finishes"
            )
            return image

        assert apply_fn_parallel_on_list(classify, ["image"], n_jobs=n_jobs) == [
            "image"
        ]


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_reported_usage_survives_an_error_and_does_not_leak_into_the_next_run(
    capsys, n_jobs
):
    def parse(image):
        report_usage(RequestUsage(provider="test", model="a", cost=2, total_tokens=300))
        message = "invalid answer"
        raise ValueError(message)

    with pytest.raises(
        ValueError if n_jobs == 1 else RuntimeError, match="invalid answer"
    ):
        apply_fn_parallel_on_list(parse, ["image"], n_jobs=n_jobs)
    assert capsys.readouterr().out.splitlines() == [
        "Est. cost: $2.000000 total | $2.000000 avg/image | $2.000000 max/image",
        "Tokens: 300 total | 300.0 avg/image | 300 max/image",
    ]

    assert apply_fn_parallel_on_list(str, [42], n_jobs=n_jobs) == ["42"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_usage_display_can_be_hidden_without_changing_results(capsys, n_jobs):
    def classify(image):
        report_usage(RequestUsage(provider="test", model="a", cost=2, total_tokens=300))
        return image

    result = apply_fn_parallel_on_list(classify, [42], n_jobs=n_jobs, show_usage=False)

    assert result == [42]
    output = capsys.readouterr()
    assert "Est. cost:" not in output.out + output.err
    assert "Tokens:" not in output.out + output.err
