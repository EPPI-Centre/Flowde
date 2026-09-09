from unittest.mock import Mock, call, patch

import pytest

from flowde.errors import MaxRetriesExceededError
from flowde.exponential_backoff import retry_with_exponential_backoff


class RetryableError(Exception):
    pass


class OtherError(Exception):
    pass


def test_retry_with_exponential_backoff_raises_if_no_errors_provided():
    with pytest.raises(ValueError) as exc_info:
        retry_with_exponential_backoff(errors=())

    msg = str(exc_info.value)

    assert "No errors provided for retry" in msg
    assert "Please provide at least one error type to retry on." in msg


def test_retry_with_exponential_backoff_returns_result_without_retry():
    fn = Mock(return_value="success")

    wrapped = retry_with_exponential_backoff(
        fn,
        errors=(RetryableError,),
    )

    result = wrapped("arg", keyword="value")

    assert result == "success"
    fn.assert_called_once_with("arg", keyword="value")


def test_retry_with_exponential_backoff_retries_then_succeeds():
    fn = Mock(
        side_effect=[
            RetryableError("temporary failure"),
            RetryableError("temporary failure again"),
            "success",
        ]
    )

    wrapped = retry_with_exponential_backoff(
        fn,
        errors=(RetryableError,),
        initial_delay=2,
        exponential_base=1.5,
        max_retries=10,
    )

    with patch("flowde.exponential_backoff.time.sleep") as mock_sleep:
        result = wrapped("arg", keyword="value")

    assert result == "success"

    assert fn.call_args_list == [
        call("arg", keyword="value"),
        call("arg", keyword="value"),
        call("arg", keyword="value"),
    ]

    assert mock_sleep.call_args_list == [
        call(2),
        call(3.0),
    ]


def test_retry_with_exponential_backoff_raises_after_max_retries():
    original_error = RetryableError("still failing")
    fn = Mock(side_effect=original_error)

    wrapped = retry_with_exponential_backoff(
        fn,
        errors=(RetryableError,),
        initial_delay=2,
        exponential_base=1.5,
        max_retries=3,
    )

    with patch("flowde.exponential_backoff.time.sleep") as mock_sleep:
        with pytest.raises(MaxRetriesExceededError) as exc_info:
            wrapped()

    assert exc_info.value.__cause__ is original_error

    assert fn.call_count == 4

    assert mock_sleep.call_args_list == [
        call(2),
        call(3.0),
        call(4.5),
    ]


def test_retry_with_exponential_backoff_does_not_retry_unlisted_errors():
    fn = Mock(side_effect=OtherError("do not retry this"))

    wrapped = retry_with_exponential_backoff(
        fn,
        errors=(RetryableError,),
        initial_delay=2,
        exponential_base=1.5,
        max_retries=3,
    )

    with patch("flowde.exponential_backoff.time.sleep") as mock_sleep:
        with pytest.raises(OtherError, match="do not retry this"):
            wrapped()

    fn.assert_called_once()
    mock_sleep.assert_not_called()


def test_retry_with_exponential_backoff_can_be_used_as_decorator_with_args():
    fn = Mock(
        side_effect=[
            RetryableError("temporary failure"),
            "success",
        ]
    )

    decorated = retry_with_exponential_backoff(
        errors=(RetryableError,),
        initial_delay=1,
        exponential_base=2,
        max_retries=3,
    )(fn)

    with patch("flowde.exponential_backoff.time.sleep") as mock_sleep:
        result = decorated()

    assert result == "success"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_retry_with_exponential_backoff_can_be_used_as_plain_decorator():
    fn = Mock(
        side_effect=[
            RetryableError("temporary failure"),
            "success",
        ]
    )

    wrapped = retry_with_exponential_backoff(
        func=fn,
        errors=(RetryableError,),
        initial_delay=1,
        exponential_base=2,
        max_retries=3,
    )

    with patch("flowde.exponential_backoff.time.sleep") as mock_sleep:
        result = wrapped()

    assert result == "success"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(1)
