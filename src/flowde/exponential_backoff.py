"""Exponential backoff retry mechanism to avoid crash on api failure."""

import time
from collections.abc import Callable

from flowde.errors import MaxRetriesExceededError


def retry_with_exponential_backoff(
    func: Callable | None = None,
    errors: tuple = (),
    initial_delay: float = 2,
    exponential_base: float = 1.5,
    max_retries: int = 10,
) -> Callable:
    """
    Retry a function with exponential backoff.

    Parameters
    ----------
    func : Callable
        Function to be retried.
    errors : tuple, optional
        Errors from func that initiate a retry, by default (InternalServerError,)
    initial_delay : float, optional
        Delay after first error, by default 2
    exponential_base : float, optional
        Multiplicative factor with which delay increases after failure, by default 2
    max_retries : int, optional
       Max number of failures before throwing the error, by default 5

    Returns
    -------
    Callable
        Wrapped function with exponential backoff retry mechanism.

    """
    if len(errors) == 0:
        msg = (
            "No errors provided for retry"
            "Please provide at least one error type to retry on."
        )
        raise ValueError(msg)

    def decorator(f: Callable) -> Callable:
        def _exponential_backoff(*args, **kwargs) -> Callable:
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return f(*args, **kwargs)

                except errors as e:
                    num_retries += 1

                    if num_retries > max_retries:
                        raise MaxRetriesExceededError(max_retries, e) from e

                    time.sleep(delay)
                    delay *= exponential_base

        return _exponential_backoff

    if func is not None:
        return decorator(func)

    return decorator
