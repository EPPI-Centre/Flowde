"""Custom errors for the flowde package."""


class MaxRetriesExceededError(RuntimeError):
    """Error raised when the maximum number of retries is exceeded."""

    def __init__(self, retries: int, last_exception: BaseException) -> None:
        """
        Initialize the error with the retry limit and last encountered exception.

        Parameters
        ----------
        retries : int
            The maximum number of retries exceeded.
        last_exception : BaseException
            The last exception encountered before exceeding retries.

        """
        super().__init__(f"Maximum number of retries ({retries}) exceeded.")
        self.retries = retries
        self.last_exception = last_exception
