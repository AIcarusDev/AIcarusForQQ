"""Stable error contracts exposed by the QQ adapter layer."""


class QQFileStreamError(RuntimeError):
    """A file-stream failure with an explicitly safe public message."""

    def __init__(self, failure_code: str, public_message: str, *, retryable: bool = True) -> None:
        super().__init__(public_message)
        self.failure_code = failure_code
        self.public_message = public_message
        self.retryable = retryable
