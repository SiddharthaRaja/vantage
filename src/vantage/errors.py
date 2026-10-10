"""Exception types. Kept narrow on purpose: a benchmark that swallows errors
silently will happily report a very fast p95 for a server that is returning
HTTP 500 to every request."""


class VantageError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(VantageError):
    """A config file or CLI combination that cannot be honoured."""


class BackendError(VantageError):
    """A backend failed in a way that is not a per-request failure."""


class RequestFailed(VantageError):
    """One request failed. Recorded in the results, never silently dropped."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class QualityGateFailed(VantageError):
    """A configuration was faster but measurably worse, and the gate is strict."""
