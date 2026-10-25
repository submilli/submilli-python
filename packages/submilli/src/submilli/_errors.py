"""The errors the SDK raises.

Every error is a `SubmilliError`. A response the server refused is an
`APIStatusError` with the status, the body, and the body's stable `code`.
Refusals a caller can act on have their own subclass, which also derives from
the status class (`UnknownSessionError` is a `NotFoundError`).

A program that fails, or that the Blueprint denies, is not an error: it is an
`ExecuteResult` whose `error` is set. A run with no result to return, such as
a durable run that is `stuck`, is a `RunStateError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "ForbiddenError",
    "IdempotencyError",
    "InternalServerError",
    "NotFoundError",
    "RunNotSuspendedError",
    "RunStateError",
    "ServiceUnavailableError",
    "SessionRequiresSecretsError",
    "SubmilliError",
    "UnknownBlueprintError",
    "UnknownCallError",
    "UnknownRunError",
    "UnknownSessionError",
]


class SubmilliError(Exception):
    """Base class of every error this package raises."""


class APIConnectionError(SubmilliError):
    """The server could not be reached, or the connection failed mid-request."""

    def __init__(self, message: str, *, request: httpx.Request | None = None) -> None:
        super().__init__(message)
        self.request = request


class APITimeoutError(APIConnectionError):
    """The request timed out."""


class APIStatusError(SubmilliError):
    """The server answered with an error status.

    Attributes:
        status_code: The HTTP status.
        body: The parsed JSON body, or the text when it is not JSON.
        code: The body's `error` field, when there is one, such as `unknown_run`.
        execution_id: The run the refusal concerns, when the body names one.
    """

    def __init__(self, status_code: int, body: Any, *, message: str | None = None) -> None:
        self.status_code = status_code
        self.body = body
        self.code: str | None = _field(body, "error")
        self.execution_id: str | None = _field(body, "execution_id")
        super().__init__(message or f"submilli-server answered {status_code}: {_describe(body)}")


class BadRequestError(APIStatusError):
    """400: the request was malformed, or names variables, secrets or tools the server refuses."""


class AuthenticationError(APIStatusError):
    """401: the API token is missing or not recognised."""


class ForbiddenError(APIStatusError):
    """403: the token's role may not use this endpoint."""


class NotFoundError(APIStatusError):
    """404."""


class ConflictError(APIStatusError):
    """409."""


class ServiceUnavailableError(APIStatusError):
    """503."""


class InternalServerError(APIStatusError):
    """5xx other than 503."""


class UnknownSessionError(NotFoundError):
    """`404 unknown session`: the session ended, usually after its idle timeout."""


class UnknownBlueprintError(NotFoundError):
    """`404`: the server holds no Blueprint of that name."""


class UnknownRunError(NotFoundError):
    """`404 unknown_run`: the server no longer holds the run, as after a restart."""


class SessionRequiresSecretsError(ConflictError):
    """`409 session_requires_secrets`: the server lost the session's harness secrets; `rebind` them."""

    @property
    def required(self) -> list[str]:
        """The harness secrets the server asks for."""
        required = self.body.get("required") if isinstance(self.body, dict) else None
        return [str(name) for name in required] if isinstance(required, list) else []


class RunNotSuspendedError(ConflictError):
    """`409 run_not_suspended`: the run waits on no call; it finished, timed out, or was cancelled."""


class UnknownCallError(BadRequestError):
    """`400 unknown_call`: the run waits on another call."""


# What each status without a result means, as servers with `durable_execution`
# report them in a run's lifecycle view.
_RUN_STATES = {
    "stuck": "it stopped, and waits to be recovered with POST /v1/runs/{run}/recover",
    "cancelled": "it was cancelled before it finished",
    "aborted": "it ended before the program started",
    "indeterminate": "it was interrupted, and its outcome is unknown",
    "expired": "it finished, but the server no longer keeps its result",
    "completed": "it finished, but the server keeps no result for it",
}


class RunStateError(SubmilliError):
    """The server reports a run in a state with no result to return.

    Servers with `durable_execution` report these: `stuck` (stopped, and held
    for `POST /v1/runs/{id}/recover`), `cancelled` (its session closed),
    `aborted` (it never started), `indeterminate` (interrupted, outcome
    unknown), `expired` (its result is no longer kept), and `completed` without
    a kept result. A status this SDK does not know raises it too.

    Attributes:
        execution_id: The run.
        session_id: The session it ran in.
        status: The run's `status`, such as `stuck`.
        reason: Why, when the server says, such as `credentials_unavailable`.
        body: The run as the server sent it, with lifecycle metadata such as `revision`.
    """

    def __init__(self, body: Mapping[str, Any]) -> None:
        self.body: dict[str, Any] = dict(body)
        self.execution_id: str = str(body.get("execution_id", ""))
        self.session_id: str = str(body.get("session_id", ""))
        self.status: str = str(body.get("status", ""))
        reason = body.get("reason")
        self.reason: str | None = reason if isinstance(reason, str) else None
        meaning = _RUN_STATES.get(self.status, "this version of the SDK does not know that status")
        because = f" ({self.reason})" if self.reason else ""
        super().__init__(
            f"run {self.execution_id} is {self.status}{because} and has no result to return: "
            + meaning.format(run=self.execution_id)
        )


class IdempotencyError(APIStatusError):
    """The server refused an `Idempotency-Key`, and ran nothing.

    `code` says why: `idempotency_conflict` (the key was used with other code),
    `idempotency_incomplete` (an earlier attempt never recorded an outcome, so
    its effects are unknown), `idempotency_in_progress` (an earlier attempt is
    still running), `idempotency_unavailable` (the record could not be read or
    written), or `idempotency_key_invalid`.
    """

    @property
    def detail(self) -> str:
        return _field(self.body, "detail") or ""


IDEMPOTENCY_CODES = frozenset(
    {
        "idempotency_conflict",
        "idempotency_incomplete",
        "idempotency_in_progress",
        "idempotency_unavailable",
        "idempotency_key_invalid",
    }
)

_BY_CODE: dict[tuple[int, str], type[APIStatusError]] = {
    (404, "unknown session"): UnknownSessionError,
    (404, "unknown blueprint"): UnknownBlueprintError,
    (404, "unknown_run"): UnknownRunError,
    (409, "session_requires_secrets"): SessionRequiresSecretsError,
    (409, "run_not_suspended"): RunNotSuspendedError,
    (400, "unknown_call"): UnknownCallError,
}

_BY_STATUS: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    503: ServiceUnavailableError,
}


def status_error(response: httpx.Response) -> APIStatusError:
    """The error for a refused response."""
    body = _body(response)
    code = _field(body, "error")
    if code in IDEMPOTENCY_CODES:
        return IdempotencyError(response.status_code, body)
    if code is not None and (response.status_code, code) in _BY_CODE:
        return _BY_CODE[(response.status_code, code)](response.status_code, body)
    if response.status_code == 404 and code == "not_found" and "blueprint" in (_field(body, "message") or ""):
        return UnknownBlueprintError(response.status_code, body)
    if response.status_code == 404 and body == "unknown run":
        # `GET /v1/runs/{id}` answers a run it holds no record of in plain text.
        return UnknownRunError(response.status_code, body)
    cls = _BY_STATUS.get(response.status_code)
    if cls is None:
        cls = InternalServerError if response.status_code >= 500 else APIStatusError
    return cls(response.status_code, body)


def _body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _field(body: Any, name: str) -> str | None:
    value = body.get(name) if isinstance(body, dict) else None
    return value if isinstance(value, str) else None


def _describe(body: Any) -> str:
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or body.get("error") or body)
    return str(body) or "(empty body)"
