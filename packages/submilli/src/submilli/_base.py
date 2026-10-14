"""What the sync and async clients share: settings, the requests, and how answers are read.

Each endpoint is a request builder and a reader, so that `Submilli` and
`AsyncSubmilli` differ only in how they send.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from submilli._errors import IDEMPOTENCY_CODES, UnknownSessionError, status_error
from submilli._types import (
    BlueprintPrompt,
    CreatedSession,
    ExecuteResult,
    LastRun,
    PackageDocs,
    RunState,
    ToolDeclaration,
    ToolResult,
    run_state,
)
from submilli._version import __version__

DEFAULT_BASE_URL = "http://127.0.0.1:8128"
# An execute may wait up to 300 seconds on the server for an earlier attempt
# with the same idempotency key; leave room for that.
DEFAULT_TIMEOUT = 330.0
DEFAULT_MAX_RETRIES = 2
# The longest `Idempotency-Key` the server accepts, in bytes.
MAX_IDEMPOTENCY_KEY_BYTES = 120


def idempotency_key(call_id: str) -> str:
    """An `Idempotency-Key` for a harness's own call id, such as a model's tool call id.

    The id as is when the server accepts it (printable ASCII, 1 to 120 bytes),
    otherwise the sha256 hex digest of its UTF-8 bytes. The same id always gives
    the same key.
    """
    if 0 < len(call_id) <= MAX_IDEMPOTENCY_KEY_BYTES and all(" " <= char <= "~" for char in call_id):
        return call_id
    return hashlib.sha256(call_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    token: str | None
    timeout: float | httpx.Timeout
    max_retries: int

    @classmethod
    def resolve(
        cls, base_url: str | None, token: str | None, timeout: float | httpx.Timeout, max_retries: int
    ) -> Settings:
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        return cls(
            base_url=(base_url or os.environ.get("SUBMILLI_SERVER") or DEFAULT_BASE_URL).rstrip("/"),
            token=token if token is not None else os.environ.get("SUBMILLI_SERVER_TOKEN"),
            timeout=timeout,
            max_retries=max_retries,
        )

    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"submilli-python/{__version__}", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    json: Any = None
    params: Any = None
    headers: Mapping[str, str] | None = None
    #: Whether sending it twice is harmless, so a failed attempt may be retried.
    retryable: bool = False


def should_retry(response: httpx.Response) -> bool:
    """Whether a response is a transient failure worth another attempt."""
    if response.status_code not in (502, 503, 504):
        return False
    try:
        code = response.json().get("error")
    except (ValueError, AttributeError):
        return True
    # An idempotency refusal is the server's answer, not a transient failure.
    return code not in IDEMPOTENCY_CODES


def retry_delay(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        after = response.headers.get("retry-after")
        if after is not None:
            try:
                return min(float(after), 30.0)
            except ValueError:
                pass
    return min(0.5 * 2.0**attempt, 8.0) * (0.75 + random.random() / 2)


def _seg(value: str) -> str:
    return quote(value, safe="")


def _json(response: httpx.Response) -> Any:
    if not response.is_success:
        raise status_error(response)
    return response.json()


# Sessions


def create_session(
    blueprint: str,
    variables: Mapping[str, str] | None,
    secrets: Mapping[str, str] | None,
    tools: Sequence[ToolDeclaration] | None,
) -> Request:
    body: dict[str, Any] = {"blueprint": blueprint, "variables": dict(variables or {})}
    if secrets:
        body["secrets"] = dict(secrets)
    if tools:
        body["tools"] = [dict(tool) for tool in tools]
    # Not retried: a lost answer would leave a session nobody holds.
    return Request("POST", "/v1/sessions", json=body)


def read_created_session(response: httpx.Response) -> CreatedSession:
    body = _json(response)
    return CreatedSession(id=str(body["session_id"]), warnings=[str(w) for w in body.get("warnings") or []])


def execute(session_id: str, code: str, key: str | None) -> Request:
    return Request(
        "POST",
        f"/v1/sessions/{_seg(session_id)}/execute",
        json={"code": code},
        headers={"Idempotency-Key": key} if key else None,
        # Without a key, a retry could run the program twice.
        retryable=key is not None,
    )


def read_run_state(response: httpx.Response) -> RunState:
    return run_state(_json(response))


def rebind(session_id: str, secrets: Mapping[str, str]) -> Request:
    return Request("POST", f"/v1/sessions/{_seg(session_id)}/rebind", json={"secrets": dict(secrets)}, retryable=True)


def read_no_content(response: httpx.Response) -> None:
    if not response.is_success:
        raise status_error(response)


def last_run(session_id: str) -> Request:
    return Request("GET", f"/v1/sessions/{_seg(session_id)}/last-run", retryable=True)


def read_last_run(response: httpx.Response) -> LastRun | None:
    # The server answers 404 with an empty body both for an unknown session and
    # for a session with no recorded run.
    if response.status_code == 404:
        return None
    return LastRun._from_json(_json(response))


def delete_session(session_id: str) -> Request:
    return Request("DELETE", f"/v1/sessions/{_seg(session_id)}", retryable=True)


def read_deleted(response: httpx.Response) -> bool:
    if response.status_code == 404:
        return False
    read_no_content(response)
    return True


def session_packages_search(session_id: str, query: str) -> Request:
    return Request("GET", f"/v1/sessions/{_seg(session_id)}/packages/search", params={"q": query}, retryable=True)


def session_packages_docs(session_id: str, name: str) -> Request:
    return Request("GET", f"/v1/sessions/{_seg(session_id)}/packages/docs", params={"name": name}, retryable=True)


# Runs


def resume(run_id: str, results: Sequence[ToolResult]) -> Request:
    if not results:
        raise ValueError("resume needs at least one result")
    # Answering a call twice changes nothing, so a resume may be retried.
    body = {"results": [result.to_json() for result in results]}
    return Request("POST", f"/v1/runs/{_seg(run_id)}/resume", json=body, retryable=True)


def retrieve_run(run_id: str) -> Request:
    return Request("GET", f"/v1/runs/{_seg(run_id)}", retryable=True)


# Blueprints


def prompt(blueprint: str) -> Request:
    return Request("GET", f"/v1/blueprints/{_seg(blueprint)}/prompt", retryable=True)


def read_prompt(response: httpx.Response) -> BlueprintPrompt:
    return BlueprintPrompt._from_json(_json(response))


def packages_search(blueprint: str, query: str) -> Request:
    return Request("GET", f"/v1/blueprints/{_seg(blueprint)}/packages/search", params={"q": query}, retryable=True)


def packages_docs(blueprint: str, name: str) -> Request:
    return Request("GET", f"/v1/blueprints/{_seg(blueprint)}/packages/docs", params={"name": name}, retryable=True)


def read_json(response: httpx.Response) -> dict[str, Any]:
    body = _json(response)
    if not isinstance(body, dict):
        raise TypeError(f"expected a JSON object from submilli-server, got {type(body).__name__}")
    return body


def read_package_docs(response: httpx.Response) -> PackageDocs:
    if response.status_code == 200:
        return PackageDocs(found=True, text=response.text)
    if response.status_code == 404:
        error = status_error(response)
        if isinstance(error, UnknownSessionError) or error.code == "not_found":
            # An unknown session, or an unknown Blueprint: the caller's mistake.
            raise error
        return PackageDocs(found=False, text=response.text)
    raise status_error(response)


def builtins(blueprint: str) -> Request:
    return Request("GET", f"/v1/blueprints/{_seg(blueprint)}/builtins", retryable=True)


def builtins_docs(blueprint: str, names: Sequence[str]) -> Request:
    return Request(
        "GET",
        f"/v1/blueprints/{_seg(blueprint)}/builtins/docs",
        params=[("name", name) for name in names],
        retryable=True,
    )


# One-off runs


def execute_once(
    blueprint: str, code: str, variables: Mapping[str, str] | None, secrets: Mapping[str, str] | None
) -> Request:
    body: dict[str, Any] = {"blueprint": blueprint, "code": code, "variables": dict(variables or {})}
    if secrets:
        body["secrets"] = dict(secrets)
    return Request("POST", "/v1/execute", json=body)


def read_execute_result(response: httpx.Response) -> ExecuteResult:
    return ExecuteResult._from_json(_json(response))
