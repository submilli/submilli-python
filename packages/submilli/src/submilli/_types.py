"""What the server sends back, and what a harness sends it, as typed values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from submilli._errors import RunStateError

__all__ = [
    "BlueprintPrompt",
    "CreatedSession",
    "ExecuteResult",
    "LastRun",
    "PackageDocs",
    "PendingCall",
    "RunError",
    "RunState",
    "RunningRun",
    "SuspendedRun",
    "ToolDeclaration",
    "ToolResult",
]


class ToolDeclaration(TypedDict):
    """One of the application's tools, as a session declares it for programs to call as `@tools`.

    The shape of an MCP `tools/list` entry. `name` must be a TypeScript function
    name. Without `output_schema`, the tool returns `unknown` in programs. The
    server does not resolve `$ref`; see `submilli.schema.inline_refs`.
    """

    name: str
    input_schema: dict[str, Any]
    description: NotRequired[str]
    output_schema: NotRequired[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CreatedSession:
    """A session `sessions.create` opened."""

    id: str
    #: What programs will not see of the declared tools, and why.
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RunError:
    """Why a run failed: a compile error, a runtime error, a limit, or a denial."""

    #: `compile_error`, `runtime_error`, `permission_denied`, `timeout`, `fuel_exhausted`,
    #: `memory_exhausted`, `blueprint_not_found`, `invalid_request`, and others.
    kind: str
    message: str
    #: On `permission_denied`: the Package refused, the capability, and who refused
    #: (`policy`, `invariant`, `read_only`, or `harness`).
    caller: str | None = None
    capability: str | None = None
    source: str | None = None
    #: On compile errors: `severity`, `line`, `column`, `message`, and `notes`.
    diagnostics: list[dict[str, Any]] | None = None
    #: The error object as the server sent it.
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_json(cls, body: dict[str, Any]) -> RunError:
        return cls(
            kind=str(body.get("kind", "")),
            message=str(body.get("message", "")),
            caller=body.get("caller"),
            capability=body.get("capability"),
            source=body.get("source"),
            diagnostics=body.get("diagnostics"),
            raw=dict(body),
        )


@dataclass(frozen=True, slots=True)
class ExecuteResult:
    """A finished run (`status: finished`).

    On a server with `durable_execution`, `GET /v1/runs/{id}` reports a finished
    run as `completed`, with this result in its `response`.
    """

    execution_id: str
    session_id: str
    #: What `main` returned: a string as is, other values as JSON text, `None` on
    #: an undefined return or a failure.
    result: str | None
    console: list[str]
    error: RunError | None
    #: Tools of an imported `@mcp/<server>` or `@tools` Package that were dropped
    #: or typed `unknown`.
    discovery_warnings: list[str] = field(default_factory=list)
    status: Literal["finished"] = "finished"

    @property
    def denied(self) -> bool:
        """Whether a denial ended the run."""
        return self.error is not None and self.error.kind == "permission_denied"

    @classmethod
    def _from_json(cls, body: dict[str, Any]) -> ExecuteResult:
        error = body.get("error")
        return cls(
            execution_id=str(body.get("execution_id", "")),
            session_id=str(body.get("session_id", "")),
            result=body.get("result"),
            console=[str(line) for line in body.get("console") or []],
            error=RunError._from_json(error) if isinstance(error, dict) else None,
            discovery_warnings=[str(w) for w in body.get("discovery_warnings") or []],
        )


@dataclass(frozen=True, slots=True)
class PendingCall:
    """A call to one of the application's tools that a suspended run waits on."""

    #: The run's id and the call's sequence number, stable across retries.
    call_id: str
    tool: str
    args: dict[str, Any]
    #: The run that made the call.
    execution_id: str


@dataclass(frozen=True, slots=True)
class SuspendedRun:
    """A run stopped at calls to the application's tools (`status: suspended`)."""

    execution_id: str
    session_id: str
    calls: list[PendingCall]
    status: Literal["suspended"] = "suspended"


@dataclass(frozen=True, slots=True)
class RunningRun:
    """A run at work (`status: running`), or waiting for its turn in the session (`status: queued`).

    `runs.retrieve` answers it, and on a server with `durable_execution` so does
    an execute still queued or working after the server's wait. `runs.complete`
    polls it.
    """

    execution_id: str
    session_id: str
    status: Literal["queued", "running"] = "running"


RunState: TypeAlias = ExecuteResult | SuspendedRun | RunningRun


def run_state(body: dict[str, Any]) -> RunState:
    """A run's state from an execute, resume or run body. A body without `status` is finished.

    Raises `RunStateError` for a run with no result to return, such as a durable
    run that is `stuck`, and for a status this SDK does not know.
    """
    status = body.get("status", "finished")
    execution_id = str(body.get("execution_id", ""))
    session_id = str(body.get("session_id", ""))
    if status == "suspended":
        calls = [
            PendingCall(
                call_id=str(call["call_id"]),
                tool=str(call["tool"]),
                args=dict(call.get("args") or {}),
                execution_id=execution_id,
            )
            for call in body.get("calls") or []
        ]
        return SuspendedRun(execution_id=execution_id, session_id=session_id, calls=calls)
    if status == "running":
        return RunningRun(execution_id=execution_id, session_id=session_id)
    if status == "queued":
        return RunningRun(execution_id=execution_id, session_id=session_id, status="queued")
    if status == "finished":
        return ExecuteResult._from_json(body)
    response = body.get("response")
    if status == "completed" and isinstance(response, dict) and response.get("status", "finished") == "finished":
        return ExecuteResult._from_json({"execution_id": execution_id, "session_id": session_id, **response})
    raise RunStateError(body)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The harness's answer to one `PendingCall`. Build one with `success`, `failed` or `denied`."""

    call_id: str
    ok: bool
    value: Any = None
    error_kind: Literal["failed", "denied"] | None = None
    message: str | None = None

    @classmethod
    def success(cls, call_id: str, value: Any) -> ToolResult:
        """The tool returned `value`, which must be JSON. The program gets it, checked against the output schema."""
        return cls(call_id=call_id, ok=True, value=value)

    @classmethod
    def failed(cls, call_id: str, message: str) -> ToolResult:
        """The tool ran and failed: the program's call throws `Error(message)`."""
        return cls(call_id=call_id, ok=False, error_kind="failed", message=message)

    @classmethod
    def denied(cls, call_id: str, message: str) -> ToolResult:
        """The application, or a person, refused the call: it throws `PermissionDeniedError`."""
        return cls(call_id=call_id, ok=False, error_kind="denied", message=message)

    def to_json(self) -> dict[str, Any]:
        """The answer as `POST /v1/runs/{id}/resume` takes it in `results`."""
        if self.ok:
            return {"call_id": self.call_id, "ok": True, "value": self.value}
        return {
            "call_id": self.call_id,
            "ok": False,
            "error": {"kind": self.error_kind or "failed", "message": self.message or ""},
        }


@dataclass(frozen=True, slots=True)
class LastRun:
    """`GET /v1/sessions/{id}/last-run`: the session's most recent run, with its full console."""

    result: str | None
    console: list[str]
    error: RunError | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_json(cls, body: dict[str, Any]) -> LastRun:
        error = body.get("error")
        return cls(
            result=body.get("result"),
            console=[str(line) for line in body.get("console") or []],
            error=RunError._from_json(error) if isinstance(error, dict) else None,
            raw=dict(body),
        )


@dataclass(frozen=True, slots=True)
class PackageDocs:
    """A Package's documentation, or the server's answer for a name it doesn't know.

    `text` is Markdown for an installed or virtual Package, and JSON text
    otherwise. For an unknown name, `found` is false and `text` carries the
    server's suggestion, which is worth showing to a model as is.
    """

    found: bool
    text: str


@dataclass(frozen=True, slots=True)
class BlueprintPrompt:
    """`GET /v1/blueprints/{name}/prompt`: what the server writes for a model about this Blueprint."""

    name: str
    #: The execute tool's description: the language, the Packages, and the rules.
    prompt: str
    #: The other tools' descriptions: `search`, `docs`, `builtins`, `builtins_list`,
    #: `builtins_docs`, `last_run`.
    tools: dict[str, str]

    @classmethod
    def _from_json(cls, body: dict[str, Any]) -> BlueprintPrompt:
        tools = body.get("tools") or {}
        return cls(
            name=str(body.get("name", "")),
            prompt=str(body["prompt"]),
            tools={str(key): str(value) for key, value in tools.items()},
        )
