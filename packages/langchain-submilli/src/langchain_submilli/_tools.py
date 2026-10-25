"""The six Submilli tools: names, argument schemas, and what the model is shown.

Nothing here knows how a session is found. The caller supplies one async
function per tool, which keeps these definitions reusable by the middleware and
by a plain toolkit later.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from submilli import BlueprintPrompt, ExecuteResult, IdempotencyError, RunStateError

# The server's own names, as its MCP endpoint lists them.
EXECUTE = "submilli__typescript__execute"
LAST_RUN = "submilli__typescript__last_run"
PACKAGES_SEARCH = "submilli__typescript__packages__search"
PACKAGES_DOCS = "submilli__typescript__packages__docs"
BUILTINS_LIST = "submilli__typescript__builtins__list"
BUILTINS_DOCS = "submilli__typescript__builtins__docs"

TOOL_NAMES = (EXECUTE, LAST_RUN, PACKAGES_SEARCH, PACKAGES_DOCS, BUILTINS_LIST, BUILTINS_DOCS)


class ExecuteArgs(BaseModel):
    code: str = Field(
        description="TypeScript-subset source with a `main()` entry point. Its return value is the tool result."
    )


class NoArgs(BaseModel):
    # langchain-core passes a tool whose schema has no fields no arguments at
    # all, the injected runtime included. This field, which the model never
    # sees, keeps it.
    runtime: Annotated[Any, InjectedToolArg] = None


class PackagesSearchArgs(BaseModel):
    query: str = Field(
        default="",
        description="Substring matched against module names, descriptions, and exported symbols. "
        "Empty lists every available package.",
    )


class PackagesDocsArgs(BaseModel):
    name: str = Field(description="Package name, e.g. `submilli:crypto`.")


class BuiltinsDocsArgs(BaseModel):
    names: list[str] = Field(
        description='Built-in names to look up, e.g. `["Array", "Map", "Temporal"]`. Each returns its '
        "declarations; unknown names come back with an `error` entry."
    )


@dataclass(frozen=True)
class _Spec:
    name: str
    args: type[BaseModel]
    # Used until the Blueprint's own descriptions have been fetched.
    fallback: str


SPECS = (
    _Spec(EXECUTE, ExecuteArgs, "Run a TypeScript program on submilli-server under the Blueprint's policy."),
    _Spec(LAST_RUN, NoArgs, "Return the most recent execution's result and full console for this session."),
    _Spec(PACKAGES_SEARCH, PackagesSearchArgs, "Search the Packages a program may import."),
    _Spec(PACKAGES_DOCS, PackagesDocsArgs, "Return a Package's documentation and TypeScript declarations."),
    _Spec(BUILTINS_LIST, NoArgs, "List the built-ins in scope without an import."),
    _Spec(BUILTINS_DOCS, BuiltinsDocsArgs, "Return the declarations of built-ins."),
)


def descriptions(prompt: BlueprintPrompt) -> dict[str, str]:
    """The description of each tool, from the Blueprint's prompt endpoint."""
    return {
        EXECUTE: prompt.prompt,
        LAST_RUN: prompt.tools["last_run"],
        PACKAGES_SEARCH: prompt.tools["search"],
        PACKAGES_DOCS: prompt.tools["docs"],
        BUILTINS_LIST: prompt.tools["builtins_list"],
        BUILTINS_DOCS: prompt.tools["builtins_docs"],
    }


# Each handler takes the tool's arguments as keywords plus `runtime`, and returns
# what the tool returns: a string, a `ToolMessage`, or a LangGraph `Command`.
Handler = Callable[..., Awaitable[Any]]


def build_tools(handlers: dict[str, Handler]) -> list[BaseTool]:
    """One `StructuredTool` per spec, each calling its handler. Async only."""
    return [_build(spec, handlers[spec.name]) for spec in SPECS]


def _build(spec: _Spec, handler: Handler) -> BaseTool:
    async def call(runtime: ToolRuntime, **kwargs: Any) -> Any:
        return await handler(runtime=runtime, **kwargs)

    def unsupported(runtime: ToolRuntime, **_: Any) -> Any:
        raise NotImplementedError(f"{spec.name} is async only: run the agent with `ainvoke` or `astream`.")

    return StructuredTool.from_function(
        name=spec.name,
        description=spec.fallback,
        func=unsupported,
        coroutine=call,
        args_schema=spec.args,
        infer_schema=False,
        metadata={"ls_code_input_language": "typescript"} if spec.name == EXECUTE else None,
    )


# What the model is shown


DENIAL_NOTE = (
    "The operator's policy denied this. The denial is final: tell the user what was refused, "
    "and do not try another route to the same effect."
)

SESSION_REPLACED_NOTE = (
    "This thread's previous Submilli session had ended (idle timeout, or the server no longer "
    "knows it), so a new session was opened for this call. Files the earlier session kept outside "
    "persistent volumes, and its `submilli:session` state, are gone."
)

IDEMPOTENCY_MESSAGES = {
    "idempotency_conflict": (
        "Nothing ran. This tool call id was already used in this session with different code. "
        "Make a new tool call to run this program."
    ),
    "idempotency_incomplete": (
        "Nothing ran now. An earlier attempt of this same tool call started, but the server stopped "
        "before it recorded the outcome, so its side effects are unknown: it may have done some, all, "
        "or none of its work. Check what it would have changed (read the files, look up the records) "
        "before running anything that repeats it, and tell the user if you cannot tell."
    ),
    "idempotency_in_progress": (
        "Nothing new ran. An earlier attempt of this same tool call is still running on the server and "
        "did not finish within the server's wait. Check its effects, or the last run, before trying again."
    ),
    "idempotency_unavailable": (
        "Nothing ran. The server could not read or write its idempotency record, so it refused to run "
        "the program rather than risk running it twice. Trying again later may work."
    ),
    "idempotency_key_invalid": "Nothing ran. The server refused this call's idempotency key.",
}


def execute_message(
    result: ExecuteResult,
    *,
    tool_call_id: str,
    notice: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ToolMessage:
    """The execute result as a `ToolMessage`, shaped as the Vercel example shapes it.

    The model gets `result`, `console`, `error` and `denied`. The execution and
    session ids go in the artifact, for applications to link to the audit trail,
    with the program's calls to the application's tools when it made any.
    """
    content: dict[str, Any] = {
        "result": result.result,
        "console": result.console,
        "error": result.error.raw if result.error else None,
        "denied": result.denied,
    }
    if result.denied:
        content["note"] = DENIAL_NOTE
    if result.discovery_warnings:
        content["discovery_warnings"] = result.discovery_warnings
    if notice:
        content["notice"] = notice
    artifact: dict[str, Any] = {"execution_id": result.execution_id, "session_id": result.session_id}
    if tool_calls:
        artifact["tool_calls"] = tool_calls
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=tool_call_id,
        name=EXECUTE,
        artifact=artifact,
        status="error" if result.error else "success",
    )


LOST_RUN_MESSAGE = (
    "The run was lost while it waited on one of the application's tools: the server restarted or no "
    "longer holds it. Its effects up to that point are unknown. Check what it would have changed before "
    "running anything that repeats it, and tell the user if you cannot tell."
)


def lost_run_message(
    *, tool_call_id: str, execution_id: str, session_id: str, calls: list[dict[str, Any]]
) -> ToolMessage:
    """A run the server lost while it was suspended."""
    content: dict[str, Any] = {
        "result": None,
        "console": [],
        "error": {"kind": "run_lost", "message": LOST_RUN_MESSAGE},
        "denied": False,
    }
    artifact: dict[str, Any] = {"execution_id": execution_id, "session_id": session_id, "refusal": "unknown_run"}
    if calls:
        artifact["tool_calls"] = calls
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=tool_call_id,
        name=EXECUTE,
        artifact=artifact,
        status="error",
    )


_UNKNOWN_EFFECTS = (
    "Check what it would have changed before running anything that repeats it, and tell the user if you cannot tell."
)
_NO_RESULT = (
    "run_result_unavailable",
    "The run finished, but the server no longer keeps its result. Check its effects, or the last run, "
    "before running anything that repeats it.",
)

# By run status: the error kind the model sees, and what it is told.
RUN_STATE_MESSAGES = {
    "stuck": (
        "run_stuck",
        "The run stopped before it finished, and the server holds it until an operator recovers it. Its "
        "effects up to that point are unknown, and programs run after it in this session wait until it is "
        "recovered. Tell the user before running anything else.",
    ),
    "cancelled": (
        "run_cancelled",
        "The run was cancelled before it finished, because its session was closed. Its effects up to that "
        "point are unknown. " + _UNKNOWN_EFFECTS,
    ),
    "aborted": ("run_aborted", "The run was abandoned before the program started, so nothing ran."),
    "indeterminate": (
        "run_indeterminate",
        "The run was interrupted, and the server cannot tell how far it got: it may have done some, all, "
        "or none of its work. " + _UNKNOWN_EFFECTS,
    ),
    "expired": _NO_RESULT,
    "completed": _NO_RESULT,
}


def run_state_message(
    error: RunStateError, *, tool_call_id: str, session_id: str, calls: list[dict[str, Any]]
) -> ToolMessage:
    """A run the server reports with no result to return, such as a durable run that is `stuck`."""
    kind, message = RUN_STATE_MESSAGES.get(
        error.status,
        (
            "run_state_unknown",
            f"The server reported the run as {error.status!r}, which this version of langchain-submilli "
            "does not know, so its outcome is unknown. " + _UNKNOWN_EFFECTS,
        ),
    )
    reported: dict[str, Any] = {"kind": kind, "message": message, "status": error.status}
    if error.reason:
        reported["reason"] = error.reason
    content: dict[str, Any] = {"result": None, "console": [], "error": reported, "denied": False}
    artifact: dict[str, Any] = {
        "execution_id": error.execution_id,
        "session_id": session_id,
        "run_status": error.status,
    }
    if calls:
        artifact["tool_calls"] = calls
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=tool_call_id,
        name=EXECUTE,
        artifact=artifact,
        status="error",
    )


def idempotency_message(error: IdempotencyError, *, tool_call_id: str, session_id: str) -> ToolMessage:
    """An idempotency refusal, explained to the model. Nothing ran."""
    message = IDEMPOTENCY_MESSAGES.get(error.code or "", f"Nothing ran. The server refused the call: {error.detail}")
    content: dict[str, Any] = {
        "result": None,
        "console": [],
        "error": {"kind": error.code, "message": message, "detail": error.detail},
        "denied": False,
    }
    artifact: dict[str, Any] = {"session_id": session_id, "refusal": error.code}
    if error.execution_id:
        artifact["execution_id"] = error.execution_id
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=tool_call_id,
        name=EXECUTE,
        artifact=artifact,
        status="error",
    )


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
