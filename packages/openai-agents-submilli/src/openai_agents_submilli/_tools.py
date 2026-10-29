"""The six Submilli tools: names, argument schemas, and what the model is shown.

Results are dictionaries here. The toolkit turns them into the JSON text the
model reads, and keeps the ids for the application in the output's
`custom_data`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from submilli import BlueprintPrompt, ExecuteResult, IdempotencyError, RunStateError

# The server's own names, as its MCP endpoint lists them.
EXECUTE = "submilli__typescript__execute"
LAST_RUN = "submilli__typescript__last_run"
PACKAGES_SEARCH = "submilli__typescript__packages__search"
PACKAGES_DOCS = "submilli__typescript__packages__docs"
BUILTINS_LIST = "submilli__typescript__builtins__list"
BUILTINS_DOCS = "submilli__typescript__builtins__docs"

TOOL_NAMES = (EXECUTE, LAST_RUN, PACKAGES_SEARCH, PACKAGES_DOCS, BUILTINS_LIST, BUILTINS_DOCS)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    # Strict mode: every property is required, and no others are allowed.
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


EXECUTE_PARAMS = _object(
    {
        "code": {
            "type": "string",
            "description": "TypeScript-subset source with a `main()` entry point. Its return value is the tool result.",
        }
    }
)
NO_PARAMS = _object({})
PACKAGES_SEARCH_PARAMS = _object(
    {
        "query": {
            "type": "string",
            "description": "Substring matched against module names, descriptions, and exported symbols. "
            'Empty ("") lists every available package.',
        }
    }
)
PACKAGES_DOCS_PARAMS = _object({"name": {"type": "string", "description": "Package name, e.g. `submilli:crypto`."}})
BUILTINS_DOCS_PARAMS = _object(
    {
        "names": {
            "type": "array",
            "items": {"type": "string"},
            "description": 'Built-in names to look up, e.g. `["Array", "Map", "Temporal"]`. Each returns its '
            "declarations; unknown names come back with an `error` entry.",
        }
    }
)


@dataclass(frozen=True)
class Spec:
    name: str
    params: dict[str, Any]
    # Used until the Blueprint's own descriptions have been fetched.
    fallback: str


SPECS = (
    Spec(EXECUTE, EXECUTE_PARAMS, "Run a TypeScript program on submilli-server under the Blueprint's policy."),
    Spec(LAST_RUN, NO_PARAMS, "Return the most recent execution's result and full console for this conversation."),
    Spec(PACKAGES_SEARCH, PACKAGES_SEARCH_PARAMS, "Search the Packages a program may import."),
    Spec(PACKAGES_DOCS, PACKAGES_DOCS_PARAMS, "Return a Package's documentation and TypeScript declarations."),
    Spec(BUILTINS_LIST, NO_PARAMS, "List the built-ins in scope without an import."),
    Spec(BUILTINS_DOCS, BUILTINS_DOCS_PARAMS, "Return the declarations of built-ins."),
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


# What the model is shown. Each builder returns `(content, data)`: the model
# reads `content`, and `data` goes to the application in `custom_data`.

Outcome = tuple[dict[str, Any], dict[str, Any]]

DENIAL_NOTE = (
    "The operator's policy denied this. The denial is final: tell the user what was refused, "
    "and do not try another route to the same effect."
)

SESSION_REPLACED_NOTE = (
    "This conversation's previous Submilli session had ended (idle timeout, or the server no longer "
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
    "idempotency_response_expired": (
        "Nothing new ran. The earlier program finished, but the server no longer keeps its result. "
        "Check its effects before running anything that repeats it; changing the key would run it again."
    ),
}


def execute_outcome(
    result: ExecuteResult, *, notice: str | None = None, tool_calls: list[dict[str, Any]] | None = None
) -> Outcome:
    """A finished run: the model gets `result`, `console`, `error`, `denied` and the tool calls."""
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
    data: dict[str, Any] = {"execution_id": result.execution_id, "session_id": result.session_id}
    _add_tool_calls(content, data, tool_calls)
    return content, data


def _add_tool_calls(content: dict[str, Any], data: dict[str, Any], tool_calls: list[dict[str, Any]] | None) -> None:
    """Report the program's calls to the application's tools, in order.

    The model gets them without their ids, so that after a refusal or a lost run it can tell
    the user which calls already took effect; the application gets them whole.
    """
    if not tool_calls:
        return
    content["tool_calls"] = [{key: value for key, value in call.items() if key != "call_id"} for call in tool_calls]
    data["tool_calls"] = tool_calls


LOST_RUN_MESSAGE = (
    "The run was lost while it waited on one of the application's tools: the server restarted or no "
    "longer holds it. Apart from the application tool calls in `tool_calls`, if any, its effects up to that "
    "point are unknown. Check what it would have changed before "
    "running anything that repeats it, and tell the user if you cannot tell."
)


def lost_run_outcome(*, execution_id: str, session_id: str, tool_calls: list[dict[str, Any]]) -> Outcome:
    """A run the server lost while it was suspended."""
    content: dict[str, Any] = {
        "result": None,
        "console": [],
        "error": {"kind": "run_lost", "message": LOST_RUN_MESSAGE},
        "denied": False,
    }
    data: dict[str, Any] = {"execution_id": execution_id, "session_id": session_id, "refusal": "unknown_run"}
    _add_tool_calls(content, data, tool_calls)
    return content, data


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


def run_state_outcome(error: RunStateError, *, session_id: str, tool_calls: list[dict[str, Any]]) -> Outcome:
    """A run the server reports with no result to return, such as a durable run that is `stuck`."""
    kind, message = RUN_STATE_MESSAGES.get(
        error.status,
        (
            "run_state_unknown",
            f"The server reported the run as {error.status!r}, which this version of openai-agents-submilli "
            "does not know, so its outcome is unknown. " + _UNKNOWN_EFFECTS,
        ),
    )
    reported: dict[str, Any] = {"kind": kind, "message": message, "status": error.status}
    if error.reason:
        reported["reason"] = error.reason
    content: dict[str, Any] = {"result": None, "console": [], "error": reported, "denied": False}
    data: dict[str, Any] = {"execution_id": error.execution_id, "session_id": session_id, "run_status": error.status}
    _add_tool_calls(content, data, tool_calls)
    return content, data


def idempotency_outcome(error: IdempotencyError, *, session_id: str) -> Outcome:
    """An idempotency refusal, explained to the model. Nothing ran."""
    message = IDEMPOTENCY_MESSAGES.get(error.code or "", f"Nothing ran. The server refused the call: {error.detail}")
    content: dict[str, Any] = {
        "result": None,
        "console": [],
        "error": {"kind": error.code, "message": message, "detail": error.detail},
        "denied": False,
    }
    data: dict[str, Any] = {"session_id": session_id, "refusal": error.code}
    if error.execution_id:
        data["execution_id"] = error.execution_id
    return content, data


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
