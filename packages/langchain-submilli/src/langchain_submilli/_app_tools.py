"""The application's own tools, for programs to call as the `@tools` Package.

The harness declares each tool when it opens a session. A program's call
suspends its run, the middleware runs the tool here, through a LangGraph
`ToolNode` as the agent would, and resumes the run with the result. The
contract is "Application tools" in submilli-runtime's HTTP API reference.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
import typing
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain.agents import AgentState
from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    InterruptOnConfig,
    ReviewConfig,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage, ToolCall, ToolMessage
from langchain_core.messages.tool import tool_call as make_tool_call
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from submilli import PendingCall, SubmilliError, ToolDeclaration, ToolResult
from submilli.schema import inline_refs

# A TypeScript function name, as the server requires of a tool's name. The
# server also drops reserved words, with a warning when the session opens.
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

_ALL_DECISIONS: list[Literal["approve", "edit", "reject"]] = ["approve", "edit", "reject"]

# What an exposed tool returns when its return type gives no schema: the text
# the model would see.
_TEXT = {"type": "string"}


def declaration(tool: BaseTool) -> ToolDeclaration:
    """`tool` as the server takes it: `{name, description, input_schema, output_schema}`.

    The input schema is the one the model sees, without injected arguments.
    The output schema comes from the function's return annotation, and is a
    string, the tool's text, when there is none or it isn't JSON. `$ref`s are
    inlined, because the server's mapper does not resolve them.
    """
    schema = tool.tool_call_schema
    if isinstance(schema, dict):
        input_schema = copy.deepcopy(schema)
    elif issubclass(schema, BaseModel):
        input_schema = schema.model_json_schema()
    else:  # a Pydantic v1 model
        input_schema = schema.schema()
    declared: ToolDeclaration = {
        "name": tool.name,
        "input_schema": inline_refs(input_schema),
        "output_schema": _output_schema(tool),
    }
    if tool.description:
        declared["description"] = tool.description
    return declared


def catalog_digest(catalog: Sequence[ToolDeclaration]) -> str | None:
    """A fingerprint of the declared tools, so a thread notices when they changed."""
    if not catalog:
        return None
    return hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()


def _output_schema(tool: BaseTool) -> dict[str, Any]:
    function = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if function is None or tool.response_format != "content":
        return dict(_TEXT)
    try:
        annotation = typing.get_type_hints(function).get("return")
    except Exception:  # unresolvable annotations mean "no schema"
        return dict(_TEXT)
    if annotation in (None, str, Any, object, type(None)) or _is_message_or_command(annotation):
        return dict(_TEXT)
    try:
        schema = TypeAdapter(annotation).json_schema(mode="serialization")
    except Exception:  # not a JSON type
        return dict(_TEXT)
    return inline_refs(schema)


def _is_message_or_command(annotation: Any) -> bool:
    candidates = [annotation, typing.get_origin(annotation), *typing.get_args(annotation)]
    return any(isinstance(item, type) and issubclass(item, (BaseMessage, Command)) for item in candidates)


# Approvals


@dataclass
class _Pauses:
    """Which call each approval pause of one run was for, in order.

    LangGraph hands a re-run tool its earlier resume values by position. The
    server, meanwhile, has moved past the calls those values answered, so the
    position of the call waiting now is looked up here.
    """

    calls: list[str] = field(default_factory=list)


class Approvals:
    """Approval of program calls to exposed tools, with LangChain's HITL shapes.

    `interrupt_on` is the `HumanInTheLoopMiddleware` setting, for the exposed
    tools. A call that needs approval interrupts the graph with a `HITLRequest`
    of one action, and the resume value is a `HITLResponse` with one decision.
    The run waits on the server meanwhile, up to the Blueprint's
    `tool_call_timeout`.
    """

    def __init__(self, interrupt_on: Mapping[str, bool | InterruptOnConfig] | None) -> None:
        self.config: dict[str, InterruptOnConfig] = {}
        for name, setting in (interrupt_on or {}).items():
            if setting is True:
                self.config[name] = InterruptOnConfig(allowed_decisions=list(_ALL_DECISIONS))
            elif setting:
                self.config[name] = setting
        self._runs: OrderedDict[str, _Pauses] = OrderedDict()

    def needed(self, request: ToolCallRequest) -> InterruptOnConfig | None:
        config = self.config.get(request.tool_call["name"])
        if config is None:
            return None
        when = config.get("when")
        if when is not None and not when(request):
            return None
        return config

    def decide(
        self, run_id: str, call: PendingCall, config: InterruptOnConfig, runtime: ToolRuntime[Any, Any], used: list[int]
    ) -> dict[str, Any]:
        """The decision for `call`, interrupting the graph when it has none yet.

        `used` counts the interrupts this tool invocation has already made, so
        that the earlier ones are skipped once each.
        """
        pauses = self._runs.get(run_id)
        if pauses is None:
            if _resume_values() and not used[0]:
                # The record of earlier pauses is gone, as when another process
                # resumed the thread: the stored decisions can't be matched to calls.
                return {
                    "type": "reject",
                    "message": (
                        "The approval for this call could not be matched to it, because the agent "
                        "restarted while it waited. Nothing ran; ask the user before trying again."
                    ),
                }
            pauses = _Pauses()
            self._remember(run_id, pauses)
        if call.call_id in pauses.calls:
            position = pauses.calls.index(call.call_id)
        else:
            position = len(pauses.calls)
            pauses.calls.append(call.call_id)
        while used[0] < position:
            interrupt(None)
            used[0] += 1
        request = HITLRequest(
            action_requests=[_action(call, config, runtime)],
            review_configs=[_review(call, config)],
        )
        response = interrupt(request)
        used[0] += 1
        return _decision(response, call, config)

    def finished(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def _remember(self, run_id: str, pauses: _Pauses) -> None:
        self._runs[run_id] = pauses
        while len(self._runs) > 1024:
            self._runs.popitem(last=False)


def _resume_values() -> list[Any]:
    """The resume values LangGraph holds for this task, newest last."""
    try:
        from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
        from langgraph.config import get_config

        return list(get_config().get("configurable", {})[CONFIG_KEY_SCRATCHPAD].resume)
    except Exception as error:
        raise SubmilliError(
            "Approving program calls needs LangGraph's interrupt support, which this LangGraph version "
            "does not provide in the expected form."
        ) from error


def _action(call: PendingCall, config: InterruptOnConfig, runtime: ToolRuntime[Any, Any]) -> ActionRequest:
    description = config.get("description")
    if callable(description):
        # The arguments `HumanInTheLoopMiddleware` gives a description factory.
        node_runtime = Runtime(context=runtime.context, store=runtime.store, stream_writer=runtime.stream_writer)
        description = description(_tool_call(call, call.args), cast("AgentState[Any]", runtime.state), node_runtime)
    if description is None:
        description = (
            f"A program asks to run this tool, and it needs your approval.\n\n"
            f"Tool: {call.tool}\nArgs: {json.dumps(call.args, ensure_ascii=False, indent=2)}"
        )
    return ActionRequest(name=call.tool, args=call.args, description=description)


def _review(call: PendingCall, config: InterruptOnConfig) -> ReviewConfig:
    review = ReviewConfig(action_name=call.tool, allowed_decisions=config["allowed_decisions"])
    if "args_schema" in config:
        review["args_schema"] = config["args_schema"]
    return review


def _decision(response: Any, call: PendingCall, config: InterruptOnConfig) -> dict[str, Any]:
    decisions = response.get("decisions") if isinstance(response, Mapping) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError(
            f"Resume a Submilli tool approval with one decision, as {{'decisions': [{{'type': 'approve'}}]}}; "
            f"got {response!r}"
        )
    decision = decisions[0]
    kind = decision.get("type")
    if kind not in config["allowed_decisions"]:
        raise ValueError(f"Decision {kind!r} is not allowed for {call.tool}; allowed: {config['allowed_decisions']}")
    if kind == "edit" and decision.get("edited_action", {}).get("name", call.tool) != call.tool:
        raise ValueError("An edit can't change which tool a program call runs; the Blueprint allowed this one.")
    return dict(decision)


# Running a call


@dataclass
class CallOutcome:
    """What answering one call produced: the answer for the server, a record, and state updates."""

    result: ToolResult
    record: dict[str, Any]
    updates: list[tuple[str, Any]] = field(default_factory=list)


class ExposedTools:
    """The tools a middleware exposes to programs, and how a call to one is answered."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        interrupt_on: Mapping[str, bool | InterruptOnConfig] | None = None,
    ) -> None:
        names = [tool.name for tool in tools]
        for name in names:
            if not _IDENTIFIER.fullmatch(name):
                raise ValueError(f"Exposed tool {name!r}: programs call tools by name, so it must be an identifier")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Exposed tools must have distinct names; repeated: {', '.join(duplicates)}")
        unknown = sorted(set(interrupt_on or {}) - set(names))
        if unknown:
            raise ValueError(f"interrupt_on names tools that are not exposed: {', '.join(unknown)}")
        self.approvals: Approvals = Approvals(interrupt_on)
        self.catalog: list[ToolDeclaration] = [declaration(tool) for tool in tools]
        self.digest: str | None = catalog_digest(self.catalog)
        self._text = {item["name"]: item.get("output_schema") == _TEXT for item in self.catalog}
        tools = [tool if self._text[tool.name] else _returning_json(tool) for tool in tools]
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self._node = ToolNode(list(tools), handle_tool_errors=True) if tools else None
        # The calls and state updates of a run interrupted for an approval, kept
        # for the invocation that carries on after it.
        self._carried: OrderedDict[str, tuple[list[dict[str, Any]], list[tuple[str, Any]]]] = OrderedDict()

    def __bool__(self) -> bool:
        return bool(self.tools)

    def prompt(self) -> str:
        """What the execute tool's description adds about `@tools`."""
        names = ", ".join(f"`{name}`" for name in self.tools)
        approval = (
            " Some calls wait for a person's approval, who may refuse them: a refusal throws "
            "`PermissionDeniedError`, and is final."
            if self.approvals.config
            else ""
        )
        return (
            "\n\nThis agent's programs can also call the application's own tools, as the `@tools` package "
            f'(`import tools from "@tools";`): {names}. The application runs each call and returns its '
            f"result to the program.{approval} Read their signatures with "
            '`submilli__typescript__packages__docs("@tools")`.'
        )

    async def answer(self, call: PendingCall, runtime: ToolRuntime[Any, Any], used: list[int]) -> CallOutcome:
        record: dict[str, Any] = {"call_id": call.call_id, "tool": call.tool, "args": call.args}
        tool = self.tools.get(call.tool)
        if tool is None or self._node is None:
            return _failed(call, record, f"the application has no tool named {call.tool!r}")
        args = call.args
        tool_call = _tool_call(call, args)
        request = ToolCallRequest(tool_call=tool_call, tool=tool, state=runtime.state, runtime=runtime)
        config = self.approvals.needed(request)
        if config is not None:
            decision = self.approvals.decide(call.execution_id, call, config, runtime, used)
            record["decision"] = decision["type"]
            if decision["type"] == "reject":
                message = decision.get("message") or f"The user rejected this call to {call.tool}."
                return _denied(call, record, message)
            if decision["type"] == "respond":
                record["outcome"] = "ok"
                return CallOutcome(ToolResult.success(call.call_id, decision["message"]), record)
            if decision["type"] == "edit":
                args = dict(decision["edited_action"].get("args") or {})
                record["args"] = args
                tool_call = _tool_call(call, args)
        try:
            output = await self._node.ainvoke(
                {"__type": "tool_call_with_context", "tool_call": tool_call, "state": runtime.state},
                runtime.config,
            )
        except GraphBubbleUp:
            raise
        except Exception as error:  # the program sees the failure
            return _failed(call, record, f"{type(error).__name__}: {error}")
        message, updates = _unpack(output, call.call_id)
        if message is None:
            return _failed(call, record, f"{call.tool} returned no result")
        if message.status == "error":
            return _failed(call, record, message.text, updates)
        record["outcome"] = "ok"
        value = _value(message, text=self._text.get(call.tool, True))
        return CallOutcome(ToolResult.success(call.call_id, value), record, updates)

    def carry(self, run_id: str, calls: list[dict[str, Any]], updates: list[tuple[str, Any]]) -> None:
        self._carried[run_id] = (list(calls), list(updates))
        while len(self._carried) > 1024:
            self._carried.popitem(last=False)

    def carried(self, run_id: str) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
        return self._carried.pop(run_id, ([], []))

    def finished(self, run_id: str) -> None:
        self.approvals.finished(run_id)
        self._carried.pop(run_id, None)


def _returning_json(tool: BaseTool) -> BaseTool:
    """`tool`, returning JSON data, so that a Pydantic model reaches the program as an object.

    A `ToolNode` writes a value that isn't JSON as its `str()`. The wrappers
    keep the function's signature, so injected arguments are still found.
    """
    if not isinstance(tool, StructuredTool):
        return tool
    update: dict[str, Any] = {}
    if tool.func is not None:
        func = tool.func

        @functools.wraps(func)
        def call(*args: Any, **kwargs: Any) -> Any:
            return to_jsonable_python(func(*args, **kwargs))

        update["func"] = call
    if tool.coroutine is not None:
        coroutine = tool.coroutine

        @functools.wraps(coroutine)
        async def acall(*args: Any, **kwargs: Any) -> Any:
            return to_jsonable_python(await coroutine(*args, **kwargs))

        update["coroutine"] = acall
    return tool.model_copy(update=update)


def _tool_call(call: PendingCall, args: dict[str, Any]) -> ToolCall:
    return make_tool_call(name=call.tool, args=args, id=call.call_id)


def _failed(
    call: PendingCall, record: dict[str, Any], message: str, updates: list[tuple[str, Any]] | None = None
) -> CallOutcome:
    record["outcome"] = "failed"
    return CallOutcome(ToolResult.failed(call.call_id, message), record, updates or [])


def _denied(call: PendingCall, record: dict[str, Any], message: str) -> CallOutcome:
    record["outcome"] = "denied"
    return CallOutcome(ToolResult.denied(call.call_id, message), record)


def _unpack(output: Any, call_id: str) -> tuple[ToolMessage | None, list[tuple[str, Any]]]:
    """The call's `ToolMessage`, and the other state updates of a tool that returned a `Command`."""
    items = output if isinstance(output, list) else [output]
    message: ToolMessage | None = None
    updates: list[tuple[str, Any]] = []
    for item in items:
        update = item.update if isinstance(item, Command) else item
        if isinstance(item, Command) and (item.goto or item.graph is not None):
            raise SubmilliError("A tool called from a program can't route the graph (`Command.goto`)")
        pairs = update.items() if isinstance(update, Mapping) else (update or [])
        for key, value in pairs:
            if key != "messages":
                updates.append((key, value))
                continue
            for candidate in value if isinstance(value, list) else [value]:
                if isinstance(candidate, ToolMessage) and candidate.tool_call_id == call_id:
                    message = candidate
    return message, updates


def _value(message: ToolMessage, *, text: bool) -> Any:
    content = message.text if not isinstance(message.content, str) else message.content
    if text:
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


def as_tools(tools: Sequence[BaseTool | Callable[..., Any]] | None) -> list[BaseTool]:
    """Exposed tools as `BaseTool`s; plain functions are wrapped as LangChain does for an agent."""
    from langchain_core.tools import tool as make_tool

    return [item if isinstance(item, BaseTool) else make_tool(item) for item in tools or []]
