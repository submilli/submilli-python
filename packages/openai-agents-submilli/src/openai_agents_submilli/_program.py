"""A program's calls to the application's tools, run as tool calls of a nested agent.

A program that calls `@tools` suspends on the server at each call. Rather than
run those calls itself, the execute tool runs a nested agent whose "model" is
the program: `ProgramModel` asks the server for the run's state and answers
with a function call for the call the run waits on. The SDK then runs that
call as it runs any tool call, with the tool's `needs_approval`, guardrails,
context, hooks and tracing, and gives the output back to the model, which
resumes the run with it. When the run finishes, the model answers with the
execute tool's result.

Approvals are therefore the SDK's own. A call that needs one interrupts the
nested run, the SDK surfaces it on the outer run as a `ToolApprovalItem` for
the application's tool, and `RunState.approve` or `reject` followed by
`Runner.run(agent, state)` carries on, as for an agent used as a tool. The
model holds no state of its own: everything it needs is in its input (the
code, the session, the idempotency key) and in the outputs the SDK recorded,
so a run resumed from a serialized `RunState` in another process works too.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents import Agent, ModelResponse, ModelSettings, RunConfig, Tool, Usage
from agents.models.interface import Model
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from openai_agents_submilli import _tools as t
from openai_agents_submilli._app_tools import ProgramTools, answer
from submilli import (
    AsyncSubmilli,
    ExecuteResult,
    IdempotencyError,
    PendingCall,
    RunState,
    RunStateError,
    ToolResult,
    UnknownRunError,
    UnknownSessionError,
)

if TYPE_CHECKING:
    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import TResponseInputItem, TResponseStreamEvent
    from agents.models.interface import ModelTracing
    from openai.types.responses.response_prompt_param import ResponsePromptParam

PROGRAM_AGENT = "submilli_program"

# Each call a program makes to the application's tools is one turn of the
# nested agent.
MAX_PROGRAM_CALLS = 10_000

# The nested model's final answer, which the execute tool unwraps.
_RESULT = "__submilli_execute_result__"


@dataclass(frozen=True)
class ProgramInput:
    """What the nested agent is given: enough to find the run again, in any process."""

    code: str
    session_id: str
    idempotency_key: str | None
    notice: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "code": self.code,
                "session_id": self.session_id,
                "idempotency_key": self.idempotency_key,
                "notice": self.notice,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> ProgramInput:
        data = json.loads(text)
        return cls(
            code=data["code"],
            session_id=data["session_id"],
            idempotency_key=data.get("idempotency_key"),
            notice=data.get("notice"),
        )


def final_text(outcome: t.Outcome) -> str:
    content, data = outcome
    return json.dumps({_RESULT: {"content": content, "data": data}}, ensure_ascii=False)


def unwrap(output: Any) -> t.Outcome | None:
    """The outcome a nested run answered with, or `None` when it stopped for an approval."""
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get(_RESULT), dict):
        return None
    result = parsed[_RESULT]
    return dict(result["content"]), dict(result["data"])


class _NeedsCall(Exception):
    """The run waits on a call that has no output yet: the nested agent must make it."""

    def __init__(self, call: PendingCall) -> None:
        super().__init__(call.call_id)
        self.call = call


class ProgramModel(Model):
    """The nested agent's model: the program's run on submilli-server."""

    def __init__(self, client: AsyncSubmilli, tools: ProgramTools) -> None:
        self.client = client
        self.tools = tools

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff[Any, Any]],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        program, calls, outputs = _read(input)
        records = [_record(call, outputs.get(call_id)) for call_id, call in calls.items()]
        session_id = program.session_id

        async def on_call(call: PendingCall) -> ToolResult:
            if call.tool not in self.tools.names:
                return ToolResult.failed(call.call_id, f"the application has no tool named {call.tool!r}")
            if call.call_id not in outputs:
                raise _NeedsCall(call)
            result, _ = answer(call.call_id, outputs[call.call_id])
            return result

        try:
            # With the same key, this answers the run's current state.
            state: RunState = await self.client.sessions.execute(
                session_id, program.code, idempotency_key=program.idempotency_key
            )
            result = await self.client.runs.complete(state, on_tool_call=on_call)
        except _NeedsCall as needs:
            return _response(_function_call(needs.call))
        except IdempotencyError as error:
            return _response(_message(final_text(t.idempotency_outcome(error, session_id=session_id))))
        except RunStateError as error:
            outcome = t.run_state_outcome(error, session_id=session_id, tool_calls=records)
            return _response(_message(final_text(outcome)))
        except (UnknownRunError, UnknownSessionError) as error:
            execution_id = getattr(error, "execution_id", None) or ""
            outcome = t.lost_run_outcome(execution_id=execution_id, session_id=session_id, tool_calls=records)
            return _response(_message(final_text(outcome)))
        if not isinstance(result, ExecuteResult):  # pragma: no cover - `complete` returns only finished runs
            raise TypeError(f"unexpected run state {result!r}")
        outcome = t.execute_outcome(result, notice=program.notice, tool_calls=records)
        return _response(_message(final_text(outcome)))

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff[Any, Any]],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("the program's nested agent is never streamed")


def program_agent(client: AsyncSubmilli, tools: ProgramTools) -> Agent[Any]:
    """The nested agent that runs a program's calls to the application's tools."""
    return Agent(name=PROGRAM_AGENT, model=ProgramModel(client, tools), tools=list(tools.tools))


def nested_run_config(parent: RunConfig | None) -> RunConfig:
    """The nested run's settings: the parent's tracing, and nothing that would swap the model.

    A parent `RunConfig.model` overrides every agent's model, the nested one's
    included, and run-level guardrails are meant for the model's turns, so
    neither is passed on.
    """
    if parent is None:
        return RunConfig()
    return RunConfig(
        tracing_disabled=parent.tracing_disabled,
        tracing=parent.tracing,
        trace_include_sensitive_data=parent.trace_include_sensitive_data,
        workflow_name=parent.workflow_name,
        group_id=parent.group_id,
        trace_metadata=parent.trace_metadata,
        tool_execution=parent.tool_execution,
    )


# Reading the nested agent's input


def _read(
    input: str | list[TResponseInputItem],
) -> tuple[ProgramInput, dict[str, ResponseFunctionToolCall], dict[str, Any]]:
    """The program, the calls made so far by call id, and their outputs by call id."""
    items: list[Any] = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
    program: ProgramInput | None = None
    calls: dict[str, ResponseFunctionToolCall] = {}
    outputs: dict[str, Any] = {}
    for item in items:
        kind = _field(item, "type")
        if program is None and _field(item, "role") == "user":
            program = ProgramInput.from_json(_content_text(_field(item, "content")))
        elif kind == "function_call":
            call = ResponseFunctionToolCall(
                type="function_call",
                call_id=str(_field(item, "call_id")),
                name=str(_field(item, "name")),
                arguments=str(_field(item, "arguments") or "{}"),
            )
            calls[call.call_id] = call
        elif kind == "function_call_output":
            outputs[str(_field(item, "call_id"))] = _field(item, "output")
    if program is None:
        raise ValueError("the program's nested agent was started without its program")
    return program, calls, outputs


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(_field(part, "text") or "") for part in content or [])


def _record(call: ResponseFunctionToolCall, output: Any) -> dict[str, Any]:
    """A program's call as the application sees it in `custom_data["tool_calls"]`."""
    try:
        args = json.loads(call.arguments)
    except ValueError:
        args = call.arguments
    record: dict[str, Any] = {"call_id": call.call_id, "tool": call.name, "args": args}
    if output is not None:
        _, record["outcome"] = answer(call.call_id, output)
    return record


def _function_call(call: PendingCall) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        call_id=call.call_id,
        name=call.tool,
        arguments=json.dumps(call.args, ensure_ascii=False),
    )


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="submilli-program",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _response(item: ResponseFunctionToolCall | ResponseOutputMessage) -> ModelResponse:
    return ModelResponse(output=[item], usage=Usage(), response_id=None)
