"""A scripted model, as in submilli-runtime's `check.py`: no API key needed."""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

from agents import FunctionTool, ModelResponse, ModelSettings, Tool, Usage
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.models.interface import Model, ModelTracing
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_prompt_param import ResponsePromptParam

EXECUTE = "submilli__typescript__execute"

_ids = itertools.count()

Call = tuple[str, dict[str, Any], str]


def call(name: str, args: dict[str, Any], id: str | None = None) -> Call:
    return (name, args, id or f"call-{next(_ids)}")


def execute(code: str, id: str | None = None) -> Call:
    return call(EXECUTE, {"code": code}, id)


class Scripted(Model):
    """Answers each model call with the next turn: tool calls, or a final text. Records what it saw."""

    def __init__(self, *turns: list[Call] | str) -> None:
        self.turns = list(turns)
        self.seen: list[list[Any]] = []
        self.tools: list[list[Tool]] = []

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
        self.seen.append([input] if isinstance(input, str) else list(input))
        self.tools.append(list(tools))
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            text = ResponseOutputText(type="output_text", text=turn, annotations=[])
            output: list[Any] = [
                ResponseOutputMessage(
                    id="message", type="message", role="assistant", status="completed", content=[text]
                )
            ]
        else:
            output = [
                ResponseFunctionToolCall(type="function_call", call_id=id, name=name, arguments=json.dumps(args))
                for name, args, id in turn
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

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
        raise NotImplementedError

    def descriptions(self, turn: int = 0) -> dict[str, str]:
        """The function tools the model was given on `turn`, by name, with their descriptions."""
        return {tool.name: tool.description for tool in self.tools[turn] if isinstance(tool, FunctionTool)}

    def tool_outputs(self) -> list[Any]:
        """The tool outputs in the last input the model saw, parsed as JSON where they are JSON."""
        outputs = []
        for item in self.seen[-1]:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                try:
                    outputs.append(json.loads(item["output"]))
                except ValueError:
                    outputs.append(item["output"])
        return outputs


def scripted(*turns: list[Call] | str) -> Scripted:
    return Scripted(*turns)


def calls_then_done(*calls: Call) -> Scripted:
    """One turn with `calls` in parallel, then "done"."""
    return Scripted(list(calls), "done")
