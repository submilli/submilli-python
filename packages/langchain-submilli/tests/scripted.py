"""A scripted chat model, as in submilli-runtime's `check.py`: no API key needed."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

EXECUTE = "submilli__typescript__execute"

_ids = itertools.count()


def call(name: str, args: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {"name": name, "args": args, "id": id or f"call-{next(_ids)}"}


def execute(code: str, id: str | None = None) -> dict[str, Any]:
    return call(EXECUTE, {"code": code}, id)


class Scripted(GenericFakeChatModel):
    """Answers with the given messages in order, and records what it was given."""

    bound: list[list[Any]] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:
        self.bound.append(list(tools))
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def tool_names(self) -> list[str]:
        return [getattr(tool, "name", None) or tool.get("name") for tool in self.bound[-1]]

    def tool_results(self) -> list[Any]:
        """The tool messages in the last request the model saw."""
        return [message for message in self.seen[-1] if message.type == "tool"]


def scripted(*turns: Iterable[dict[str, Any]] | str) -> Scripted:
    """One AI message per turn: a list of tool calls, or a final text."""
    messages = [
        AIMessage(content=turn) if isinstance(turn, str) else AIMessage(content="", tool_calls=list(turn))
        for turn in turns
    ]
    return Scripted(messages=iter(messages))


def calls_then_done(*calls: dict[str, Any]) -> Scripted:
    """One turn with `calls` in parallel, then "done"."""
    return scripted(list(calls), "done")
