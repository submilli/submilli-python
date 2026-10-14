"""Runs agent.py with a scripted model in place of Claude, so it needs no API key.

Start the server first (see README.md). It asserts that two users who write the
same note each see only their own.
"""

import asyncio
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import agent
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

PROGRAM = (pathlib.Path(__file__).parent / "note.ts").read_text()


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "Scripted":
        names = {tool.name for tool in tools}
        assert "submilli__typescript__execute" in names
        assert "write_file" not in names, "deepagents' file tools are hidden"
        return self


def scripted(call_id: str) -> Scripted:
    call = {"name": "submilli__typescript__execute", "args": {"code": PROGRAM}, "id": call_id}
    return Scripted(messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]))


async def note(user_id: str) -> str:
    graph = create_deep_agent(
        model=scripted(f"call-{time.time_ns()}"),
        middleware=[agent.submilli],
        system_prompt=agent.INSTRUCTIONS,
        context_schema=agent.Context,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "note"}]}, context=agent.Context(user_id=user_id)
    )
    return next(str(m.text) for m in result["messages"] if m.type == "tool")


async def main() -> None:
    ada, grace = f"ada_{time.time_ns()}", f"grace_{time.time_ns()}"
    assert "notes before: none" in await note(ada)
    assert "notes before: none" in await note(grace)
    assert "notes before: check.md" in await note(ada)
    print("research: ok")


asyncio.run(main())
