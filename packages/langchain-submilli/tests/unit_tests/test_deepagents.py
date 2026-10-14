"""SubmilliMiddleware inside create_deep_agent."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import pytest
import respx
from deepagents import create_deep_agent

from langchain_submilli import SubmilliMiddleware
from tests.scripted import EXECUTE, call, calls_then_done, execute
from tests.unit_tests.fake_server import SERVER, TOKEN, FakeServer

FILE_TOOLS = {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


async def invoke(graph: Any) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, context={"user_id": "u_ada"}),
    )


async def test_the_one_line_setup_runs_programs_on_submilli(server: FakeServer) -> None:
    model = calls_then_done(execute("a"))
    graph = create_deep_agent(model=model, middleware=[SubmilliMiddleware("research", server=SERVER, token=TOKEN)])
    result = await invoke(graph)

    assert EXECUTE in model.tool_names()
    # Without the flag, deepagents keeps its own file tools.
    assert set(model.tool_names()) >= FILE_TOOLS
    [message] = [m for m in result["messages"] if m.type == "tool"]
    assert json.loads(message.text)["result"] == "ok"


async def test_hide_filesystem_tools_removes_and_refuses_them(server: FakeServer) -> None:
    model = calls_then_done(call("write_file", {"file_path": "/notes/a.md", "content": "x"}))
    submilli = SubmilliMiddleware("research", server=SERVER, token=TOKEN, hide_filesystem_tools=True)
    result = await invoke(create_deep_agent(model=model, middleware=[submilli]))

    offered = set(model.tool_names())
    assert not (FILE_TOOLS | {"execute"}) & offered
    assert EXECUTE in offered
    [message] = [m for m in result["messages"] if m.type == "tool"]
    assert message.status == "error"
    assert "not available" in message.content
    assert not result.get("files")
