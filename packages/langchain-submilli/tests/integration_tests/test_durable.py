"""Durable tool callbacks through an agent, including approvals across restarts."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from tests.integration_tests.conftest import ManagedServer
from tests.integration_tests.test_app_tools import THREAD, agent, body, decide, first_tool, sent, start
from tests.scripted import execute

pytestmark = [pytest.mark.integration, pytest.mark.durable, pytest.mark.parametrize("managed", [True], indirect=True)]

PROGRAM = """import tools from "@tools";
function main(): string {
    const first = tools.send_email({to: "first@x", body: "one"});
    const second = tools.send_email({to: "second@x", body: "two"});
    return first + ";" + second;
}
"""


@pytest.fixture(autouse=True)
def clear_sent() -> None:
    sent.clear()


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_approval_survives_restart_without_repeating_an_earlier_tool(
    managed: ManagedServer, decision: str
) -> None:
    graph = agent(managed, [execute(PROGRAM)], "done", interrupt_on={"send_email": True})
    paused = await start(graph)
    assert paused["__interrupt__"] and sent == []
    paused = await decide(graph, {"type": "approve"})
    assert paused["__interrupt__"]
    assert sent == [{"to": "first@x", "body": "one"}]
    session = (await graph.aget_state(THREAD)).values["submilli_session"]
    managed.restart()
    result = await decide(graph, {"type": decision})
    message = first_tool(result)
    if decision == "approve":
        assert body(message)["result"] == "sent to first@x;sent to second@x"
        assert sent == [{"to": "first@x", "body": "one"}, {"to": "second@x", "body": "two"}]
    else:
        assert body(message)["denied"] is True
        assert body(message)["error"]["source"] == "harness"
        assert sent == [{"to": "first@x", "body": "one"}]
    assert (await graph.aget_state(THREAD)).values["submilli_session"] == session
    assert len(message.artifact["tool_calls"]) == 2


async def test_concurrent_agents_keep_durable_tool_calls_separate(managed: ManagedServer) -> None:
    source = 'import tools from "@tools"; function main(): string { return tools.send_email({to: "TO", body: "hi"}); }'

    # Each agent gets its own scripted model iterator and middleware instance.
    # This also checks several clients driving the same server simultaneously.
    async def independent(number: int) -> dict[str, Any]:
        independent_graph = agent(managed, [execute(source.replace("TO", f"user-{number}@x"))], "done")
        return cast(
            "dict[str, Any]",
            await independent_graph.ainvoke(
                {"messages": [{"role": "user", "content": "go"}]},
                {"configurable": {"thread_id": f"t-{number}"}},
                context={"user_id": f"u-{number}"},
            ),
        )

    results = await asyncio.wait_for(asyncio.gather(*(independent(i) for i in range(12))), timeout=60)
    assert sorted(item["to"] for item in sent) == sorted(f"user-{i}@x" for i in range(12))
    assert len({first_tool(result).artifact["execution_id"] for result in results}) == 12
    assert all(body(first_tool(result))["result"].startswith("sent to user-") for result in results)
