"""Programs calling the application's tools (`@tools`) on a real submilli-server.

These start their own server (`managed`), which must be 0.4.1 or later.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_submilli import SubmilliMiddleware
from tests.integration_tests.conftest import ManagedServer
from tests.scripted import call, execute, scripted

pytestmark = pytest.mark.integration

BLUEPRINT = """kind: blueprint
name: shop
variables:
  userId:
    required: true
default: deny
permissions:
  main:
  - capability: tools.call
    filter: tool == "query_orders" or tool == "send_email" or tool == "broken"
    action: allow
"""

THREAD: RunnableConfig = {"configurable": {"thread_id": "t"}}
USER = {"user_id": "u_ada"}

sent: list[dict[str, str]] = []


class Orders(TypedDict):
    count: int
    for_user: int


@tool
def query_orders(customer: str, runtime: ToolRuntime[dict[str, str]]) -> Orders:
    """Orders of one customer."""
    return {"count": len(customer), "for_user": len(runtime.context["user_id"])}


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    sent.append({"to": to, "body": body})
    return f"sent to {to}"


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("database is down")


@tool
def refund(order: str) -> str:
    """Refund an order. The Blueprint does not allow it."""
    raise AssertionError("a tool the Blueprint denies must never run")


@pytest.fixture(autouse=True)
def clear_sent() -> None:
    sent.clear()


def agent(managed: ManagedServer, *turns: Any, interrupt_on: dict[str, Any] | None = None) -> Any:
    managed.apply(BLUEPRINT)
    submilli = SubmilliMiddleware(
        "shop",
        server=managed.url,
        token=managed.token,
        expose_tools=[query_orders, send_email, broken, refund],
        interrupt_on=interrupt_on,
    )
    return create_agent(model=scripted(*turns), middleware=[submilli], checkpointer=InMemorySaver())


async def start(graph: Any) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, THREAD, context=USER)
    )


async def decide(graph: Any, decision: dict[str, Any]) -> dict[str, Any]:
    [pending] = (await graph.aget_state(THREAD)).interrupts
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(Command(resume={pending.id: {"decisions": [decision]}}), THREAD, context=USER),
    )


def first_tool(result: dict[str, Any]) -> ToolMessage:
    return cast("ToolMessage", next(m for m in result["messages"] if m.type == "tool"))


def body(message: ToolMessage) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(message.text))


ORDERS = """
import tools from "@tools";

function main(): string {
    const orders = tools.query_orders({ customer: "c_42" });
    return `${orders.count} orders, user ${orders.for_user}`;
}
"""


async def test_a_program_calls_a_typed_tool_and_uses_its_fields(managed: ManagedServer) -> None:
    result = await start(agent(managed, [execute(ORDERS)], "done"))
    message = first_tool(result)
    assert body(message)["error"] is None, body(message)
    assert body(message)["result"] == "4 orders, user 5"
    assert message.artifact["tool_calls"][0]["tool"] == "query_orders"


async def test_package_docs_show_the_tools_signatures(managed: ManagedServer) -> None:
    graph = agent(managed, [call("submilli__typescript__packages__docs", {"name": "@tools"})], "done")
    text = first_tool(await start(graph)).content
    assert "function query_orders(args: { customer: string }): { count: number; for_user: number };" in text, text


CATCHES = """
import tools from "@tools";

function main(): string {
    try {
        tools.broken();
        return "no error";
    } catch (e) {
        return `caught: ${(e as Error).message}`;
    }
}
"""


async def test_a_failing_tool_throws_in_the_program(managed: ManagedServer) -> None:
    message = first_tool(await start(agent(managed, [execute(CATCHES)], "done")))
    assert "caught:" in body(message)["result"] and "database is down" in body(message)["result"], body(message)


REFUND = """
import tools from "@tools";

function main(): string {
    return tools.refund({ order: "o_1" });
}
"""


async def test_the_blueprint_decides_which_tools_programs_may_call(managed: ManagedServer) -> None:
    message = first_tool(await start(agent(managed, [execute(REFUND)], "done")))
    # `@tools` lists every declared tool; the Blueprint refuses the call, and
    # the application never sees it.
    error = body(message)["error"]
    assert error["kind"] == "permission_denied" and error["source"] == "policy", error
    assert body(message)["denied"] is True
    assert "tool_calls" not in (message.artifact or {})


TWO_EMAILS = """
import tools from "@tools";

function main(): string {
    const first = tools.send_email({ to: "first@example.com", body: "1" });
    let second = "";
    try {
        second = tools.send_email({ to: "second@example.com", body: "2" });
    } catch (e) {
        second = `refused: ${(e as Error).message}`;
    }
    return `${first} | ${second}`;
}
"""


async def test_each_call_waits_for_its_own_approval(managed: ManagedServer) -> None:
    graph = agent(managed, [execute(TWO_EMAILS)], "done", interrupt_on={"send_email": True})

    paused = await start(graph)
    assert paused["__interrupt__"][0].value["action_requests"][0]["args"]["to"] == "first@example.com"
    paused = await decide(graph, {"type": "approve"})
    assert paused["__interrupt__"][0].value["action_requests"][0]["args"]["to"] == "second@example.com"
    result = await decide(graph, {"type": "reject", "message": "Only one email today."})

    assert sent == [{"to": "first@example.com", "body": "1"}]
    message = first_tool(result)
    assert body(message)["error"] is None, body(message)
    assert body(message)["result"].startswith("sent to first@example.com | refused: ")
    assert "Only one email today." in body(message)["result"]
    assert [record.get("decision") for record in message.artifact["tool_calls"]] == ["approve", "reject"]
