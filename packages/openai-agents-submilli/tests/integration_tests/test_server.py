"""`SubmilliTools` on a real submilli-server, with a scripted model.

These start their own server (`managed`), which must be 0.4.1 or later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, RunContextWrapper, Runner, RunResult, RunState, function_tool, set_tracing_disabled
from agents.items import ToolCallOutputItem
from typing_extensions import TypedDict

from openai_agents_submilli import SQLiteSessionStore, SubmilliSessionMismatchError, SubmilliTools
from tests.integration_tests.conftest import ManagedServer
from tests.scripted import Scripted, call, calls_then_done, execute

pytestmark = pytest.mark.integration

set_tracing_disabled(True)

SHOP = """kind: blueprint
name: shop
vfs: none
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

sent: list[dict[str, str]] = []


class Orders(TypedDict):
    count: int
    for_user: int


@function_tool
def query_orders(wrapper: RunContextWrapper[dict[str, str]], customer: str) -> Orders:
    """Orders of one customer."""
    return {"count": len(customer), "for_user": len(wrapper.context["user_id"])}


@function_tool(needs_approval=True)
def send_email(to: str, body: str) -> str:
    """Send an email."""
    sent.append({"to": to, "body": body})
    return f"sent to {to}"


@function_tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("database is down")


@function_tool
def refund(order: str) -> str:
    """Refund an order. The Blueprint does not allow it."""
    raise AssertionError("a tool the Blueprint denies must never run")


@pytest.fixture(autouse=True)
def clear_sent() -> None:
    sent.clear()


def ada(conversation: str = "c_1") -> dict[str, str]:
    return {"user_id": "u_ada", "conversation_id": conversation}


def shop(managed: ManagedServer, model: Scripted, store: SQLiteSessionStore | None = None) -> Agent[Any]:
    managed.apply(SHOP)
    submilli = SubmilliTools(
        "shop",
        server=managed.url,
        token=managed.token,
        store=store,
        ptc=[query_orders, send_email, broken, refund],
    )
    return Agent(name="assistant", model=model, tools=list(submilli.tools))


def tool_outputs(result: RunResult) -> list[ToolCallOutputItem]:
    return [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]


def body(item: ToolCallOutputItem) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(item.output)
    return parsed


ORDERS = """
import tools from "@tools";

function main(): string {
    const orders = tools.query_orders({ customer: "c_42" });
    return `${orders.count} orders, user ${orders.for_user}`;
}
"""


async def test_a_program_calls_a_typed_tool_with_the_runs_context(managed: ManagedServer) -> None:
    result = await Runner.run(shop(managed, calls_then_done(execute(ORDERS))), "go", context=ada())
    [output] = tool_outputs(result)
    assert body(output)["error"] is None, body(output)
    assert body(output)["result"] == "4 orders, user 5"
    assert output.custom_data is not None
    assert output.custom_data["tool_calls"][0]["tool"] == "query_orders"


async def test_package_docs_show_the_tools_signatures(managed: ManagedServer) -> None:
    model = calls_then_done(call("submilli__typescript__packages__docs", {"name": "@tools"}))
    [output] = tool_outputs(await Runner.run(shop(managed, model), "go", context=ada()))
    assert "function query_orders(args: { customer: string }): { count: number; for_user: number };" in output.output


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
    [output] = tool_outputs(await Runner.run(shop(managed, calls_then_done(execute(CATCHES))), "go", context=ada()))
    assert "caught:" in body(output)["result"] and "database is down" in body(output)["result"], body(output)


REFUND = """
import tools from "@tools";

function main(): string {
    return tools.refund({ order: "o_1" });
}
"""


async def test_the_blueprint_decides_which_tools_programs_may_call(managed: ManagedServer) -> None:
    [output] = tool_outputs(await Runner.run(shop(managed, calls_then_done(execute(REFUND))), "go", context=ada()))
    # `@tools` lists every declared tool; the Blueprint refuses the call, and
    # the application never sees it.
    error = body(output)["error"]
    assert error["kind"] == "permission_denied" and error["source"] == "policy", error
    assert body(output)["denied"] is True
    assert "tool_calls" not in (output.custom_data or {})


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


async def test_each_call_waits_for_its_own_approval_across_processes(managed: ManagedServer, tmp_path: Path) -> None:
    database = tmp_path / "app.db"
    assistant = shop(managed, calls_then_done(execute(TWO_EMAILS)), SQLiteSessionStore(database))
    paused = await Runner.run(assistant, "go", context=ada())
    [first] = paused.interruptions
    assert first.tool_name == "send_email"
    assert json.loads(first.arguments or "{}")["to"] == "first@example.com"
    saved = paused.to_state().to_string()

    # Another process: a new toolkit and agent, the saved state, the same database.
    assistant = shop(managed, Scripted("done"), SQLiteSessionStore(database))
    state = await RunState.from_string(assistant, saved)
    state.approve(state.get_interruptions()[0])
    paused = await Runner.run(assistant, state)
    [second] = paused.interruptions
    assert json.loads(second.arguments or "{}")["to"] == "second@example.com"
    state = paused.to_state()
    state.reject(second, rejection_message="Only one email today.")
    result = await Runner.run(assistant, state)

    assert sent == [{"to": "first@example.com", "body": "1"}]
    [output] = tool_outputs(result)
    assert body(output)["error"] is None, body(output)
    assert body(output)["result"].startswith("sent to first@example.com | refused: ")
    assert "Only one email today." in body(output)["result"]
    assert output.custom_data is not None
    assert [record["outcome"] for record in output.custom_data["tool_calls"]] == ["ok", "denied"]


async def test_an_approved_call_resumes_its_run_even_without_the_store(managed: ManagedServer) -> None:
    paused = await Runner.run(shop(managed, calls_then_done(execute(TWO_EMAILS))), "go", context=ada())
    saved = paused.to_state().to_string()

    # Another process with a fresh in-memory store: the saved nested run names its session.
    assistant = shop(managed, Scripted("done"))
    state = await RunState.from_string(assistant, saved)
    state.approve(state.get_interruptions()[0])
    paused = await Runner.run(assistant, state)
    [second] = paused.interruptions
    assert json.loads(second.arguments or "{}")["to"] == "second@example.com"
    state = paused.to_state()
    state.approve(second)
    [output] = tool_outputs(await Runner.run(assistant, state))

    assert [email["to"] for email in sent] == ["first@example.com", "second@example.com"]
    assert body(output)["result"] == "sent to first@example.com | sent to second@example.com", body(output)


def scratch_blueprint(name: str, idle_timeout: str = "1h") -> str:
    return f"""kind: blueprint
name: {name}
idle_timeout: '{idle_timeout}'
variables:
  userId:
    required: true
secrets:
  USER_TOKEN:
    harness:
      required: true
vfs:
  mode: per_session
default: deny
permissions:
  main:
  - {{capability: fs.read, action: allow}}
  - {{capability: fs.write, action: allow}}
  - {{capability: fs.stat, action: allow}}
  - {{capability: fs.list, action: allow}}
"""


SCRATCH = """
import fs from "submilli:fs";

function main(): string {
    const before = fs.exists("scratch.txt");
    fs.writeText("scratch.txt", "x");
    return before ? "scratch was there" : "scratch was not there";
}
"""


def scratch(managed: ManagedServer, blueprint: str) -> SubmilliTools:
    return SubmilliTools(
        blueprint,
        server=managed.url,
        token=managed.token,
        secrets=lambda context: {"USER_TOKEN": context["token"]},
    )


async def ask(submilli: SubmilliTools, context: dict[str, str], **call: Any) -> dict[str, Any]:
    model = calls_then_done(execute(SCRATCH, **call))
    [output] = tool_outputs(
        await Runner.run(Agent(name="a", model=model, tools=list(submilli.tools)), "go", context=context)
    )
    return {**body(output), "data": output.custom_data}


async def test_a_conversation_keeps_its_session_and_others_do_not_share_it(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("scratch"))
    submilli = scratch(managed, "scratch")
    first = await ask(submilli, {**ada(), "token": "t"})
    again = await ask(submilli, {**ada(), "token": "t"})
    other = await ask(submilli, {**ada("c_2"), "token": "t"})
    assert first["result"] == "scratch was not there"
    assert again["result"] == "scratch was there"
    assert other["result"] == "scratch was not there"
    assert first["data"]["session_id"] == again["data"]["session_id"] != other["data"]["session_id"]
    assert await submilli.aclose_conversation("c_1") is True
    closed = await ask(submilli, {**ada(), "token": "t"})
    assert closed["result"] == "scratch was not there"


async def test_the_same_tool_call_runs_once(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("scratch"))
    submilli = scratch(managed, "scratch")
    first = await ask(submilli, {**ada(), "token": "t"}, id="call_once")
    replay = await ask(submilli, {**ada(), "token": "t"}, id="call_once")
    assert first["result"] == replay["result"] == "scratch was not there"
    assert first["data"]["execution_id"] == replay["data"]["execution_id"]


async def test_another_user_in_the_conversation_is_refused(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("scratch"))
    submilli = scratch(managed, "scratch")
    await ask(submilli, {**ada(), "token": "t"})
    with pytest.raises(SubmilliSessionMismatchError):
        await ask(submilli, {"user_id": "u_grace", "conversation_id": "c_1", "token": "t"})


async def test_a_server_restart_keeps_the_session(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("scratch"))
    submilli = scratch(managed, "scratch")
    first = await ask(submilli, {**ada(), "token": "t"})
    managed.restart()
    second = await ask(submilli, {**ada(), "token": "t"})
    assert second["result"] == "scratch was there"
    assert "notice" not in second
    assert second["data"]["session_id"] == first["data"]["session_id"]


async def test_an_expired_session_is_replaced_and_the_model_told(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("expiring", idle_timeout="1s"))
    submilli = scratch(managed, "expiring")
    first = await ask(submilli, {**ada(), "token": "t"})

    # Idle past the timeout while the server is down; it reaps on start.
    managed.stop()
    time.sleep(1.5)
    managed.start()
    second = await ask(submilli, {**ada(), "token": "t"})

    assert second["result"] == "scratch was not there"
    assert "earlier session" in second["notice"]
    assert second["data"]["session_id"] != first["data"]["session_id"]
    stored = await submilli.store.get("c_1")
    assert stored is not None and stored["id"] == second["data"]["session_id"]
