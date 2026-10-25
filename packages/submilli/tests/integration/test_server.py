"""The SDK against a real submilli-server (0.4.1 or later)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from submilli import (
    AsyncSubmilli,
    ConflictError,
    PendingCall,
    Submilli,
    SuspendedRun,
    ToolDeclaration,
    ToolResult,
    UnknownSessionError,
)
from tests.integration.conftest import ManagedServer

pytestmark = pytest.mark.integration

BLUEPRINT = """kind: blueprint
name: shop
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
  - {capability: fs.read, action: allow}
  - {capability: fs.write, action: allow}
  - {capability: fs.stat, action: allow}
  - capability: tools.call
    filter: tool == "query_orders"
    action: allow
"""

ORDERS: ToolDeclaration = {
    "name": "query_orders",
    "description": "Orders of one customer.",
    "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]},
    "output_schema": {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]},
}
REFUND: ToolDeclaration = {
    "name": "refund",
    "input_schema": {"type": "object", "properties": {"order": {"type": "string"}}},
}

COUNTER = """
import fs from "submilli:fs";

function main(): string {
    const before = fs.exists("n.txt") ? (fs.readText("n.txt") ?? "") : "";
    fs.writeText("n.txt", before + "x");
    return `${before.length + 1}`;
}
"""

ORDERS_PROGRAM = """
import tools from "@tools";

function main(): string {
    const orders = tools.query_orders({ customer: "c_42" });
    return `${orders.count} orders`;
}
"""

REFUND_PROGRAM = """
import tools from "@tools";

function main(): string {
    tools.refund({ order: "o_1" });
    return "refunded";
}
"""

USER = {"userId": "u_ada"}
SECRETS = {"USER_TOKEN": "tok"}


@pytest.fixture
async def client(managed: ManagedServer) -> AsyncIterator[AsyncSubmilli]:
    managed.apply("shop", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        yield client


async def test_a_session_keeps_state_and_a_key_runs_once(client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS)
    first = await client.sessions.run(session.id, COUNTER, idempotency_key="call-1")
    replay = await client.sessions.run(session.id, COUNTER, idempotency_key="call-1")
    second = await client.sessions.run(session.id, COUNTER, idempotency_key="call-2")
    assert (first.result, replay.result, second.result) == ("1", "1", "2")
    assert replay.execution_id == first.execution_id
    last = await client.sessions.last_run(session.id)
    assert last is not None and last.result == "2"
    assert await client.sessions.delete(session.id)
    with pytest.raises(UnknownSessionError):
        await client.sessions.execute(session.id, COUNTER)


async def test_a_program_calls_the_applications_tool(client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS, tools=[ORDERS, REFUND])
    # `refund` declares no output schema, so programs get `unknown` from it.
    assert session.warnings == [
        "@tools: 1 tool(s) return unknown: result schemas are unavailable or unrepresentable; "
        "consult package docs for signatures"
    ]
    calls: list[PendingCall] = []

    def answer(call: PendingCall) -> ToolResult:
        calls.append(call)
        return ToolResult.success(call.call_id, {"count": 3})

    result = await client.sessions.run(session.id, ORDERS_PROGRAM, on_tool_call=answer)
    assert result.error is None, result.error
    assert result.result == "3 orders"
    assert [(call.tool, call.args) for call in calls] == [("query_orders", {"customer": "c_42"})]

    docs = await client.sessions.package_docs(session.id, "@tools")
    assert docs.found and "query_orders(args: { customer: string }): { count: number }" in docs.text


async def test_the_blueprint_refuses_tools_it_does_not_grant(client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS, tools=[ORDERS, REFUND])
    result = await client.sessions.run(session.id, REFUND_PROGRAM, on_tool_call=lambda c: pytest.fail("ran"))
    assert result.denied and result.error is not None and result.error.source == "policy"


async def test_a_denied_answer_is_a_harness_denial(client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS, tools=[ORDERS])
    result = await client.sessions.run(
        session.id, ORDERS_PROGRAM, on_tool_call=lambda c: ToolResult.denied(c.call_id, "not today")
    )
    assert result.denied and result.error is not None and result.error.source == "harness"
    assert "not today" in result.error.message


async def test_a_suspended_run_resumes_by_key(client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS, tools=[ORDERS])
    state = await client.sessions.execute(session.id, ORDERS_PROGRAM, idempotency_key="k")
    assert isinstance(state, SuspendedRun)
    again = await client.sessions.execute(session.id, ORDERS_PROGRAM, idempotency_key="k")
    assert again == state
    assert await client.runs.retrieve(state.execution_id) == state
    done = await client.runs.complete(again, on_tool_call=lambda c: ToolResult.success(c.call_id, {"count": 7}))
    assert done.result == "7 orders"


async def test_a_session_and_its_secrets_survive_a_restart(managed: ManagedServer, client: AsyncSubmilli) -> None:
    # The server keeps harness secrets encrypted with its secret-store key.
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS)
    await client.sessions.run(session.id, COUNTER)
    managed.restart()
    assert (await client.sessions.run(session.id, COUNTER)).result == "2"


async def test_secrets_sealed_with_another_key_are_refused(managed: ManagedServer, client: AsyncSubmilli) -> None:
    session = await client.sessions.create("shop", variables=USER, secrets=SECRETS)
    managed.stop()
    managed.rotate_store_key()
    managed.start()
    with pytest.raises(ConflictError) as refused:
        await client.sessions.run(session.id, COUNTER)
    assert refused.value.status_code == 409
    assert "session secrets" in str(refused.value)


async def test_discovery_and_one_off_runs(client: AsyncSubmilli) -> None:
    prompt = await client.blueprints.prompt("shop")
    assert prompt.name == "shop" and "last_run" in prompt.tools
    assert "namespaces" in await client.blueprints.builtins("shop")
    once = await client.execute(
        "shop", 'export function main(): string { return "hi"; }', variables=USER, secrets=SECRETS
    )
    assert once.result == "hi", once.error
    missing = await client.execute("nope", "export function main(): number { return 1; }")
    assert missing.error is not None and missing.error.kind == "blueprint_not_found"


def test_the_sync_client(managed: ManagedServer) -> None:
    managed.apply("shop", BLUEPRINT)
    with Submilli(base_url=managed.url, token=managed.token) as client:
        session = client.sessions.create("shop", variables=USER, secrets=SECRETS, tools=[ORDERS])
        result = client.sessions.run(
            session.id, ORDERS_PROGRAM, on_tool_call=lambda c: ToolResult.success(c.call_id, {"count": 1})
        )
    assert json.loads(json.dumps(result.result)) == "1 orders"
