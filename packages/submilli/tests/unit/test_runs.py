"""Runs that call the application's tools: suspend, answer, resume."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from submilli import (
    AsyncSubmilli,
    ExecuteResult,
    PendingCall,
    RunningRun,
    RunStateError,
    Submilli,
    SubmilliError,
    SuspendedRun,
    ToolDeclaration,
    ToolResult,
    UnknownRunError,
)
from tests.unit.fake_server import SERVER, TOKEN, FakeServer

ORDERS: ToolDeclaration = {
    "name": "query_orders",
    "description": "Orders of one customer.",
    "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]},
    "output_schema": {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]},
}


async def session_with_tools(server: FakeServer, client: AsyncSubmilli) -> str:
    server.open_warnings = ["@tools: something programs will not see"]
    created = await client.sessions.create("research", variables={"userId": "u"}, tools=[ORDERS])
    assert created.warnings == ["@tools: something programs will not see"]
    assert json.loads(server.requests[-1].content)["tools"] == [ORDERS]
    return created.id


async def test_execute_answers_a_suspended_run(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "c_42"})]
    session_id = await session_with_tools(server, client)
    state = await client.sessions.execute(session_id, "orders", idempotency_key="k")

    assert isinstance(state, SuspendedRun)
    [call] = state.calls
    assert (call.tool, call.args, call.execution_id) == ("query_orders", {"customer": "c_42"}, state.execution_id)
    assert call.call_id == f"{state.execution_id}:1"
    # Retrying with the same key answers the run's current state.
    assert await client.sessions.execute(session_id, "orders", idempotency_key="k") == state

    finished = await client.runs.resume(state.execution_id, [ToolResult.success(call.call_id, {"count": 3})])
    assert finished.status == "finished"


async def test_run_answers_each_call_with_the_handler(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["two"] = [("query_orders", {"customer": "a"}), ("query_orders", {"customer": "bb"})]
    session_id = await session_with_tools(server, client)
    seen: list[PendingCall] = []

    async def answer(call: PendingCall) -> ToolResult:
        seen.append(call)
        return ToolResult.success(call.call_id, {"count": len(call.args["customer"])})

    result = await client.sessions.run(session_id, "two", on_tool_call=answer)
    assert [call.args for call in seen] == [{"customer": "a"}, {"customer": "bb"}]
    assert result.result == json.dumps([{"count": 1}, {"count": 2}])


async def test_failed_and_denied_answers(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["mail"] = [("query_orders", {"customer": "a"})]
    session_id = await session_with_tools(server, client)

    denied = await client.sessions.run(session_id, "mail", on_tool_call=lambda c: ToolResult.denied(c.call_id, "no"))
    assert denied.denied and denied.error is not None and denied.error.source == "harness"
    [resumed] = server.calls("POST", r"/v1/runs/.*/resume")
    assert json.loads(resumed.content)["results"][0] == {
        "call_id": json.loads(resumed.content)["results"][0]["call_id"],
        "ok": False,
        "error": {"kind": "denied", "message": "no"},
    }

    failed = await client.sessions.run(
        session_id, "mail", on_tool_call=lambda c: ToolResult.failed(c.call_id, "db down")
    )
    assert failed.error is not None and failed.error.kind == "runtime_error"


async def test_a_handler_that_raises_leaves_the_run_waiting(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "a"})]
    session_id = await session_with_tools(server, client)

    def interrupted(call: PendingCall) -> ToolResult:
        raise KeyboardInterrupt  # stands in for a framework's interrupt

    with pytest.raises(KeyboardInterrupt):
        await client.sessions.run(session_id, "orders", idempotency_key="k", on_tool_call=interrupted)
    # Later, the same key picks the run up at the call it waits on.
    result = await client.sessions.run(
        session_id, "orders", idempotency_key="k", on_tool_call=lambda c: ToolResult.success(c.call_id, {"count": 1})
    )
    assert result.result == json.dumps([{"count": 1}])
    assert server.total_runs() == 1


async def test_a_suspended_run_without_a_handler_raises(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "a"})]
    session_id = await session_with_tools(server, client)
    with pytest.raises(SubmilliError, match="no on_tool_call"):
        await client.sessions.run(session_id, "orders")


async def test_a_handler_must_answer_the_call_it_was_given(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "a"})]
    session_id = await session_with_tools(server, client)
    with pytest.raises(SubmilliError, match="must return a ToolResult"):
        await client.sessions.run(session_id, "orders", on_tool_call=lambda c: ToolResult.success("other", 1))


async def test_a_lost_run_raises_unknown_run(server: FakeServer, client: AsyncSubmilli) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "a"})]
    session_id = await session_with_tools(server, client)

    def restart(call: PendingCall) -> ToolResult:
        server.suspended.clear()
        return ToolResult.success(call.call_id, {"count": 1})

    with pytest.raises(UnknownRunError) as lost:
        await client.sessions.run(session_id, "orders", on_tool_call=restart)
    assert lost.value.execution_id


def running(run_id: str) -> httpx.Response:
    return httpx.Response(200, json={"status": "running", "execution_id": run_id, "session_id": "s"})


def finished(run_id: str, result: str) -> httpx.Response:
    body: dict[str, Any] = {"status": "finished", "execution_id": run_id, "session_id": "s"}
    return httpx.Response(200, json={**body, "result": result, "console": [], "error": None})


async def test_a_timed_out_call_is_looked_up_and_a_running_run_polled(router: respx.MockRouter) -> None:
    suspended = {
        "status": "suspended",
        "execution_id": "r",
        "session_id": "s",
        "calls": [{"call_id": "r:1", "tool": "t", "args": {}}],
    }
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=httpx.Response(200, json=suspended))
    router.post(f"{SERVER}/v1/runs/r/resume").mock(
        return_value=httpx.Response(409, json={"error": "run_not_suspended", "execution_id": "r", "message": "x"})
    )
    poll = router.get(f"{SERVER}/v1/runs/r").mock(side_effect=[running("r"), finished("r", "timed out")])

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        result = await client.runs.complete(
            await client.sessions.execute("s", "code"),
            on_tool_call=lambda c: ToolResult.success(c.call_id, 1),
            poll_interval=0,
        )
    assert result.result == "timed out"
    assert poll.call_count == 2


# Durable runs: the shapes of crates/submilli-server/src/handlers/durable_runs.rs (`view`, `observe`).


def lifecycle(status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    """A durable run's lifecycle view, as execute, resume and `GET /v1/runs/{id}` answer it."""
    return {
        "execution_id": "r",
        "session_id": "s",
        "status": status,
        "reason": reason,
        "durable": True,
        "revision": 3,
        "attempt": 1,
        "admission_order": 7,
        "created_at_unix_ms": 1_761_000_000_000,
        "completed_at_unix_ms": None,
        **extra,
    }


RETAINED = {
    "status": "finished",
    "execution_id": "r",
    "session_id": "s",
    "result": None,
    "console": ["before"],
    "error": {"kind": "runtime_error", "message": "Error: boom"},
}

# Each run state with no result to return: (view, status, reason).
WITHOUT_RESULT = [
    pytest.param(lifecycle("stuck", "credentials_unavailable"), "stuck", "credentials_unavailable", id="stuck"),
    pytest.param(lifecycle("cancelled", "session_closed"), "cancelled", "session_closed", id="cancelled"),
    pytest.param(lifecycle("aborted", "never_dispatched"), "aborted", "never_dispatched", id="aborted"),
    pytest.param(
        lifecycle("indeterminate", "interrupted_without_history", durable=False),
        "indeterminate",
        "interrupted_without_history",
        id="indeterminate",
    ),
    pytest.param(lifecycle("expired"), "expired", None, id="expired"),
    pytest.param(lifecycle("completed", durable=False), "completed", None, id="completed-without-response"),
    pytest.param(lifecycle("paused", "operator"), "paused", "operator", id="unknown-status"),
]


def check_without_result(error: RunStateError, view: dict[str, Any], status: str, reason: str | None) -> None:
    assert (error.execution_id, error.session_id, error.status, error.reason) == ("r", "s", status, reason)
    assert error.body == view
    assert f"run r is {status}" in str(error)
    if reason:
        assert reason in str(error)


def reply(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body)


async def test_a_queued_run_is_polled_until_the_result_its_durable_view_keeps(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(lifecycle("queued")))
    poll = router.get(f"{SERVER}/v1/runs/r").mock(
        side_effect=[reply(lifecycle("running")), reply(lifecycle("completed", response=RETAINED))]
    )

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        state = await client.sessions.execute("s", "code", idempotency_key="k")
        assert state == RunningRun(execution_id="r", session_id="s", status="queued")
        result = await client.runs.complete(state, poll_interval=0)

    assert isinstance(result, ExecuteResult)
    assert (result.execution_id, result.session_id, result.result, result.console) == ("r", "s", None, ["before"])
    assert result.error is not None and result.error.kind == "runtime_error"
    assert poll.call_count == 2


def test_the_sync_client_polls_a_queued_run_until_its_result(router: respx.MockRouter) -> None:
    retained = {**RETAINED, "result": "2", "console": [], "error": None}
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(lifecycle("queued")))
    router.get(f"{SERVER}/v1/runs/r").mock(
        side_effect=[reply(lifecycle("queued")), reply(lifecycle("completed", response=retained))]
    )

    with Submilli(base_url=SERVER, token=TOKEN) as client:
        result = client.runs.complete(client.sessions.execute("s", "code"), poll_interval=0)

    assert isinstance(result, ExecuteResult)
    assert (result.result, result.error) == ("2", None)


@pytest.mark.parametrize(("view", "status", "reason"), WITHOUT_RESULT)
async def test_an_execute_without_a_result_raises_run_state_error(
    router: respx.MockRouter, view: dict[str, Any], status: str, reason: str | None
) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(view))

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        with pytest.raises(RunStateError) as raised:
            await client.sessions.run("s", "code", idempotency_key="k")
    check_without_result(raised.value, view, status, reason)


@pytest.mark.parametrize(("view", "status", "reason"), WITHOUT_RESULT)
def test_the_sync_client_raises_run_state_error_while_polling(
    router: respx.MockRouter, view: dict[str, Any], status: str, reason: str | None
) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(lifecycle("queued")))
    router.get(f"{SERVER}/v1/runs/r").mock(return_value=reply(view))

    with Submilli(base_url=SERVER, token=TOKEN) as client:
        state = client.sessions.execute("s", "code")
        with pytest.raises(RunStateError) as raised:
            client.runs.complete(state, poll_interval=0)
    check_without_result(raised.value, view, status, reason)


SUSPENDED = {
    "status": "suspended",
    "execution_id": "r",
    "session_id": "s",
    "calls": [{"call_id": "r:effect-1", "tool": "t", "args": {}}],
}


async def test_a_resume_that_answers_a_stopped_run_raises(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(SUSPENDED))
    stuck = lifecycle("stuck", "execution_interrupted")
    router.post(f"{SERVER}/v1/runs/r/resume").mock(return_value=reply(stuck))

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        with pytest.raises(RunStateError) as raised:
            await client.sessions.run("s", "code", on_tool_call=lambda c: ToolResult.success(c.call_id, 1))
    check_without_result(raised.value, stuck, "stuck", "execution_interrupted")


def test_the_sync_client_raises_when_a_resume_answers_a_stopped_run(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").mock(return_value=reply(SUSPENDED))
    cancelled = lifecycle("cancelled", "session_closed")
    router.post(f"{SERVER}/v1/runs/r/resume").mock(return_value=reply(cancelled))

    with Submilli(base_url=SERVER, token=TOKEN) as client, pytest.raises(RunStateError) as raised:
        client.sessions.run("s", "code", on_tool_call=lambda c: ToolResult.success(c.call_id, 1))
    check_without_result(raised.value, cancelled, "cancelled", "session_closed")


async def test_a_completed_run_whose_response_is_no_result_raises(router: respx.MockRouter) -> None:
    view = lifecycle("completed", response="legacy bytes")
    router.get(f"{SERVER}/v1/runs/r").mock(return_value=reply(view))

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        with pytest.raises(RunStateError) as raised:
            await client.runs.retrieve("r")
    check_without_result(raised.value, view, "completed", None)


async def test_a_run_the_durable_view_does_not_know_is_unknown_run(router: respx.MockRouter) -> None:
    router.get(f"{SERVER}/v1/runs/r").mock(return_value=httpx.Response(404, text="unknown run"))

    async with AsyncSubmilli(base_url=SERVER, token=TOKEN) as client:
        with pytest.raises(UnknownRunError) as lost:
            await client.runs.retrieve("r")
    assert lost.value.status_code == 404 and lost.value.body == "unknown run"


def test_the_sync_client_answers_calls(server: FakeServer) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "a"})]
    with Submilli(base_url=SERVER, token=TOKEN) as client:
        created = client.sessions.create("research", variables={"userId": "u"}, tools=[ORDERS])
        result = client.sessions.run(
            created.id, "orders", on_tool_call=lambda c: ToolResult.success(c.call_id, {"count": 2})
        )
    assert result.result == json.dumps([{"count": 2}])


def test_resume_needs_a_result() -> None:
    with Submilli(base_url=SERVER) as client, pytest.raises(ValueError):
        client.runs.resume("r", [])
