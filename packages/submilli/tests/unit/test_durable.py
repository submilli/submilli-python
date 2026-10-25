"""Durable lifecycle transitions, retries, and callback replay over mocked HTTP."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from submilli import AsyncSubmilli, ExecuteResult, IdempotencyError, PendingCall, Submilli, SubmilliError, ToolResult
from tests.unit.fake_server import SERVER


def view(status: str, run: str = "r", **extra: Any) -> dict[str, Any]:
    return {"status": status, "execution_id": run, "session_id": "s", "durable": True, **extra}


def suspended(sequence: int, run: str = "r") -> dict[str, Any]:
    return view(
        "suspended",
        run,
        calls=[{"call_id": f"{run}:effect-{sequence}", "tool": "echo", "args": {"value": sequence}}],
    )


def completed(run: str = "r") -> dict[str, Any]:
    return view(
        "completed", run, response={"status": "finished", "result": "done", "console": ["saved"], "error": None}
    )


async def test_run_waits_through_long_queues_and_recovery_between_tools(
    router: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("submilli._async_client.asyncio.sleep", sleep)
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=view("queued"))
    polls = router.get(f"{SERVER}/v1/runs/r").mock(
        side_effect=[
            httpx.Response(200, json=body)
            for body in [*[view("queued")] * 40, suspended(1), view("queued"), suspended(2), completed()]
        ]
    )
    resumes = router.post(f"{SERVER}/v1/runs/r/resume").respond(200, json=view("running"))
    calls: list[PendingCall] = []

    async def answer(call: PendingCall) -> ToolResult:
        calls.append(call)
        return ToolResult.success(call.call_id, call.args["value"])

    async with AsyncSubmilli(base_url=SERVER) as client:
        result = await client.sessions.run("s", "code", idempotency_key="k", on_tool_call=answer)
    assert (result.result, result.console, result.execution_id, result.session_id) == ("done", ["saved"], "r", "s")
    assert [call.call_id for call in calls] == ["r:effect-1", "r:effect-2"]
    assert polls.call_count == 44 and resumes.call_count == 2
    assert max(args.args[0] for args in sleep.await_args_list) <= 2


def test_sync_run_waits_through_queued_running_and_tools(
    router: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("submilli._client.time.sleep", lambda _: None)
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=view("running"))
    polls = router.get(f"{SERVER}/v1/runs/r").mock(
        side_effect=[
            httpx.Response(200, json=body) for body in [view("queued"), suspended(1), view("running"), completed()]
        ]
    )
    router.post(f"{SERVER}/v1/runs/r/resume").respond(200, json=view("queued"))
    calls: list[str] = []

    def answer(call: PendingCall) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult.success(call.call_id, 1)

    with Submilli(base_url=SERVER) as client:
        result = client.sessions.run("s", "code", on_tool_call=answer)
    assert result.result == "done" and calls == ["r:effect-1"] and polls.call_count == 4


async def test_a_running_request_does_not_finish_until_the_server_finishes(router: respx.MockRouter) -> None:
    polling = asyncio.Event()
    release = asyncio.Event()
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=view("queued"))

    async def retrieve(_: httpx.Request) -> httpx.Response:
        polling.set()
        await release.wait()
        return httpx.Response(200, json=completed())

    router.get(f"{SERVER}/v1/runs/r").mock(side_effect=retrieve)
    async with AsyncSubmilli(base_url=SERVER) as client:
        task = asyncio.create_task(client.sessions.run("s", "code"))
        try:
            await asyncio.wait_for(polling.wait(), timeout=5)
            assert not task.done()
            release.set()
            assert (await asyncio.wait_for(task, timeout=5)).result == "done"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_polling_can_be_cancelled_without_resubmitting_execution(router: respx.MockRouter) -> None:
    entered = asyncio.Event()
    post = router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=view("running"))

    async def retrieve(_: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    router.get(f"{SERVER}/v1/runs/r").mock(side_effect=retrieve)
    async with AsyncSubmilli(base_url=SERVER) as client:
        task = asyncio.create_task(client.sessions.run("s", "code", idempotency_key="k"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert post.call_count == 1


@pytest.mark.parametrize("refusal", ["run_not_suspended", "unknown_call"])
async def test_recovered_uncommitted_call_resends_the_answer_without_running_the_tool_twice(
    router: respx.MockRouter, refusal: str
) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=suspended(1))
    router.get(f"{SERVER}/v1/runs/r").respond(200, json=suspended(1))
    resume = router.post(f"{SERVER}/v1/runs/r/resume").mock(
        side_effect=[
            httpx.Response(409 if refusal == "run_not_suspended" else 400, json={"error": refusal}),
            httpx.Response(200, json=completed()),
        ]
    )
    calls: list[str] = []

    def answer(call: PendingCall) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult.success(call.call_id, len(calls))

    async with AsyncSubmilli(base_url=SERVER) as client:
        result = await client.sessions.run("s", "code", on_tool_call=answer)
    assert result.result == "done"
    assert calls == ["r:effect-1"]
    assert resume.calls[0].request.content == resume.calls[1].request.content


async def test_lost_resume_response_retries_the_same_answer(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=suspended(1))
    resume = router.post(f"{SERVER}/v1/runs/r/resume").mock(
        side_effect=[
            httpx.ReadError("response lost after commit"),
            httpx.Response(503, json={"error": "run_unavailable"}, headers={"Retry-After": "0"}),
            httpx.Response(200, json=completed()),
        ]
    )
    seen: list[str] = []

    def answer(call: PendingCall) -> ToolResult:
        seen.append(call.call_id)
        return ToolResult.success(call.call_id, 1)

    async with AsyncSubmilli(base_url=SERVER) as client:
        assert (await client.sessions.run("s", "code", on_tool_call=answer)).result == "done"
    assert seen == ["r:effect-1"] and resume.call_count == 3
    assert len({call.request.content for call in resume.calls}) == 1


def test_sync_recovery_resends_the_same_tool_answer(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=suspended(1))
    router.get(f"{SERVER}/v1/runs/r").respond(200, json=suspended(1))
    resume = router.post(f"{SERVER}/v1/runs/r/resume").mock(
        side_effect=[
            httpx.Response(409, json={"error": "run_not_suspended"}),
            httpx.Response(200, json=completed()),
        ]
    )
    seen: list[str] = []

    def answer(call: PendingCall) -> ToolResult:
        seen.append(call.call_id)
        return ToolResult.success(call.call_id, len(seen))

    with Submilli(base_url=SERVER) as client:
        assert client.sessions.run("s", "code", on_tool_call=answer).result == "done"
    assert seen == ["r:effect-1"]
    assert resume.calls[0].request.content == resume.calls[1].request.content


async def test_recovery_cannot_change_the_arguments_of_an_answered_call(router: respx.MockRouter) -> None:
    router.post(f"{SERVER}/v1/sessions/s/execute").respond(200, json=suspended(1))
    changed = suspended(1)
    changed["calls"][0]["args"] = {"value": 99}
    router.get(f"{SERVER}/v1/runs/r").respond(200, json=changed)
    router.post(f"{SERVER}/v1/runs/r/resume").respond(409, json={"error": "run_not_suspended"})
    async with AsyncSubmilli(base_url=SERVER) as client:
        with pytest.raises(SubmilliError, match="changed the tool or arguments"):
            await client.sessions.run("s", "code", on_tool_call=lambda c: ToolResult.success(c.call_id, 1))


async def test_an_expired_idempotency_response_raises_without_reexecution(router: respx.MockRouter) -> None:
    execute = router.post(f"{SERVER}/v1/sessions/s/execute").respond(
        409, json={"error": "idempotency_response_expired", "execution_id": "r"}
    )
    async with AsyncSubmilli(base_url=SERVER) as client:
        with pytest.raises(IdempotencyError) as raised:
            await client.sessions.run("s", "code", idempotency_key="original-key")
    assert raised.value.code == "idempotency_response_expired" and execute.call_count == 1


async def test_concurrent_runs_keep_polling_and_callback_answers_isolated() -> None:
    positions: Counter[str] = Counter()
    handlers: Counter[str] = Counter()
    submitted: Counter[str] = Counter()

    def transport(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        if parts[-1] == "execute":
            run = json.loads(request.content)["code"]
            return httpx.Response(200, json=view("queued", run))
        run = parts[3]
        if parts[-1] == "resume":
            [answer] = json.loads(request.content)["results"]
            assert answer["call_id"] == f"{run}:effect-{submitted[run] + 1}"
            assert answer["value"] == {"run": run, "sequence": submitted[run] + 1}
            submitted[run] += 1
            return httpx.Response(200, json=view("running", run))
        positions[run] += 1
        if positions[run] % 3:
            body = view("queued" if positions[run] % 2 else "running", run)
        elif submitted[run] < 8:
            body = suspended(submitted[run] + 1, run)
        else:
            body = completed(run)
        return httpx.Response(200, json=body)

    async def answer(call: PendingCall) -> ToolResult:
        handlers[call.execution_id] += 1
        await asyncio.sleep(0)
        return ToolResult.success(call.call_id, {"run": call.execution_id, "sequence": call.args["value"]})

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http,
        AsyncSubmilli(base_url=SERVER, http_client=http) as client,
    ):

        async def run(number: int) -> ExecuteResult:
            state = await client.sessions.execute("s", f"r-{number}", idempotency_key=f"k-{number}")
            return await client.runs.complete(state, on_tool_call=answer, poll_interval=0)

        results = await asyncio.wait_for(asyncio.gather(*(run(i) for i in range(64))), timeout=20)
    assert len({result.execution_id for result in results}) == 64
    assert all(result.result == "done" for result in results)
    assert set(handlers.values()) == set(submitted.values()) == {8}
    assert set(positions.values()) == {27}
