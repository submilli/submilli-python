"""Crash recovery, queued admission and callback durability against a real server."""

from __future__ import annotations

import asyncio
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest

from submilli import (
    AsyncSubmilli,
    ExecuteResult,
    IdempotencyError,
    PendingCall,
    RunningRun,
    RunState,
    RunStateError,
    Submilli,
    SuspendedRun,
    ToolDeclaration,
    ToolResult,
)
from tests.integration.conftest import ManagedServer

pytestmark = [pytest.mark.integration, pytest.mark.durable, pytest.mark.parametrize("managed", [True], indirect=True)]

BLUEPRINT = """name: durable
default: allow
vfs: none
tool_call_timeout: 2m
"""

ECHO: ToolDeclaration = {
    "name": "echo",
    "input_schema": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
    "output_schema": {"type": "number"},
}
ONE = 'import tools from "@tools"; function main(): number { return tools.echo({value: 1}); }'
TWO = """import tools from "@tools";
function main(): number { const first = tools.echo({value: 1}); return first + tools.echo({value: first}); }
"""


async def next_call(client: AsyncSubmilli, state: RunState) -> SuspendedRun:
    async with asyncio.timeout(15):
        while isinstance(state, RunningRun):
            await asyncio.sleep(0.05)
            state = await client.runs.retrieve(state.execution_id)
    assert isinstance(state, SuspendedRun), state
    return state


async def test_unanswered_call_and_committed_prefix_survive_two_crashes(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        session = await client.sessions.create("durable", tools=[ECHO])
        first = await next_call(client, await client.sessions.execute(session.id, TWO, idempotency_key="k"))
        [call] = first.calls
        managed.crash()
        managed.start()
        recovered = await next_call(client, await client.runs.retrieve(first.execution_id))
        assert recovered == first
        assert await client.sessions.execute(session.id, TWO, idempotency_key="k") == first
        second = await next_call(
            client, await client.runs.resume(first.execution_id, [ToolResult.success(call.call_id, 6)])
        )
        assert second.calls[0].args == {"value": 6}
        assert second.calls[0].call_id != call.call_id
        managed.crash()
        managed.start()
        recovered = await next_call(client, await client.runs.retrieve(first.execution_id))
        assert recovered == second
        # A conflicting retry cannot replace a committed answer or answer the next call.
        assert await client.runs.resume(first.execution_id, [ToolResult.success(call.call_id, 99)]) == second
        done = await client.runs.complete(second, on_tool_call=lambda c: ToolResult.success(c.call_id, 7))
        assert done.error is None and done.result == "13"
        managed.restart()
        retained = await client.runs.retrieve(first.execution_id)
        assert retained == done
        assert await client.sessions.run(session.id, TWO, idempotency_key="k") == done


async def test_concurrent_duplicate_resumes_commit_one_answer(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        session = await client.sessions.create("durable", tools=[ECHO])
        first = await next_call(client, await client.sessions.execute(session.id, TWO, idempotency_key="k"))
        answer = ToolResult.success(first.calls[0].call_id, 3)
        duplicates = await asyncio.gather(*(client.runs.resume(first.execution_id, [answer]) for _ in range(16)))
        assert all(state == duplicates[0] for state in duplicates)
        second = await next_call(client, duplicates[0])
        assert second.calls[0].args == {"value": 3}
        result = await client.runs.complete(second, on_tool_call=lambda c: ToolResult.success(c.call_id, 4))
        assert result.result == "7" and result.error is None
        terminal = await asyncio.gather(*(client.runs.resume(first.execution_id, [answer]) for _ in range(8)))
        assert all(state == result for state in terminal)


async def test_queued_http_response_is_waited_to_completion(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token, timeout=45) as client:
        session = await client.sessions.create("durable", tools=[ECHO])
        blocker = await next_call(client, await client.sessions.execute(session.id, ONE, idempotency_key="first"))
        # The suspended run retains session admission. This POST must return
        # real queued metadata after the server's 25-second observation bound.
        queued = await client.sessions.execute(session.id, ONE, idempotency_key="second")
        assert isinstance(queued, RunningRun) and queued.status == "queued"
        assert queued.execution_id != blocker.execution_id
        answers: list[str] = []

        def answer(call: PendingCall) -> ToolResult:
            answers.append(call.call_id)
            return ToolResult.success(call.call_id, 9)

        waiting = asyncio.create_task(client.runs.complete(queued, on_tool_call=answer, poll_interval=0.01))
        try:
            await asyncio.sleep(0.2)
            assert not waiting.done() and not answers
            await client.runs.complete(blocker, on_tool_call=lambda c: ToolResult.success(c.call_id, 1))
            result = await asyncio.wait_for(waiting, timeout=15)
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        assert result.error is None and result.result == "9"
        assert len(answers) == 1 and answers[0].startswith(f"{queued.execution_id}:")
        assert await client.runs.retrieve(result.execution_id) == result


async def test_running_http_response_waits_and_retains_the_native_outcome(managed: ManagedServer) -> None:
    entered = Event()
    release = Event()
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            entered.set()
            if not release.wait(timeout=45):
                self.send_error(504)
                return
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"done")

        def log_message(self, format: str, *args: object) -> None:
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        managed.stop()
        config = managed.home / "server.yaml"
        config.write_text(config.read_text() + "network:\n  allow_localhost: true\n")
        managed.start()
        managed.apply("durable", BLUEPRINT + "allow_insecure_http: true\n")
        code = (
            'import { get } from "submilli:http"; '
            'function main(): string { return get("http://127.0.0.1:'
            f'{upstream.server_port}/slow").body; }}'
        )
        async with AsyncSubmilli(base_url=managed.url, token=managed.token, timeout=40) as client:
            session = await client.sessions.create("durable")
            state = await client.sessions.execute(session.id, code, idempotency_key="slow-http")
            if isinstance(state, ExecuteResult):
                pytest.fail(str(state.error))
            assert entered.is_set(), state
            assert isinstance(state, RunningRun) and state.status == "running", state
            waiting = asyncio.create_task(client.runs.complete(state, poll_interval=0.01))
            try:
                await asyncio.sleep(0.1)
                assert not waiting.done()
                release.set()
                done = await asyncio.wait_for(waiting, timeout=10)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            assert done.result == "done" and done.error is None
            managed.restart()
            assert await client.runs.retrieve(done.execution_id) == done
            assert await client.sessions.run(session.id, code, idempotency_key="slow-http") == done
            assert hits == ["/slow"]
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


async def test_many_sessions_and_many_tool_calls_remain_isolated(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    source = """import tools from "@tools";
function main(): number {
    let total = 0;
    for (let i = 0; i < 5; i++) { total += tools.echo({value: i}); }
    return total;
}
"""
    seen: Counter[str] = Counter()
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:

        async def run(number: int) -> ExecuteResult:
            session = await client.sessions.create("durable", tools=[ECHO])

            async def answer(call: PendingCall) -> ToolResult:
                seen[call.call_id] += 1
                await asyncio.sleep(0.01)
                return ToolResult.success(call.call_id, number + call.args["value"])

            result = await client.sessions.run(session.id, source, idempotency_key="same-key", on_tool_call=answer)
            assert result.result == str(5 * number + 10) and result.error is None
            assert await client.sessions.run(session.id, source, idempotency_key="same-key") == result
            return result

        results = await asyncio.wait_for(asyncio.gather(*(run(i) for i in range(24))), timeout=60)
        assert len({result.execution_id for result in results}) == 24
    assert len(seen) == 120 and set(seen.values()) == {1}


@pytest.mark.parametrize("outcome", ["failed", "denied", "bad-schema"])
async def test_durable_tool_failures_remain_failures_after_restart(managed: ManagedServer, outcome: str) -> None:
    managed.apply("durable", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        session = await client.sessions.create("durable", tools=[ECHO])

        def answer(call: PendingCall) -> ToolResult:
            if outcome == "failed":
                return ToolResult.failed(call.call_id, "database down")
            if outcome == "denied":
                return ToolResult.denied(call.call_id, "user refused")
            return ToolResult.success(call.call_id, "not a number")

        result = await client.sessions.run(session.id, ONE, on_tool_call=answer)
        assert result.error is not None
        assert result.denied == (outcome == "denied")
        if outcome == "denied":
            assert result.error.source == "harness"
        managed.restart()
        assert await client.runs.retrieve(result.execution_id) == result


async def test_session_deletion_cancels_a_durable_tool_wait(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        session = await client.sessions.create("durable", tools=[ECHO])
        state = await next_call(client, await client.sessions.execute(session.id, ONE))
        assert await client.sessions.delete(session.id)
        for operation in (
            client.runs.retrieve(state.execution_id),
            client.runs.resume(state.execution_id, [ToolResult.success(state.calls[0].call_id, 1)]),
        ):
            with pytest.raises(RunStateError) as raised:
                await operation
            assert raised.value.status == "cancelled"


async def test_expired_result_is_reported_without_starting_another_run(managed: ManagedServer) -> None:
    managed.stop()
    config = managed.home / "server.yaml"
    config.write_text(config.read_text() + "idempotency_retention: 0\nidempotency_keep_last: 1\n")
    managed.start()
    managed.apply("durable", BLUEPRINT)
    code = "function main(): number { return 1; }"
    async with AsyncSubmilli(base_url=managed.url, token=managed.token) as client:
        session = await client.sessions.create("durable")
        first = await client.sessions.run(session.id, code, idempotency_key="first")
        assert first.error is None
        await client.sessions.run(session.id, code, idempotency_key="second")
        # Expiry is performed by the background reaper. Startup triggers its
        # first sweep without waiting for the normal reaper interval.
        managed.restart()
        async with asyncio.timeout(5):
            while True:
                try:
                    await client.runs.retrieve(first.execution_id)
                except RunStateError as expired:
                    assert expired.status == "expired"
                    break
                await asyncio.sleep(0.05)
        with pytest.raises(IdempotencyError) as refused:
            await client.sessions.run(session.id, code, idempotency_key="first")
        assert refused.value.code == "idempotency_response_expired"
        with pytest.raises(RunStateError) as raised:
            await client.runs.retrieve(first.execution_id)
        assert raised.value.status == "expired"


def test_sync_durable_tools_and_retained_results(managed: ManagedServer) -> None:
    managed.apply("durable", BLUEPRINT)
    with Submilli(base_url=managed.url, token=managed.token) as client:
        session = client.sessions.create("durable", tools=[ECHO])
        result = client.sessions.run(session.id, TWO, on_tool_call=lambda c: ToolResult.success(c.call_id, 2))
        assert result.result == "4" and result.error is None
        managed.restart()
        assert client.runs.retrieve(result.execution_id) == result
