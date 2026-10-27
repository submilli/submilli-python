"""`SubmilliTools` against a fake of the HTTP API, with a scripted model."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
import respx
from agents import Agent, Runner, RunResult, UserError, set_tracing_disabled
from agents.items import ToolCallOutputItem

from openai_agents_submilli import (
    SQLiteSessionStore,
    SubmilliConversationError,
    SubmilliSessionMismatchError,
    SubmilliTools,
)
from submilli import AuthenticationError, SubmilliError, UnknownBlueprintError
from tests.scripted import EXECUTE, Scripted, call, calls_then_done, execute, scripted
from tests.unit_tests.fake_server import SERVER, TOKEN, FakeServer

set_tracing_disabled(True)

ADA = {"user_id": "u_ada", "conversation_id": "c_1"}


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


def toolkit(**kwargs: Any) -> SubmilliTools:
    return SubmilliTools(blueprint="research", server=SERVER, token=TOKEN, **kwargs)


async def run(submilli: SubmilliTools, model: Scripted, context: Any = None, *, question: str = "go") -> RunResult:
    agent = Agent(name="assistant", model=model, tools=list(submilli.tools))
    return await Runner.run(agent, question, context=ADA if context is None else context)


def outputs(result: RunResult) -> list[ToolCallOutputItem]:
    return [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]


def content(item: ToolCallOutputItem) -> Any:
    return json.loads(item.output)


@contextmanager
def raises_from_tool(error: type[Exception], match: str | None = None) -> Iterator[None]:
    """The SDK raises a tool's exception as the cause of a `UserError`."""
    with pytest.raises(UserError) as raised:
        yield
    assert isinstance(raised.value.__cause__, error), repr(raised.value.__cause__)
    if match is not None:
        assert re.search(match, str(raised.value.__cause__))


async def test_execute_opens_a_session_bound_to_the_user_and_sends_the_tool_call_id(server: FakeServer) -> None:
    result = await run(toolkit(), calls_then_done(execute("main()", id="call_abc")))
    [output] = outputs(result)
    assert content(output) == {"result": "ok", "console": [], "error": None, "denied": False}
    [session] = server.sessions.values()
    assert session.variables == {"userId": "u_ada"}
    [request] = server.calls("POST", r"/v1/sessions/[^/]+/execute")
    assert request.headers["idempotency-key"] == "call_abc"
    assert output.custom_data is not None
    assert output.custom_data["session_id"] in server.sessions
    assert output.custom_data["execution_id"]


async def test_tools_carry_the_blueprints_descriptions_fetched_once(server: FakeServer) -> None:
    submilli = toolkit()
    assert [tool.name for tool in submilli.tools] == [
        EXECUTE,
        "submilli__typescript__last_run",
        "submilli__typescript__packages__search",
        "submilli__typescript__packages__docs",
        "submilli__typescript__builtins__list",
        "submilli__typescript__builtins__docs",
    ]
    model = scripted("hi")
    await run(submilli, model)
    await run(submilli, scripted("again"))
    described = model.descriptions()
    assert described[EXECUTE].startswith("Run a submilli program")
    assert described["submilli__typescript__last_run"] == "The last run."
    assert server.prompt_fetches == 1


async def test_a_conversation_keeps_its_session_across_turns(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a")))
    await run(submilli, calls_then_done(execute("b")))
    assert len(server.sessions) == 1
    await run(submilli, calls_then_done(execute("c")), {"user_id": "u_ada", "conversation_id": "c_2"})
    assert len(server.sessions) == 2


async def test_parallel_calls_share_one_session(server: FakeServer) -> None:
    await run(toolkit(), calls_then_done(execute("a"), execute("b"), execute("c")))
    assert len(server.sessions) == 1


async def test_different_variables_in_a_conversation_are_refused_before_the_model_runs(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a")))
    model = scripted("never")
    with pytest.raises(SubmilliSessionMismatchError, match="differing: userId"):
        await run(submilli, model, {"user_id": "u_grace", "conversation_id": "c_1"})
    assert model.seen == []


async def test_a_context_without_a_conversation_fails_clearly(server: FakeServer) -> None:
    with raises_from_tool(SubmilliConversationError, match="conversation_id"):
        await run(toolkit(), calls_then_done(execute("a")), {"user_id": "u_ada"})


async def test_a_custom_conversation_id(server: FakeServer) -> None:
    @dataclass
    class Context:
        user_id: str
        thread: str

    submilli = toolkit(conversation_id=lambda context: context.thread)
    await run(submilli, calls_then_done(execute("a")), Context("u_ada", "t_1"))
    await run(submilli, calls_then_done(execute("b")), Context("u_ada", "t_1"))
    assert len(server.sessions) == 1
    assert await submilli.store.get("t_1") is not None


async def test_discovery_tools_do_not_open_a_session(server: FakeServer) -> None:
    model = calls_then_done(
        call("submilli__typescript__packages__search", {"query": "jina"}),
        call("submilli__typescript__packages__docs", {"name": "@submilli/jina"}),
        call("submilli__typescript__builtins__list", {}),
        call("submilli__typescript__builtins__docs", {"names": ["Map"]}),
        call("submilli__typescript__last_run", {}),
    )
    result = await run(toolkit(), model)
    texts = [item.output for item in outputs(result)]
    assert json.loads(texts[0])["results"][0]["name"] == "@submilli/jina"
    assert texts[1].startswith("# jina")
    assert texts[4] == "No program has run in this conversation yet."
    assert server.sessions == {}


async def test_last_run_reads_the_conversations_session(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a")))
    result = await run(submilli, calls_then_done(call("submilli__typescript__last_run", {})))
    assert json.loads(outputs(result)[0].output)["result"] == "ok"


async def test_permission_denied_is_marked_for_the_model(server: FakeServer) -> None:
    server.outcome = {
        "result": None,
        "console": [],
        "error": {"kind": "permission_denied", "message": "denied", "capability": "net.fetch", "source": "policy"},
    }
    result = await run(toolkit(), calls_then_done(execute("a")))
    reported = content(outputs(result)[0])
    assert reported["denied"] is True
    assert "final" in reported["note"]


async def test_an_ended_session_is_replaced_and_the_model_told(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a")))
    first = await submilli.store.get("c_1")
    assert first is not None
    server.sessions.clear()
    result = await run(submilli, calls_then_done(execute("b")))
    assert "new session" in content(outputs(result)[0])["notice"]
    second = await submilli.store.get("c_1")
    assert second is not None and second["id"] != first["id"]
    assert list(server.sessions) == [second["id"]]


async def test_lost_secrets_are_sent_again_with_rebind(server: FakeServer) -> None:
    server.required_secrets = ["apiKey"]
    submilli = toolkit(secrets=lambda context: {"apiKey": "k-" + context["user_id"]})
    await run(submilli, calls_then_done(execute("a")))
    [session_id] = server.sessions
    server.lost_secrets.add(session_id)
    result = await run(submilli, calls_then_done(execute("b")))
    assert content(outputs(result)[0])["result"] == "ok"
    assert server.sessions[session_id].secrets == {"apiKey": "k-u_ada"}
    stored = await submilli.store.get("c_1")
    assert stored is not None and "apiKey" not in json.dumps(stored)


@pytest.mark.parametrize(
    ("status", "code", "phrase"),
    [
        (409, "idempotency_conflict", "different code"),
        (409, "idempotency_incomplete", "side effects are unknown"),
        (409, "idempotency_in_progress", "still running"),
        (503, "idempotency_unavailable", "risk running it twice"),
    ],
)
async def test_idempotency_refusals_are_explained(server: FakeServer, status: int, code: str, phrase: str) -> None:
    server.refuse_next = [(status, {"error": code, "detail": "the detail"})]
    result = await run(toolkit(), calls_then_done(execute("a")))
    [output] = outputs(result)
    reported = content(output)
    assert reported["error"]["kind"] == code
    assert phrase in reported["error"]["message"]
    assert output.custom_data is not None and output.custom_data["refusal"] == code


async def test_a_durable_run_without_a_result_is_reported_not_raised(server: FakeServer) -> None:
    server.durable_next = [{"status": "stuck", "execution_id": "run-1", "reason": "worker crashed"}]
    result = await run(toolkit(), calls_then_done(execute("a")))
    [output] = outputs(result)
    reported = content(output)
    assert reported["error"]["kind"] == "run_stuck"
    assert reported["error"]["reason"] == "worker crashed"
    assert output.custom_data is not None and output.custom_data["run_status"] == "stuck"


async def test_the_same_tool_call_replays(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a", id="call_same")))
    await run(submilli, calls_then_done(execute("a", id="call_same")))
    assert server.total_runs() == 1


async def test_a_long_tool_call_id_is_hashed(server: FakeServer) -> None:
    await run(toolkit(), calls_then_done(execute("a", id="call_" + "x" * 200)))
    [request] = server.calls("POST", r"/v1/sessions/[^/]+/execute")
    assert len(request.headers["idempotency-key"]) == 64


async def test_a_dataclass_context_works(server: FakeServer) -> None:
    @dataclass
    class Context:
        user_id: str
        conversation_id: str

    await run(toolkit(), calls_then_done(execute("a")), Context("u_ada", "c_9"))
    [session] = server.sessions.values()
    assert session.variables == {"userId": "u_ada"}


async def test_custom_variables(server: FakeServer) -> None:
    await run(toolkit(variables={"userId": "fixed"}), calls_then_done(execute("a")))
    [session] = server.sessions.values()
    assert session.variables == {"userId": "fixed"}


async def test_variables_must_be_strings(server: FakeServer) -> None:
    with raises_from_tool(TypeError, match="must be strings"):
        await run(toolkit(variables=lambda context: {"userId": 7}), calls_then_done(execute("a")))


async def test_aclose_conversation_ends_the_session(server: FakeServer) -> None:
    submilli = toolkit()
    await run(submilli, calls_then_done(execute("a")))
    assert await submilli.aclose_conversation("c_1") is True
    assert server.sessions == {}
    assert await submilli.store.get("c_1") is None
    assert await submilli.aclose_conversation("c_1") is False
    await run(submilli, calls_then_done(execute("b")), {"user_id": "u_grace", "conversation_id": "c_1"})
    [session] = server.sessions.values()
    assert session.variables == {"userId": "u_grace"}


async def test_a_session_for_another_blueprint_or_server_is_not_reused(server: FakeServer) -> None:
    submilli = toolkit()
    await submilli.store.set(
        "c_1", {"id": "elsewhere", "server": "http://other", "blueprint": "research", "variables": {"userId": "u_ada"}}
    )
    await run(submilli, calls_then_done(execute("a")))
    stored = await submilli.store.get("c_1")
    assert stored is not None and stored["id"] in server.sessions


async def test_sessions_can_live_in_sqlite(server: FakeServer, tmp_path: Any) -> None:
    path = tmp_path / "conversations.db"
    await run(toolkit(store=SQLiteSessionStore(path)), calls_then_done(execute("a")))
    # Another process: the same database, a new toolkit.
    await run(toolkit(store=SQLiteSessionStore(path)), calls_then_done(execute("b")))
    assert len(server.sessions) == 1


async def test_server_and_token_come_from_the_environment(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBMILLI_SERVER", SERVER)
    monkeypatch.setenv("SUBMILLI_SERVER_TOKEN", TOKEN)
    await run(SubmilliTools(blueprint="research"), calls_then_done(execute("a")))
    assert len(server.sessions) == 1


async def test_application_errors_raise(server: FakeServer) -> None:
    with pytest.raises(AuthenticationError):
        await run(SubmilliTools(blueprint="research", server=SERVER, token="wrong"), scripted("hi"))
    # Found when the descriptions are fetched, before the model runs.
    with pytest.raises(UnknownBlueprintError):
        await run(SubmilliTools(blueprint="nope", server=SERVER, token=TOKEN), calls_then_done(execute("a")))
    with raises_from_tool(SubmilliError, match="userId"):
        await run(toolkit(), calls_then_done(execute("a")), {"conversation_id": "c_1"})
