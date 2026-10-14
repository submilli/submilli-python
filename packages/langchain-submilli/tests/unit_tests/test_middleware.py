"""SubmilliMiddleware in an agent, against a fake of the HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import pytest
import respx
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_submilli import SubmilliMiddleware, SubmilliSessionMismatchError
from submilli import AuthenticationError, BadRequestError
from tests.scripted import EXECUTE, Scripted, call, calls_then_done, execute, scripted
from tests.unit_tests.fake_server import SERVER, TOKEN, FakeServer


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


def middleware(**kwargs: Any) -> SubmilliMiddleware:
    return SubmilliMiddleware("research", server=SERVER, token=TOKEN, **kwargs)


def agent(model: Scripted, submilli: SubmilliMiddleware, checkpointer: Any = None) -> Any:
    return create_agent(model=model, middleware=[submilli], checkpointer=checkpointer)


async def run(graph: Any, *, user: str = "u_ada", thread: str | None = None, context: Any = None) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}} if thread else None
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]},
            config,
            context=context if context is not None else {"user_id": user},
        ),
    )


def tool_message(result: dict[str, Any], index: int = 0) -> ToolMessage:
    return cast("ToolMessage", [m for m in result["messages"] if m.type == "tool"][index])


def content(message: ToolMessage) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(message.text))


async def test_execute_opens_a_session_bound_to_the_user_and_sends_the_tool_call_id(server: FakeServer) -> None:
    result = await run(agent(calls_then_done(execute("code", id="toolu_01")), middleware()))

    [opened] = server.calls("POST", "/v1/sessions")
    assert json.loads(opened.content) == {"blueprint": "research", "variables": {"userId": "u_ada"}}
    [executed] = server.calls("POST", r"/v1/sessions/.*/execute")
    assert executed.headers["idempotency-key"] == "toolu_01"
    assert json.loads(executed.content) == {"code": "code"}

    message = tool_message(result)
    assert content(message) == {"result": "ok", "console": [], "error": None, "denied": False}
    assert message.status == "success"
    # The execution id is for the application, not the model.
    assert set(message.artifact) == {"execution_id", "session_id"}
    assert message.artifact["execution_id"] not in message.content
    assert result["messages"][-1].content == "done"


async def test_tools_carry_the_blueprints_descriptions_fetched_once(server: FakeServer) -> None:
    submilli = middleware()
    first = calls_then_done(execute("a"))
    await run(agent(first, submilli))
    await run(agent(calls_then_done(execute("b")), submilli))

    names = first.tool_names()
    assert names == [
        "submilli__typescript__execute",
        "submilli__typescript__last_run",
        "submilli__typescript__packages__search",
        "submilli__typescript__packages__docs",
        "submilli__typescript__builtins__list",
        "submilli__typescript__builtins__docs",
    ]
    described = {tool.name: tool.description for tool in first.bound[-1]}
    assert "strict TypeScript subset" in described[EXECUTE]
    assert described["submilli__typescript__last_run"] == "The last run."
    assert server.prompt_fetches == 1


async def test_a_thread_keeps_its_session_across_turns(server: FakeServer) -> None:
    submilli = middleware()
    graph = agent(scripted([execute("a")], "one", [execute("b")], "two"), submilli, InMemorySaver())
    await run(graph, thread="t1")
    await run(graph, thread="t1")

    assert len(server.calls("POST", "/v1/sessions")) == 1
    sessions = {r.url.path.split("/")[3] for r in server.calls("POST", r"/v1/sessions/.*/execute")}
    assert len(sessions) == 1
    state = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values["submilli_session"]["id"] in sessions
    assert state.values["submilli_session"]["variables"] == {"userId": "u_ada"}
    # Ending the turn does not end the session.
    assert server.calls("DELETE", r"/v1/sessions/.*") == []


async def test_parallel_calls_share_one_session(server: FakeServer) -> None:
    await run(agent(calls_then_done(execute("a"), execute("b"), execute("c")), middleware()))
    assert len(server.calls("POST", "/v1/sessions")) == 1
    assert server.total_runs() == 3


async def test_different_variables_on_a_thread_are_refused(server: FakeServer) -> None:
    graph = agent(scripted([execute("a")], "one", [execute("b")], "two"), middleware(), InMemorySaver())
    await run(graph, thread="t1", user="u_ada")
    with pytest.raises(SubmilliSessionMismatchError, match="differing: userId"):
        await run(graph, thread="t1", user="u_grace")
    assert server.total_runs() == 1
    assert len(server.calls("POST", "/v1/sessions")) == 1


async def test_discovery_tools_do_not_open_a_session(server: FakeServer) -> None:
    model = calls_then_done(
        call("submilli__typescript__packages__search", {"query": "jina"}),
        call("submilli__typescript__packages__docs", {"name": "@submilli/jina"}),
        call("submilli__typescript__packages__docs", {"name": "nope"}),
        call("submilli__typescript__builtins__list", {}),
        call("submilli__typescript__builtins__docs", {"names": ["Map", "JSON"]}),
        call("submilli__typescript__last_run", {}),
    )
    result = await run(agent(model, middleware()))

    outputs = [m.content for m in result["messages"] if m.type == "tool"]
    assert json.loads(outputs[0])["results"][0]["name"] == "@submilli/jina"
    assert "searchJson" in outputs[1]
    assert json.loads(outputs[2])["error"] == "unknown_package"
    assert json.loads(outputs[3])["types"] == ["Map"]
    assert [r["name"] for r in json.loads(outputs[4])["results"]] == ["Map", "JSON"]
    assert outputs[5] == "No program has run in this thread yet."
    assert server.calls("POST", "/v1/sessions") == []
    [docs] = [r for r in server.calls("GET", r".*/builtins/docs")]
    assert docs.url.params.get_list("name") == ["Map", "JSON"]


async def test_last_run_reads_the_threads_session(server: FakeServer) -> None:
    server.outcome = {"result": None, "console": [], "error": None}
    model = scripted([execute("a")], [call("submilli__typescript__last_run", {})], "done")
    result = await run(agent(model, middleware()))
    assert json.loads(tool_message(result, 1).text) == {"result": None, "console": [], "error": None}


async def test_permission_denied_is_marked_for_the_model(server: FakeServer) -> None:
    server.outcome = {
        "result": None,
        "console": [],
        "error": {
            "kind": "permission_denied",
            "message": "permission denied",
            "caller": "main",
            "capability": "example.com/op",
            "source": "policy",
        },
    }
    message = tool_message(await run(agent(calls_then_done(execute("a")), middleware())))
    body = content(message)
    assert body["denied"] is True
    assert body["error"]["capability"] == "example.com/op"
    assert "do not try another route" in body["note"]
    assert message.status == "error"


async def test_a_runtime_error_is_not_a_denial(server: FakeServer) -> None:
    server.outcome = {"result": None, "console": ["before"], "error": {"kind": "runtime_error", "message": "boom"}}
    body = content(tool_message(await run(agent(calls_then_done(execute("a")), middleware()))))
    assert body["denied"] is False
    assert "note" not in body
    assert body["console"] == ["before"]


async def test_an_ended_session_is_replaced_and_the_model_told(server: FakeServer) -> None:
    graph = agent(scripted([execute("a")], "one", [execute("b"), execute("c")], "two"), middleware(), InMemorySaver())
    await run(graph, thread="t1")
    config = {"configurable": {"thread_id": "t1"}}
    old = (await graph.aget_state(config)).values["submilli_session"]["id"]
    server.sessions.pop(old)  # as the idle timeout does

    result = await run(graph, thread="t1")

    replaced = [content(m) for m in result["messages"][-3:-1]]
    assert all("new session was opened" in body["notice"] for body in replaced)
    new = (await graph.aget_state(config)).values["submilli_session"]["id"]
    assert new != old
    # The two parallel calls that found it gone opened one replacement.
    assert len(server.calls("POST", "/v1/sessions")) == 2
    assert server.sessions[new].runs == 2


async def test_lost_secrets_are_sent_again_with_rebind(server: FakeServer) -> None:
    server.required_secrets = ["USER_TOKEN"]
    submilli = middleware(secrets=lambda context: {"USER_TOKEN": context["token"]})
    graph = agent(scripted([execute("a")], "one", [execute("b")], "two"), submilli, InMemorySaver())
    context = {"user_id": "u_ada", "token": "tok_1"}
    await run(graph, thread="t1", context=context)
    [opened] = server.calls("POST", "/v1/sessions")
    assert json.loads(opened.content)["secrets"] == {"USER_TOKEN": "tok_1"}
    session_id = next(iter(server.sessions))
    server.lost_secrets.add(session_id)  # as after a server restart

    result = await run(graph, thread="t1", context={**context, "token": "tok_2"})

    [rebind] = server.calls("POST", r"/v1/sessions/.*/rebind")
    assert json.loads(rebind.content) == {"secrets": {"USER_TOKEN": "tok_2"}}
    assert content(tool_message(result, 1))["result"] == "ok"
    assert server.sessions[session_id].runs == 2
    # Secrets are never kept in the agent's state.
    state = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert "tok_" not in json.dumps(state.values["submilli_session"])


@pytest.mark.parametrize(
    ("status", "code", "phrase"),
    [
        (409, "idempotency_conflict", "different code"),
        (409, "idempotency_incomplete", "side effects are unknown"),
        (503, "idempotency_in_progress", "still running"),
        (503, "idempotency_unavailable", "could not read or write"),
    ],
)
async def test_idempotency_refusals_are_explained(server: FakeServer, status: int, code: str, phrase: str) -> None:
    server.refuse_next.append((status, {"error": code, "detail": "server detail", "execution_id": "e-1"}))
    message = tool_message(await run(agent(calls_then_done(execute("a")), middleware())))
    body = content(message)
    assert body["error"]["kind"] == code
    assert phrase in body["error"]["message"]
    assert body["error"]["message"].startswith("Nothing")
    assert message.status == "error"
    assert message.artifact["refusal"] == code
    assert message.artifact["execution_id"] == "e-1"


async def test_incomplete_tells_the_model_to_check_before_retrying(server: FakeServer) -> None:
    server.refuse_next.append((409, {"error": "idempotency_incomplete", "detail": "never recorded"}))
    body = content(tool_message(await run(agent(calls_then_done(execute("a")), middleware()))))
    assert "Check what it would have changed" in body["error"]["message"]


async def test_the_same_tool_call_replays(server: FakeServer) -> None:
    submilli = middleware()
    graph = agent(
        scripted([execute("a", id="same")], "one", [execute("a", id="same")], "two"), submilli, InMemorySaver()
    )
    first = await run(graph, thread="t1")
    second = await run(graph, thread="t1")
    assert server.total_runs() == 1
    assert tool_message(first).artifact["execution_id"] == tool_message(second, 1).artifact["execution_id"]


async def test_a_long_tool_call_id_is_hashed(server: FakeServer) -> None:
    await run(agent(calls_then_done(execute("a", id="x" * 200)), middleware()))
    [executed] = server.calls("POST", r"/v1/sessions/.*/execute")
    key = executed.headers["idempotency-key"]
    assert len(key) == 64 and key != "x" * 64


async def test_a_context_without_the_required_variable_fails_clearly(server: FakeServer) -> None:
    with pytest.raises(BadRequestError, match="required variable 'userId'"):
        await run(agent(calls_then_done(execute("a")), middleware()), context={})


async def test_a_dataclass_context_works(server: FakeServer) -> None:
    from dataclasses import dataclass

    @dataclass
    class Context:
        user_id: str

    await run(agent(calls_then_done(execute("a")), middleware()), context=Context("u_dc"))
    assert next(iter(server.sessions.values())).variables == {"userId": "u_dc"}


async def test_custom_variables(server: FakeServer) -> None:
    submilli = middleware(variables=lambda context: {"userId": context["account"], "region": "eu"})
    await run(agent(calls_then_done(execute("a")), submilli), context={"account": "acct_7"})
    assert next(iter(server.sessions.values())).variables == {"userId": "acct_7", "region": "eu"}


async def test_aclose_thread_ends_the_session(server: FakeServer) -> None:
    submilli = middleware()
    graph = agent(scripted([execute("a")], "one", [execute("b")], "two"), submilli, InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    await run(graph, thread="t1")
    old = (await graph.aget_state(config)).values["submilli_session"]["id"]

    assert await submilli.aclose_thread(graph, config) is True
    assert old not in server.sessions
    assert (await graph.aget_state(config)).values["submilli_session"] is None
    assert await submilli.aclose_thread(graph, config) is False

    # The next call opens a new session, and may bind another user.
    await run(graph, thread="t1", user="u_grace")
    assert [s.variables for s in server.sessions.values()] == [{"userId": "u_grace"}]


async def test_server_and_token_come_from_the_environment(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBMILLI_SERVER", SERVER + "/")
    monkeypatch.setenv("SUBMILLI_SERVER_TOKEN", TOKEN)
    submilli = SubmilliMiddleware(blueprint="research")
    assert submilli.client.base_url == SERVER
    await run(agent(calls_then_done(execute("a")), submilli))
    assert server.total_runs() == 1


async def test_a_wrong_token_fails_clearly(server: FakeServer) -> None:
    submilli = SubmilliMiddleware("research", server=SERVER, token="wrong")
    with pytest.raises(AuthenticationError, match="401"):
        await run(agent(calls_then_done(execute("a")), submilli))
