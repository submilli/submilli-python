"""Programs calling the application's own tools (`@tools`), against a fake of the HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Annotated, Any, TypedDict, cast

import pytest
import respx
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from langchain_submilli import SubmilliMiddleware
from langchain_submilli._app_tools import declaration
from tests.scripted import Scripted, call, calls_then_done, execute
from tests.unit_tests.fake_server import SERVER, TOKEN, FakeServer


class Context(TypedDict):
    user_id: str


USER: Context = {"user_id": "u_ada"}


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


ran: list[tuple[str, Any]] = []


@pytest.fixture(autouse=True)
def clear_ran() -> None:
    ran.clear()


@tool
def query_orders(customer: str, runtime: ToolRuntime[dict[str, str]]) -> dict[str, int]:
    """Orders of one customer."""
    ran.append(("query_orders", {"customer": customer, "user": runtime.context["user_id"]}))
    return {"count": 3}


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    ran.append(("send_email", {"to": to, "body": body}))
    return f"sent to {to}"


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("database is down")


class FilesState(AgentState):
    files: dict[str, str]


@tool
def write_note(path: str, text: str, runtime: ToolRuntime) -> Command[Any]:
    """Write a note into the agent's files."""
    return Command(
        update={
            "files": {path: text},
            "messages": [ToolMessage(content=f"wrote {path}", tool_call_id=runtime.tool_call_id)],
        }
    )


def middleware(**kwargs: Any) -> SubmilliMiddleware:
    return SubmilliMiddleware("research", server=SERVER, token=TOKEN, **kwargs)


def agent(model: Scripted, submilli: SubmilliMiddleware, **kwargs: Any) -> Any:
    return create_agent(model=model, middleware=[submilli], checkpointer=InMemorySaver(), **kwargs)


THREAD: RunnableConfig = {"configurable": {"thread_id": "t"}}


async def start(graph: Any) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, THREAD, context=USER)
    )


async def decide(graph: Any, *decisions: dict[str, Any]) -> dict[str, Any]:
    [pending] = (await graph.aget_state(THREAD)).interrupts
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(Command(resume={pending.id: {"decisions": list(decisions)}}), THREAD, context=USER),
    )


def executed(result: dict[str, Any]) -> ToolMessage:
    return cast("ToolMessage", next(m for m in result["messages"] if m.type == "tool"))


# Declaring tools


def test_a_tool_is_declared_with_the_models_schema_and_its_return_type() -> None:
    declared = declaration(query_orders)
    assert declared["name"] == "query_orders"
    assert declared.get("description") == "Orders of one customer."
    # The injected runtime is not an argument a program passes.
    assert set(declared["input_schema"]["properties"]) == {"customer"}
    assert declared["input_schema"]["required"] == ["customer"]
    # A `dict[str, int]` is declared as it is; the server types it `unknown`.
    assert declared.get("output_schema") == {"type": "object", "additionalProperties": {"type": "integer"}}
    # Without a JSON return type, a tool returns its text.
    assert declaration(send_email).get("output_schema") == {"type": "string"}
    assert declaration(write_note).get("output_schema") == {"type": "string"}


class Line(BaseModel):
    sku: str
    quantity: int


class Order(BaseModel):
    id: str
    lines: list[Line]


@tool
def get_order(order: Annotated[str, "The order id"]) -> Order:
    """One order."""
    return Order(id=order, lines=[])


def test_nested_models_are_inlined_because_the_server_resolves_no_refs() -> None:
    output = declaration(get_order).get("output_schema", {})
    assert "$ref" not in json.dumps(output) and "$defs" not in output
    assert output["properties"]["lines"]["items"]["properties"]["quantity"] == {"title": "Quantity", "type": "integer"}


def test_exposed_tools_must_be_callable_by_name_once() -> None:
    @tool("send-email")
    def dashed(to: str) -> str:
        """Send."""
        return to

    with pytest.raises(ValueError, match="identifier"):
        middleware(ptc=[dashed])
    with pytest.raises(ValueError, match="distinct"):
        middleware(ptc=[send_email, send_email])
    with pytest.raises(TypeError, match="not tool names"):
        middleware(ptc=["send_email"])
    with pytest.raises(ValueError, match="not exposed"):
        middleware(ptc=[send_email], interrupt_on={"query_orders": True})


async def test_the_session_declares_the_tools_and_logs_what_programs_cannot_see(
    server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    server.open_warnings = ["@tools: blueprint `research` grants no `tools.call`, so programs cannot see these tools"]
    with caplog.at_level(logging.WARNING):
        await start(agent(calls_then_done(execute("plain")), middleware(ptc=[query_orders])))

    [opened] = server.calls("POST", "/v1/sessions")
    body = json.loads(opened.content)
    assert [item["name"] for item in body["tools"]] == ["query_orders"]
    assert "grants no `tools.call`" in caplog.text


async def test_the_execute_description_tells_the_model_about_tools(server: FakeServer) -> None:
    model = calls_then_done(execute("plain"))
    await start(agent(model, middleware(ptc=[query_orders, send_email], interrupt_on={"send_email": True})))
    [described] = [tool for tool in model.bound[-1] if tool.name == "submilli__typescript__execute"]
    assert '`import tools from "@tools";`' in described.description
    assert "`query_orders`, `send_email`" in described.description
    assert "approval" in described.description
    # Exposed tools are for programs; the model is not given them.
    assert "query_orders" not in model.tool_names()


# Running calls


async def test_a_program_call_runs_the_tool_with_the_runs_context_and_returns_its_value(server: FakeServer) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "c_42"})]
    result = await start(agent(calls_then_done(execute("orders", id="toolu_1")), middleware(ptc=[query_orders])))

    assert ran == [("query_orders", {"customer": "c_42", "user": "u_ada"})]
    [resumed] = server.calls("POST", r"/v1/runs/.*/resume")
    [answer] = json.loads(resumed.content)["results"]
    assert answer["ok"] is True and answer["value"] == {"count": 3}
    assert answer["call_id"].endswith(":1")

    message = executed(result)
    assert json.loads(message.text)["result"] == json.dumps([{"count": 3}])
    assert message.artifact["tool_calls"] == [
        {"call_id": answer["call_id"], "tool": "query_orders", "args": {"customer": "c_42"}, "outcome": "ok"}
    ]


async def test_a_tool_that_raises_fails_the_call_in_the_program(server: FakeServer) -> None:
    server.programs["boom"] = [("broken", {})]
    result = await start(agent(calls_then_done(execute("boom")), middleware(ptc=[broken])))

    [resumed] = server.calls("POST", r"/v1/runs/.*/resume")
    [answer] = json.loads(resumed.content)["results"]
    assert answer["ok"] is False and answer["error"]["kind"] == "failed"
    assert "database is down" in answer["error"]["message"]
    assert json.loads(executed(result).text)["error"]["kind"] == "runtime_error"


async def test_state_updates_of_called_tools_reach_the_agents_state(server: FakeServer) -> None:
    server.programs["notes"] = [
        ("write_note", {"path": "/a.md", "text": "A"}),
        ("write_note", {"path": "/b.md", "text": "B"}),
    ]
    submilli = middleware(ptc=[write_note])
    graph = agent(calls_then_done(execute("notes")), submilli, state_schema=FilesState)
    result = await start(graph)

    assert result["files"] == {"/a.md": "A", "/b.md": "B"}
    assert json.loads(executed(result).text)["result"] == json.dumps(["wrote /a.md", "wrote /b.md"])


async def test_a_run_the_server_lost_is_reported_without_raising(server: FakeServer) -> None:
    server.programs["orders"] = [("query_orders", {"customer": "c_42"})]
    original = server.resume

    def restart_then_resume(run_id: str, body: dict[str, Any]) -> Any:
        server.suspended.clear()
        return original(run_id, body)

    server.resume = restart_then_resume  # type: ignore[method-assign]
    result = await start(agent(calls_then_done(execute("orders")), middleware(ptc=[query_orders])))

    message = executed(result)
    assert json.loads(message.text)["error"]["kind"] == "run_lost"
    assert message.status == "error"
    assert message.artifact["refusal"] == "unknown_run"


async def test_a_changed_tool_catalog_opens_a_new_session(server: FakeServer) -> None:
    graph = agent(calls_then_done(execute("a")), middleware(ptc=[query_orders]))
    await start(graph)
    first = (await graph.aget_state(THREAD)).values["submilli_session"]
    assert first["tools"]

    # The same thread, served by a build that exposes another tool set.
    later = create_agent(
        model=calls_then_done(execute("b")),
        middleware=[middleware(ptc=[query_orders, send_email])],
        checkpointer=graph.checkpointer,
        context_schema=Context,
    )
    await later.ainvoke({"messages": [{"role": "user", "content": "again"}]}, THREAD, context=USER)
    second = (await later.aget_state(THREAD)).values["submilli_session"]
    assert second["id"] != first["id"]
    assert len(server.calls("POST", "/v1/sessions")) == 2


# Discovery


async def test_package_docs_include_the_sessions_tools(server: FakeServer) -> None:
    model = calls_then_done(call("submilli__typescript__packages__docs", {"name": "@tools"}))
    result = await start(agent(model, middleware(ptc=[query_orders])))

    assert server.calls("GET", r"/v1/sessions/.*/packages/docs")
    assert "function query_orders" in executed(result).content
    # Discovery of `@tools` needs the thread's session, so it opened one.
    assert len(server.calls("POST", "/v1/sessions")) == 1


async def test_without_exposed_tools_discovery_stays_on_the_blueprint(server: FakeServer) -> None:
    model = calls_then_done(call("submilli__typescript__packages__search", {"query": "x"}))
    await start(agent(model, middleware()))
    assert server.calls("GET", "/v1/blueprints/research/packages/search")
    assert not server.calls("POST", "/v1/sessions")


# Approvals


async def test_an_approved_call_runs_after_the_graph_resumes(server: FakeServer) -> None:
    server.programs["mail"] = [("send_email", {"to": "ada@example.com", "body": "hi"})]
    graph = agent(calls_then_done(execute("mail")), middleware(ptc=[send_email], interrupt_on={"send_email": True}))

    paused = await start(graph)
    [pending] = paused["__interrupt__"]
    assert pending.value["action_requests"][0]["name"] == "send_email"
    assert pending.value["action_requests"][0]["args"] == {"to": "ada@example.com", "body": "hi"}
    assert pending.value["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert ran == []

    result = await decide(graph, {"type": "approve"})
    assert ran == [("send_email", {"to": "ada@example.com", "body": "hi"})]
    message = executed(result)
    assert json.loads(message.text)["result"] == json.dumps(["sent to ada@example.com"])
    assert message.artifact["tool_calls"][0]["decision"] == "approve"
    # The run was started once and resumed, not run again.
    assert server.total_runs() == 1


async def test_a_rejected_call_is_a_denial_in_the_program(server: FakeServer) -> None:
    server.programs["mail"] = [("send_email", {"to": "ada@example.com", "body": "hi"})]
    graph = agent(calls_then_done(execute("mail")), middleware(ptc=[send_email], interrupt_on={"send_email": True}))
    await start(graph)
    result = await decide(graph, {"type": "reject", "message": "Not to customers."})

    assert ran == []
    [resumed] = server.calls("POST", r"/v1/runs/.*/resume")
    assert json.loads(resumed.content)["results"][0]["error"] == {"kind": "denied", "message": "Not to customers."}
    body = json.loads(executed(result).text)
    assert body["denied"] is True and body["error"]["source"] == "harness"


async def test_an_edit_runs_the_edited_arguments_and_respond_answers_for_the_tool(server: FakeServer) -> None:
    server.programs["mail"] = [("send_email", {"to": "a@x", "body": "hi"}), ("send_email", {"to": "b@x", "body": "yo"})]
    submilli = middleware(ptc=[send_email], interrupt_on={"send_email": {"allowed_decisions": ["edit", "respond"]}})
    graph = agent(calls_then_done(execute("mail")), submilli)
    await start(graph)
    await decide(
        graph, {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "a@x", "body": "hello"}}}
    )
    result = await decide(graph, {"type": "respond", "message": "skipped"})

    assert ran == [("send_email", {"to": "a@x", "body": "hello"})]
    assert json.loads(executed(result).text)["result"] == json.dumps(["sent to a@x", "skipped"])


async def test_two_approvals_in_one_program_each_get_their_own_decision(server: FakeServer) -> None:
    # The second pause re-runs the tool with both decisions stored, while the
    # server has moved past the first call. Each decision must reach its call.
    server.programs["mail"] = [
        ("send_email", {"to": "first@x", "body": "1"}),
        ("query_orders", {"customer": "c_1"}),
        ("send_email", {"to": "second@x", "body": "2"}),
    ]
    submilli = middleware(ptc=[send_email, query_orders], interrupt_on={"send_email": True})
    graph = agent(calls_then_done(execute("mail")), submilli)

    await start(graph)
    paused = await decide(graph, {"type": "approve"})
    assert paused["__interrupt__"][0].value["action_requests"][0]["args"]["to"] == "second@x"
    result = await decide(graph, {"type": "reject", "message": "no"})

    assert ran == [
        ("send_email", {"to": "first@x", "body": "1"}),
        ("query_orders", {"customer": "c_1", "user": "u_ada"}),
    ]
    message = executed(result)
    assert [(record["tool"], record["outcome"]) for record in message.artifact["tool_calls"]] == [
        ("send_email", "ok"),
        ("query_orders", "ok"),
        ("send_email", "denied"),
    ]
    assert json.loads(message.text)["denied"] is True


async def test_a_call_whose_approval_cannot_be_matched_is_refused_not_guessed(server: FakeServer) -> None:
    server.programs["mail"] = [("send_email", {"to": "a@x", "body": "1"}), ("send_email", {"to": "b@x", "body": "2"})]
    submilli = middleware(ptc=[send_email], interrupt_on={"send_email": True})
    graph = agent(calls_then_done(execute("mail")), submilli)
    await start(graph)
    await decide(graph, {"type": "approve"})
    # As when another process picks the thread up: the record of pauses is gone.
    submilli._exposed.approvals._runs.clear()
    result = await decide(graph, {"type": "approve"})

    assert ran == [("send_email", {"to": "a@x", "body": "1"})]
    [answer] = json.loads(server.calls("POST", r"/v1/runs/.*/resume")[-1].content)["results"]
    assert answer["error"]["kind"] == "denied"
    assert "could not be matched" in answer["error"]["message"]
    assert json.loads(executed(result).text)["denied"] is True


async def test_when_decides_which_calls_need_approval(server: FakeServer) -> None:
    server.programs["mail"] = [("send_email", {"to": "me@corp", "body": "1"})]
    submilli = middleware(
        ptc=[send_email],
        interrupt_on={
            "send_email": {
                "allowed_decisions": ["approve", "reject"],
                "when": lambda request: not request.tool_call["args"]["to"].endswith("@corp"),
            }
        },
    )
    result = await start(agent(calls_then_done(execute("mail")), submilli))
    assert "__interrupt__" not in result
    assert ran == [("send_email", {"to": "me@corp", "body": "1"})]


async def test_a_pydantic_result_reaches_the_program_as_an_object(server: FakeServer) -> None:
    server.programs["order"] = [("get_order", {"order": "o_1"})]
    await start(agent(calls_then_done(execute("order")), middleware(ptc=[get_order])))

    [resumed] = server.calls("POST", r"/v1/runs/.*/resume")
    assert json.loads(resumed.content)["results"][0]["value"] == {"id": "o_1", "lines": []}
    # The caller's tool object is left as it was.
    assert isinstance(get_order, StructuredTool) and get_order.func is not None
    assert get_order.func(order="o_2") == Order(id="o_2", lines=[])
