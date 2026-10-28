"""Programs calling the application's own tools (`@tools`), against a fake of the HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
import respx
from agents import Agent, RunContextWrapper, Runner, RunResult, RunState, function_tool, set_tracing_disabled
from agents.items import ToolCallOutputItem
from agents.testing.model import ScriptedModel, assistant_message, function_call
from pydantic import BaseModel
from typing_extensions import TypedDict

from openai_agents_submilli import SubmilliTools
from openai_agents_submilli._app_tools import declaration
from tests.scripted import Scripted, call, calls_then_done, execute
from tests.unit_tests.fake_server import SERVER, TOKEN, FakeServer

set_tracing_disabled(True)

ADA = {"user_id": "u_ada", "conversation_id": "c_1"}


class Order(TypedDict):
    id: str
    total: float


class Customer(BaseModel):
    name: str
    orders: list[Order]


@function_tool
def lookup_order(order_id: str) -> Order:
    """Look up an order."""
    return {"id": order_id, "total": 12.5}


@function_tool
def customer(customer_id: str) -> Customer:
    """Look up a customer."""
    return Customer(name="Ada", orders=[{"id": "o_1", "total": 3.0}])


@function_tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"hello {name}"


@function_tool
def broken(order_id: str) -> str:
    """Always fails."""
    raise ValueError(f"no order {order_id}")


@function_tool
def whoami(wrapper: RunContextWrapper[Any]) -> str:
    """The run's user."""
    return str(wrapper.context["user_id"])


@function_tool(needs_approval=True)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


def toolkit(*ptc: Any) -> SubmilliTools:
    return SubmilliTools(blueprint="research", server=SERVER, token=TOKEN, ptc=list(ptc))


def agent(submilli: SubmilliTools, model: Scripted) -> Agent[Any]:
    return Agent(name="assistant", model=model, tools=list(submilli.tools))


def execute_output(result: RunResult) -> ToolCallOutputItem:
    [output] = [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    return output


def test_declarations_carry_input_and_output_schemas() -> None:
    order = declaration(lookup_order)
    assert order["name"] == "lookup_order"
    assert order.get("description") == "Look up an order."
    assert order["input_schema"]["properties"] == {"order_id": {"title": "Order Id", "type": "string"}}
    assert order.get("output_schema", {})["properties"]["total"] == {"title": "Total", "type": "number"}
    nested = declaration(customer).get("output_schema", {})
    assert "$defs" not in json.dumps(nested)
    assert nested["properties"]["orders"]["items"]["properties"]["id"] == {"title": "Id", "type": "string"}
    assert declaration(greet).get("output_schema") == {"type": "string"}


def test_ptc_rejects_names_programs_cannot_call() -> None:
    @function_tool(name_override="not-an-identifier")
    def dashed() -> str:
        return ""

    with pytest.raises(ValueError, match="identifier"):
        toolkit(dashed)
    with pytest.raises(ValueError, match="repeated: greet"):
        toolkit(greet, greet)
    with pytest.raises(TypeError, match="not tool names"):
        toolkit("greet")


async def test_a_program_calls_the_applications_tools(server: FakeServer) -> None:
    server.programs["main()"] = [("lookup_order", {"order_id": "o_7"}), ("customer", {"customer_id": "c_1"})]
    submilli = toolkit(lookup_order, customer)
    model = calls_then_done(execute("main()"))
    result = await Runner.run(agent(submilli, model), "go", context=ADA)
    output = execute_output(result)
    reported = json.loads(output.output)
    assert json.loads(reported["result"]) == [
        {"id": "o_7", "total": 12.5},
        {"name": "Ada", "orders": [{"id": "o_1", "total": 3.0}]},
    ]
    assert output.custom_data is not None
    assert [(c["tool"], c["outcome"]) for c in output.custom_data["tool_calls"]] == [
        ("lookup_order", "ok"),
        ("customer", "ok"),
    ]
    [session] = server.sessions.values()
    assert [tool["name"] for tool in session.tools] == ["lookup_order", "customer"]
    # The model sees the tools in the execute description, not as tools of its own.
    described = model.descriptions()
    assert "`lookup_order`, `customer`" in described["submilli__typescript__execute"]
    assert "lookup_order" not in described


async def test_a_tool_that_raises_fails_the_programs_call(server: FakeServer) -> None:
    server.programs["main()"] = [("broken", {"order_id": "o_7"})]
    result = await Runner.run(agent(toolkit(broken), calls_then_done(execute("main()"))), "go", context=ADA)
    output = execute_output(result)
    reported = json.loads(output.output)
    assert reported["error"]["kind"] == "runtime_error"
    assert "ValueError: no order o_7" in reported["error"]["message"]
    assert output.custom_data is not None and output.custom_data["tool_calls"][0]["outcome"] == "failed"


async def test_the_tool_gets_the_runs_context(server: FakeServer) -> None:
    server.programs["main()"] = [("whoami", {})]
    result = await Runner.run(agent(toolkit(whoami), calls_then_done(execute("main()"))), "go", context=ADA)
    assert json.loads(json.loads(execute_output(result).output)["result"]) == ["u_ada"]


async def test_an_approval_interrupts_the_run_with_the_applications_tool(server: FakeServer) -> None:
    server.programs["main()"] = [("refund", {"order_id": "o_7", "amount": 18})]
    submilli = toolkit(refund)
    assistant = agent(submilli, calls_then_done(execute("main()")))
    result = await Runner.run(assistant, "go", context=ADA)
    [approval] = result.interruptions
    assert approval.tool_name == "refund"
    assert json.loads(approval.arguments or "{}") == {"order_id": "o_7", "amount": 18}
    state = result.to_state()
    state.approve(approval)
    result = await Runner.run(assistant, state)
    assert result.interruptions == []
    assert json.loads(json.loads(execute_output(result).output)["result"]) == ["refunded 18.0 on o_7"]
    assert server.total_runs() == 1


async def test_a_rejection_is_a_denial_in_the_program(server: FakeServer) -> None:
    server.programs["main()"] = [("refund", {"order_id": "o_7", "amount": 42})]
    assistant = agent(toolkit(refund), calls_then_done(execute("main()")))
    result = await Runner.run(assistant, "go", context=ADA)
    state = result.to_state()
    state.reject(result.interruptions[0], rejection_message="Refunds over $25 need a manager.")
    result = await Runner.run(assistant, state)
    output = execute_output(result)
    reported = json.loads(output.output)
    assert reported["denied"] is True
    assert reported["error"]["source"] == "harness"
    assert "need a manager" in reported["error"]["message"]
    assert output.custom_data is not None and output.custom_data["tool_calls"][0]["outcome"] == "denied"
    # The model sees the calls too, without the ids that are only for the audit trail.
    assert reported["tool_calls"] == [
        {"tool": "refund", "args": {"order_id": "o_7", "amount": 42}, "outcome": "denied"}
    ]


async def test_two_approvals_resume_from_a_serialized_state(server: FakeServer) -> None:
    server.programs["main()"] = [
        ("refund", {"order_id": "o_1", "amount": 5}),
        ("lookup_order", {"order_id": "o_2"}),
        ("refund", {"order_id": "o_2", "amount": 7}),
    ]
    submilli = toolkit(refund, lookup_order)
    assistant = agent(submilli, calls_then_done(execute("main()")))
    result = await Runner.run(assistant, "go", context=ADA)
    approved = []
    while result.interruptions:
        # As if in another process: only the serialized state and the store carry over.
        state = await RunState.from_string(assistant, result.to_state().to_string())
        for item in state.get_interruptions():
            approved.append(json.loads(item.arguments or "{}")["order_id"])
            state.approve(item)
        result = await Runner.run(assistant, state)
    assert approved == ["o_1", "o_2"]
    values = json.loads(json.loads(execute_output(result).output)["result"])
    assert values == ["refunded 5.0 on o_1", {"id": "o_2", "total": 12.5}, "refunded 7.0 on o_2"]
    assert server.total_runs() == 1


async def test_discovery_includes_the_sessions_tools(server: FakeServer) -> None:
    model = calls_then_done(
        call("submilli__typescript__packages__search", {"query": ""}),
        call("submilli__typescript__packages__docs", {"name": "@tools"}),
    )
    result = await Runner.run(agent(toolkit(greet), model), "go", context=ADA)
    texts = [item.output for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    assert "@tools" in [entry["name"] for entry in json.loads(texts[0])["results"]]
    assert "function greet" in texts[1]
    assert len(server.sessions) == 1


async def test_a_session_opened_with_other_tools_is_not_reused(server: FakeServer) -> None:
    first = toolkit(greet)
    await Runner.run(agent(first, calls_then_done(execute("a"))), "go", context=ADA)
    second = SubmilliTools(
        blueprint="research", server=SERVER, token=TOKEN, ptc=[greet, lookup_order], store=first.store
    )
    await Runner.run(agent(second, calls_then_done(execute("b"))), "go", context=ADA)
    assert len(server.sessions) == 2


async def test_plain_functions_are_wrapped(server: FakeServer) -> None:
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    server.programs["main()"] = [("double", {"n": 21})]
    result = await Runner.run(agent(toolkit(double), calls_then_done(execute("main()"))), "go", context=ADA)
    assert json.loads(json.loads(execute_output(result).output)["result"]) == [42]


async def test_approvals_work_in_a_streamed_run(server: FakeServer) -> None:
    server.programs["main()"] = [("refund", {"order_id": "o_7", "amount": 18})]
    model = ScriptedModel(
        [
            [function_call("submilli__typescript__execute", {"code": "main()"}, call_id="call_1")],
            [assistant_message("done")],
        ]
    )
    assistant = Agent(name="assistant", model=model, tools=list(toolkit(refund).tools))
    streamed = Runner.run_streamed(assistant, "go", context=ADA)
    async for _ in streamed.stream_events():
        pass
    [approval] = streamed.interruptions
    assert approval.tool_name == "refund"
    state = streamed.to_state()
    state.approve(approval)
    streamed = Runner.run_streamed(assistant, state)
    async for _ in streamed.stream_events():
        pass
    assert streamed.final_output == "done"
    [output] = [item for item in streamed.new_items if isinstance(item, ToolCallOutputItem)]
    assert json.loads(json.loads(output.output)["result"]) == ["refunded 18.0 on o_7"]
