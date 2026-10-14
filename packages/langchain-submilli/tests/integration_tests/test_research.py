"""SubmilliMiddleware against a real submilli-server, with a scripted model.

No model API key is needed: the model is scripted, as in submilli-runtime's
`check.py`, and the programs only touch the user's notes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, cast

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_submilli import SubmilliMiddleware, SubmilliSessionMismatchError
from tests.integration_tests.conftest import EXAMPLE, ManagedServer
from tests.scripted import EXECUTE, Scripted, execute, scripted

pytestmark = pytest.mark.integration

NOTE = (EXAMPLE / "note.ts").read_text()

# Appends a line to the user's notes and returns how many lines there are, so a
# second run shows up as a second line.
APPEND = """
import fs from "submilli:fs";

function main(): string {
    fs.appendText("runs.txt", "run\\n");
    let count = 0;
    for (const line of fs.lines("runs.txt")) if (line.length > 0) count++;
    return `${count}`;
}
"""

COUNT = """
import fs from "submilli:fs";

function main(): string {
    let count = 0;
    if (fs.exists("runs.txt")) for (const line of fs.lines("runs.txt")) if (line.length > 0) count++;
    return `${count}`;
}
"""

DENIED = """
import { check } from "submilli:security";

function main(): void { check("example.com/op", {}); }
"""


def unique(prefix: str) -> str:
    return f"{prefix}_{time.time_ns()}_{uuid.uuid4().hex[:6]}"


def deep_agent(model: Scripted, submilli: SubmilliMiddleware, checkpointer: Any = None) -> Any:
    instructions = (EXAMPLE / "prompt.txt").read_text()
    return create_deep_agent(model=model, system_prompt=instructions, middleware=[submilli], checkpointer=checkpointer)


async def ask(graph: Any, user: str, thread: str | None = None, **context: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}} if thread else None
    return cast(
        "dict[str, Any]",
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]}, config, context={"user_id": user, **context}
        ),
    )


def outputs(result: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if m.type == "tool"]


def body(message: ToolMessage) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(message.text))


@pytest.fixture
def submilli(research: tuple[str, str]) -> SubmilliMiddleware:
    server, token = research
    return SubmilliMiddleware("research", server=server, token=token, hide_filesystem_tools=True)


async def test_two_users_in_two_threads_see_only_their_own_notes(submilli: SubmilliMiddleware) -> None:
    ada, grace = unique("ada"), unique("grace")

    async def note(user: str, thread: str) -> str:
        model = scripted([execute(NOTE)], "done")
        [message] = outputs(await ask(deep_agent(model, submilli), user, thread))
        assert "strict TypeScript subset" in next(t for t in model.bound[-1] if t.name == EXECUTE).description
        return cast("str", body(message)["result"])

    assert await note(ada, "a1") == "notes before: none"
    assert await note(grace, "g1") == "notes before: none"
    assert await note(ada, "a2") == "notes before: check.md"


async def test_the_same_tool_call_runs_once(submilli: SubmilliMiddleware) -> None:
    user, call_id = unique("idem"), unique("toolu")
    model = scripted(
        [execute(APPEND, id=call_id)], "one", [execute(APPEND, id=call_id)], "two", [execute(COUNT)], "three"
    )
    graph = deep_agent(model, submilli, InMemorySaver())

    first = outputs(await ask(graph, user, "t"))[0]
    second = outputs(await ask(graph, user, "t"))[1]
    count = outputs(await ask(graph, user, "t"))[2]

    assert body(first)["result"] == body(second)["result"] == "1"
    assert first.artifact["execution_id"] == second.artifact["execution_id"]
    assert body(count)["result"] == "1"


async def test_a_resumed_thread_reuses_its_session_and_does_not_rerun(
    research: tuple[str, str], submilli: SubmilliMiddleware
) -> None:
    user, saver = unique("resume"), InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    graph = deep_agent(scripted([execute(APPEND)], "done"), submilli, saver)
    [ran] = outputs(await ask(graph, user, "t"))
    session = (await graph.aget_state(config)).values["submilli_session"]

    # A new process: a new middleware and agent over the same checkpoints. It
    # resumes from the checkpoint taken just before the tools ran, as after a
    # crash mid-call, so the same tool call runs again.
    server, token = research
    restarted = SubmilliMiddleware("research", server=server, token=token, hide_filesystem_tools=True)
    graph = deep_agent(scripted("done again"), restarted, saver)
    history = [snapshot async for snapshot in graph.aget_state_history(config)]
    before_tools = next(snapshot for snapshot in history if snapshot.next == ("tools",))
    assert before_tools.values["submilli_session"] == session
    resumed = await graph.ainvoke(None, before_tools.config, context={"user_id": user})

    replayed = outputs(resumed)[-1]
    assert replayed.tool_call_id == ran.tool_call_id
    assert replayed.artifact == ran.artifact
    assert body(replayed)["result"] == "1"
    assert (await graph.aget_state(config)).values["submilli_session"] == session

    # The program ran once: one line in the user's notes.
    count = deep_agent(scripted([execute(COUNT)], "done"), restarted)
    assert body(outputs(await ask(count, user))[0])["result"] == "1"


async def test_different_variables_on_a_thread_are_refused(submilli: SubmilliMiddleware) -> None:
    graph = deep_agent(scripted([execute(NOTE)], "one", [execute(NOTE)], "two"), submilli, InMemorySaver())
    await ask(graph, unique("ada"), "t")
    with pytest.raises(SubmilliSessionMismatchError):
        await ask(graph, unique("grace"), "t")


async def test_a_denied_call_comes_back_marked(submilli: SubmilliMiddleware) -> None:
    [message] = outputs(await ask(deep_agent(scripted([execute(DENIED)], "done"), submilli), unique("deny")))
    result = body(message)
    assert result["denied"] is True
    assert result["error"]["kind"] == "permission_denied"
    assert result["error"]["capability"] == "example.com/op"
    assert result["error"]["source"] == "policy"
    assert message.artifact["execution_id"]


async def test_a_failed_program_is_not_a_denial(submilli: SubmilliMiddleware) -> None:
    program = 'function main(): void { console.log("before"); throw new Error("boom"); }'
    [message] = outputs(await ask(deep_agent(scripted([execute(program)], "done"), submilli), unique("fail")))
    result = body(message)
    assert result["denied"] is False
    assert result["error"]["kind"] == "runtime_error"
    assert result["console"] == ["before"]


async def test_discovery_tools(submilli: SubmilliMiddleware) -> None:
    from tests.scripted import call

    model = scripted(
        [
            call("submilli__typescript__packages__search", {"query": "jina"}),
            call("submilli__typescript__packages__docs", {"name": "@submilli/jina"}),
            call("submilli__typescript__builtins__list", {}),
            call("submilli__typescript__builtins__docs", {"names": ["Map", "Temporal.Instant"]}),
        ],
        [execute('function main(): void { console.log("seen"); }')],
        [call("submilli__typescript__last_run", {})],
        "done",
    )
    found, docs, listed, builtins, _, last = outputs(await ask(deep_agent(model, submilli), unique("disc")))
    assert any(hit["name"] == "@submilli/jina" for hit in json.loads(found.text)["results"])
    assert "searchJson" in docs.content
    assert "Map" in json.loads(listed.text)["types"]
    assert len(json.loads(builtins.text)["results"]) == 2
    assert json.loads(last.text)["console"] == ["seen"]


# Restart and expiry, on a server these tests control.


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


def scratch_agent(managed: ManagedServer, blueprint: str, saver: InMemorySaver) -> Any:
    submilli = SubmilliMiddleware(
        blueprint,
        server=managed.url,
        token=managed.token,
        secrets=lambda context: {"USER_TOKEN": context["token"]},
        hide_filesystem_tools=True,
    )
    model = scripted([execute(SCRATCH)], "one", [execute(SCRATCH)], "two")
    return create_deep_agent(model=model, middleware=[submilli], checkpointer=saver)


async def test_a_server_restart_keeps_the_session(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("scratch"))
    graph, config = scratch_agent(managed, "scratch", InMemorySaver()), {"configurable": {"thread_id": "t"}}
    first = outputs(await ask(graph, "u_ada", "t", token="tok_1"))[0]
    assert body(first)["result"] == "scratch was not there"
    session = (await graph.aget_state(config)).values["submilli_session"]

    # The server keeps sessions, and their harness secrets sealed with its
    # secret-store key, across a restart, so no rebind is needed.
    managed.restart()
    second = outputs(await ask(graph, "u_ada", "t", token="tok_2"))[1]

    assert body(second)["result"] == "scratch was there"
    assert "notice" not in body(second)
    assert (await graph.aget_state(config)).values["submilli_session"] == session


async def test_an_expired_session_is_replaced_and_the_model_told(managed: ManagedServer) -> None:
    managed.apply(scratch_blueprint("expiring", idle_timeout="1s"))
    graph, config = scratch_agent(managed, "expiring", InMemorySaver()), {"configurable": {"thread_id": "t"}}
    await ask(graph, "u_ada", "t", token="tok")
    old = (await graph.aget_state(config)).values["submilli_session"]["id"]

    # Idle past the timeout while the server is down; it reaps on start.
    managed.stop()
    time.sleep(1.5)
    managed.start()
    second = outputs(await ask(graph, "u_ada", "t", token="tok"))[1]

    assert body(second)["result"] == "scratch was not there"
    assert "earlier session" in body(second)["notice"]
    new = (await graph.aget_state(config)).values["submilli_session"]["id"]
    assert new != old
    assert second.artifact["session_id"] == new
