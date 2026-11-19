import json
from typing import cast

import httpx
import respx
from agents import Agent, Runner, set_tracing_disabled

from openai_agents_submilli import SubmilliTools
from tests.scripted import call, execute, scripted
from tests.unit_tests.fake_server import PROMPT, SERVER, TOKEN, FakeServer

DOCS = "submilli__typescript__packages__docs"
FUNCTIONS = "submilli__typescript__packages__functions"
REQUESTS = [
    {"name": "@submilli/jina", "functions": ["searchJson"]},
    {"name": "submilli:crypto", "functions": ["sha256"]},
]


def install(router: respx.MockRouter) -> None:
    def prompt(request: httpx.Request) -> httpx.Response:
        assert request.url.params["function_docs"] == "true"
        return httpx.Response(
            200,
            json={**PROMPT, "tools": {**cast("dict[str, str]", PROMPT["tools"]), "functions": "Fetch declarations."}},
        )

    router.get(f"{SERVER}/v1/blueprints/research/prompt").mock(side_effect=prompt)

    def docs(request: httpx.Request) -> httpx.Response:
        assert request.url.params["view"] == "index"
        assert len(request.url.params.get_list("names[]")) == 2
        return httpx.Response(
            200, json={"results": [{"name": "@submilli/jina", "found": True, "text": "searchJson: Search"}]}
        )

    def functions(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.url.params["requests"]) == REQUESTS
        return httpx.Response(
            200, json={"results": [{"name": "@submilli/jina", "found": True, "text": "function searchJson(): string;"}]}
        )

    router.get(f"{SERVER}/v1/blueprints/research/packages/docs").mock(side_effect=docs)
    router.get(f"{SERVER}/v1/blueprints/research/packages/functions").mock(side_effect=functions)
    FakeServer().install(router)


set_tracing_disabled(True)


async def test_openai_function_docs() -> None:
    with respx.mock(assert_all_called=False) as router:
        install(router)
        model = scripted(
            [call(DOCS, {"names": [r["name"] for r in REQUESTS]})],
            [call(FUNCTIONS, {"requests": REQUESTS})],
            [execute("export function main(): string { return 'ok'; }")],
            "done",
        )
        toolkit = SubmilliTools("research", server=SERVER, token=TOKEN, function_docs=True)
        assert FUNCTIONS in [tool.name for tool in toolkit.tools]
        await Runner.run(
            Agent(name="assistant", model=model, tools=list(toolkit.tools)),
            "go",
            context={"user_id": "u_ada", "conversation_id": "c_1"},
        )
