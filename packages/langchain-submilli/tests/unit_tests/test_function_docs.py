import json
from typing import cast

import httpx
import respx
from deepagents import create_deep_agent

from langchain_submilli import SubmilliMiddleware
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


async def test_deepagents_function_docs() -> None:
    with respx.mock(assert_all_called=False) as router:
        install(router)
        model = scripted(
            [call(DOCS, {"names": [r["name"] for r in REQUESTS]})],
            [call(FUNCTIONS, {"requests": REQUESTS})],
            [
                execute(
                    'import jina from "@submilli/jina"; export function main(): string { return jina.searchJson(); }'
                )
            ],
            "done",
        )
        middleware = SubmilliMiddleware("research", server=SERVER, token=TOKEN, function_docs=True)
        graph = create_deep_agent(model=model, middleware=[middleware])
        result = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, context={"user_id": "u_ada"})
        messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(messages) == 3
        assert all(m.status == "success" for m in messages)
        assert FUNCTIONS in model.tool_names()
