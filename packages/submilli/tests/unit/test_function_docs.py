"""Wire contracts for opt-in docs and cross-package selections."""

import json

import httpx
import pytest

from submilli import AsyncSubmilli, Submilli

REQUESTS = [{"name": "submilli:crypto", "functions": ["sha256"]}, {"name": "submilli:http", "functions": ["get"]}]


def respond(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/functions"):
        assert json.loads(request.url.params["requests"]) == REQUESTS
        return httpx.Response(
            200, json={"results": [{"name": r["name"], "found": True, "text": "declarations"} for r in REQUESTS]}
        )
    assert request.url.params["view"] == "index"
    assert request.url.params.get_list("names[]") == [r["name"] for r in REQUESTS]
    return httpx.Response(200, json={"results": []})


def test_sync_batch_function_docs() -> None:
    with Submilli(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        assert len(client.blueprints.package_functions("blueprint", REQUESTS)["results"]) == 2
        assert len(client.sessions.package_functions("session", REQUESTS)["results"]) == 2
        names = ["submilli:crypto", "submilli:http"]
        assert client.blueprints.package_docs_batch("blueprint", names, view="index") == {"results": []}
        assert client.sessions.package_docs_batch("session", names, view="index") == {"results": []}


@pytest.mark.asyncio
async def test_async_batch_function_docs() -> None:
    async with AsyncSubmilli(
        base_url="http://test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        assert len((await client.blueprints.package_functions("blueprint", REQUESTS))["results"]) == 2
        assert len((await client.sessions.package_functions("session", REQUESTS))["results"]) == 2
        names = ["submilli:crypto", "submilli:http"]
        assert await client.blueprints.package_docs_batch("blueprint", names, view="index") == {"results": []}
        assert await client.sessions.package_docs_batch("session", names, view="index") == {"results": []}
