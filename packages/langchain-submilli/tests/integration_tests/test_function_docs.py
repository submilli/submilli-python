"""Function discovery followed by execution against a managed local server."""

import json

import pytest
from deepagents import create_deep_agent
from langchain_core.tools import tool

from langchain_submilli import SubmilliMiddleware
from tests.integration_tests.conftest import ManagedServer
from tests.scripted import call, execute, scripted

pytestmark = [pytest.mark.integration, pytest.mark.function_docs]


async def test_function_docs_executes_fetched_function(managed: ManagedServer) -> None:
    managed.apply("name: function-docs\ndefault: allow\n")

    @tool
    def greeting(name: str) -> str:
        """Greet a person."""
        return f"Hello {name}"

    functions = "submilli__typescript__packages__functions"
    model = scripted(
        [call("submilli__typescript__packages__docs", {"names": ["@tools", "submilli:crypto"]})],
        [
            call(
                functions,
                {
                    "requests": [
                        {"name": "@tools", "functions": ["greeting"]},
                        {"name": "submilli:crypto", "functions": ["sha256"]},
                    ]
                },
            )
        ],
        [
            execute(
                'import crypto from "submilli:crypto"; '
                'export function main(): string { return crypto.sha256("hello").toHex(); }'
            )
        ],
        "done",
    )
    middleware = SubmilliMiddleware(
        "function-docs", server=managed.url, token=managed.token, variables={}, function_docs=True, ptc=[greeting]
    )
    graph = create_deep_agent(model=model, middleware=[middleware])
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, context={"user_id": "u_ada"})
    messages = [message for message in result["messages"] if message.type == "tool"]
    assert len(messages) == 3
    assert all(message.status == "success" for message in messages), [message.text for message in messages]
    selected = json.loads(messages[1].text)["results"]
    assert all(entry["found"] for entry in selected)
    assert "greeting" in selected[0]["text"]
    assert "sha256" in selected[1]["text"]
    assert "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" in messages[2].text
    await middleware.client.aclose()
