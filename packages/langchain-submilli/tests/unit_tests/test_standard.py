"""LangChain's standard unit tests, for each of the six tools.

The integration half of the standard suite invokes a tool synchronously and on
its own, without the agent's runtime. These tools are async and need the
thread's state, so it does not apply; tests/integration_tests covers them in an
agent instead.
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_submilli import SubmilliMiddleware
from tests.unit_tests.fake_server import SERVER, TOKEN

TOOLS = {tool.name: tool for tool in SubmilliMiddleware("research", server=SERVER, token=TOKEN).tools}


class _Base(ToolsUnitTests):
    name: str
    example: ClassVar[dict[str, Any]] = {}

    @property
    def tool_constructor(self) -> BaseTool:
        return TOOLS[self.name]

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return self.example


class TestExecute(_Base):
    name = "submilli__typescript__execute"
    example: ClassVar[dict[str, Any]] = {"code": 'export function main(): string { return "hi"; }'}


class TestLastRun(_Base):
    name = "submilli__typescript__last_run"


class TestPackagesSearch(_Base):
    name = "submilli__typescript__packages__search"
    example: ClassVar[dict[str, Any]] = {"query": "jina"}


class TestPackagesDocs(_Base):
    name = "submilli__typescript__packages__docs"
    example: ClassVar[dict[str, Any]] = {"name": "@submilli/jina"}


class TestBuiltinsList(_Base):
    name = "submilli__typescript__builtins__list"


class TestBuiltinsDocs(_Base):
    name = "submilli__typescript__builtins__docs"
    example: ClassVar[dict[str, Any]] = {"names": ["Map", "Temporal"]}
