"""The application's own tools, for programs to call as the `@tools` Package.

The toolkit declares each tool when it opens a session. A program's call
suspends its run; the call is then run as an ordinary `FunctionTool` call of a
nested agent (see `_program.py`), so the SDK's approvals, guardrails, context
and tracing apply, and the run is resumed with the result. The contract is
"Application tools" in submilli-runtime's HTTP API reference.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import typing
from collections.abc import Callable, Sequence
from typing import Any

from agents import FunctionTool, function_tool
from agents.tool_context import ToolContext
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from submilli import ToolDeclaration, ToolResult
from submilli.schema import inline_refs

# A TypeScript function name, as the server requires of a tool's name. The
# server also drops reserved words, with a warning when the session opens.
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

# What a tool returns when its return type gives no schema: the text the
# model would see.
_TEXT = {"type": "string"}

# Tool outputs are wrapped in an envelope with this key, so that what the tool
# returned or raised can be told apart from text the SDK writes in its place,
# such as a rejected approval.
_ENVELOPE = "__submilli_program_call__"


def as_function_tools(tools: Sequence[FunctionTool | Callable[..., Any]] | None) -> list[FunctionTool]:
    """The `ptc` tools as `FunctionTool`s; plain functions are wrapped with `function_tool`."""
    names = [str(item) for item in tools or [] if isinstance(item, str)]
    if names:
        raise TypeError(f"ptc takes tools, not tool names; pass the tool objects for: {', '.join(names)}")
    converted: list[FunctionTool] = []
    for item in tools or []:
        if isinstance(item, FunctionTool):
            converted.append(item)
        elif callable(item):
            converted.append(function_tool(item))
        else:
            raise TypeError(f"ptc takes FunctionTools or functions; got {type(item).__name__}")
    return converted


def declaration(tool: FunctionTool) -> ToolDeclaration:
    """`tool` as the server takes it: `{name, description, input_schema, output_schema}`.

    The input schema is the one the model would see. The output schema is the
    tool's `output_json_schema`, else the one of the function's return
    annotation, else a string: the tool's text. `$ref`s are inlined, because
    the server's mapper does not resolve them.
    """
    declared: ToolDeclaration = {
        "name": tool.name,
        "input_schema": inline_refs(copy.deepcopy(tool.params_json_schema)),
        "output_schema": output_schema(tool),
    }
    if tool.description:
        declared["description"] = tool.description
    return declared


def output_schema(tool: FunctionTool) -> dict[str, Any]:
    if tool.output_json_schema is not None:
        return inline_refs(copy.deepcopy(tool.output_json_schema))
    try:
        function = tool.__wrapped__
    except AttributeError:  # not made by `function_tool`
        return dict(_TEXT)
    try:
        annotation = typing.get_type_hints(function).get("return")
    except Exception:  # unresolvable annotations mean "no schema"
        return dict(_TEXT)
    if annotation in (None, str, Any, object, type(None)):
        return dict(_TEXT)
    try:
        schema = TypeAdapter(annotation).json_schema(mode="serialization")
    except Exception:  # not a JSON type
        return dict(_TEXT)
    return inline_refs(schema)


def catalog_digest(catalog: Sequence[ToolDeclaration]) -> str | None:
    """A fingerprint of the declared tools, so a conversation notices when they changed."""
    if not catalog:
        return None
    return hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()


class ProgramTools:
    """The tools programs may call, as declared to the server and as the nested agent runs them."""

    def __init__(self, tools: Sequence[FunctionTool]) -> None:
        names = [tool.name for tool in tools]
        for name in names:
            if not _IDENTIFIER.fullmatch(name):
                raise ValueError(f"ptc tool {name!r}: programs call tools by name, so it must be an identifier")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"ptc tools must have distinct names; repeated: {', '.join(duplicates)}")
        self.catalog: list[ToolDeclaration] = [declaration(tool) for tool in tools]
        self.digest: str | None = catalog_digest(self.catalog)
        text = {item["name"]: item.get("output_schema") == _TEXT for item in self.catalog}
        self.approvals: bool = any(tool.needs_approval is not False for tool in tools)
        self.tools: list[FunctionTool] = [_enveloped(tool, text=text[tool.name]) for tool in tools]
        self.names: frozenset[str] = frozenset(names)

    def __bool__(self) -> bool:
        return bool(self.tools)

    def prompt(self) -> str:
        """What the execute tool's description adds about `@tools`."""
        names = ", ".join(f"`{tool.name}`" for tool in self.tools)
        approval = (
            " Some calls wait for a person's approval, who may refuse them: a refusal throws "
            "`PermissionDeniedError`, and is final."
            if self.approvals
            else ""
        )
        return (
            "\n\nThis agent's programs can also call the application's own tools, as the `@tools` package "
            f'(`import tools from "@tools";`): {names}. The application runs each call and returns its '
            f"result to the program.{approval} Read their signatures with "
            '`submilli__typescript__packages__docs("@tools")`.'
        )


def _enveloped(tool: FunctionTool, *, text: bool) -> FunctionTool:
    """A copy of `tool` whose output, or failure, comes back wrapped in an envelope.

    The copy lets exceptions through to the wrapper instead of formatting them
    for a model, and drops the output schema, which the SDK would check the
    envelope against. Everything else, approvals included, is the tool's own.
    """
    raising = dataclasses.replace(
        tool,
        output_json_schema=None,
        _output_type_adapter=None,
        _failure_error_function=None,
        _use_default_failure_error_function=False,
    )
    invoke = raising.on_invoke_tool

    async def on_invoke(context: ToolContext[Any], arguments: str) -> str:
        try:
            output = await invoke(context, arguments)
        except Exception as error:  # the program sees the failure
            return json.dumps({_ENVELOPE: "failed", "message": f"{type(error).__name__}: {error}"})
        value = _text(output) if text else to_jsonable_python(output)
        return json.dumps({_ENVELOPE: "ok", "value": value}, ensure_ascii=False)

    return dataclasses.replace(raising, on_invoke_tool=on_invoke)


def _text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    try:
        return json.dumps(to_jsonable_python(output), ensure_ascii=False)
    except Exception:  # not JSON: what the SDK would show the model
        return str(output)


def answer(call_id: str, output: Any) -> tuple[ToolResult, str]:
    """The server's answer to a call, from the output the nested agent recorded, and its outcome.

    Text in place of an envelope was written by the SDK instead of the tool:
    a rejected approval, a guardrail's refusal, or a timeout. The program sees
    it as a denial.
    """
    text = output if isinstance(output, str) else json.dumps(output)
    try:
        envelope = json.loads(text)
    except ValueError:
        envelope = None
    if isinstance(envelope, dict) and envelope.get(_ENVELOPE) == "ok":
        return ToolResult.success(call_id, envelope.get("value")), "ok"
    if isinstance(envelope, dict) and envelope.get(_ENVELOPE) == "failed":
        return ToolResult.failed(call_id, str(envelope.get("message", ""))), "failed"
    return ToolResult.denied(call_id, text), "denied"
