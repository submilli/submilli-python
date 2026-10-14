"""Helpers for the JSON Schemas of application tools."""

from __future__ import annotations

from typing import Any

__all__ = ["inline_refs"]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` with its local references (`#/$defs/...`, `#/definitions/...`) replaced by what they name.

    The server's mapper does not resolve `$ref`, and types a field that uses
    one as `unknown`. Pydantic and most generators put nested models behind
    references, so inline them before declaring a tool. A recursive reference
    becomes `{}`, which the server types `unknown`.
    """
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
            name = ref.rsplit("/", 1)[1]
            if name in seen or name not in defs:
                return {}
            rest = {key: value for key, value in node.items() if key != "$ref"}
            return resolve({**defs[name], **rest}, (*seen, name))
        return {key: resolve(value, seen) for key, value in node.items() if key not in ("$defs", "definitions")}

    resolved: dict[str, Any] = resolve(schema, ())
    return resolved
