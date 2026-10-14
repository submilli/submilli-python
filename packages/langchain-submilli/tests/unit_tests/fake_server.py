"""A stateful stand-in for submilli-server's HTTP API, served through respx."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx

SERVER = "http://submilli.test"
TOKEN = "test-token"

PROMPT = {
    "name": "research",
    "prompt": "Run a submilli program — strict TypeScript subset compiled to WebAssembly.",
    "tools": {
        "search": "Search packages.",
        "docs": "Package docs.",
        "builtins": "Built-ins.",
        "builtins_list": "List built-ins.",
        "builtins_docs": "Built-in docs.",
        "last_run": "The last run.",
    },
}


@dataclass
class Session:
    blueprint: str
    variables: dict[str, str]
    secrets: dict[str, str]
    runs: int = 0
    last: dict[str, Any] | None = None
    # (key) -> (code, body), or (code, run id) for a run that may suspend
    ledger: dict[str, tuple[str, Any]] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Run:
    """A program that calls `@tools`: one suspension per call, in order."""

    id: str
    session_id: str
    calls: list[tuple[str, dict[str, Any]]]
    answers: list[dict[str, Any]] = field(default_factory=list)
    finished: dict[str, Any] | None = None


@dataclass
class FakeServer:
    """Behaves as the HTTP API reference says, for the parts the package uses.

    Programs are not run: `outcome` decides what an execute returns, and
    `refuse_next` makes the next execute answer one of the refusals.
    """

    sessions: dict[str, Session] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    required_secrets: list[str] = field(default_factory=list)
    lost_secrets: set[str] = field(default_factory=set)
    refuse_next: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=lambda: {"result": "ok", "console": [], "error": None})
    prompt_fetches: int = 0
    # Code -> the `@tools` calls it makes, (tool, args) in order. Programs in a
    # session with tools whose code is listed here suspend at each call, and
    # return the answers they got as JSON.
    programs: dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    suspended: dict[str, Run] = field(default_factory=dict)
    open_warnings: list[str] = field(default_factory=list)

    def install(self, router: respx.MockRouter) -> None:
        router.route(host="submilli.test").mock(side_effect=self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return _json(401, {"error": "unauthorized", "message": "missing or unrecognised API token"})
        path, method = request.url.path, request.method
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if method == "POST" and path == "/v1/sessions":
            return self.open(body)
        if match := re.fullmatch(r"/v1/sessions/([^/]+)/execute", path):
            return self.execute(match[1], body, request.headers.get("idempotency-key"))
        if match := re.fullmatch(r"/v1/sessions/([^/]+)/rebind", path):
            return self.rebind(match[1], body)
        if match := re.fullmatch(r"/v1/sessions/([^/]+)/packages/(search|docs)", path):
            return self.session_discovery(match[1], match[2], request)
        if match := re.fullmatch(r"/v1/runs/([^/]+)/resume", path):
            return self.resume(match[1], body)
        if match := re.fullmatch(r"/v1/runs/([^/]+)", path):
            run = self.suspended.get(match[1])
            if run is None:
                return _json(404, {"error": "unknown_run", "execution_id": match[1], "message": "no such run"})
            return _json(200, self.state(run))
        if match := re.fullmatch(r"/v1/sessions/([^/]+)/last-run", path):
            session = self.sessions.get(match[1])
            if session is None or session.last is None:
                return httpx.Response(404)
            return _json(200, session.last)
        if method == "DELETE" and (match := re.fullmatch(r"/v1/sessions/([^/]+)", path)):
            return httpx.Response(204 if self.sessions.pop(match[1], None) else 404)
        if match := re.fullmatch(r"/v1/blueprints/([^/]+)/(.*)", path):
            if match[1] != "research":
                return _json(404, {"error": "not_found", "message": f"blueprint '{match[1]}' is not registered"})
            return self.discovery(match[2], request)
        return httpx.Response(500, text=f"unexpected {method} {path}")

    def open(self, body: dict[str, Any]) -> httpx.Response:
        if body["blueprint"] != "research":
            return _json(404, {"error": "unknown blueprint", "message": f"unknown blueprint: {body['blueprint']}"})
        variables = body.get("variables") or {}
        if "userId" not in variables:
            return _json(400, {"error": "invalid variables: required variable 'userId' was not supplied"})
        secrets = body.get("secrets") or {}
        for name in self.required_secrets:
            if not secrets.get(name):
                return _json(400, {"error": f"invalid secrets: required harness secret '{name}' is missing or empty"})
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(body["blueprint"], variables, secrets, tools=body.get("tools") or [])
        opened: dict[str, Any] = {"session_id": session_id}
        if body.get("tools"):
            opened["warnings"] = list(self.open_warnings)
        return _json(200, opened)

    def execute(self, session_id: str, body: dict[str, Any], key: str | None) -> httpx.Response:
        session = self.sessions.get(session_id)
        if session is None:
            return _json(404, {"error": "unknown session", "session_id": session_id})
        if session_id in self.lost_secrets:
            return _json(
                409, {"error": "session_requires_secrets", "required": self.required_secrets, "session_id": session_id}
            )
        if self.refuse_next:
            status, refusal = self.refuse_next.pop(0)
            return _json(status, {**refusal, "session_id": session_id})
        code = body["code"]
        if key is not None and key in session.ledger:
            recorded_code, recorded = session.ledger[key]
            if recorded_code != code:
                return _json(
                    409, {"error": "idempotency_conflict", "detail": "different code", "session_id": session_id}
                )
            if isinstance(recorded, str):
                run = self.suspended.get(recorded)
                if run is None:
                    return _json(409, {"error": "idempotency_incomplete", "session_id": session_id})
                return _json(200, self.state(run))
            return _json(200, recorded)
        if session.tools and code in self.programs:
            session.runs += 1
            run = Run(str(uuid.uuid4()), session_id, list(self.programs[code]))
            self.suspended[run.id] = run
            if key is not None:
                session.ledger[key] = (code, run.id)
            return _json(200, self.state(run))
        session.runs += 1
        result = {"execution_id": str(uuid.uuid4()), "session_id": session_id, **self.outcome}
        if key is not None:
            session.ledger[key] = (code, result)
        session.last = {key: result[key] for key in ("result", "console", "error")}
        return _json(200, result)

    def state(self, run: Run) -> dict[str, Any]:
        if run.finished is not None:
            return run.finished
        position = len(run.answers)
        if position < len(run.calls):
            tool, args = run.calls[position]
            call = {"call_id": f"{run.id}:{position + 1}", "tool": tool, "args": args}
            return {"status": "suspended", "execution_id": run.id, "session_id": run.session_id, "calls": [call]}
        error = None
        for answer in run.answers:
            if not answer["ok"] and error is None:
                kind = answer["error"]["kind"]
                message = answer["error"]["message"]
                if kind == "denied":
                    error = {
                        "kind": "permission_denied",
                        "message": f"PermissionDeniedError: @tools: {message}",
                        "caller": "main",
                        "capability": "tools.call",
                        "source": "harness",
                    }
                else:
                    error = {"kind": "runtime_error", "message": f"Error: {message}"}
        values = [answer.get("value") for answer in run.answers]
        run.finished = {
            "status": "finished",
            "execution_id": run.id,
            "session_id": run.session_id,
            "result": None if error else json.dumps(values),
            "console": [],
            "error": error,
        }
        self.sessions[run.session_id].last = {key: run.finished[key] for key in ("result", "console", "error")}
        return run.finished

    def resume(self, run_id: str, body: dict[str, Any]) -> httpx.Response:
        run = self.suspended.get(run_id)
        if run is None:
            return _json(404, {"error": "unknown_run", "execution_id": run_id, "message": "no such run"})
        for result in body["results"]:
            sequence = int(result["call_id"].rsplit(":", 1)[1])
            if sequence <= len(run.answers):
                continue  # answered already: changes nothing
            if run.finished is not None:
                return _json(409, {"error": "run_not_suspended", "execution_id": run_id, "message": "finished"})
            if sequence != len(run.answers) + 1:
                return _json(400, {"error": "unknown_call", "execution_id": run_id, "message": "another call"})
            run.answers.append(result)
        return _json(200, self.state(run))

    def session_discovery(self, session_id: str, rest: str, request: httpx.Request) -> httpx.Response:
        session = self.sessions.get(session_id)
        if session is None:
            return _json(404, {"error": "unknown session", "session_id": session_id})
        names = [tool["name"] for tool in session.tools]
        if rest == "search":
            results = [{"name": "@submilli/jina", "description": request.url.params.get("q"), "source": "registry"}]
            if session.tools:
                results.append({"name": "@tools", "description": "The application's tools.", "source": "tools"})
            return _json(200, {"results": results})
        if request.url.params["name"] == "@tools":
            if not session.tools:
                return _json(404, {"error": "unknown_package", "message": "unknown package: @tools"})
            return httpx.Response(
                200, text="# @tools\n" + "\n".join(f"function {name}(args): unknown;" for name in names)
            )
        return self.discovery("packages/docs", request)

    def rebind(self, session_id: str, body: dict[str, Any]) -> httpx.Response:
        if session_id not in self.sessions:
            return _json(404, {"error": "unknown session", "session_id": session_id})
        for name in self.required_secrets:
            if not body["secrets"].get(name):
                return _json(400, {"error": f"invalid secrets: required harness secret '{name}' is missing or empty"})
        self.sessions[session_id].secrets = body["secrets"]
        self.lost_secrets.discard(session_id)
        return httpx.Response(204)

    def discovery(self, rest: str, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if rest == "prompt":
            self.prompt_fetches += 1
            return _json(200, PROMPT)
        if rest == "packages/search":
            return _json(
                200, {"results": [{"name": "@submilli/jina", "description": params.get("q"), "source": "registry"}]}
            )
        if rest == "packages/docs":
            name = params["name"]
            if name == "@submilli/jina":
                return httpx.Response(200, text="# jina\nfunction searchJson(): string;")
            return _json(404, {"error": "unknown_package", "message": f"unknown package: {name}"})
        if rest == "builtins":
            return _json(200, {"namespaces": ["JSON"], "types": ["Map"]})
        if rest == "builtins/docs":
            return _json(200, {"results": [{"name": name, "declarations": "..."} for name in params.get_list("name")]})
        return httpx.Response(404)

    # Helpers for assertions

    def calls(self, method: str, pattern: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and re.fullmatch(pattern, r.url.path)]

    def total_runs(self) -> int:
        return sum(session.runs for session in self.sessions.values())


def _json(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body)
