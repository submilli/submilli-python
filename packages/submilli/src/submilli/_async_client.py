"""`AsyncSubmilli`: the HTTP API of one submilli-server, for asyncio."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx

from submilli import _base as b
from submilli._errors import (
    APIConnectionError,
    APITimeoutError,
    RunNotSuspendedError,
    SubmilliError,
    UnknownCallError,
)
from submilli._types import (
    BlueprintPrompt,
    CreatedSession,
    ExecuteResult,
    LastRun,
    PackageDocs,
    PendingCall,
    RunningRun,
    RunState,
    ToolDeclaration,
    ToolResult,
)

AsyncToolHandler = Callable[[PendingCall], "ToolResult | Awaitable[ToolResult]"]


class AsyncSubmilli:
    """The HTTP API of one submilli-server, for one API token.

    Args:
        base_url: The server. Defaults to `$SUBMILLI_SERVER`, then `http://127.0.0.1:8128`.
        token: API token. Defaults to `$SUBMILLI_SERVER_TOKEN`. A server started
            with `--allow-unauthenticated` needs none.
        timeout: Seconds per request, or an `httpx.Timeout`.
        max_retries: Further attempts after a connection failure or a 502, 503
            or 504, for requests that are safe to repeat: reads, deletes,
            rebinds, resumes, and executes with an idempotency key.
        http_client: An `httpx.AsyncClient` to send with. Its lifetime is yours.
            Without one, the client keeps one per event loop and closes it in
            `aclose`.

    Example:
        ```python
        async with AsyncSubmilli() as client:
            session = await client.sessions.create("research", variables={"userId": "u_ada"})
            result = await client.sessions.run(session.id, code, idempotency_key="call-1")
        ```
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | httpx.Timeout = b.DEFAULT_TIMEOUT,
        max_retries: int = b.DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = b.Settings.resolve(base_url, token, timeout, max_retries)
        self._external = http_client
        # An httpx client can't outlive the loop it was first used on, and
        # applications and tests often run several loops.
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self.sessions: AsyncSessions = AsyncSessions(self)
        self.runs: AsyncRuns = AsyncRuns(self)
        self.blueprints: AsyncBlueprints = AsyncBlueprints(self)

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    async def execute(
        self,
        blueprint: str,
        code: str,
        *,
        variables: Mapping[str, str] | None = None,
        secrets: Mapping[str, str] | None = None,
    ) -> ExecuteResult:
        """`POST /v1/execute`: run one program in a session that ends when it returns.

        An unknown Blueprint or invalid variables come back as the result's
        `error`, not as an exception. One-off runs have no application tools.
        """
        return b.read_execute_result(await self._send(b.execute_once(blueprint, code, variables, secrets)))

    async def aclose(self) -> None:
        """Close the connections this client opened."""
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            if not client.is_closed:
                await client.aclose()

    async def __aenter__(self) -> AsyncSubmilli:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._external is not None:
            return self._external
        loop = asyncio.get_running_loop()
        for stale in [other for other in self._clients if other.is_closed()]:
            del self._clients[stale]
        client = self._clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(timeout=self._settings.timeout)
            self._clients[loop] = client
        return client

    async def _send(self, request: b.Request) -> httpx.Response:
        headers = {**self._settings.headers(), **(request.headers or {})}
        url = f"{self._settings.base_url}{request.path}"
        attempts = 1 + (self._settings.max_retries if request.retryable else 0)
        for attempt in range(attempts):
            last = attempt == attempts - 1
            response: httpx.Response | None = None
            try:
                response = await self._http().request(
                    request.method, url, json=request.json, params=request.params, headers=headers
                )
            except httpx.TimeoutException as error:
                if last:
                    raise APITimeoutError(f"request to {url} timed out", request=error.request) from error
            except httpx.TransportError as error:
                if last:
                    raise APIConnectionError(f"could not reach {url}: {error}", request=error.request) from error
            else:
                if last or not b.should_retry(response):
                    return response
            await asyncio.sleep(b.retry_delay(attempt, response))
        raise AssertionError("unreachable")


class AsyncSessions:
    """Sessions: a Blueprint bound to variables and secrets, where programs run one after another."""

    def __init__(self, client: AsyncSubmilli) -> None:
        self._client = client

    async def create(
        self,
        blueprint: str,
        *,
        variables: Mapping[str, str] | None = None,
        secrets: Mapping[str, str] | None = None,
        tools: Sequence[ToolDeclaration] | None = None,
    ) -> CreatedSession:
        """`POST /v1/sessions`: open a session.

        `tools` are the application's tools that programs may call as `@tools`,
        fixed for the session's life. The Blueprint must grant `tools.call`.
        """
        return b.read_created_session(await self._client._send(b.create_session(blueprint, variables, secrets, tools)))

    async def execute(self, session_id: str, code: str, *, idempotency_key: str | None = None) -> RunState:
        """`POST /v1/sessions/{id}/execute`: start a program and answer its first state.

        In a session with application tools a run may answer `SuspendedRun`;
        `run` answers its calls for you. With an `idempotency_key`, a repeated
        request answers the same run instead of running the program again. On a
        server with `durable_execution`, a run still queued or working after the
        server's wait answers `RunningRun`, and a run with no result to return
        raises `RunStateError`.
        """
        return b.read_run_state(await self._client._send(b.execute(session_id, code, idempotency_key)))

    async def run(
        self,
        session_id: str,
        code: str,
        *,
        idempotency_key: str | None = None,
        on_tool_call: AsyncToolHandler | None = None,
    ) -> ExecuteResult:
        """Run a program to the end, answering its calls to the application's tools with `on_tool_call`.

        An exception from `on_tool_call` propagates and leaves the run waiting
        on the server. Executing again with the same `idempotency_key` picks it
        up at the call it waits on. Raises `RunStateError` when the run has no
        result to return, such as a durable run that is `stuck`.
        """
        state = await self.execute(session_id, code, idempotency_key=idempotency_key)
        return await self._client.runs.complete(state, on_tool_call=on_tool_call)

    async def rebind(self, session_id: str, secrets: Mapping[str, str]) -> None:
        """`POST /v1/sessions/{id}/rebind`: replace the session's harness secrets, as after a server restart."""
        b.read_no_content(await self._client._send(b.rebind(session_id, secrets)))

    async def last_run(self, session_id: str) -> LastRun | None:
        """`GET /v1/sessions/{id}/last-run`. `None` when no run is recorded or the session is unknown."""
        return b.read_last_run(await self._client._send(b.last_run(session_id)))

    async def delete(self, session_id: str) -> bool:
        """`DELETE /v1/sessions/{id}`: end the session. `False` when it had already ended."""
        return b.read_deleted(await self._client._send(b.delete_session(session_id)))

    async def search_packages(self, session_id: str, query: str = "") -> dict[str, Any]:
        """The Blueprint's Package search, plus the session's own `@tools`."""
        return b.read_json(await self._client._send(b.session_packages_search(session_id, query)))

    async def package_docs(self, session_id: str, name: str) -> PackageDocs:
        """A Package's docs, where `@tools` is the session's own."""
        return b.read_package_docs(await self._client._send(b.session_packages_docs(session_id, name)))


class AsyncRuns:
    """Runs that wait on the application's tools."""

    def __init__(self, client: AsyncSubmilli) -> None:
        self._client = client

    async def resume(self, run_id: str, results: Sequence[ToolResult]) -> RunState:
        """`POST /v1/runs/{id}/resume`: answer the run's pending calls, then wait for its next state.

        Answering a call twice changes nothing. Raises `UnknownRunError`,
        `RunNotSuspendedError` and `UnknownCallError`, and `RunStateError` for
        a run with no result to return.
        """
        return b.read_run_state(await self._client._send(b.resume(run_id, results)))

    async def retrieve(self, run_id: str) -> RunState:
        """`GET /v1/runs/{id}`: the run's state now.

        Raises `UnknownRunError`, and `RunStateError` for a run with no result
        to return, such as a durable run that is `stuck`.
        """
        return b.read_run_state(await self._client._send(b.retrieve_run(run_id)))

    async def complete(
        self,
        state: RunState,
        *,
        on_tool_call: AsyncToolHandler | None = None,
        poll_interval: float = 0.1,
        max_poll_interval: float = 2.0,
    ) -> ExecuteResult:
        """Drive a run from `state` until it finishes.

        Each pending call goes to `on_tool_call`, sync or async, and its
        `ToolResult` goes back with `resume`. A run that is `running` or
        `queued` is polled. Raises `UnknownRunError` when the server lost the
        run, as after a restart, and `RunStateError` when the run has no result
        to return, such as a durable run that is `stuck` or `cancelled`.
        """
        delay = poll_interval
        while not isinstance(state, ExecuteResult):
            if isinstance(state, RunningRun):
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_poll_interval)
                state = await self.retrieve(state.execution_id)
                continue
            if on_tool_call is None:
                raise SubmilliError(
                    f"run {state.execution_id} called the application's tool {state.calls[0].tool!r}, "
                    "and no on_tool_call was given to answer it"
                )
            results = []
            for call in state.calls:
                result = on_tool_call(call)
                if inspect.isawaitable(result):
                    result = await result
                if not isinstance(result, ToolResult) or result.call_id != call.call_id:
                    raise SubmilliError(f"on_tool_call must return a ToolResult for call {call.call_id!r}")
                results.append(result)
            try:
                state = await self.resume(state.execution_id, results)
            except (RunNotSuspendedError, UnknownCallError):
                # A call timed out, or another attempt answered it: look again.
                state = await self.retrieve(state.execution_id)
        return state


class AsyncBlueprints:
    """What a model needs to write programs for one Blueprint."""

    def __init__(self, client: AsyncSubmilli) -> None:
        self._client = client

    async def prompt(self, blueprint: str) -> BlueprintPrompt:
        """`GET /v1/blueprints/{name}/prompt`: the tool descriptions the server writes for this Blueprint."""
        return b.read_prompt(await self._client._send(b.prompt(blueprint)))

    async def search_packages(self, blueprint: str, query: str = "") -> dict[str, Any]:
        """`GET /v1/blueprints/{name}/packages/search`. An empty query lists every Package."""
        return b.read_json(await self._client._send(b.packages_search(blueprint, query)))

    async def package_docs(self, blueprint: str, name: str) -> PackageDocs:
        """`GET /v1/blueprints/{name}/packages/docs`."""
        return b.read_package_docs(await self._client._send(b.packages_docs(blueprint, name)))

    async def builtins(self, blueprint: str) -> dict[str, Any]:
        """`GET /v1/blueprints/{name}/builtins`: the built-ins in scope without an import."""
        return b.read_json(await self._client._send(b.builtins(blueprint)))

    async def builtin_docs(self, blueprint: str, names: Sequence[str]) -> dict[str, Any]:
        """`GET /v1/blueprints/{name}/builtins/docs`: declarations of built-ins."""
        return b.read_json(await self._client._send(b.builtins_docs(blueprint, names)))


__all__ = ["AsyncBlueprints", "AsyncRuns", "AsyncSessions", "AsyncSubmilli", "AsyncToolHandler"]
