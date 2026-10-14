"""`SubmilliMiddleware`: Submilli code mode for deepagents and LangChain v1 agents."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from langchain_submilli import _tools as t
from langchain_submilli._app_tools import ExposedTools, as_tools
from submilli import (
    AsyncSubmilli,
    ExecuteResult,
    IdempotencyError,
    PendingCall,
    RunState,
    SessionRequiresSecretsError,
    SubmilliError,
    ToolResult,
    UnknownRunError,
    UnknownSessionError,
    idempotency_key,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# deepagents' own file and shell tools, which `hide_filesystem_tools` removes.
FILESYSTEM_TOOLS = frozenset({"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"})

ContextMapper = Callable[[Any], Mapping[str, str]]


class SubmilliSessionMismatchError(SubmilliError):
    """The run's context binds different variables than the thread's session was opened with."""


class SubmilliSession(TypedDict):
    """The session a thread runs its programs in, as kept in the agent's state."""

    id: str
    server: str
    blueprint: str
    variables: dict[str, str]
    # A fingerprint of the application tools the session was opened with.
    tools: NotRequired[str | None]


def _replace(_: Any, new: Any) -> Any:
    # Parallel tool calls that each replace an ended session report the same new
    # one, so more than one write in a step must be allowed.
    return new


class SubmilliState(AgentState):
    """State schema for `SubmilliMiddleware`."""

    submilli_session: NotRequired[Annotated[SubmilliSession | None, PrivateStateAttr, _replace]]


def user_id_variables(context: Any) -> dict[str, str]:
    """The default `variables`: `user_id` in the run context becomes `userId`.

    Reads a mapping key or an attribute, so a dict, a dataclass, and a Pydantic
    model all work. A context without `user_id` binds no variables.
    """
    user_id = _context_value(context, "user_id")
    return {} if user_id is None else {"userId": str(user_id)}


def no_secrets(_: Any) -> dict[str, str]:
    """The default `secrets`: none."""
    return {}


# Any context and any response format: the middleware reads the context only
# through `variables` and `secrets`.
class SubmilliMiddleware(AgentMiddleware[SubmilliState, Any, Any]):
    """Runs the agent's tool calls as TypeScript programs on submilli-server.

    Adds the server's six tools. Every program runs in one session per thread,
    under the Blueprint's policy, bound to the variables this run's context
    gives. Build the agent once and share it; the user comes with each run.

    Args:
        blueprint: The registered Blueprint every program runs under.
        server: Base URL of submilli-server. Defaults to `SUBMILLI_SERVER`,
            then `http://127.0.0.1:8128`.
        token: API token. Defaults to `SUBMILLI_SERVER_TOKEN`.
        variables: Values for the Blueprint's variables, from the run's
            `runtime.context`: a callable taking the context and returning
            `dict[str, str]`, or a fixed mapping. The default maps `user_id` in
            the context to `userId`.
        secrets: Values for the Blueprint's `harness` secrets, the same way.
            Sent when a session opens and when the server asks for them again
            after a restart; never stored in the agent's state. Default: none.
        hide_filesystem_tools: Remove deepagents' own file and shell tools
            (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`,
            `grep`, `execute`) from this agent, so that files go through
            Submilli. It does not reach deepagents' subagents.
        client: A `submilli.AsyncSubmilli` to use instead of building one
            from `server` and `token`.
        expose_tools: The application's own tools, which programs may call
            as the `@tools` Package: LangChain tools, or functions to wrap as
            tools. Each call runs here, through a `ToolNode` with the run's
            state and context, and its result goes back to the program. The
            Blueprint must grant `tools.call`. The model is not given these
            tools directly; pass them to the agent as well for that.
        interrupt_on: Which exposed tools need a person's approval when a
            program calls them, as for `HumanInTheLoopMiddleware`: `True`, or
            an `InterruptOnConfig`. The graph is interrupted with a
            `HITLRequest` of one action; resume it with a `HITLResponse` of one
            decision. The agent needs a checkpointer.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_submilli import SubmilliMiddleware

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-5",
            middleware=[SubmilliMiddleware(blueprint="research")],
        )
        await agent.ainvoke({"messages": [...]}, context={"user_id": "u_ada"})
        ```
    """

    state_schema = SubmilliState

    def __init__(
        self,
        blueprint: str,
        server: str | None = None,
        token: str | None = None,
        variables: ContextMapper | Mapping[str, str] | None = None,
        secrets: ContextMapper | Mapping[str, str] | None = None,
        *,
        hide_filesystem_tools: bool = False,
        client: AsyncSubmilli | None = None,
        expose_tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
        interrupt_on: Mapping[str, bool | InterruptOnConfig] | None = None,
    ) -> None:
        super().__init__()
        self.blueprint: str = blueprint
        self.client: AsyncSubmilli = client or AsyncSubmilli(base_url=server, token=token)
        self._variables = _mapper(variables, user_id_variables)
        self._secrets = _mapper(secrets, no_secrets)
        self.hide_filesystem_tools: bool = hide_filesystem_tools
        self._exposed = ExposedTools(as_tools(expose_tools), interrupt_on)
        self._described = False
        self._describe_lock: asyncio.Lock | None = None
        # Sessions opened to replace an ended one, by the ended one's id, so that
        # parallel calls that all find it gone open one replacement between them.
        self._replacements: OrderedDict[str, SubmilliSession] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self.tools: Sequence[BaseTool] = t.build_tools(
            {
                t.EXECUTE: self._execute,
                t.LAST_RUN: self._last_run,
                t.PACKAGES_SEARCH: self._packages_search,
                t.PACKAGES_DOCS: self._packages_docs,
                t.BUILTINS_LIST: self._builtins_list,
                t.BUILTINS_DOCS: self._builtins_docs,
            }
        )

    # Closing a thread's session

    async def aclose_thread(self, agent: Any, config: Any) -> bool:
        """End the session of the thread `config` names, and forget it.

        `agent` is the compiled agent, with a checkpointer. Returns `False` when
        the thread had no session or the server had already ended it. The
        thread's next tool call opens a new session.
        """
        snapshot = await agent.aget_state(config)
        session = snapshot.values.get("submilli_session")
        if not session:
            return False
        deleted = await self.client.sessions.delete(session["id"])
        await agent.aupdate_state(config, {"submilli_session": None})
        return deleted

    # Hooks

    async def abefore_agent(self, state: SubmilliState, runtime: Runtime[Any]) -> None:
        """Refuse a run whose variables differ from those of the thread's session."""
        session = self._own_session(state)
        if session is not None:
            self._check_binding(session, self._variables(runtime.context))

    async def aafter_model(self, state: SubmilliState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Open the thread's session when the model first asks to run a program.

        Opening it here, rather than in the tool, records it in a checkpoint
        before any program runs. A thread resumed after a crash mid-call then
        retries in the same session, where the idempotency key replays the run.
        """
        last = state["messages"][-1] if state.get("messages") else None
        if not isinstance(last, AIMessage) or not any(
            call["name"] in self._session_tools() for call in last.tool_calls
        ):
            return None
        if self._own_session(state) is not None:
            return None
        session = await self._open(runtime.context)
        return {"submilli_session": session}

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        """Give the tools the Blueprint's descriptions, and hide deepagents' file tools if asked."""
        await self._describe()
        if self.hide_filesystem_tools:
            request = request.override(
                tools=[tool for tool in request.tools if _tool_name(tool) not in FILESYSTEM_TOOLS]
            )
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Refuse calls to deepagents' file tools when they are hidden."""
        name = request.tool_call["name"]
        if self.hide_filesystem_tools and name in FILESYSTEM_TOOLS:
            return ToolMessage(
                content=f"`{name}` is not available to this agent. Work with files in a program, through Submilli.",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return await handler(request)

    # Tools

    async def _execute(self, code: str, runtime: ToolRuntime) -> ToolMessage | Command[Any]:
        stored = self._own_session(runtime.state)
        session = stored
        if session is None:
            # `aafter_model` normally opened it. Without that hook, as when the
            # tool is called directly, open it here.
            session = await self._open_for_thread(runtime)
        variables = self._variables(runtime.context)
        self._check_binding(session, variables)
        tool_call_id = runtime.tool_call_id or ""
        key = idempotency_key(tool_call_id) if tool_call_id else None
        notice = None
        rebound = False
        while True:
            try:
                result = await self.client.sessions.execute(session["id"], code, idempotency_key=key)
                break
            except UnknownSessionError:
                if notice is not None:
                    raise
                session = await self._replace_session(session, runtime.context)
                notice = t.SESSION_REPLACED_NOTE
            except SessionRequiresSecretsError as error:
                if rebound:
                    raise SubmilliError(
                        f"submilli-server still asks for harness secrets {error.required} after a rebind; "
                        "check the `secrets` given to SubmilliMiddleware"
                    ) from error
                await self.client.sessions.rebind(session["id"], self._secrets(runtime.context))
                rebound = True
            except IdempotencyError as error:
                message = t.idempotency_message(error, tool_call_id=tool_call_id, session_id=session["id"])
                return _with_session(message, session, stored)
        calls: list[dict[str, Any]] = []
        updates: list[tuple[str, Any]] = []
        if not isinstance(result, ExecuteResult):
            try:
                result = await self._finish(result, runtime, calls, updates)
            except UnknownRunError:
                message = t.lost_run_message(
                    tool_call_id=tool_call_id, execution_id=result.execution_id, session_id=session["id"], calls=calls
                )
                return _with_session(message, session, stored, updates)
        message = t.execute_message(result, tool_call_id=tool_call_id, notice=notice, tool_calls=calls)
        return _with_session(message, session, stored, updates)

    async def _finish(
        self,
        state: RunState,
        runtime: ToolRuntime,
        calls: list[dict[str, Any]],
        updates: list[tuple[str, Any]],
    ) -> ExecuteResult:
        """Answer the run's calls to the application's tools until it finishes.

        An approval interrupts the graph here and the run waits on the server.
        When the graph resumes, the tool runs again, its execute replays to the
        run's current state, and this carries on from the call it waits on.
        """
        run_id = state.execution_id
        carried_calls, carried_updates = self._exposed.carried(run_id)
        calls.extend(carried_calls)
        updates.extend(carried_updates)
        # Interrupts this invocation has made, for `Approvals.decide`.
        used = [0]

        async def answer(call: PendingCall) -> ToolResult:
            outcome = await self._exposed.answer(call, runtime, used)
            calls.append(outcome.record)
            updates.extend(outcome.updates)
            return outcome.result

        try:
            result = await self.client.runs.complete(state, on_tool_call=answer)
        except GraphBubbleUp:
            self._exposed.carry(run_id, calls, updates)
            raise
        self._exposed.finished(run_id)
        return result

    async def _last_run(self, runtime: ToolRuntime) -> str:
        session = self._own_session(runtime.state)
        if session is None:
            return "No program has run in this thread yet."
        run = await self.client.sessions.last_run(session["id"])
        if run is None:
            return (
                "No run is recorded for this thread's session. "
                "The server keeps runs in memory, so a restart clears them."
            )
        return t.as_text(run.raw)

    async def _packages_search(self, runtime: ToolRuntime, query: str = "") -> str | ToolMessage | Command[Any]:
        if not self._exposed:
            return t.as_text(await self.client.blueprints.search_packages(self.blueprint, query))

        async def search(session_id: str) -> str:
            return t.as_text(await self.client.sessions.search_packages(session_id, query))

        return await self._in_session(t.PACKAGES_SEARCH, runtime, search)

    async def _packages_docs(self, name: str, runtime: ToolRuntime) -> str | ToolMessage | Command[Any]:
        if not self._exposed:
            return (await self.client.blueprints.package_docs(self.blueprint, name)).text

        async def docs(session_id: str) -> str:
            return (await self.client.sessions.package_docs(session_id, name)).text

        return await self._in_session(t.PACKAGES_DOCS, runtime, docs)

    async def _in_session(
        self, tool: str, runtime: ToolRuntime, fetch: Callable[[str], Awaitable[str]]
    ) -> str | ToolMessage | Command[Any]:
        """Discovery that includes the session's own `@tools`, so it needs the thread's session."""
        stored = self._own_session(runtime.state)
        session = stored or await self._open_for_thread(runtime)
        self._check_binding(session, self._variables(runtime.context))
        try:
            text = await fetch(session["id"])
        except UnknownSessionError:
            session = await self._replace_session(session, runtime.context)
            text = await fetch(session["id"])
        if session == stored:
            return text
        message = ToolMessage(content=text, tool_call_id=runtime.tool_call_id or "", name=tool)
        return _with_session(message, session, stored)

    async def _builtins_list(self, runtime: ToolRuntime) -> str:
        return t.as_text(await self.client.blueprints.builtins(self.blueprint))

    async def _builtins_docs(self, names: list[str], runtime: ToolRuntime) -> str:
        return t.as_text(await self.client.blueprints.builtin_docs(self.blueprint, names))

    # Sessions

    def _own_session(self, state: Mapping[str, Any]) -> SubmilliSession | None:
        """The thread's session, if it was opened on this server for this Blueprint."""
        session = state.get("submilli_session")
        if not session or session.get("server") != self.client.base_url or session.get("blueprint") != self.blueprint:
            return None
        if session.get("tools") != self._exposed.digest:
            # Opened with other application tools, which a session keeps for life.
            return None
        return cast(SubmilliSession, session)

    def _session_tools(self) -> frozenset[str]:
        """The tools whose first call opens the thread's session."""
        if self._exposed:
            return frozenset({t.EXECUTE, t.PACKAGES_SEARCH, t.PACKAGES_DOCS})
        return frozenset({t.EXECUTE})

    def _check_binding(self, session: SubmilliSession, variables: dict[str, str]) -> None:
        if session["variables"] == variables:
            return
        differing = sorted(
            name
            for name in set(session["variables"]) | set(variables)
            if session["variables"].get(name) != variables.get(name)
        )
        raise SubmilliSessionMismatchError(
            "This thread's Submilli session was opened with different variables than this run's context "
            f"gives (differing: {', '.join(differing)}). A session is never shared between bindings: "
            "run this context in a new thread, or end the thread's session first with "
            "`SubmilliMiddleware.aclose_thread`."
        )

    async def _open(self, context: Any) -> SubmilliSession:
        variables = self._variables(context)
        created = await self.client.sessions.create(
            self.blueprint, variables=variables, secrets=self._secrets(context), tools=self._exposed.catalog or None
        )
        for warning in created.warnings:
            logger.warning("submilli-server: %s", warning)
        logger.debug("opened Submilli session %s for blueprint %s", created.id, self.blueprint)
        session = SubmilliSession(
            id=created.id, server=self.client.base_url, blueprint=self.blueprint, variables=variables
        )
        if self._exposed:
            session["tools"] = self._exposed.digest
        return session

    async def _open_for_thread(self, runtime: ToolRuntime) -> SubmilliSession:
        thread = (runtime.config or {}).get("configurable", {}).get("thread_id")
        if thread is None:
            return await self._open(runtime.context)
        key = f"thread:{thread}"
        async with self._lock(key):
            if key in self._replacements:
                return self._replacements[key]
            session = await self._open(runtime.context)
            self._remember(key, session)
            return session

    async def _replace_session(self, ended: SubmilliSession, context: Any) -> SubmilliSession:
        key = f"session:{ended['id']}"
        async with self._lock(key):
            if key in self._replacements:
                return self._replacements[key]
            session = await self._open(context)
            self._remember(key, session)
            return session

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def _remember(self, key: str, session: SubmilliSession) -> None:
        self._replacements[key] = session
        while len(self._replacements) > 1024:
            old, _ = self._replacements.popitem(last=False)
            self._locks.pop(old, None)

    async def _describe(self) -> None:
        """Fetch the Blueprint's tool descriptions once, and put them on the tools."""
        if self._described:
            return
        if self._describe_lock is None:
            self._describe_lock = asyncio.Lock()
        async with self._describe_lock:
            if self._described:
                return
            described = t.descriptions(await self.client.blueprints.prompt(self.blueprint))
            if self._exposed:
                described[t.EXECUTE] += self._exposed.prompt()
            for tool in self.tools:
                tool.description = described[tool.name]
            self._described = True


def _with_session(
    message: ToolMessage,
    session: SubmilliSession,
    stored: SubmilliSession | None,
    updates: Sequence[tuple[str, Any]] = (),
) -> ToolMessage | Command[Any]:
    """`message`, with the thread's new session and the state updates of the tools a program called."""
    if session == stored and not updates:
        return message
    update: dict[str, Any] = {}
    for key, value in updates:
        # One update per key reaches the graph. Dicts, such as deepagents'
        # `files`, are merged; otherwise the last write wins.
        previous = update.get(key)
        update[key] = {**previous, **value} if isinstance(previous, dict) and isinstance(value, dict) else value
    update["messages"] = [message]
    if session != stored:
        update["submilli_session"] = session
    return Command(update=update)


def _mapper(value: ContextMapper | Mapping[str, str] | None, default: ContextMapper) -> Callable[[Any], dict[str, str]]:
    if value is None:
        source: ContextMapper = default
    elif callable(value):
        source = value
    else:
        fixed = dict(value)
        source = lambda _: fixed  # noqa: E731

    def mapped(context: Any) -> dict[str, str]:
        values = dict(source(context) or {})
        for name, item in values.items():
            if not isinstance(name, str) or not isinstance(item, str):
                raise TypeError(f"Submilli variables and secrets must be strings; got {name!r}: {type(item).__name__}")
        return values

    return mapped


def _context_value(context: Any, name: str) -> Any:
    if context is None:
        return None
    if isinstance(context, Mapping):
        return context.get(name)
    return getattr(context, name, None)


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)
