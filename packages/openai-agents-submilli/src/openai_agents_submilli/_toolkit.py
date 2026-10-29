"""`SubmilliTools`: Submilli code mode for OpenAI Agents SDK agents."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from agents import FunctionTool, RunContextWrapper
from agents.tool import FunctionToolCustomDataContext
from agents.tool_context import ToolContext

from openai_agents_submilli import _tools as t
from openai_agents_submilli._app_tools import ProgramTools, as_function_tools
from openai_agents_submilli._program import (
    MAX_PROGRAM_CALLS,
    ProgramInput,
    nested_run_config,
    program_agent,
    unwrap,
)
from openai_agents_submilli._sessions import InMemorySessionStore, SessionStore, SubmilliSession
from submilli import (
    AsyncSubmilli,
    ExecuteResult,
    IdempotencyError,
    RunStateError,
    SessionRequiresSecretsError,
    SubmilliError,
    UnknownSessionError,
    idempotency_key,
)

logger = logging.getLogger(__name__)

ContextMapper = Callable[[Any], Mapping[str, str]]
ConversationMapper = Callable[[Any], "str | None"]

# Outcomes kept for `custom_data` until the SDK asks for them, by tool call id.
_MAX_PENDING_DATA = 1024


class SubmilliSessionMismatchError(SubmilliError):
    """The run's context binds different variables than the conversation's session was opened with."""


class SubmilliConversationError(SubmilliError):
    """The run's context names no conversation, so there is no session to run programs in."""


def user_id_variables(context: Any) -> dict[str, str]:
    """The default `variables`: `user_id` in the run context becomes `userId`.

    Reads a mapping key or an attribute, so a dict, a dataclass, and a Pydantic
    model all work. A context without `user_id` binds no variables.
    """
    user_id = _context_value(context, "user_id")
    return {} if user_id is None else {"userId": str(user_id)}


def context_conversation_id(context: Any) -> str | None:
    """The default `conversation_id`: `conversation_id` in the run context."""
    value = _context_value(context, "conversation_id")
    return None if value is None else str(value)


def no_secrets(_: Any) -> dict[str, str]:
    """The default `secrets`: none."""
    return {}


class SubmilliTools:
    """The server's six tools for an OpenAI Agents SDK agent, with one Submilli session per conversation.

    Every program runs under the Blueprint's policy, in the conversation's
    session, bound to the variables the run's context gives. Build it and the
    agent once and share them; the user and the conversation come with each
    run, in its context.

    Args:
        blueprint: The registered Blueprint every program runs under.
        server: Base URL of submilli-server. Defaults to `SUBMILLI_SERVER`,
            then `http://127.0.0.1:8128`.
        token: API token. Defaults to `SUBMILLI_SERVER_TOKEN`.
        variables: Values for the Blueprint's variables, from the run's
            context (`RunContextWrapper.context`): a callable taking the
            context and returning `dict[str, str]`, or a fixed mapping. The
            default maps `user_id` in the context to `userId`.
        secrets: Values for the Blueprint's `harness` secrets, the same way.
            Sent when a session opens and when the server asks for them again
            after a restart; never stored. Default: none.
        conversation_id: The conversation a run belongs to, from its context.
            Programs of one conversation share a session. The default reads
            `conversation_id` from the context.
        store: Where each conversation's session is kept between turns.
            Defaults to an `InMemorySessionStore`; use `SQLiteSessionStore`, or
            your own `SessionStore`, to keep sessions across restarts.
        client: A `submilli.AsyncSubmilli` to use instead of building one
            from `server` and `token`.
        ptc: Programmatic tool calling: the application's own tools, which
            programs may call as the `@tools` Package. `FunctionTool`s, or
            functions to wrap with `function_tool`. Each call runs here as an
            ordinary tool call, with the run's context, the tool's guardrails,
            and its `needs_approval`, and its result goes back to the program.
            The Blueprint must grant `tools.call`. The model is not given these
            tools directly; add them to the agent as well for that.

    Example:
        ```python
        from agents import Agent, Runner
        from openai_agents_submilli import SubmilliTools

        submilli = SubmilliTools(blueprint="research")
        agent = Agent(name="researcher", tools=submilli.tools)
        await Runner.run(agent, "...", context={"user_id": "u_ada", "conversation_id": "c_1"})
        ```
    """

    def __init__(
        self,
        blueprint: str,
        server: str | None = None,
        token: str | None = None,
        variables: ContextMapper | Mapping[str, str] | None = None,
        secrets: ContextMapper | Mapping[str, str] | None = None,
        *,
        conversation_id: ConversationMapper | None = None,
        store: SessionStore | None = None,
        client: AsyncSubmilli | None = None,
        ptc: Sequence[FunctionTool | Callable[..., Any]] | None = None,
    ) -> None:
        self.blueprint: str = blueprint
        self.client: AsyncSubmilli = client or AsyncSubmilli(base_url=server, token=token)
        self.store: SessionStore = store if store is not None else InMemorySessionStore()
        self._owns_client = client is None
        self._variables = _mapper(variables, user_id_variables)
        self._secrets = _mapper(secrets, no_secrets)
        self._conversation = conversation_id or context_conversation_id
        self._program_tools = ProgramTools(as_function_tools(ptc))
        self._program_agent = program_agent(self.client, self._program_tools) if self._program_tools else None
        self._described = False
        self._describe_lock: asyncio.Lock | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.tools: list[FunctionTool] = self._build_tools()

    # Conversations

    async def aclose_conversation(self, conversation_id: str) -> bool:
        """End the conversation's session, and forget it.

        Returns `False` when the conversation had no session or the server had
        already ended it. Its next program opens a new session, which may bind
        another user.
        """
        session = await self.store.get(conversation_id)
        if session is None:
            return False
        deleted = await self.client.sessions.delete(session["id"])
        await self.store.set(conversation_id, None)
        return deleted

    async def aclose(self) -> None:
        """Close the HTTP connections of the client this toolkit built. Sessions stay open."""
        if self._owns_client:
            await self.client.aclose()

    # Tools

    def _build_tools(self) -> list[FunctionTool]:
        handlers: dict[str, Callable[[ToolContext[Any], dict[str, Any]], Awaitable[str]]] = {
            t.EXECUTE: self._execute,
            t.LAST_RUN: self._last_run,
            t.PACKAGES_SEARCH: self._packages_search,
            t.PACKAGES_DOCS: self._packages_docs,
            t.BUILTINS_LIST: self._builtins_list,
            t.BUILTINS_DOCS: self._builtins_docs,
        }
        tools: list[FunctionTool] = []
        for spec in t.SPECS:
            handler = handlers[spec.name]

            async def invoke(
                context: ToolContext[Any],
                arguments: str,
                handler: Callable[[ToolContext[Any], dict[str, Any]], Awaitable[str]] = handler,
            ) -> str:
                return await handler(context, json.loads(arguments or "{}"))

            tool = FunctionTool(
                name=spec.name,
                description=spec.fallback,
                params_json_schema=spec.params,
                on_invoke_tool=invoke,
                is_enabled=self._prepare,
            )
            if spec.name == t.EXECUTE:
                tool = self._execute_tool(tool)
            tools.append(tool)
        return tools

    def _execute_tool(self, tool: FunctionTool) -> FunctionTool:
        """The execute tool, which reports its ids in `custom_data`.

        With application tools it is built as the nested agent's tool, so that
        the SDK keeps, serializes and resumes the nested run of a program
        interrupted for an approval.
        """
        if self._program_agent is not None:
            template = self._program_agent.as_tool(t.EXECUTE, tool.description, failure_error_function=None)
            tool = dataclasses.replace(
                template,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=tool.on_invoke_tool,
                is_enabled=self._prepare_execute,
            )
        else:
            tool.is_enabled = self._prepare_execute
        tool.custom_data_extractor = self._custom_data
        return tool

    async def _prepare(self, context: RunContextWrapper[Any], agent: Any) -> bool:
        """Before each model call: put the Blueprint's descriptions on the tools."""
        await self._describe()
        return True

    async def _prepare_execute(self, context: RunContextWrapper[Any], agent: Any) -> bool:
        """Before each model call: also refuse a run whose variables differ from the session's."""
        await self._describe()
        conversation = self._conversation(context.context)
        if conversation is not None:
            session = self._own(await self.store.get(conversation))
            if session is not None:
                self._check_binding(session, self._variables(context.context))
        return True

    def _custom_data(self, data: FunctionToolCustomDataContext) -> dict[str, Any] | None:
        return self._data.pop(data.tool_context.tool_call_id, None)

    async def _execute(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        code = str(arguments["code"])
        conversation = self._require_conversation(context.context)
        session = await self._session(conversation, context.context)
        key = idempotency_key(context.tool_call_id) if context.tool_call_id else None
        notice = None
        rebound = False
        while True:
            try:
                output = await self._run(context, code, session["id"], key, notice)
                break
            except UnknownSessionError:
                if notice is not None:
                    raise
                session = await self._replace(conversation, session, context.context)
                notice = t.SESSION_REPLACED_NOTE
            except SessionRequiresSecretsError as error:
                if rebound:
                    raise SubmilliError(
                        f"submilli-server still asks for harness secrets {error.required} after a rebind; "
                        "check the `secrets` given to SubmilliTools"
                    ) from error
                await self.client.sessions.rebind(session["id"], self._secrets(context.context))
                rebound = True
        if isinstance(output, str):
            return output
        content, data = output
        self._remember(context.tool_call_id, data)
        return json.dumps(content, ensure_ascii=False)

    async def _run(
        self, context: ToolContext[Any], code: str, session_id: str, key: str | None, notice: str | None
    ) -> t.Outcome | str:
        """Run the program once: its outcome, or the text to hand the SDK when an approval interrupted it.

        Raises `UnknownSessionError` and `SessionRequiresSecretsError` for the
        caller to replace the session or rebind its secrets, which needs the
        run's context.
        """
        if self._program_agent is None:
            try:
                state = await self.client.sessions.execute(session_id, code, idempotency_key=key)
            except IdempotencyError as error:
                return t.idempotency_outcome(error, session_id=session_id)
            except RunStateError as error:
                return t.run_state_outcome(error, session_id=session_id, tool_calls=[])
            if not isinstance(state, ExecuteResult):
                raise SubmilliError(
                    f"run {state.execution_id} did not finish, in a session opened without application tools"
                )
            return t.execute_outcome(state, notice=notice)
        # With application tools the nested agent runs the program, from its
        # first execute on. A program resumed after an approval continues from
        # the nested run the SDK saved, whose input names the session it began
        # in, so it is never started again in another session.
        program = ProgramInput(code=code, session_id=session_id, idempotency_key=key, notice=notice)
        nested = self._program_agent.as_tool(
            t.EXECUTE,
            "",
            failure_error_function=None,
            run_config=nested_run_config(context.run_config),
            max_turns=MAX_PROGRAM_CALLS,
        )
        output = await nested.on_invoke_tool(context, json.dumps({"input": program.to_json()}))
        outcome = unwrap(output)
        # `None` when interrupted for an approval: the SDK holds the output back.
        return outcome if outcome is not None else t.as_text(output)

    async def _last_run(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        conversation = self._require_conversation(context.context)
        session = self._own(await self.store.get(conversation))
        if session is None:
            return "No program has run in this conversation yet."
        run = await self.client.sessions.last_run(session["id"])
        if run is None:
            return (
                "No run is recorded for this conversation's session. "
                "The server keeps runs in memory, so a restart clears them."
            )
        return t.as_text(run.raw)

    async def _packages_search(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "")
        if not self._program_tools:
            return t.as_text(await self.client.blueprints.search_packages(self.blueprint, query))

        async def search(session_id: str) -> str:
            return t.as_text(await self.client.sessions.search_packages(session_id, query))

        return await self._in_session(context, search)

    async def _packages_docs(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        if not self._program_tools:
            return (await self.client.blueprints.package_docs(self.blueprint, name)).text

        async def docs(session_id: str) -> str:
            return (await self.client.sessions.package_docs(session_id, name)).text

        return await self._in_session(context, docs)

    async def _in_session(self, context: ToolContext[Any], fetch: Callable[[str], Awaitable[str]]) -> str:
        """Discovery that includes the session's own `@tools`, so it needs the conversation's session."""
        conversation = self._require_conversation(context.context)
        session = await self._session(conversation, context.context)
        try:
            return await fetch(session["id"])
        except UnknownSessionError:
            session = await self._replace(conversation, session, context.context)
            return await fetch(session["id"])

    async def _builtins_list(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        return t.as_text(await self.client.blueprints.builtins(self.blueprint))

    async def _builtins_docs(self, context: ToolContext[Any], arguments: dict[str, Any]) -> str:
        names = [str(name) for name in arguments.get("names") or []]
        return t.as_text(await self.client.blueprints.builtin_docs(self.blueprint, names))

    # Sessions

    def _require_conversation(self, context: Any) -> str:
        conversation = self._conversation(context)
        if conversation is None:
            raise SubmilliConversationError(
                "The run's context names no conversation, so Submilli can't tell which session its programs "
                "belong to. Pass a context with a `conversation_id` (for example the id of the agent's SDK "
                "session), or give SubmilliTools a `conversation_id` function."
            )
        return conversation

    def _own(self, session: SubmilliSession | None) -> SubmilliSession | None:
        """The stored session, if it was opened on this server for this Blueprint and these tools."""
        if session is None:
            return None
        if session.get("server") != self.client.base_url or session.get("blueprint") != self.blueprint:
            return None
        if session.get("tools") != self._program_tools.digest:
            # Opened with other application tools, which a session keeps for life.
            return None
        return session

    async def _session(self, conversation: str, context: Any) -> SubmilliSession:
        """The conversation's session, opened if it has none."""
        variables = self._variables(context)
        session = self._own(await self.store.get(conversation))
        if session is None:
            async with self._lock(conversation):
                session = self._own(await self.store.get(conversation))
                if session is None:
                    session = await self._open(context)
                    await self.store.set(conversation, session)
        self._check_binding(session, variables)
        return session

    async def _replace(self, conversation: str, ended: SubmilliSession, context: Any) -> SubmilliSession:
        """A new session for a conversation whose session ended, shared by calls that found it ended."""
        async with self._lock(conversation):
            current = self._own(await self.store.get(conversation))
            if current is not None and current["id"] != ended["id"]:
                return current
            session = await self._open(context)
            await self.store.set(conversation, session)
            return session

    def _check_binding(self, session: SubmilliSession, variables: dict[str, str]) -> None:
        if session["variables"] == variables:
            return
        differing = sorted(
            name
            for name in set(session["variables"]) | set(variables)
            if session["variables"].get(name) != variables.get(name)
        )
        raise SubmilliSessionMismatchError(
            "This conversation's Submilli session was opened with different variables than this run's context "
            f"gives (differing: {', '.join(differing)}). A session is never shared between bindings: run this "
            "context in a new conversation, or end the conversation's session first with "
            "`SubmilliTools.aclose_conversation`."
        )

    async def _open(self, context: Any) -> SubmilliSession:
        variables = self._variables(context)
        created = await self.client.sessions.create(
            self.blueprint,
            variables=variables,
            secrets=self._secrets(context),
            tools=self._program_tools.catalog or None,
        )
        for warning in created.warnings:
            logger.warning("submilli-server: %s", warning)
        logger.debug("opened Submilli session %s for blueprint %s", created.id, self.blueprint)
        session = SubmilliSession(
            id=created.id, server=self.client.base_url, blueprint=self.blueprint, variables=variables
        )
        if self._program_tools:
            session["tools"] = self._program_tools.digest
        return session

    def _lock(self, conversation: str) -> asyncio.Lock:
        lock = self._locks.get(conversation)
        if lock is None:
            if len(self._locks) > 1024:
                for idle in [name for name, held in self._locks.items() if not held.locked()]:
                    del self._locks[idle]
            lock = self._locks[conversation] = asyncio.Lock()
        return lock

    def _remember(self, tool_call_id: str, data: dict[str, Any]) -> None:
        self._data[tool_call_id] = data
        while len(self._data) > _MAX_PENDING_DATA:
            self._data.popitem(last=False)

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
            if self._program_tools:
                described[t.EXECUTE] += self._program_tools.prompt()
            for tool in self.tools:
                tool.description = described[tool.name]
            self._described = True


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
