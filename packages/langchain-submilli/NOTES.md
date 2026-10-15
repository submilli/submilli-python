# Notes

These are the decisions the brief left open, and what in submilli-runtime got in the way. The
package was built and tested against submilli-runtime at `60f2a857` (server 0.4.0, built from
source; the installed 0.3.0 binary was not used). Application tools (`ptc`, first named `expose_tools`) were added
for, and tested against, the published 0.4.1 release binaries, which also pass the restart and
expiry tests. The HTTP client now lives in its own package, `submilli`
(`packages/submilli` in the same workspace), described below. It was also tested against deepagents 0.7.24, langchain 1.4.5, langchain-core
1.6.10, and langgraph 1.2.15.

## Decisions

### Tools

- **Names.** The tools keep the server's own names (`submilli__typescript__execute` and the
  rest), without a prefix of ours. The server's responses refer to its tools by these names.
  For example, a package search with no hits points to `submilli__typescript__builtins__docs`.
  A renamed tool would contradict the text the model reads. The names also match the MCP
  endpoint and the Vercel example.
- **Arguments.** `code`, `query` (default `""`), `name`, and `names`, with the field
  descriptions of the server's MCP tools.
- **Descriptions.** They come from `GET /v1/blueprints/{name}/prompt`, which the middleware
  fetches on the first model call, in `awrap_model_call`, and caches for the instance. It does
  not fetch them in `__init__`, because that would need a blocking request while the agent is
  built. The descriptions are set on the middleware's own tool objects. Until then, each tool
  has a short fallback description, so it can be inspected before any run.
- **Result shape.** The content is JSON text with `result`, `console`, `error`, and `denied`, as
  in the Vercel example. A denial adds `note`, which tells the model the denial is final. A
  replaced session adds `notice`. `discovery_warnings` is passed through when present.
  `ToolMessage.status` is `"error"` whenever `error` is set.
- **Artifact.** The artifact holds `{execution_id, session_id}`. An idempotency refusal adds
  `refusal`, and `execution_id` when the server sent one.
- **Discovery tools never open a session.** `last_run` in a thread with no session answers in
  text and does not open one.
- **langchain-core quirk.** A `StructuredTool` whose argument schema has no fields gets no
  arguments at all, the injected `ToolRuntime` included. `last_run` and `builtins__list`
  therefore carry a hidden `runtime` field marked `InjectedToolArg`.

### Sessions

- **Where the session opens.** The session opens in `aafter_model`, when the model's message
  first asks for `submilli__typescript__execute`. It does not open inside the tool. The update
  from the `after_model` node is checkpointed before the tools node runs. A process that
  crashes mid-call and resumes from that checkpoint retries in the same session, and the
  idempotency key replays the run. Had the session opened inside the tool, the resumed call
  would find no session, open a new one, and run the program a second time. If the tool runs
  without the hook, it opens the session itself, with a per-thread lock, and returns it in a
  `Command`.
- **State.** The state key is `submilli_session`: `{id, server, blueprint, variables}`. It is
  marked `PrivateStateAttr`, so it is left out of the agent's input and output schemas, and
  applications read it with `aget_state`. Its reducer replaces the old value. Parallel tool
  calls that all find the session gone each write the same replacement, and the default
  channel would reject more than one write in a step.
- **Variables are stored, secrets are not.** The variables are kept as values, so a mismatch
  is an exact comparison. Secrets are recomputed from the context whenever they are needed.
- **Mismatch.** `abefore_agent` raises `SubmilliSessionMismatchError` before the model is
  called. The message names the variables that differ, not their values. A mismatch is the
  application's error, so it is raised rather than shown to the model. The tool checks again
  before every execute.
- **Blueprint or server change on a thread.** When the stored session was opened for another
  Blueprint or server, it is ignored and a new session is opened. The old session is left to
  its idle timeout.
- **Closing.** `await middleware.aclose_thread(agent, config)` deletes the thread's session and
  sets `submilli_session` to `None` with `aupdate_state`. The next execute opens a new session,
  and it may bind another user. `middleware.client.sessions.delete(id)` is the low-level form.
  Nothing closes a session automatically.
- **Retries.** On `404 unknown session`, the middleware opens a new session once per call, and
  the next call reuses that replacement. On `409 session_requires_secrets`, it rebinds once and
  retries. A second `409` raises `SubmilliError`.
- **Errors that are the application's raise; errors that are the model's are tool results.**
  These raise: 401, invalid variables or secrets, an unknown Blueprint, and an unreachable
  server. These come back as tool results: program failures, denials, and idempotency
  refusals.

### Idempotency

- The key is the tool call id when it is printable ASCII of at most 120 bytes. Otherwise it is
  the sha256 hex digest of the id's UTF-8 bytes.
- This assumes tool call ids are unique within a thread, as Anthropic's and OpenAI's are. A
  model that reuses an id, as the scripted model in `check.py` does (`call-1` for every call),
  would get the recorded result of an earlier call with the same code. The tests use unique
  ids.
- Across a session replacement nothing is deduplicated, because the ledger belongs to the
  session.

### Configuration

- `variables` and `secrets` accept a callable over `runtime.context` or a fixed mapping. The
  default reads `user_id` from a dict key or an attribute, so a dict, a dataclass, and a
  Pydantic model all work, and maps it to `userId`. Values must be strings; anything else raises
  `TypeError`.
- `client=` takes a ready `submilli.AsyncSubmilli`, which can wrap your own `httpx.AsyncClient`.
  The managed client keeps one `httpx.AsyncClient` per event loop, because applications and tests
  often run several loops.
- The default request timeout is 330 s. The server may wait up to 300 s for a running request
  with the same idempotency key.
- **Async only.** The tools' sync entry points raise `NotImplementedError` with a pointer to
  `ainvoke`. deepagents does not require a sync API.
- One `SubmilliMiddleware` per agent: two instances would share the `submilli_session` key.

### deepagents' file tools

- The middleware can hide them cleanly. `hide_filesystem_tools=True` filters `ls`, `read_file`,
  `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute` out of `request.tools` in
  `awrap_model_call`. deepagents' own `_ToolExclusionMiddleware` uses the same mechanism. The
  flag also refuses a call to one of them in `awrap_tool_call`. deepagents 0.7.24's
  `FilesystemMiddleware` adds no system-prompt text of its own, so no prose is left describing
  the hidden tools. The flag is opt-in, and the example uses it.
- **Limit.** deepagents' built-in `general-purpose` subagent is built with its own middleware
  list, and user middleware does not reach it. It keeps deepagents' file tools and gets no
  Submilli tools. The README says so, and points to `permissions=` for denying the file tools
  there too.
- The flag also hides any tool of the app's own with one of those names, `execute` included.

### Application tools (`ptc`)

Server 0.4.1 shipped SUB-1694: a session declares `tools`, programs call them through `@tools`,
each call suspends the run in memory, and the harness answers with `POST /v1/runs/{id}/resume`.

- **The option is `ptc`, not `tools`.** `AgentMiddleware.tools` is already the list of
  tools a middleware gives the model. These tools are not given to the model. It was first
  `expose_tools`; it was renamed before the first release to match `langchain-quickjs`'s
  `CodeInterpreterMiddleware(ptc=...)`. Unlike there, it takes tool objects only, not names:
  the session declares its tools when it opens, from the execute tool, which does not see the
  agent's tool list.
- **Declaring.** The input schema is `tool.tool_call_schema`, so injected arguments
  (`ToolRuntime`, `InjectedState`, `InjectedStore`) are not part of it. The output schema comes
  from the function's return annotation through a pydantic `TypeAdapter`. It is `string` when
  there is no annotation, when the annotation is a message, a `Command`, `Any`, or `str`, when
  the tool is not `response_format="content"`, or when the annotation has no JSON schema.
  Local `$ref`s are inlined, because the server's mapper does not resolve `$ref` and would type
  the field `unknown`. A recursive reference becomes `{}`.
- **What the server can't type.** `dict[str, X]` (`additionalProperties`) comes back
  `unknown`, with a warning when the session opens. It is declared anyway, because a program
  can still narrow an `unknown`. The README tells users to return a `TypedDict` or a model.
- **Running a call.** It goes through a `ToolNode` built over the exposed tools, invoked with a
  `ToolCallWithContext` and the execute tool's own config. That gives the agent's own injection,
  error handling (`handle_tool_errors=True`, so a raising tool becomes a `failed` answer), and
  `Command` validation. The nested call's id is the server's `call_id`. Other middleware's
  `wrap_tool_call` hooks do not run around nested calls.
- **Values.** A tool declared as text returns the `ToolMessage` text. A structured tool returns
  the parsed JSON of that text. A `ToolNode` writes a value that isn't JSON as its `str()`, so
  structured tools run through a copy whose function returns `to_jsonable_python(...)`. That
  makes a Pydantic model reach the program as an object. The copy keeps the signature
  (`functools.wraps`), so injected arguments are still detected.
- **State updates.** A nested tool that returns a `Command` has its non-message updates
  applied with the execute result. `ToolNode` only accepts a dict as a Command's update, so
  updates to the same key are combined: dicts are merged (deepagents' `files`), and anything
  else keeps the last value. A nested `Command.goto` is refused.
- **Approvals use LangGraph `interrupt()` inside the execute tool.** The request and decision
  shapes are `HumanInTheLoopMiddleware`'s, one action per pause, so existing approval UIs work.
  The hard part is that LangGraph gives a re-run task its earlier resume values by position,
  while the server has already moved past the calls those values answered. Re-running the
  tool replays the execute with the same `Idempotency-Key`, which answers the run's current
  state, but the first interrupt of the re-run would get the first stored value. The middleware
  therefore records, for each run, which call each pause was for. On a re-run it skips the
  interrupts that belong to earlier calls (`interrupt(None)` returns a stored value without
  raising) before it asks for the current one. Two approvals in one program are tested against
  the fake and against a real 0.4.1 server.
- **The limit of that record.** It lives in the middleware's memory. When the thread is resumed
  in another process, for example another worker, and the task already has resume values,
  they can't be matched to calls. Then the call is answered `denied`, saying the approval
  could not be matched; it is never matched by guessing. Whether the task has resume values is
  read from LangGraph's scratchpad (`langgraph._internal._constants.CONFIG_KEY_SCRATCHPAD`),
  which is private. A server-side history of a run's answered calls would make this
  stateless.
- **The record also carries** the call records and state updates of a run interrupted
  mid-way, so the invocation that finishes it reports every call, not only those after the
  last pause.
- **Nested interrupts are not supported.** An exposed tool that calls `interrupt()` itself
  would take resume values meant for approvals.
- **Lost runs.** A suspended run is lost when the server restarts (`404 unknown_run` on
  resume). The model gets a `run_lost` result saying its effects are unknown, as for
  `idempotency_incomplete`. `409 run_not_suspended` (a call timed out) and `400 unknown_call`
  make the middleware read the run with `GET /v1/runs/{id}` and carry on. A `running` state is
  polled with backoff.
- **Discovery.** With exposed tools, `packages__search` and `packages__docs` use
  `/v1/sessions/{id}/packages/...`, so `@tools` is included, and those calls open the session
  (in `aafter_model` too). Without exposed tools nothing changes.
- **Sessions keep their catalog.** `submilli_session.tools` holds a sha256 of the declared
  catalog, and a session whose fingerprint differs is not reused. Sessions stored before this
  field have none, which matches a middleware without exposed tools.
- **Seen on the real server.** `@tools` docs list every declared tool, including those the
  Blueprint's filter refuses. The filter is checked on each call, which then fails with
  `permission_denied`, `source: policy`, and the application never sees it. A tool with no
  arguments is called as `tools.name()`.

### Packaging, tests, CI

- The license is Apache-2.0, as submilli-runtime's. Python is 3.11 or later. Dependencies have
  lower bounds at the tested versions, under the next major version (deepagents under 0.8),
  and `uv.lock` pins the tested set. deepagents is an optional extra
  (`langchain-submilli[deepagents]`), because the middleware works with any LangChain v1 agent.
- The package lives in the `github.com/submilli/submilli-python` workspace, next to `submilli`,
  and its URLs point there.
- **Standard tests.** `ToolsUnitTests` runs for each of the six tools. `ToolsIntegrationTests`
  invokes a tool synchronously and on its own, without an agent runtime, so it cannot apply
  to these async, state-bound tools. It will apply to the future toolkit.
- **Integration tests.** Most run against an external server with the `research` Blueprint and
  are skipped when none is reachable. The restart and expiry tests start their own server from
  `SUBMILLI_BIN` or `PATH`, with a small Blueprint that needs no Package and declares a harness
  secret. A restart does not force a `409`: see "Harness secrets survive restarts" below, so the
  rebind path is covered by the unit tests only. Expiry uses `idle_timeout: 1s`: the test stops the server
  for 1.5 s, and the server reaps the session when it starts again. The resume test reopens the
  thread with a new middleware and agent over the same checkpointer, from the checkpoint taken
  just before the tools node.
- **Typing.** The package ships `py.typed`. `mypy --strict` and pyright check the sources,
  the tests, and the example, and `pyright --verifytypes` requires the public API to be fully
  typed and documented (100%). The internal modules are private (`_middleware.py`, `_tools.py`,
  `_app_tools.py`), and the public API is what `langchain_submilli` exports. Typing found two
  things worth changing. `SubmilliMiddleware` was generic over a context type it could never
  bind, so an agent built with `context_schema=` together with it failed type inference; it is
  now `AgentMiddleware[SubmilliState, Any, Any]`. And the example now declares a `Context`
  schema, so its `ainvoke(context=...)` type-checks.
- **CI** is the workspace's: ruff, mypy, pyright and `--verifytypes`, and the unit tests on
  Python 3.11 to 3.13. Its integration job downloads the released `submilli-server` and runs
  the tests that start their own server; those that need the `research` Blueprint's Packages
  are skipped there. The integration fixture registers Blueprints over the admin API, so the
  `submilli` CLI, which has no Linux release build, is not needed.
- The example keeps the brief's model id, `anthropic:claude-sonnet-5`. It uses an
  `InMemorySaver` and closes the thread's session when it exits.

### The `submilli` SDK

The HTTP client moved out of this package into `submilli`, a separate package in the same
workspace (`packages/submilli`). It knows the HTTP API and nothing about LangChain, so other
harnesses can use it. Both are published to PyPI on their own, from one repository, so a
change that spans both, such as application tools, is one pull request.

- **What moved.** The requests, the typed results and errors, `idempotency_key`, `$ref`
  inlining, and the suspend and resume loop (`runs.complete`). The loop takes an
  `on_tool_call` callback and lets an exception from it propagate, which leaves the run
  waiting on the server. That is what a LangGraph interrupt needs.
- **What stayed.** The middleware, the six tools, the state schema, turning LangChain tools
  into declarations, running nested calls through a `ToolNode`, approvals, and merging state
  updates.
- **Names.** Some names changed: `IdempotencyRefusedError` is `submilli.IdempotencyError`, and
  `SubmilliHTTPError` is `submilli.APIStatusError` with a subclass for each status. The
  middleware no longer re-exports client types.
- **The dependency.** `submilli>=0.1.0,<0.2` on PyPI. In the workspace,
  `[tool.uv.sources]` resolves it to `packages/submilli`. The published wheel names the PyPI
  requirement, which CI checks.

### Harness secrets survive restarts

The HTTP API reference says harness secrets are kept in memory only and do not survive a
restart. On 0.4.0 and 0.4.1 that is not what happens. The server stores them encrypted with
its secret-store key, in SQLite (`encrypted_harness_bindings`). It refuses session secrets
when no key is configured (`409`, "configure the secret-store encryption key before supplying
session secrets"). After a restart with a different key, it answers `409` "cannot decrypt
session secrets", not `session_requires_secrets`. The earlier restart test passed without ever
rebinding; it now says what it checks. The middleware's rebind on `session_requires_secrets`
stays, because the API still documents that answer.

## What in submilli-runtime got in the way

Each item below is filed in Linear, in the Submilli team's `server` project.

1. **MCP execute ignores idempotency.** (SUB-1685) `mcp/server.rs` passes `idempotency_key: None`. It
   already reads Claude Code's `_meta["claudecode/toolUseId"]`, but only for the run record. If
   the server took a key from MCP `_meta`, as a generic `submilli/idempotencyKey` or by reusing
   the tool call id, MCP harnesses would get the same retry safety as this package.
2. **HTTP execute cannot carry the tool call id.** (SUB-1686) `handlers/sessions.rs` records
   `tool_call_id: None`. An HTTP harness has no way to put its tool call id in the run's record
   and audit trail, except as the `Idempotency-Key`, which this package hashes when the id is
   long. A `Submilli-Tool-Call-Id` header would close the gap.
3. **A replay is not marked.** (SUB-1687) A replayed response is byte-identical to the original. A
   `replayed: true` field, or a response header, would let the harness tell the model or the
   application that nothing ran this time.
4. **Error codes are sometimes prose.** (SUB-1688) `404 {"error":"unknown session"}`,
   `"blueprint no longer exists"`, `"unknown blueprint"`, and the `400` bodies of
   `POST /v1/sessions` (`"invalid variables: ..."`) put a sentence where other endpoints put a
   stable snake_case code. The client has to match those strings.
5. **`execution_id` on refusals is undocumented.** (SUB-1689) Refusals carry an `execution_id`:
   `404 unknown session`, `409 session_requires_secrets`, and the idempotency refusals. The
   HTTP API reference shows bodies without it.
6. **`GET last-run` is ambiguous.** (SUB-1690) It answers `404` with an empty body both for an unknown
   session and for a session with no recorded run.
7. **`idle_timeout` is approximate.** (SUB-1691) A session idle past its timeout keeps answering until the
   next reaper sweep, every 30 s (`REAP_INTERVAL` in `app.rs`), or a restart. That is harmless
   for users, but a test cannot observe real expiry without restarting the server.
8. **Two names for the server URL.** (SUB-1692) The `submilli server ...` commands read
   `SUBMILLI_SERVER_URL`, while the harness examples, and so this package, use
   `SUBMILLI_SERVER`.
9. **The HTTP API has no file endpoints.** (SUB-1693) The MCP endpoint has `submilli__files__*`; the HTTP
   API does not. This package therefore has no file tools, and the model reads files with
   programs.
