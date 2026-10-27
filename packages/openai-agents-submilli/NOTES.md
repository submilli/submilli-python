# Notes

These are the decisions the brief (Linear SUB-2053) left open, and what got in the way. The
package mirrors `langchain-submilli` where the frameworks allow it; where they differ, this file
says why. It was built and tested against openai-agents 0.24.0, the `submilli` SDK 0.1.2 from this
workspace, and submilli-server 0.4.1 and 0.4.3 release binaries.

## Decisions

### The shape: a toolkit, not middleware

The Agents SDK has no middleware. Its extension point for a code-mode harness is a list of
`FunctionTool`s on `Agent(tools=...)`, so the package is `SubmilliTools`, whose `tools` are the
server's six tools. Names, arguments, and the result shape are the same as `langchain-submilli`'s:
the server's own tool names (its responses refer to them), the MCP tools' field descriptions,
and `result`, `console`, `error`, `denied` (plus `note`, `notice`, `discovery_warnings`) as JSON
text for the model.

- **Strict schemas.** The SDK sends function tools in strict mode by default, where every
  property is required. `packages__search` therefore has a required `query`; its description
  says that `""` lists everything.
- **Descriptions.** They come from `GET /v1/blueprints/{name}/prompt` before the first model
  call, and are cached for the toolkit. The SDK has no hook before a model call, but it calls a
  tool's `is_enabled` callable (async allowed) on every turn before it reads the tools, so the
  toolkit fetches there and sets the descriptions on its tool objects. Until then each tool has a
  short fallback description. It does not fetch in `__init__`, which would need a blocking
  request while the agent is built.
- **What the application gets.** LangChain has message artifacts; the SDK has
  `ToolCallOutputItem.custom_data`, filled by a tool's `custom_data_extractor`, which the model
  never sees and `RunState` serializes. It holds `execution_id`, `session_id`, `tool_calls`, and
  on refusals `refusal` or `run_status`, as the LangChain artifact does.
- **Errors.** Application errors raise. Those found in `is_enabled` (a bad token, an unknown
  Blueprint, a variables mismatch) come out of `Runner.run` as they are. Those found while a
  tool runs (invalid variables, a missing conversation id) come out as the cause of the SDK's
  `UserError("Error running tool ...")`, because the SDK wraps every exception a tool raises that
  is not its own. Program failures, denials, idempotency refusals, and lost runs are tool
  results.
- **Nothing to hide.** The SDK adds no file or shell tools of its own (its hosted and shell tools
  are opt-in), so there is no `hide_filesystem_tools`.

### Sessions

- **Keyed by a conversation id from the run context.** A tool sees `ToolContext.context`, the
  application's own context object, but not the SDK `Session` the run uses, nor any thread id.
  So the conversation id comes from the context: `conversation_id` by default (key or
  attribute), or a function. Without one, the tool raises `SubmilliConversationError`: falling
  back to a session per process or per context object would silently share sessions, or leak
  one per turn.
- **Stored in a `SessionStore`.** LangChain keeps the session in checkpointed graph state. The
  SDK has no such state for tools, so the toolkit keeps `{id, server, blueprint, variables,
  tools}` per conversation in a store: `InMemorySessionStore` by default, `SQLiteSessionStore`
  (its own table, which can live in the same file as the SDK's `SQLiteSession`), or the
  application's own implementation of the two-method protocol. Secrets are never stored.
- **Where the session opens.** Inside the tool, under a per-conversation lock, and it is written
  to the store before any program runs. The SDK does not checkpoint between the model's turn and
  the tools, so there is no earlier point that would survive a crash. A tool call resumed from a
  `RunState` finds the stored session and replays its run with the same key.
- **Mismatch.** The execute tool's `is_enabled` hook compares the context's variables with the
  stored session's before every model call, and raises `SubmilliSessionMismatchError` before the
  model runs. The tools check again before each call.
- **Blueprint, server, or catalog change.** A stored session opened for another Blueprint,
  server, or set of application tools is ignored and replaced, as in LangChain.
- **Closing.** `aclose_conversation(id)` deletes the session and forgets it. Nothing closes a
  session automatically.
- **Retries.** As in LangChain: `404 unknown session` opens a replacement once per call (calls
  that find the same session gone share one replacement) and tells the model; `409
  session_requires_secrets` rebinds once.

### Idempotency

The key is `submilli.idempotency_key(ToolContext.tool_call_id)`: the id as is when it is
printable ASCII of at most 120 bytes, otherwise its sha256. OpenAI's call ids are unique within a
conversation. The SDK's own model contract requires call ids to be stable and unique for an
invocation across resumes, which is what the key needs.

### Application tools (`ptc`) and approvals

This is where the design differs most from LangChain, and it is worth reading before changing
`_program.py`.

- **The problem.** A program suspends at each `@tools` call. The application's tool should run
  as the SDK would run it, with `needs_approval`, guardrails, timeouts, context injection, and
  tracing, and an approval should pause the whole agent run so that a person can decide, possibly
  much later and in another process. The SDK can only pause a run for an approval *before* it
  invokes a tool, judged on the tool call's arguments. A program's calls happen *inside* the
  execute tool, and are not known before it runs. `needs_approval` on the execute tool would
  pause once per program, for the code, not once per call.
- **The answer: a nested agent whose "model" is the program.** The SDK supports agents used as
  tools (`Agent.as_tool`), and pauses the outer run for approvals inside the nested run: the
  outer `RunResult.interruptions` holds the nested tool's `ToolApprovalItem`, `RunState.approve`
  and `reject` apply to it, and `to_string` / `from_string` serialize the nested state with the
  outer. With application tools, the execute tool is built that way. Its nested agent has the
  `ptc` tools, and its model is `ProgramModel`, which calls no LLM: on each turn it replays the
  execute with the same idempotency key to get the run's current state, answers the calls whose
  outputs the SDK recorded, and either returns a function call for the call the run waits on
  next (with the server's `call_id`), or, when the run has finished, a final message holding the
  execute result. The SDK runs each function call as a normal tool call, approvals included.
- **Stateless model.** Everything `ProgramModel` needs is in its input: the first message carries
  the code, the session id, and the key, and the SDK's recorded function calls and outputs carry
  the rest. That is what makes a resume in another process work. The session id travels in the
  nested input as well as in the store, so a resumed program finds its run even if the store
  changed.
- **Always through the nested agent.** With application tools, the nested agent runs the
  program from its first execute on. On a resume, the SDK holds the nested run's pending result
  and expects the agent tool to consume it; returning without it made the SDK raise the same
  approvals again. An earlier version executed once in the outer tool first, which on a resume
  with a lost store started the program again in a new session. Session replacement and secret
  rebinds still happen in the outer tool, which has the run's context: the nested model lets
  `404 unknown session` and `409 session_requires_secrets` propagate on its first turn, and the
  outer tool retries. A `404` after the program has made calls is a lost run, reported to the
  model, never a restart.
- **A resume without the store** still finishes the program in the session it began in, because
  the nested input names it. The outer tool resolves the conversation's session before handing
  over, so with an empty store it opens a session that this call does not use; later calls in
  the conversation use it.
- **Tool outputs.** Each `ptc` tool is copied so that its exceptions are raised instead of
  formatted for a model, and its output is wrapped as `{"__submilli_program_call__": "ok",
  value}` or `"failed"`, `message`. Anything else the SDK records in its place, which is the
  approval's rejection message, a guardrail's refusal, or a timeout message, is answered as
  `denied` with that text, so the program throws `PermissionDeniedError`. The copy drops the
  tool's `output_json_schema` and output adapter, which the SDK would check the envelope against
  (the schema is still declared to the server). These are private dataclass fields of
  `FunctionTool` (`_failure_error_function`, `_use_default_failure_error_function`,
  `_output_type_adapter`), set through `dataclasses.replace`. That, and building the execute tool
  from an `as_tool` template so that `RunState.from_string` finds the nested agent through the
  tool's `_agent_instance`, are the package's two dependencies on SDK internals. The dependency
  is pinned below the next minor release, and the tests exercise both paths.
- **Declaring.** The input schema is the tool's `params_json_schema` (what the model would see,
  without the injected context). The output schema is the tool's `output_json_schema`, else
  the return annotation's through a pydantic `TypeAdapter`, else `string`, where the program gets
  the text the model would get. Local `$ref`s are inlined, because the server's mapper does not
  resolve them.
- **The nested run's settings.** The SDK passes the outer `RunConfig` to an agent tool unless it
  is given one, and `RunConfig.model` overrides every agent's model, which would send the
  program's turns to an LLM. The nested run gets a `RunConfig` with the outer run's tracing
  settings only. Each program call is one nested turn; `max_turns` is 10,000.
- **What the model is told.** The execute description gains the same paragraph as in LangChain,
  naming the tools and, when any needs approval, saying that a refusal is final.
- **Discovery** includes `@tools` through `/v1/sessions/{id}/packages/...` when there are
  application tools, which opens the conversation's session.
- **Not done: deciding at the call.** LangChain's `interrupt_on` is not mirrored. The SDK's own
  setting on the tool, `needs_approval` (a bool, or an async function of the arguments), already
  says which calls need approval, and it applies unchanged.

### Packaging, tests, CI

- Name `openai-agents-submilli`, module `openai_agents_submilli`, as `langchain-submilli` is
  named. Apache-2.0, Python 3.11 or later. `openai-agents>=0.24.0,<0.25`: the SDK is below 1.0,
  so a minor release may break, and the package touches the internals named above.
- Unit tests run the SDK's real `Runner` with a scripted model against the same respx fake of
  the HTTP API as `langchain-submilli` (copied, not shared, so each package's tests stand alone).
  They cover sessions, refusals, replays, approvals (approve, reject, two in one program, a
  serialized `RunState`, and a streamed run), and `run_sync`.
- Integration tests start their own server from `SUBMILLI_BIN` or `PATH` with Blueprints
  registered over the admin API, so CI runs all of them. They cover typed tools with the run's
  context, the Blueprint refusing a tool, a failing tool, two approvals resumed in "another
  process" (a new toolkit and agent, `RunState.from_string`, a shared `SQLiteSessionStore`),
  a resume without the store, session reuse and isolation, replay, a mismatch, a restart, and an
  expired session.
- Typing: `mypy --strict`, pyright, and `pyright --verifytypes` at 100%, as for the other
  packages.

### What this shares with `langchain-submilli`

The model-facing texts (denial note, idempotency and run-state messages) and the session
policy are the same, copied rather than shared, because they are private to each package. If a
third integration needs them, they belong in the `submilli` SDK.

## What in submilli-runtime got in the way

Nothing new. The gaps `langchain-submilli` recorded (SUB-1685..1693) apply unchanged; two matter
more here:

1. **A run's answered calls are not readable.** (as in langchain-submilli's notes) The nested
   model rebuilds a program's progress from the SDK's recorded outputs. A server-side history of
   a run's answered calls would make the records in `custom_data["tool_calls"]` authoritative.
2. **HTTP execute cannot carry the tool call id** (SUB-1686), so the run's audit record has the
   idempotency key, which is the tool call id unless it was hashed.
