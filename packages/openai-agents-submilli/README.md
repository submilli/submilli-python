# openai-agents-submilli

[Submilli](https://github.com/submilli/submilli-runtime) is a dedicated secure runtime for
code-mode agents, with built-in isolation, governance, and permissions.

`openai-agents-submilli` is the Python integration for the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). It lets your agent execute
TypeScript code in Submilli's runtime, and lets that code call your existing function tools
under the Blueprint's policy and the SDK's own approvals.

## Install

Python 3.11 or later and submilli-server 0.4.1 or later are required.

```sh
pip install openai-agents-submilli
```

Follow [Connect OpenAI Agents](https://submilli.ai/docs/tutorials/connect-openai-agents/) for a
complete setup with a server, Blueprint, model provider, agent, and session cleanup.

## Use it

```python
from agents import Agent, Runner
from openai_agents_submilli import SubmilliTools

# Server and token come from SUBMILLI_SERVER and SUBMILLI_SERVER_TOKEN.
submilli = SubmilliTools(blueprint="research")

# Built once and shared: the user and the conversation come with each run.
agent = Agent(name="researcher", instructions="...", tools=submilli.tools)

result = await Runner.run(
    agent,
    "What is new in the latest Rust release?",
    context={"user_id": "u_ada", "conversation_id": "c_1"},
)

# When the conversation is over:
await submilli.aclose_conversation("c_1")
```

## Toolkit reference

The tools are async. `Runner.run`, `Runner.run_streamed`, and `Runner.run_sync` all work.

```python
SubmilliTools(
    blueprint="research",
    server=None,
    token=None,
    variables=None,
    secrets=None,
    conversation_id=None,
    store=None,
    client=None,
    ptc=None,
)
```

| Setting | Contract |
| --- | --- |
| `blueprint` | Name of a Blueprint already [registered on the server](https://submilli.ai/docs/server/register-a-blueprint/); used for every program |
| `server` | submilli-server URL; defaults to `$SUBMILLI_SERVER`, then `http://127.0.0.1:8128` |
| `token` | Server token; defaults to `$SUBMILLI_SERVER_TOKEN` |
| `variables` | Fixed `dict[str, str]`, or a function from the run context to one; defaults to mapping context `user_id` to Blueprint variable `userId` |
| `secrets` | Fixed `dict[str, str]`, or a function from the run context to one, for the Blueprint's `harness` secrets |
| `conversation_id` | Function from the run context to the conversation's id; defaults to reading context `conversation_id` |
| `store` | Where each conversation's session id is kept: `InMemorySessionStore` (default), `SQLiteSessionStore`, or your own `SessionStore` |
| `client` | Existing `submilli.AsyncSubmilli`; replaces `server` and `token` |
| `ptc` | Application tools that programs may call through the `@tools` Package |

The run context is the `context` you pass to `Runner.run`. It may be a dict, a dataclass, or a
Pydantic model; the defaults read `user_id` and `conversation_id` as keys or attributes. The
model never sees it.

`submilli.tools` holds the six tools below. Before the first model call, the toolkit fetches the
instructions for using them from the Submilli server.

| Tool | Does |
| --- | --- |
| `submilli__typescript__execute` | Runs a program |
| `submilli__typescript__last_run` | Returns the conversation's last run and full console |
| `submilli__typescript__packages__search` | Searches Packages available to the program |
| `submilli__typescript__packages__docs` | Returns a Package's documentation and declarations |
| `submilli__typescript__builtins__list` | Lists built-ins available without an import |
| `submilli__typescript__builtins__docs` | Returns declarations for selected built-ins |

## Sessions, retries, and results

- The first program in a conversation opens a Submilli session, and later turns of the same
  conversation reuse it. The session id is kept in the `store`. The default store lives in
  memory; to reuse sessions after your application restarts, or from another worker, use
  `SQLiteSessionStore("conversations.db")` (it can share a file with the SDK's `SQLiteSession`)
  or implement `SessionStore` over your own database. Close the session with
  `await submilli.aclose_conversation(conversation_id)`.
- Blueprint variables are fixed when the session opens. A later run in the same conversation
  with different values raises `SubmilliSessionMismatchError` before the model is called.
- Harness secrets are sent when a session opens and are never stored. End the session before
  using rotated secrets.
- Each execute call sends an `Idempotency-Key` derived from the SDK's tool call id. A repeated
  call replays the recorded result. The toolkit reports conflicts and incomplete, in-progress,
  or unavailable replays without silently rerunning the program.
- When a session has ended, the toolkit opens a replacement and tells the model. If a restarted
  server asks for the session's secrets, it rebinds them once and retries.
- On a server with durable execution, the toolkit waits for a queued run. A run that stops
  without a result, such as one that is stuck or cancelled, comes back to the model as an error
  with the run's `status` and `reason`.
- The execute output the model reads contains `result`, `console`, `error`, `denied`, and the
  program's application-tool calls (`tool_calls`: tool, arguments, outcome), so that after a
  refusal the model can say which calls already took effect. The output item's `custom_data`,
  which the model never sees, contains `execution_id`, `session_id`, and the same calls with
  their `call_id`s, for linking to the audit trail.
- Errors that are the application's raise: a bad token, an unknown Blueprint, invalid variables,
  a missing conversation id, an unreachable server. Errors found while a tool runs reach you as
  the cause of the SDK's `UserError`. Program failures and denials are tool results.

Submilli enforces the Blueprint only for code that runs through Submilli. Other tools you give
the agent keep their own permission boundaries.

## Application tools

Let programs reuse your existing function tools. A model-written program can combine calls,
loop over results, branch on conditions, and process the results in one run.

```python
from agents import function_tool


@function_tool
def lookup_order(order_id: str) -> Order: ...


@function_tool(needs_approval=True)
def refund(order_id: str, amount: float) -> str: ...


submilli = SubmilliTools(blueprint="support", ptc=[lookup_order, refund])
agent = Agent(name="support", tools=submilli.tools)
```

The `ptc` parameter (programmatic tool calling) tells the toolkit which tools programs can use.
Submilli makes them available to the program through the
[`@tools` Package](https://submilli.ai/docs/tutorials/connect-openai-agents/#your-tools-in-programs),
so the program can import and call them. In your Blueprint, grant [`tools.call` with a filter for
each allowed tool name](https://submilli.ai/docs/reference/http-api#application-tools).

Each call runs as an ordinary tool call of the SDK, with the run's context, the tool's
guardrails, timeout, and tracing. Input types come from the tool's schema, and return types
come from its `output_json_schema` or return annotation. Use a `TypedDict` (from
`typing_extensions` before Python 3.12) or a Pydantic model for typed objects. A raised
exception becomes an `Error` in the TypeScript program.

Tools in `ptc` are not given to the model directly. If you also pass them to the agent, direct
model calls bypass Submilli's Blueprint checks.

A session keeps the application-tool catalog it opened with. Changing the catalog causes the
conversation's next call to open a new session.

### Approvals

A tool with `needs_approval` pauses the program at each call that needs approval, the same way
the SDK pauses at a direct tool call. The run result's `interruptions` holds a
`ToolApprovalItem` for your tool, with the program's arguments. Approve or reject it on the run
state and run again:

```python
result = await Runner.run(agent, "Refund order o_7", context=context)
while result.interruptions:
    state = result.to_state()
    for item in result.interruptions:
        if ask_a_person(item.tool_name, item.arguments):
            state.approve(item)
        else:
            state.reject(item, rejection_message="Refunds over $25 need a manager.")
    result = await Runner.run(agent, state)
```

The program waits on the server meanwhile, up to the Blueprint's `tool_call_timeout`. A
rejection throws `PermissionDeniedError` in the program with your message. Each call that needs
approval pauses on its own, so a program that refunds three orders asks three times. The state
can be saved with `to_string()` and resumed with `RunState.from_string` in another process. The
program finishes in the session it started in; share the `store` too so that later turns of the
conversation keep using that session.

A run left waiting is not an answer. To go on without one, reject the pending calls on the state
and run it, which lets the program finish, or drop the unanswered call from the input you pass to
the next run: the Responses API refuses an input that holds a tool call without its output. A
dropped program keeps waiting on the server until `tool_call_timeout`, and its call never runs.

## Submilli documentation

- [Install the CLI and server](https://submilli.ai/docs/install/)
- [Create and register a Blueprint](https://submilli.ai/docs/blueprints/start-a-blueprint/)
- [Configure permissions](https://submilli.ai/docs/reference/permissions/)
- [Run the server](https://submilli.ai/docs/server/run-the-server/)
- [Inspect the audit trail](https://submilli.ai/docs/reference/audit-trail/)
- [Use the HTTP API](https://submilli.ai/docs/reference/http-api)

## Development

This package lives in the [submilli-python](https://github.com/submilli/submilli-python)
workspace. See the repository README for workspace setup, checks, and releases.

```sh
uv run pytest                          # unit tests; the HTTP API is mocked
uv run pytest tests/integration_tests  # starts a server from $SUBMILLI_BIN or PATH
```

The integration tests use a scripted model and need no model provider key.

Implementation notes are in [NOTES.md](https://github.com/submilli/submilli-python/blob/main/packages/openai-agents-submilli/NOTES.md).

## License

Apache-2.0.
