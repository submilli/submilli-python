# langchain-submilli

`langchain-submilli` runs a LangChain agent's code-mode tool calls as TypeScript programs on
[submilli-server](https://github.com/submilli/submilli-runtime). Each program runs under the
policy of a Blueprint, on behalf of the user named in the run's context.

v1 ships `SubmilliMiddleware`, which works with
[deepagents](https://docs.langchain.com/oss/python/deepagents/overview) and any LangChain v1
agent that takes middleware.

## Install

```sh
pip install langchain-submilli
```

deepagents is not a dependency, because the middleware works with any LangChain v1 agent.
Install it with `pip install "langchain-submilli[deepagents]"`. The HTTP API itself is the
[`submilli`](https://github.com/submilli/submilli-python) SDK, which this package depends on;
its client, results, and errors are imported from `submilli`.

```python
from deepagents import create_deep_agent
from langchain_submilli import SubmilliMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[SubmilliMiddleware(blueprint="research")],
)
await agent.ainvoke({"messages": [...]}, context={"user_id": "u_ada"})
```

The middleware needs a running submilli-server with the Blueprint registered. See
[Run a server](#run-a-server). It is async only: use `ainvoke` or `astream`.

## What the agent gets

Six tools, with the server's names and with descriptions the server writes for the Blueprint:

| Tool | Does |
| --- | --- |
| `submilli__typescript__execute` | Runs a program. Its description teaches the model the language, with the Blueprint's Packages and rules filled in |
| `submilli__typescript__last_run` | The thread's last run, with its full console |
| `submilli__typescript__packages__search` | Searches the Packages a program may import |
| `submilli__typescript__packages__docs` | A Package's documentation and declarations |
| `submilli__typescript__builtins__list` | The built-ins in scope without an import |
| `submilli__typescript__builtins__docs` | Declarations of built-ins |

The descriptions come from `GET /v1/blueprints/{name}/prompt`. Each middleware instance fetches
them once, before the first model call. The model writes code and nothing else. The Blueprint,
the user, and any secrets are set by your application, where the model cannot reach them.

## What happens on each tool call

**One session per thread.** The first time the model calls `execute` in a thread, the
middleware opens a session on the server. It binds the Blueprint and the variables from this
run's context. The session's id and variables go into the agent's state as `submilli_session`.
With a checkpointer, they are saved before any program runs, so a resumed thread, or one
continued in another process, keeps using the same session. Discovery tools need no session,
unless you [expose your own tools](#your-tools-in-programs).

Ending a turn does not end the session, because the session belongs to the thread. End it with
`await middleware.aclose_thread(agent, config)`. Otherwise the Blueprint's `idle_timeout`
(default 24 hours) ends it.

**One user per session.** Before each run, the middleware compares the context's variables with
the variables the thread's session was opened with. If they differ, it raises
`SubmilliSessionMismatchError` before the model is called. One user's programs never run in
another user's session.

**Idempotent retries.** Every execute call sends an `Idempotency-Key` derived from the
LangChain tool call id. The id is used as is when it is printable ASCII of at most 120 bytes;
otherwise its sha256 hex digest is used. When the same tool call reaches the server twice, the
program runs once and the second call gets the recorded result. This happens when a thread is
resumed from a checkpoint taken before its tools finished, or when a retry middleware repeats
a call. Both tool messages carry the same `execution_id`. When the server refuses a key, the
model gets a tool result that says nothing ran and what to do. The refusals are
`idempotency_conflict`, `idempotency_incomplete`, `idempotency_in_progress`, and
`idempotency_unavailable`. On `idempotency_incomplete`, the model is told the earlier attempt's
side effects are unknown, and to check before it retries.

**Policy and denials.** The server checks every capability a program uses against the
Blueprint, such as a file write, an HTTP request, a secret, or a model call. The tool result
is JSON with `result`, `console`, `error`, and `denied`, the same shape submilli-runtime's
Vercel AI SDK example returns. When the policy refuses a call, `error.kind` is
`permission_denied` and names the `caller`, `capability`, and `source`. `denied` is `true`,
and a note tells the model to report the denial to the user rather than route around it.
`execution_id` and `session_id` are in the tool message's `artifact`, not in the model's
content, so your application can link each run to the server's
[audit trail](https://submilli.ai/docs/reference/audit-trail).

**When the server loses something.** When a session has ended, for example by idle timeout,
the server answers `404 unknown session`. The middleware opens a new session and runs the
program there. The tool result tells the model that the earlier session's files are gone.
Sessions survive a server restart, and so do their harness secrets, which the server keeps
encrypted with its secret-store key. If the server answers `409 session_requires_secrets`
anyway, the middleware sends the context's secrets with `rebind` and retries once.

`interrupt_on` approvals apply to `submilli__typescript__execute` as they do to any other tool:
a person approves the whole program before it runs. To approve single calls a program makes to
your own tools, see [Your tools in programs](#your-tools-in-programs).

## Your tools in programs

With submilli-server 0.4.1 or later, programs can call your application's own tools, such as a
LangChain `@tool` that queries your database or sends an email. Pass them as `expose_tools`:

```python
from typing import TypedDict

from langchain.tools import ToolRuntime, tool


class Orders(TypedDict):
    count: int
    total: float


@tool
def query_orders(customer: str, runtime: ToolRuntime) -> Orders:
    """Orders of one customer."""
    return db.orders(customer, user=runtime.context["user_id"])


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    ...


middleware = SubmilliMiddleware(
    blueprint="shop",
    expose_tools=[query_orders, send_email],
    interrupt_on={"send_email": True},
)
```

Programs import them as the `@tools` Package, typed from your tools' schemas:

```ts
import tools from "@tools";

function main(): string {
    const orders = tools.query_orders({ customer: "c_42" });
    if (orders.count > 10) {
        tools.send_email({ to: "sales@example.com", body: `c_42 has ${orders.count} orders` });
    }
    return `${orders.count} orders`;
}
```

- **Running a call.** Each call stops the run on the server. The middleware runs the tool
  through a LangGraph `ToolNode`, with the run's state, context, and store, as the agent would,
  and resumes the run with the result. Time spent waiting doesn't count against the run's time
  limit.
- **Return values.** The argument types come from the tool's argument schema. The return type
  comes from its return annotation. Return a `TypedDict` or a Pydantic model for a typed
  object. A `dict[str, X]` reaches the program as `unknown`. A tool with no JSON return
  annotation returns its text.
- **Errors.** A tool that raises throws `Error` in the program, which the program can catch.
- **Policy.** The Blueprint decides which tools programs may call, through the `tools.call`
  capability and its `tool` filter. It is denied by default, and the server logs a warning
  when a session declares tools the Blueprint can't use:

  ```yaml
  permissions:
    main:
    - capability: tools.call
      filter: tool == "query_orders" or tool == "send_email"
      action: allow
  ```

- **Approvals.** `interrupt_on` takes the `HumanInTheLoopMiddleware` settings for exposed
  tools: `True`, or an `InterruptOnConfig` with `allowed_decisions`, `description`, and
  `when`. A call that needs approval interrupts the graph with a `HITLRequest` of one action,
  while the run waits on the server. Resume it with one decision:

  ```python
  [pending] = (await agent.aget_state(config)).interrupts
  await agent.ainvoke(Command(resume={pending.id: {"decisions": [{"type": "approve"}]}}), config, context=...)
  ```

  `approve` runs the call. `edit` runs it with edited arguments, for the same tool. `respond`
  answers for the tool without running it. `reject` throws `PermissionDeniedError` in the
  program, with the rejection message, and the model is told the denial is final. A program
  that makes several such calls pauses once for each. The agent needs a checkpointer. A run
  waits up to the Blueprint's `tool_call_timeout` (10 minutes by default) for each answer, so
  raise it for approvals that take longer.

  This approval is separate from deepagents' and LangChain's own `interrupt_on`, which only
  sees tool calls the model makes. If you approve a tool there, approve it here too before you
  expose it.

- **What the application sees.** The execute tool message's `artifact` lists the program's
  calls under `tool_calls`, each with its arguments, its outcome, and the decision, when there
  was one.
- **State updates.** A tool that returns a `Command`, such as one that writes deepagents'
  `files`, has its state update applied with the execute result. When several calls write
  the same key, dict values are merged and anything else keeps the last value.
- **Discovery.** With exposed tools, `packages__search` and `packages__docs` answer for the
  thread's session, which includes `@tools`. The first such call opens the session.
- **Sessions.** A session keeps the tools it was opened with. When the exposed tools change,
  the thread's next call opens a new session.

The exposed tools are not given to the model directly. Pass them to the agent as well if the
model should call them itself.

## Configuration

```python
SubmilliMiddleware(
    blueprint="research",
    server=None,  # default: $SUBMILLI_SERVER, then http://127.0.0.1:8128
    token=None,  # default: $SUBMILLI_SERVER_TOKEN
    variables=None,  # default: {"userId": context.user_id}
    secrets=None,  # default: none
    hide_filesystem_tools=False,
    client=None,  # a submilli.AsyncSubmilli, instead of server and token
    expose_tools=None,  # your tools, for programs to call as @tools
    interrupt_on=None,  # which of them need approval, as for HumanInTheLoopMiddleware
)
```

`variables` and `secrets` take the run's `runtime.context` and return `dict[str, str]`. A fixed
mapping also works. Build the agent once and share it; each run brings its own user:

```python
middleware = SubmilliMiddleware(
    blueprint="research",
    variables=lambda context: {"userId": context.account_id, "region": context.region},
    secrets=lambda context: {"USER_TOKEN": context.github_token},
)
```

By default, `user_id` in the context becomes the Blueprint variable `userId`. The context can be a
dict, a dataclass, or a Pydantic model. A context without `user_id` binds no variables. A
Blueprint that requires `userId` then refuses the session, and the run raises
`submilli.BadRequestError`.

Secrets go to the Blueprint's `harness` secrets. They are sent when a session opens and when
the server asks for them again. They are never stored in the agent's state. A session keeps
the secrets it was opened with: when they change, for example a rotated user token, end the
thread's session with `aclose_thread` so the next call opens one with the new values.

`hide_filesystem_tools=True` removes deepagents' own file and shell tools from the agent:
`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`. The
model then reaches files only through programs. Those tools keep files in the conversation's
state, so without the flag a model asked to "save a note" may save it there, and the note
never reaches the server. The middleware filters the tools from each model request, the way
deepagents' own harness profiles do, and refuses a call that names one. The flag does not
reach deepagents' built-in `general-purpose` subagent, which does not receive your middleware.
To deny the file tools there too, pass deepagents'
`permissions=[FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny")]`
as well.

## Run a server

Install Submilli and start a server as
[Connect a harness](https://submilli.ai/docs/tutorials/connect-a-harness) describes. Then
register the Blueprint. `examples/research/` holds the tutorial's `blueprint.yaml`,
`prompt.txt`, and `note.ts`. Export the server's token where the agent runs:

```sh
export SUBMILLI_SERVER=http://127.0.0.1:8128
export SUBMILLI_SERVER_TOKEN=...
```

The [HTTP API reference](https://submilli.ai/docs/reference/http-api) describes every endpoint
this package calls, and the [Blueprint file reference](https://submilli.ai/docs/reference/blueprint-file)
describes policies, variables, secrets, and `idle_timeout`.

For development, `scripts/test-server.sh` starts a throwaway server with the `research`
Blueprint in `.submilli-test/` at the repository root.

## Compared with `CodeInterpreterMiddleware`

[`CodeInterpreterMiddleware`](https://docs.langchain.com/oss/python/deepagents/interpreters)
from `langchain-quickjs` fills the same slot. It adds a code tool to a deep agent, and each
thread keeps its own state.

| | `CodeInterpreterMiddleware` | `SubmilliMiddleware` |
| --- | --- | --- |
| Language | JavaScript | A strict TypeScript subset, compiled to WebAssembly |
| Where code runs | QuickJS, in the agent's process | submilli-server, a separate process |
| Reaching the outside | Agent tools exposed through PTC | Packages and capabilities the Blueprint grants: files, HTTP, secrets, model calls; and the tools you expose |
| Permission boundary | The PTC allowlist | The Blueprint's policy, checked on each capability call, exposed tools included |
| Approvals | PTC calls skip `interrupt_on` | `interrupt_on` for the whole program, and per call to an exposed tool |
| Per-user binding | — | Variables and secrets from the run's context, fixed per session |
| Retries | — | `Idempotency-Key` per tool call; a repeated call replays |
| Record of runs | — | `execution_id` per run, in the server's audit trail |
| Needs | `pip install` | A running submilli-server |

The `langchain-quickjs` documentation says its interpreter is "not a host-memory isolation
boundary".

## Development

This package lives in the [submilli-python](https://github.com/submilli/submilli-python)
workspace, next to the `submilli` SDK it depends on. See the repository's README for setup,
checks, and releases. In `packages/langchain-submilli`:

```sh
uv run pytest                          # unit tests; the HTTP API is mocked
uv run pytest tests/integration_tests  # needs a server; see below
```

The integration tests use a scripted model, so they need no model API key. Most of them run
against a server at `$SUBMILLI_SERVER` with the `research` Blueprint, and are skipped when none
is reachable. The others start their own server from `$SUBMILLI_BIN` or `PATH`.

```sh
export SUBMILLI_SERVER_TOKEN=$(openssl rand -hex 32)
scripts/test-server.sh &
SUBMILLI_SERVER=http://127.0.0.1:8128 uv run pytest tests/integration_tests
```

Design notes, and the problems in submilli-runtime found while building this package, are in
[NOTES.md](https://github.com/submilli/submilli-python/blob/main/packages/langchain-submilli/NOTES.md).

## License

Apache-2.0, as submilli-runtime.
