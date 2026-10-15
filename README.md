# langchain-submilli

Code mode for [LangChain](https://docs.langchain.com/oss/python/langchain/overview) and
[deepagents](https://docs.langchain.com/oss/python/deepagents/overview), with a policy the
model can't talk its way around.

`SubmilliMiddleware` gives your agent a tool that runs TypeScript programs on
[submilli-server](https://github.com/submilli/submilli-runtime). Every program runs under a
Blueprint, a policy you write in YAML, on behalf of the user your application names. Before a
program touches the outside world, such as an HTTP request, a file, a secret, a model call, or
one of your own tools, the server checks the call against the Blueprint.

[Docs](https://submilli.ai/docs/) ·
[Connect deepagents](https://submilli.ai/docs/tutorials/connect-deepagents/) ·
[Package README](packages/langchain-submilli#readme) ·
[Submilli runtime](https://github.com/submilli/submilli-runtime) ·
[Discord](https://discord.gg/VphpukeGGj) ·
[Website](https://submilli.ai)

[![PyPI: langchain-submilli](https://img.shields.io/pypi/v/langchain-submilli?label=langchain-submilli)](https://pypi.org/project/langchain-submilli/)
[![PyPI: submilli](https://img.shields.io/pypi/v/submilli?label=submilli)](https://pypi.org/project/submilli/)
[![CI](https://github.com/submilli/submilli-python/actions/workflows/ci.yml/badge.svg)](https://github.com/submilli/submilli-python/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Discord](https://img.shields.io/discord/1553732232928165958?label=discord)](https://discord.gg/VphpukeGGj)

```python
from deepagents import create_deep_agent
from langchain_submilli import SubmilliMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[SubmilliMiddleware(blueprint="support")],
)
await agent.ainvoke({"messages": [...]}, context={"user_id": "u_ada"})
```

## Why

Think about someone who wants their customer support agent to issue a refund of up to $500
for platinum clients, and up to $100 for every other tier. They could add it to the prompt as a
safeguard, but because of the nature of models, it will likely work only *some* of the time.
When Submilli runs the agent's code, the owner of the workflow defines these guardrails in
advance, and the runtime enforces them, outside the model's control.

Code mode and programmatic tool calling started a movement toward agents that write code
instead of calling tools one by one. Submilli is a runtime built for those agents. The agent
submits TypeScript, and Submilli compiles it and runs it in WebAssembly for isolation. There is
no `node:http` or `node:fs`: programs reach the outside only through Packages, and each
Package defines its operations in terms you can write rules about, such as "allow a refund up
to $500, only for customer 123".

![A support agent creates a replacement with $18 shipping. A $42 replacement is blocked until the developer raises the Blueprint limit from $25 to $50.](https://raw.githubusercontent.com/submilli/submilli-runtime/main/.github/assets/submilli-demo.gif)

## Packages

| Package | Install | What it is |
| --- | --- | --- |
| [`langchain-submilli`](packages/langchain-submilli) | `pip install langchain-submilli` | `SubmilliMiddleware` for deepagents and any LangChain v1 agent |
| [`submilli`](packages/submilli) | `pip install submilli` | The SDK for submilli-server's HTTP API, sync and async, for any Python harness. `langchain-submilli` is built on it |

Both need Python 3.11 or later and submilli-server 0.4.1 or later. Each package has its own
README, changelog, and version, and is released on its own.

## Quick start

**1. Install Submilli and start the server.** The installer brings the `submilli` CLI and
`submilli-server`:

```sh
curl -fsSL https://submilli.ai/install.sh | sh     # macOS and Linux; Windows: irm https://submilli.ai/install.ps1 | iex
export SUBMILLI_SERVER_TOKEN=$(openssl rand -hex 32)
submilli-server &                                  # listens on 127.0.0.1:8128
```

**2. Register a Blueprint.** This one allows only computation, which is enough to try the
middleware. It declares `userId`, which the middleware fills from `user_id` in the run's
context:

```yaml
# blueprint.yaml
kind: blueprint
name: hello
variables:
  userId:
    required: true
default: deny
```

```sh
submilli server blueprint apply blueprint.yaml
```

**3. Run an agent.** In the shell where `SUBMILLI_SERVER_TOKEN` is set:

```sh
pip install "langchain-submilli[deepagents]" "langchain-anthropic>=1.0"
```

```python
import asyncio

from deepagents import create_deep_agent
from langchain_submilli import SubmilliMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    middleware=[SubmilliMiddleware(blueprint="hello")],
)


async def main() -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the 40th Fibonacci number? Compute it."}]},
        context={"user_id": "u_ada"},
    )
    print(result["messages"][-1].text)


asyncio.run(main())
```

The middleware is async only: use `ainvoke` or `astream`. It reads the server from
`SUBMILLI_SERVER` (default `http://127.0.0.1:8128`) and the token from `SUBMILLI_SERVER_TOKEN`.

## What the agent gets

Six tools, with descriptions the server writes for the Blueprint:

| Tool | Does |
| --- | --- |
| `submilli__typescript__execute` | Runs a program. Its description teaches the model the language, with the Blueprint's Packages and rules filled in |
| `submilli__typescript__last_run` | The thread's last run, with its full console |
| `submilli__typescript__packages__search` | Searches the Packages a program may import |
| `submilli__typescript__packages__docs` | A Package's documentation and declarations |
| `submilli__typescript__builtins__list` | The built-ins in scope without an import |
| `submilli__typescript__builtins__docs` | Declarations of built-ins |

The model writes code and nothing else. The Blueprint, the user, and any secrets are set by
your application, where the model can't reach them.

## Examples

### A rule on an operation's arguments

A Blueprint's `permissions` block says what programs may do. Package authors publish the
capabilities their Package supports, and you grant all or some of them. Variables, bound by
your application for each session, let a rule depend on who the agent is working for. Here the
agent may list charges and apply credits of up to $500, only for the customer the application
bound. A different customer, or a larger amount, fails:

```yaml
variables:
  customerId:
    required: true

permissions:
  main:
  - capability: acme.com/charges.list
    filter: customerId == ${vars.customerId}
    action: allow
  - capability: acme.com/credits.apply
    filter: customerId == ${vars.customerId} and amount <= 50000  # cents
    action: allow
```

The middleware fills `customerId` from the run's context, so one shared agent serves every
customer:

```python
middleware = SubmilliMiddleware(
    blueprint="support",
    variables=lambda context: {"customerId": context["customer_id"]},
)
```

When the policy refuses a call, the tool result has `error.kind` `permission_denied`, naming
the capability, and tells the model to report the denial rather than route around it.

### A deep agent with a notebook per user

The [research example](packages/langchain-submilli/examples/research) searches the web, reads
pages, and keeps notes that outlive the conversation, one directory per user. The Blueprint
mounts each user's notes at `/notes` and grants the web search Package:

```yaml
variables:
  userId:
    required: true

packages:
- '@submilli/jina'

vfs:
  mode: per_session
  cwd: /notes
  mounts:
    /notes:
      mode: named
      volume: notes
      subPath: ${vars.userId}   # each user sees only their own notes

default: deny

permissions:
  main:
  - capability: jina.ai/search
    action: allow
  - capability: jina.ai/read
    action: allow
  - capability: fs.read
    action: allow
  - capability: fs.write
    action: allow
  # ... and the Package's own HTTP and secret grants; see the full file
```

The agent is built once and shared. Each run brings its user in the context, and each thread
keeps one server session, saved by the checkpointer:

```python
from typing import TypedDict

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_submilli import SubmilliMiddleware


class Context(TypedDict):
    user_id: str


submilli = SubmilliMiddleware(
    blueprint="research",
    # deepagents' own file tools keep files in the conversation's state. Hide them,
    # so that notes go through Submilli.
    hide_filesystem_tools=True,
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    system_prompt=INSTRUCTIONS,
    middleware=[submilli],
    checkpointer=InMemorySaver(),
    context_schema=Context,
)

config = {"configurable": {"thread_id": "t1"}}
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What's new in Rust? Save a note with your sources."}]},
    config,
    context=Context(user_id="u_ada"),
)
await submilli.aclose_thread(agent, config)  # when the conversation is over
```

Its `check.py` runs the agent with a scripted model, so it needs no API key, and asserts that
two users who write the same note each see only their own.

### Your own tools in programs, with approvals

Programs can call your application's LangChain tools (programmatic tool calling). Pass them as
`ptc`, and name the ones a person must approve in `interrupt_on`:

```python
from typing import TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_submilli import SubmilliMiddleware


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


agent = create_agent(  # any LangChain v1 agent, not only deepagents
    model="anthropic:claude-sonnet-5",
    middleware=[
        SubmilliMiddleware(
            blueprint="shop",
            ptc=[query_orders, send_email],
            interrupt_on={"send_email": True},
        )
    ],
    checkpointer=InMemorySaver(),
)
```

Programs import them as the `@tools` Package, typed from your tools' schemas and return
annotations:

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

The Blueprint decides which tools programs may call, through the `tools.call` capability. It
is denied by default:

```yaml
permissions:
  main:
  - capability: tools.call
    filter: tool == "query_orders" or tool == "send_email"
    action: allow
```

Each `send_email` call pauses the graph with LangChain's human-in-the-loop request, while the
run waits on the server. `approve`, `edit`, `respond`, and `reject` work as they do for
`HumanInTheLoopMiddleware`. See [Your tools in programs](packages/langchain-submilli#your-tools-in-programs).

### Per-user secrets

`secrets` maps the run's context to the Blueprint's `harness` secrets. They are sent to the
server when a session opens and are never stored in the agent's state, so a program can act
with the user's own token without the model ever seeing it:

```python
middleware = SubmilliMiddleware(
    blueprint="github",
    variables=lambda context: {"userId": context.account_id},
    secrets=lambda context: {"GITHUB_TOKEN": context.github_token},
)
```

### Without LangChain

The `submilli` SDK talks to the same server from any Python code:

```python
from submilli import AsyncSubmilli

async with AsyncSubmilli() as client:
    session = await client.sessions.create("hello", variables={"userId": "u_ada"})
    result = await client.sessions.run(
        session.id,
        "export function main(): string { return `${1 + 1}`; }",
        idempotency_key="call-1",
    )
    print(result.result)  # "2"
```

See the [`submilli` README](packages/submilli#readme) for sessions, files, tool calls, and
errors.

## How it behaves

- **One session per thread.** The first `execute` in a thread opens a server session bound to
  the Blueprint and the run's variables. With a checkpointer, a resumed thread, or one continued
  in another process, keeps the same session and its files.
- **One user per session.** A thread whose variables change raises
  `SubmilliSessionMismatchError` before the model is called. One user's programs never run in
  another user's session.
- **Safe retries.** Every execute call sends an `Idempotency-Key` derived from the tool call id,
  so a repeated call replays the recorded result instead of running twice.
- **An audit trail.** Each tool message's `artifact` carries the run's `execution_id`, its place
  in the server's [audit trail](https://submilli.ai/docs/reference/audit-trail).

The [package README](packages/langchain-submilli#readme) covers configuration, lost sessions,
and how `SubmilliMiddleware` compares with `CodeInterpreterMiddleware` from
`langchain-quickjs`.

## What Submilli brings

- [Blueprints](https://submilli.ai/docs/blueprints) with rules on an operation's arguments,
  bound per session, and everything denied by default.
- [Packages](https://submilli.ai/docs/packages) that wrap your APIs and hold the credentials, so
  generated code never sees a secret. Curated Packages for GitHub, Slack, Gmail, Google Drive
  and Calendar, Linear, Notion, Sentry, and web search are
  [included](https://github.com/submilli/submilli-runtime/blob/main/packages/README.md).
- [Limits](https://submilli.ai/docs/server/set-limits) on memory, time, stack depth, model
  tokens, and more. A failing run ends alone, and the rest of the server keeps serving.
- An [audit trail](https://submilli.ai/docs/reference/audit-trail) of every refusal, run,
  session, and admin change.
- MCP servers as Packages, with rules on their tools.
- HTTPS, API tokens with admin and user roles, and an encrypted secret store.

To run the server in production, use
[Docker Compose](https://submilli.ai/docs/server/deploy-with-compose), the
[Helm chart](https://submilli.ai/docs/server/deploy-on-kubernetes), or
[systemd](https://submilli.ai/docs/server/deploy-on-linux). The
[Submilli docs](https://submilli.ai/docs/) cover
[writing Blueprints](https://submilli.ai/docs/blueprints/start-a-blueprint/) and the
[HTTP API](https://submilli.ai/docs/reference/http-api).

## Status

Submilli is young and moving quickly. Breaking changes are called out in each package's
changelog. If you're thinking of using it, we'd love to talk to you: contact us at
[hello@submilli.ai](mailto:hello@submilli.ai) or on [Discord](https://discord.gg/VphpukeGGj).

## Development

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).
In the workspace, `langchain-submilli` uses `packages/submilli` directly. When published, it
depends on `submilli` from PyPI.

```sh
uv sync --all-packages --all-groups
```

Formatting and linting run over the whole repository:

```sh
uv run ruff check .
uv run ruff format .
```

Type checks and tests run inside each package, with that package's settings:

```sh
cd packages/submilli            # or packages/langchain-submilli
uv run mypy                     # strict
uv run pyright
uv run pyright --verifytypes submilli --ignoreexternal   # public API fully typed; langchain_submilli there
uv run pytest                   # unit tests, against a fake of the HTTP API
```

The integration tests start their own submilli-server from `$SUBMILLI_BIN`, a directory that
holds `submilli-server`, or from `PATH`. Put a release build in `.submilli-bin/`, which git
ignores, and point at it:

```sh
SUBMILLI_BIN=$PWD/../../.submilli-bin uv run pytest tests/integration         # submilli
SUBMILLI_BIN=$PWD/../../.submilli-bin uv run pytest tests/integration_tests   # langchain-submilli
```

Some `langchain-submilli` tests also need a server with the `research` Blueprint;
`packages/langchain-submilli/scripts/test-server.sh` starts one.

After changing `packages/submilli/src/submilli/_async_client.py`, regenerate the sync client
with `uv run python packages/submilli/scripts/generate_sync.py`. CI checks it is up to date.

## CI and releases

CI runs ruff, then for each package mypy, pyright, `--verifytypes`, and the unit tests on
every supported Python. It also runs the integration tests against the submilli-server
release named in `.github/workflows/ci.yml`, and checks that each wheel builds and installs on
its own.

A package is released by pushing a tag with its name and version, matching its
`pyproject.toml`: `submilli-v0.1.0` or `langchain-submilli-v0.1.0`. The release workflow builds
that package and publishes it to PyPI through trusted publishing. Release `submilli` before a
`langchain-submilli` that needs its new version.

## License

[Apache License 2.0](LICENSE).
