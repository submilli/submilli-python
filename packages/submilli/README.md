# submilli

The Python SDK for [submilli-server](https://github.com/submilli/submilli-runtime). It runs
TypeScript programs under a Blueprint's policy, on behalf of a user, from any Python harness:
an agent framework, a job queue, or a script.

```sh
pip install submilli
```

It needs Python 3.11 or later and submilli-server 0.4.1 or later. For LangChain and
deepagents, [`langchain-submilli`](https://github.com/submilli/submilli-python/tree/main/packages/langchain-submilli)
builds on it.

## Get a server

The installer puts `submilli`, the CLI, and `submilli-server` on your `PATH`. Start the
server with a token:

```sh
curl -fsSL https://submilli.ai/install.sh | sh     # macOS and Linux; Windows: irm https://submilli.ai/install.ps1 | iex
export SUBMILLI_SERVER_TOKEN=$(openssl rand -hex 32)
submilli-server &                                  # listens on 127.0.0.1:8128
```

Then register a Blueprint, the policy programs run under. This one allows only computation and
declares the `userId` variable:

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

[Install](https://submilli.ai/docs/install/), [Run the server](https://submilli.ai/docs/server/run-the-server/),
and [Register a Blueprint](https://submilli.ai/docs/server/register-a-blueprint/) cover the
details, and the [Blueprint file reference](https://submilli.ai/docs/reference/blueprint-file/)
lists what a Blueprint can grant: HTTP, files, secrets, model calls, and your own tools.

## Quick start

```python
import asyncio

from submilli import AsyncSubmilli

PROGRAM = """
export function main(): string {
    return `${1 + 1}`;
}
"""


async def main() -> None:
    async with AsyncSubmilli() as client:
        session = await client.sessions.create("hello", variables={"userId": "u_ada"})
        result = await client.sessions.run(session.id, PROGRAM, idempotency_key="call-1")
        print(result.result)  # "2"


asyncio.run(main())
```

`Submilli` has the same API without `await`:

```python
from submilli import Submilli

with Submilli() as client:
    session = client.sessions.create("hello", variables={"userId": "u_ada"})
    print(client.sessions.run(session.id, PROGRAM).result)
```

The client reads the server's URL from `$SUBMILLI_SERVER` (default `http://127.0.0.1:8128`) and
its API token from `$SUBMILLI_SERVER_TOKEN`, or from `base_url=` and `token=`.

## Sessions

A session binds a Blueprint to one user's variables and harness secrets. Programs in it run one
after another and share its files.

```python
session = await client.sessions.create(
    "research",
    variables={"userId": "u_ada"},
    secrets={"GITHUB_TOKEN": token},  # the Blueprint's `harness` secrets
)
state = await client.sessions.execute(session.id, code, idempotency_key=call_id)
await client.sessions.rebind(session.id, {"GITHUB_TOKEN": new_token})
last = await client.sessions.last_run(session.id)  # the full console, or None
await client.sessions.delete(session.id)
```

**Results, not exceptions.** A program that fails or is denied is a result:

```python
result = await client.sessions.run(session.id, code)
if result.denied:
    print(result.error.capability, result.error.source)  # e.g. "http.get", "policy"
elif result.error:
    print(result.error.kind, result.error.message)  # e.g. "compile_error"
else:
    print(result.result, result.console)
```

**Idempotency.** Pass an `idempotency_key` with every execute that may be retried. When the
same request reaches the server twice, the program runs once and the retry gets the recorded
result. `idempotency_key(call_id)` turns any id, such as a model's tool call id, into a key the
server accepts.

## Your tools in programs

A session can declare the application's own tools. Programs import them as `@tools`, typed from
their schemas, and the server checks each call against the Blueprint's `tools.call` capability:

```python
from submilli import PendingCall, ToolResult

QUERY_ORDERS = {
    "name": "query_orders",
    "description": "Orders of one customer.",
    "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]},
    "output_schema": {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]},
}


async def answer(call: PendingCall) -> ToolResult:
    orders = await db.orders(call.args["customer"])
    return ToolResult.success(call.call_id, {"count": len(orders)})


session = await client.sessions.create("shop", variables={"userId": "u_ada"}, tools=[QUERY_ORDERS])
result = await client.sessions.run(session.id, PROGRAM, idempotency_key="call-1", on_tool_call=answer)
```

```ts
import tools from "@tools";

function main(): string {
    return `${tools.query_orders({ customer: "c_42" }).count} orders`;
}
```

- **The loop.** Each call stops the run on the server. `sessions.run` hands the call to
  `on_tool_call`, sync or async, and resumes the run with the answer, until the program finishes.
  The time a run waits doesn't count against its time limit.
- **Answers.** `ToolResult.success(call_id, value)` returns `value` to the program.
  `ToolResult.failed(call_id, message)` throws `Error` there. `ToolResult.denied(call_id,
  message)` throws `PermissionDeniedError`, reported with `source: "harness"`.
- **Pausing.** If `on_tool_call` raises, the exception propagates and the run keeps waiting on the
  server, up to the Blueprint's `tool_call_timeout`. That is how a harness pauses for a person's
  approval. Executing again with the same `idempotency_key` answers the run's current state, and
  `sessions.run` carries on from the call it waits on.
- **Declarations.** `session.warnings` lists what programs won't see of the declared tools, such
  as a Blueprint that grants no `tools.call`, or a tool whose result is typed `unknown`. The
  server doesn't resolve `$ref`, so pass Pydantic schemas through `submilli.schema.inline_refs`.
- **Lower level.** `runs.resume`, `runs.retrieve`, and `runs.complete` drive a run yourself.
- **Restarts.** A suspended run is lost if the server restarts, and resuming it raises
  `UnknownRunError`.

## Discovery

What a model needs to write programs for a Blueprint:

```python
prompt = await client.blueprints.prompt("research")  # tool descriptions for a model
await client.blueprints.search_packages("research", "github")
docs = await client.blueprints.package_docs("research", "@submilli/jina")  # docs.found, docs.text
await client.blueprints.builtins("research")
await client.blueprints.builtin_docs("research", ["Map", "Temporal"])
```

`sessions.search_packages` and `sessions.package_docs` answer the same for one session, including
its `@tools`.

## Errors

Every error is a `SubmilliError`:

| Error | When |
| --- | --- |
| `APIConnectionError`, `APITimeoutError` | The server couldn't be reached, or didn't answer in time |
| `AuthenticationError` | 401: the token is missing or wrong |
| `UnknownSessionError` | The session ended, usually after its idle timeout. Open a new one |
| `SessionRequiresSecretsError` | The server asks for the session's harness secrets again. `rebind` them; `.required` names them |
| `IdempotencyError` | The server refused the key and ran nothing. `.code` says why |
| `UnknownRunError` | The server no longer holds a suspended run |
| `RunNotSuspendedError`, `UnknownCallError` | A resume that no longer applies. `runs.complete` handles these |
| `UnknownBlueprintError` | No Blueprint of that name |
| `BadRequestError`, `ForbiddenError`, `NotFoundError`, `ConflictError`, `ServiceUnavailableError`, `InternalServerError` | Other refusals, by status |

Each `APIStatusError` carries `status_code`, `body`, the body's `code`, and the `execution_id`
it concerns, when there is one.

Harness secrets are kept encrypted with the server's secret-store key, so sessions keep them
across restarts. A server whose key changed refuses such a session with a `ConflictError`
("cannot decrypt session secrets").

## Retries and timeouts

A connection failure, or a 502, 503, or 504, is retried up to `max_retries` times (default 2),
with backoff, but only for requests that are safe to repeat: reads, deletes, rebinds, resumes,
and executes that carry an idempotency key. An execute without a key and `sessions.create` are
never retried. The default timeout is 330 seconds, because an execute may wait up to 300 seconds
for an earlier attempt with the same key.

```python
client = AsyncSubmilli(timeout=60.0, max_retries=0, http_client=httpx.AsyncClient(...))
```

## Development

This package lives in the [submilli-python](https://github.com/submilli/submilli-python)
workspace, next to `langchain-submilli`. See the repository's README for setup, checks, and
releases. In `packages/submilli`:

```sh
uv run pytest                                   # unit tests, against a fake of the HTTP API
SUBMILLI_BIN=/path/to/bin uv run pytest tests/integration   # starts its own submilli-server
uv run python scripts/generate_sync.py          # after changing _async_client.py
```

## License

Apache-2.0.
