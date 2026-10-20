# langchain-submilli

[Submilli](https://github.com/submilli/submilli-runtime) is a dedicated secure runtime for
code-mode agents, with built-in isolation, governance, and permissions.

`langchain-submilli` is the Python integration for
[LangChain](https://docs.langchain.com/oss/python/langchain/overview) and
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview). It lets your agent
execute TypeScript code in Submilli's runtime.

## Install

Python 3.11 or later and submilli-server 0.4.1 or later are required.

```sh
pip install langchain-submilli
```

Deep Agents is optional. Install the extra when you use it:

```sh
pip install "langchain-submilli[deepagents]"
```

Follow [Connect Deep Agents](https://submilli.ai/docs/tutorials/connect-deepagents/) for a
complete setup with a server, Blueprint, model provider, agent, and session cleanup.

## Middleware reference

`SubmilliMiddleware` is async only. Use `ainvoke` or `astream`.

```python
SubmilliMiddleware(
    blueprint="research",
    server=None,
    token=None,
    variables=None,
    secrets=None,
    hide_filesystem_tools=False,
    client=None,
    ptc=None,
    interrupt_on=None,
)
```

| Setting | Contract |
| --- | --- |
| `blueprint` | Name of a Blueprint already [registered on the server](https://submilli.ai/docs/server/register-a-blueprint/); used for every program |
| `server` | submilli-server URL; defaults to `$SUBMILLI_SERVER`, then `http://127.0.0.1:8128` |
| `token` | Server token; defaults to `$SUBMILLI_SERVER_TOKEN` |
| `variables` | Fixed `dict[str, str]`, or a function from the run context to one; defaults to mapping context `user_id` to Blueprint variable `userId` |
| `secrets` | Fixed `dict[str, str]`, or a function from the run context to one, for the Blueprint's `harness` secrets |
| `hide_filesystem_tools` | Hides and refuses Deep Agents' file and shell tools on this agent. Subagents need separate configuration. |
| `client` | Existing `submilli.AsyncSubmilli`; replaces `server` and `token` |
| `ptc` | Application tools that programs may call through the `@tools` Package |
| `interrupt_on` | LangChain human-in-the-loop settings for calls to tools in `ptc` |

The middleware gives your agent the six tools below. Before the first model call, it fetches the
instructions for using them from the Submilli server.

| Tool | Does |
| --- | --- |
| `submilli__typescript__execute` | Runs a program |
| `submilli__typescript__last_run` | Returns the thread's last run and full console |
| `submilli__typescript__packages__search` | Searches Packages available to the program |
| `submilli__typescript__packages__docs` | Returns a Package's documentation and declarations |
| `submilli__typescript__builtins__list` | Lists built-ins available without an import |
| `submilli__typescript__builtins__docs` | Returns declarations for selected built-ins |

## Sessions, retries, and results

- The first program run opens a Submilli session for the conversation. LangGraph can save the
  session ID with the conversation state using a [checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence),
  so later turns reuse it. To reuse it after your application restarts, use persistent storage
  for the checkpointer. Close the session with `await middleware.aclose_thread(agent, config)`.
- Blueprint variables are fixed when the session opens. A later run on the same thread with
  different values raises `SubmilliSessionMismatchError` before the model is called.
- Harness secrets are sent when a session opens and are never stored in agent state. End the
  session before using rotated secrets.
- Each execute call sends an `Idempotency-Key` derived from the LangChain tool call id. A
  repeated call replays the recorded result. The middleware reports conflicts and incomplete,
  in-progress, or unavailable replays without silently rerunning the program.
- When a session has ended, the middleware opens a replacement. If a restarted server asks for
  the session's secrets, it rebinds them once and retries.
- The execute tool message contains `result`, `console`, `error`, and `denied`. Its artifact
  contains `execution_id`, `session_id`, and any application-tool calls.

Submilli enforces the Blueprint only for code that runs through Submilli. Other LangChain
tools keep their own permission boundaries.

## Application tools

Let programs reuse your existing LangChain tools. A model-written program can combine calls,
loop over results, branch on conditions, and process the results in one run.

The `ptc` parameter (programmatic tool calling) tells `SubmilliMiddleware` which LangChain tools
programs can use. Submilli makes those tools available to the
program through the [`@tools` Package](https://submilli.ai/docs/tutorials/connect-deepagents/#your-tools-in-programs),
so the program can import and call them. In your Blueprint, grant [`tools.call` with a filter for
each allowed tool name](https://submilli.ai/docs/reference/http-api#application-tools).

The middleware runs each call through a LangGraph `ToolNode` with the run's state, context, and
store. Input types come from the tool schema, and return types come from annotations. Use a
`TypedDict` or Pydantic model for typed objects. A raised exception becomes an `Error` in the
TypeScript program.

Tools in `ptc` are not given to the model directly. If you also pass them to the agent, direct
model calls bypass Submilli's Blueprint checks.

A session keeps the application-tool catalog it opened with. Changing the catalog causes the
thread's next call to open a new session.

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
uv run pytest tests/integration_tests  # requires a server
```

The integration tests use a scripted model and need no model provider key. Most use a server
at `$SUBMILLI_SERVER` with the `research` Blueprint; the others start a server from
`$SUBMILLI_BIN` or `PATH`.

```sh
export SUBMILLI_SERVER_TOKEN=$(openssl rand -hex 32)
scripts/test-server.sh &
SUBMILLI_SERVER=http://127.0.0.1:8128 uv run pytest tests/integration_tests
```

Implementation notes are in [NOTES.md](https://github.com/submilli/submilli-python/blob/main/packages/langchain-submilli/NOTES.md).

## License

Apache-2.0.
