# langchain-submilli

[Submilli](https://github.com/submilli/submilli-runtime) is a dedicated secure runtime for
code-mode agents, with built-in isolation, governance, and permissions.

`langchain-submilli` is the Python integration for
[LangChain](https://docs.langchain.com/oss/python/langchain/overview) and
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview). It lets your agent
execute TypeScript code in Submilli's runtime.

[Docs](https://submilli.ai/docs/) ·
[Connect Deep Agents](https://submilli.ai/docs/tutorials/connect-deepagents/) ·
[Package reference](packages/langchain-submilli#readme) ·
[Submilli runtime](https://github.com/submilli/submilli-runtime) ·
[Discord](https://discord.gg/VphpukeGGj) ·
[Website](https://submilli.ai)

[![PyPI: langchain-submilli](https://img.shields.io/pypi/v/langchain-submilli?label=langchain-submilli)](https://pypi.org/project/langchain-submilli/)
[![PyPI: submilli](https://img.shields.io/pypi/v/submilli?label=submilli)](https://pypi.org/project/submilli/)
[![CI](https://github.com/submilli/submilli-python/actions/workflows/ci.yml/badge.svg)](https://github.com/submilli/submilli-python/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

## Let your coding agents do the work

Copy this prompt into your coding agent to connect your LangChain or Deep Agents project to
Submilli:

<details>
<summary>Show prompt</summary>

```text
Connect this project’s LangChain or Deep Agents agent to Submilli. Follow
https://submilli.ai/docs/tutorials/connect-deepagents/.
Keep the existing model. Show me how existing application tools will be
available to the agent, and which calls will go through Submilli.
Show me the Blueprint and code changes. Test with mocked model responses.
```

</details>

## Why

[Code mode](https://submilli.ai/docs/why/#the-industry-already-agrees) lets agents write code
instead of calling tools one at a time. The agent submits code, and
Submilli compiles and runs it in WebAssembly for isolation. Programs have no direct access to the
network or host filesystem. Programs use [Packages](https://submilli.ai/docs/packages/) to access
external systems, and [Blueprints](https://submilli.ai/docs/blueprints/) define which operations
are allowed. See [the execution challenges](https://submilli.ai/docs/why/#this-code-is-a-stranger)
for why isolation alone is not enough.

## Example

Suppose your support agent can send replacements with shipping costs up to $25. A prompt can ask
the model to respect that limit, but the model can still generate code that exceeds it. Set the
limit in a Blueprint, and Submilli enforces it. In the video, a replacement with $18 shipping is
allowed and a replacement with $42 shipping is blocked. That replacement succeeds after the
developer raises the limit to $50.

https://github.com/user-attachments/assets/e601c541-400b-4ac1-ad96-23dc0b0e6f90

## Install

Python 3.11 or later is required.

```sh
pip install "langchain-submilli[deepagents]"
```

For a different LangChain v1 agent, install `langchain-submilli` without the `deepagents`
extra. To use submilli-server directly from Python, install the
[`submilli` SDK](packages/submilli/README.md) instead.

Follow [Connect Deep Agents](https://submilli.ai/docs/tutorials/connect-deepagents/) for the
complete setup, including the server, Blueprint, model provider, agent, and session cleanup.

## Use your existing LangChain tools

Your agent can call your existing LangChain tools from the code it runs through Submilli. Pass
those tools to `SubmilliMiddleware` with `ptc` to make them available to the program.

The [integration guide](https://submilli.ai/docs/tutorials/connect-deepagents/#your-tools-in-programs)
explains how these calls work. See the [LangChain configuration](packages/langchain-submilli#application-tools)
for the `ptc` option.

## Next steps

To control what the agent may do, [create a Blueprint](https://submilli.ai/docs/blueprints/start-a-blueprint/)
and [register it on your server](https://submilli.ai/docs/server/register-a-blueprint/). The
[permissions reference](https://submilli.ai/docs/reference/permissions/) covers rules for the
operations your agent needs.

To pass application context or credentials to Submilli, see the [middleware settings](packages/langchain-submilli#middleware-reference).
The [session reference](packages/langchain-submilli#sessions-retries-and-results) explains reuse
across turns and cleanup.

## What the Submilli runtime provides

- [Blueprints](https://submilli.ai/docs/blueprints/) define rules for operation arguments,
  bind values per session, and deny everything else by default.
- [Packages](https://submilli.ai/docs/packages/) wrap APIs in operations the Blueprint can
  govern. Credentials can be attached by the host without being returned to the program, and
  [curated integrations](https://submilli.ai/docs/reference/curated-packages/) cover common
  services.
- [Resource limits](https://submilli.ai/docs/server/set-limits/) bound memory, execution time,
  stack depth, model tokens, and more.
- The [audit trail](https://submilli.ai/docs/reference/audit-trail/) records refusals, runs,
  sessions, and administrative changes.
- [MCP servers](https://submilli.ai/docs/reference/mcp-servers/) can be exposed as Packages, so
  Blueprint rules apply to their tools.
- [Server controls](https://submilli.ai/docs/server/run-the-server/) include HTTPS, API tokens
  with administrative and user roles, and an encrypted secret store.

For production, deploy with
[Docker Compose](https://submilli.ai/docs/server/deploy-with-compose/),
[Kubernetes](https://submilli.ai/docs/server/deploy-on-kubernetes/), or
[Linux and systemd](https://submilli.ai/docs/server/deploy-on-linux/).

## Status

Submilli is young and moving quickly. Breaking changes are recorded in the
[`langchain-submilli` changelog](packages/langchain-submilli/CHANGELOG.md) and
[`submilli` changelog](packages/submilli/CHANGELOG.md). To talk with us, email
[hello@submilli.ai](mailto:hello@submilli.ai) or join
[Discord](https://discord.gg/VphpukeGGj).

## Development

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).
In the workspace, `langchain-submilli` uses `packages/submilli` directly. When published, it
depends on `submilli` from PyPI.

```sh
uv sync --all-packages --all-groups
uv run ruff check .
uv run ruff format .
```

Run type checks and tests inside each package:

```sh
cd packages/submilli            # or packages/langchain-submilli
uv run mypy
uv run pyright
uv run pyright --verifytypes submilli --ignoreexternal             # packages/submilli
uv run pyright --verifytypes langchain_submilli --ignoreexternal   # packages/langchain-submilli
uv run pytest
```

Integration tests start submilli-server from `$SUBMILLI_BIN` or `PATH`. A release build can
live in the ignored `.submilli-bin/` directory:

```sh
SUBMILLI_BIN=$PWD/../../.submilli-bin uv run pytest tests/integration -m 'not durable'         # submilli
SUBMILLI_BIN=$PWD/../../.submilli-bin uv run pytest tests/integration_tests -m 'not durable'   # langchain-submilli
```

Durable integration tests require an upstream runtime build containing durable callbacks
(tested at `5f67954152a8fa812f64ffc9cb1d17cd4ccd82a8`, after v0.4.3). Set `SUBMILLI_BIN`
to its binary directory and run the same commands with `-m durable`. CI builds that pinned
revision separately and exercises queued/running HTTP responses, crash recovery, duplicate
resumes, concurrent tools, and approvals after restart. Each test starts an isolated server.

Some `langchain-submilli` integration tests also need the `research` Blueprint.
`packages/langchain-submilli/scripts/test-server.sh` starts a suitable server.

From the repository root, after changing `packages/submilli/src/submilli/_async_client.py`,
regenerate the sync client:

```sh
uv run python packages/submilli/scripts/generate_sync.py
```

## CI and releases

CI runs formatting, linting, type checks, unit tests, integration tests, and standalone wheel
installation checks.

Release a package by pushing a tag that matches its `pyproject.toml` version:
`submilli-v0.1.0` or `langchain-submilli-v0.1.0`. The release workflow publishes that package
to PyPI through trusted publishing. Release `submilli` before a `langchain-submilli` version
that depends on it.

## License

[Apache License 2.0](LICENSE).
