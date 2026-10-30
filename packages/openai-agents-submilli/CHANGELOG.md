# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-30

First release, for submilli-server 0.4.1, the `submilli` SDK 0.1.2, and openai-agents 0.24.

### Added

- `SubmilliTools` for OpenAI Agents SDK agents: the server's six tools as `FunctionTool`s,
  with the Blueprint's descriptions, and one session per conversation, bound to the variables
  and secrets from the run's context.
- Sessions kept in a `SessionStore`: `InMemorySessionStore` (default) or `SQLiteSessionStore`.
  `aclose_conversation` ends a conversation's session.
- Retries with an `Idempotency-Key` per tool call. A session that ended is replaced, and the
  model is told. A conversation whose variables changed is refused before the model runs.
- Execution and session ids, and the program's application-tool calls, in the output item's
  `custom_data`.
- `ptc` (programmatic tool calling): programs call the application's function tools as
  `@tools`. Each call runs as an SDK tool call, and a tool's `needs_approval` pauses the run
  with a `ToolApprovalItem` for that call, resumed with `RunState.approve` or `reject`.
