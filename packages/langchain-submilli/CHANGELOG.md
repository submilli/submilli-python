# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

First release, for submilli-server 0.4.1 and the `submilli` SDK 0.1.

### Added

- `SubmilliMiddleware` for deepagents and any LangChain v1 agent: the server's
  six tools, with the Blueprint's descriptions, and one session per thread,
  bound to the variables and secrets from the run's context.
- Retries with an `Idempotency-Key` per tool call. A session that ended is
  replaced, and the model is told. A thread whose variables changed is refused.
- `aclose_thread` to end a thread's session.
- `hide_filesystem_tools` to remove deepagents' own file and shell tools.
- `ptc` (programmatic tool calling): programs call the application's LangChain tools as `@tools`,
  run through a `ToolNode` with the run's state and context.
- `interrupt_on` for `ptc` tools: a person approves each call, with
  LangChain's human-in-the-loop request and decision shapes.
