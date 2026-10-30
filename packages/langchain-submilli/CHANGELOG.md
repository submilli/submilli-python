# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.3] - 2026-10-30

### Changed

- Align the SDK and both integration packages on version 0.1.3.
- Require `submilli>=0.1.3,<0.2`.

## [0.1.2] - 2026-10-25

### Fixed

- Expired idempotency responses reach the model as errors that advise checking
  effects before repeating the program.
- On a server with durable execution, a run that stops without a result
  (`stuck`, `cancelled`, `aborted`, `indeterminate`, `expired`, or a state the
  SDK does not know) reaches the model as an error result with the run's status
  and reason, as a lost run does, instead of an empty result. The artifact's
  `run_status` names the state. A queued run is waited for. Requires
  `submilli>=0.1.2`, which adds `RunStateError` and prevents duplicate tool
  execution when recovery exposes an uncommitted call again.

## [0.1.1] - 2026-10-15

### Changed

- The README explains how to get a server: install Submilli, start
  `submilli-server`, and register a minimal Blueprint that declares `userId`.

## [0.1.0] - 2026-10-15

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
