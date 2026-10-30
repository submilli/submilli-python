# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.3] - 2026-10-30

### Changed

- Align the SDK and both integration packages on version 0.1.3.

## [0.1.2] - 2026-10-25

### Added

- `RunStateError`, raised for a run with no result to return. Servers with
  `durable_execution` report runs that are `stuck`, `cancelled`, `aborted`,
  `indeterminate`, `expired`, or `completed` without a kept result. A status
  the SDK does not know raises it too. It carries the run's `execution_id`,
  `session_id`, `status`, `reason`, and the `body` the server sent.

### Changed

- `RunningRun.status` is `"queued"` or `"running"`.

### Fixed

- Recovery exposing the same uncommitted tool call during `runs.complete`
  resends its cached answer instead of executing the Python tool again.
- An expired idempotency response raises `IdempotencyError` with code
  `idempotency_response_expired`.
- Runs on a server with `durable_execution` no longer read as finished runs
  with no result. A `queued` run is a `RunningRun`, which `runs.complete` and
  `sessions.run` poll as they poll a `running` one. A `completed` run from
  `GET /v1/runs/{id}` returns the `ExecuteResult` kept in its `response`.
- `GET /v1/runs/{id}` answers an unknown run with a plain-text `404`, which now
  raises `UnknownRunError` as the JSON `unknown_run` does.

## [0.1.1] - 2026-10-15

### Changed

- The README explains how to get a server: install Submilli, start
  `submilli-server`, and register a minimal Blueprint.

## [0.1.0] - 2026-10-15

First release, for submilli-server 0.4.1.

### Added

- `Submilli` and `AsyncSubmilli`, the same API, blocking and asyncio. The sync
  client is generated from the async one.
- Sessions: `create` (with variables, harness secrets, and application tools),
  `execute`, `run`, `rebind`, `last_run`, `delete`, and session-scoped Package
  search and docs.
- Application tools: `runs.resume`, `runs.retrieve`, and `runs.complete`,
  which answers a run's calls with a callback until it finishes.
  `sessions.run` does the same from the first execute.
- Blueprint discovery: `prompt`, `search_packages`, `package_docs`,
  `builtins`, and `builtin_docs`.
- One-off runs: `execute`.
- Typed results (`ExecuteResult`, `SuspendedRun`, `RunningRun`, `RunError`) and
  typed errors for each refusal a caller can act on.
- `idempotency_key()`, which turns any call id into a valid `Idempotency-Key`.
- `submilli.schema.inline_refs`, for declaring tools from Pydantic schemas.
- Retries with backoff, only for requests that are safe to repeat.
