# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
