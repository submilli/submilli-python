# submilli-python

Python packages for [Submilli](https://github.com/submilli/submilli-runtime): run an agent's
code as TypeScript programs on submilli-server, under a Blueprint's policy, on behalf of a user.

| Package | PyPI | What it is |
| --- | --- | --- |
| [`submilli`](packages/submilli) | `pip install submilli` | The SDK for submilli-server's HTTP API, sync and async. For any Python harness |
| [`langchain-submilli`](packages/langchain-submilli) | `pip install langchain-submilli` | `SubmilliMiddleware` for deepagents and LangChain v1 agents. Built on `submilli` |

Each package has its own README, changelog, and version, and is released on its own.

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

Apache-2.0.
