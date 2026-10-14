# Research agent

This is the research agent from submilli-runtime's
[Connect deepagents](https://submilli.ai/docs/tutorials/connect-deepagents) tutorial, rewritten
to use `langchain-submilli`. It answers questions by searching the web and reading pages, and
it keeps a notebook for each user on the server.

`blueprint.yaml`, `prompt.txt`, and `note.ts` are copies of the files in submilli-runtime's
`examples/harnesses/`.

## Run it

Start a server with this directory's Blueprint, as
[Connect a harness](https://submilli.ai/docs/tutorials/connect-a-harness) describes. Store real
Jina and Anthropic keys in the server's secret store. Then, from `packages/langchain-submilli`:

```sh
uv sync --all-packages --all-groups
export SUBMILLI_SERVER=http://127.0.0.1:8128 SUBMILLI_SERVER_TOKEN=...
ANTHROPIC_API_KEY=... uv run python examples/research/agent.py
```

To ask your own question, pass it as arguments to `agent.py`.

## Check it without a model

```sh
cd examples/research && uv run python check.py
```

`check.py` runs the agent with a scripted model in place of Claude, so it needs no API key. It
asserts that two users who write the same note each see only their own.
