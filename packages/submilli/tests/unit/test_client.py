from __future__ import annotations

import json

import httpx
import pytest
import respx

from submilli import (
    DEFAULT_BASE_URL,
    APIConnectionError,
    AsyncSubmilli,
    AuthenticationError,
    ExecuteResult,
    IdempotencyError,
    NotFoundError,
    SessionRequiresSecretsError,
    Submilli,
    UnknownBlueprintError,
    UnknownSessionError,
    __version__,
    idempotency_key,
)
from submilli.schema import inline_refs
from tests.unit.fake_server import SERVER, TOKEN, FakeServer


def test_a_short_printable_id_is_the_key() -> None:
    assert idempotency_key("toolu_01ABC") == "toolu_01ABC"
    assert idempotency_key("x" * 120) == "x" * 120


@pytest.mark.parametrize("call_id", ["", "x" * 121, "call\n1", "callé"])
def test_other_ids_are_hashed(call_id: str) -> None:
    key = idempotency_key(call_id)
    assert len(key) == 64 and key == idempotency_key(call_id)


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUBMILLI_SERVER", raising=False)
    assert AsyncSubmilli().base_url == DEFAULT_BASE_URL
    monkeypatch.setenv("SUBMILLI_SERVER", "http://elsewhere:1/")
    assert Submilli().base_url == "http://elsewhere:1"
    with pytest.raises(ValueError):
        Submilli(max_retries=-1)


async def test_requests_carry_the_token_and_the_sdk_version(server: FakeServer, client: AsyncSubmilli) -> None:
    await client.blueprints.builtins("research")
    [request] = server.requests
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["user-agent"] == f"submilli-python/{__version__}"


async def test_a_bad_token_raises_authentication_error(server: FakeServer) -> None:
    async with AsyncSubmilli(base_url=SERVER, token="wrong") as client:
        with pytest.raises(AuthenticationError) as raised:
            await client.blueprints.builtins("research")
    assert raised.value.status_code == 401 and raised.value.code == "unauthorized"


async def test_a_session_runs_a_program(server: FakeServer, client: AsyncSubmilli) -> None:
    session = await client.sessions.create("research", variables={"userId": "u_ada"}, secrets={"T": "s"})
    assert json.loads(server.requests[0].content) == {
        "blueprint": "research",
        "variables": {"userId": "u_ada"},
        "secrets": {"T": "s"},
    }
    result = await client.sessions.execute(session.id, "code", idempotency_key="k1")
    assert isinstance(result, ExecuteResult)
    assert (result.result, result.console, result.error, result.denied) == ("ok", [], None, False)
    assert result.session_id == session.id
    assert server.requests[-1].headers["idempotency-key"] == "k1"
    # The same key replays the recorded run.
    again = await client.sessions.execute(session.id, "code", idempotency_key="k1")
    assert again == result and server.total_runs() == 1


async def test_a_denial_is_a_result_not_an_exception(server: FakeServer, client: AsyncSubmilli) -> None:
    server.outcome = {
        "result": None,
        "console": [],
        "error": {
            "kind": "permission_denied",
            "message": "permission denied",
            "caller": "main",
            "capability": "http.get",
            "source": "policy",
        },
    }
    session = await client.sessions.create("research", variables={"userId": "u"})
    result = await client.sessions.run(session.id, "code")
    assert result.denied
    assert result.error is not None
    assert (result.error.capability, result.error.source) == ("http.get", "policy")
    assert result.error.raw["caller"] == "main"


async def test_refusals_raise_typed_errors(server: FakeServer, client: AsyncSubmilli) -> None:
    with pytest.raises(UnknownSessionError) as unknown:
        await client.sessions.execute("nope", "code")
    assert isinstance(unknown.value, NotFoundError)

    session = await client.sessions.create("research", variables={"userId": "u"})
    server.required_secrets = ["USER_TOKEN"]
    server.lost_secrets.add(session.id)
    with pytest.raises(SessionRequiresSecretsError) as lost:
        await client.sessions.execute(session.id, "code")
    assert lost.value.required == ["USER_TOKEN"]
    await client.sessions.rebind(session.id, {"USER_TOKEN": "t"})
    assert isinstance(await client.sessions.execute(session.id, "code"), ExecuteResult)

    server.refuse_next.append((409, {"error": "idempotency_incomplete", "execution_id": "e1"}))
    with pytest.raises(IdempotencyError) as refused:
        await client.sessions.execute(session.id, "code", idempotency_key="k")
    assert (refused.value.code, refused.value.execution_id) == ("idempotency_incomplete", "e1")

    with pytest.raises(UnknownBlueprintError):
        await client.sessions.create("nope", variables={"userId": "u"})


async def test_package_docs_for_an_unknown_name_carry_the_servers_answer(
    server: FakeServer, client: AsyncSubmilli
) -> None:
    docs = await client.blueprints.package_docs("research", "@submilli/jina")
    assert docs.found and docs.text.startswith("# jina")
    missing = await client.blueprints.package_docs("research", "@nope/x")
    assert not missing.found and "unknown package" in missing.text
    with pytest.raises(UnknownBlueprintError):
        await client.blueprints.package_docs("nope", "@submilli/jina")


async def test_discovery(server: FakeServer, client: AsyncSubmilli) -> None:
    prompt = await client.blueprints.prompt("research")
    assert prompt.name == "research" and prompt.tools["docs"] == "Package docs."
    assert (await client.blueprints.search_packages("research", "jina"))["results"][0]["name"] == "@submilli/jina"
    assert (await client.blueprints.builtins("research"))["namespaces"] == ["JSON"]
    docs = await client.blueprints.builtin_docs("research", ["Map", "Set"])
    assert [entry["name"] for entry in docs["results"]] == ["Map", "Set"]


async def test_last_run_and_delete(server: FakeServer, client: AsyncSubmilli) -> None:
    session = await client.sessions.create("research", variables={"userId": "u"})
    assert await client.sessions.last_run(session.id) is None
    await client.sessions.execute(session.id, "code")
    last = await client.sessions.last_run(session.id)
    assert last is not None and last.result == "ok"
    assert await client.sessions.delete(session.id) is True
    assert await client.sessions.delete(session.id) is False


async def test_a_one_off_run_reports_an_unknown_blueprint_in_its_result(
    server: FakeServer, client: AsyncSubmilli
) -> None:
    assert (await client.execute("research", "code")).result == "ok"
    missing = await client.execute("nope", "code")
    assert missing.error is not None and missing.error.kind == "blueprint_not_found"


def test_the_sync_client_has_the_same_surface(server: FakeServer, sync_client: Submilli) -> None:
    session = sync_client.sessions.create("research", variables={"userId": "u"})
    assert sync_client.sessions.run(session.id, "code", idempotency_key="k").result == "ok"
    with pytest.raises(UnknownSessionError):
        sync_client.sessions.execute("nope", "code")


# Retries


async def test_safe_requests_are_retried_after_a_connection_failure(router: respx.MockRouter) -> None:
    route = router.get(f"{SERVER}/v1/blueprints/research/builtins").mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json={"namespaces": []})]
    )
    async with AsyncSubmilli(base_url=SERVER, token=TOKEN, max_retries=1) as client:
        assert await client.blueprints.builtins("research") == {"namespaces": []}
    assert route.call_count == 2


async def test_an_execute_without_a_key_is_never_retried(router: respx.MockRouter) -> None:
    route = router.post(f"{SERVER}/v1/sessions/s/execute").mock(side_effect=httpx.ConnectError("refused"))
    async with AsyncSubmilli(base_url=SERVER, token=TOKEN, max_retries=3) as client:
        with pytest.raises(APIConnectionError):
            await client.sessions.execute("s", "code")
    assert route.call_count == 1


async def test_an_idempotency_refusal_is_an_answer_not_a_transient_failure(router: respx.MockRouter) -> None:
    route = router.post(f"{SERVER}/v1/sessions/s/execute").mock(
        return_value=httpx.Response(503, json={"error": "idempotency_in_progress"})
    )
    async with AsyncSubmilli(base_url=SERVER, token=TOKEN, max_retries=3) as client:
        with pytest.raises(IdempotencyError):
            await client.sessions.execute("s", "code", idempotency_key="k")
    assert route.call_count == 1


def test_a_gateway_error_is_retried(router: respx.MockRouter) -> None:
    route = router.get(f"{SERVER}/v1/runs/r").mock(
        side_effect=[
            httpx.Response(502, headers={"retry-after": "0"}),
            httpx.Response(200, json={"status": "running", "execution_id": "r", "session_id": "s"}),
        ]
    )
    with Submilli(base_url=SERVER, token=TOKEN, max_retries=1) as client:
        assert client.runs.retrieve("r").status == "running"
    assert route.call_count == 2


# Schemas


def test_inline_refs_resolves_local_definitions_and_stops_at_recursion() -> None:
    schema = {
        "type": "object",
        "properties": {"line": {"$ref": "#/$defs/Line"}, "node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Line": {"type": "object", "properties": {"sku": {"type": "string"}}},
            "Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}},
        },
    }
    assert inline_refs(schema) == {
        "type": "object",
        "properties": {
            "line": {"type": "object", "properties": {"sku": {"type": "string"}}},
            "node": {"type": "object", "properties": {"next": {}}},
        },
    }
