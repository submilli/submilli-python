from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import respx

from submilli import AsyncSubmilli, Submilli
from tests.unit.fake_server import SERVER, TOKEN, FakeServer


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def server(router: respx.MockRouter) -> FakeServer:
    fake = FakeServer()
    fake.install(router)
    return fake


@pytest.fixture
async def client() -> AsyncIterator[AsyncSubmilli]:
    async with AsyncSubmilli(base_url=SERVER, token=TOKEN, max_retries=0) as client:
        yield client


@pytest.fixture
def sync_client() -> Iterator[Submilli]:
    with Submilli(base_url=SERVER, token=TOKEN, max_retries=0) as client:
        yield client
