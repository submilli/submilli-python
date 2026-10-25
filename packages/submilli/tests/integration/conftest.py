"""A submilli-server these tests start, stop and restart themselves.

It needs the `submilli-server` binary, from `$SUBMILLI_BIN` (a directory) or
`PATH`; the tests are skipped without it. Blueprints are registered over the
admin API, so the `submilli` CLI is not needed.
"""

from __future__ import annotations

import base64
import os
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import httpx
import pytest


def _binary() -> str | None:
    directory = os.environ.get("SUBMILLI_BIN")
    if directory:
        path = Path(directory) / "submilli-server"
        return str(path) if path.exists() else None
    return shutil.which("submilli-server")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


class ManagedServer:
    """A submilli-server in a temporary home."""

    def __init__(self, home: Path, binary: str, *, durable: bool = False) -> None:
        self.binary = binary
        self.home = home
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.token = secrets.token_hex(32)
        self.process: subprocess.Popen[bytes] | None = None
        self.log: IO[bytes] | None = None
        self.rotate_store_key()
        config = f"secret_store:\n  key_file: {home / 'store.key'}\n"
        if durable:
            config += "durable_execution: true\nmax_durable_runs: 2\n"
        (home / "server.yaml").write_text(config)
        self.env = {**os.environ, "SUBMILLI_HOME": str(home), "SUBMILLI_SERVER_TOKEN": self.token}

    def rotate_store_key(self) -> None:
        """Give the server a new secret-store key; secrets sealed with the old one can't be read."""
        (self.home / "store.key").write_text(base64.b64encode(os.urandom(32)).decode())

    def start(self) -> None:
        self.log = open(self.home / "server.log", "ab")  # noqa: SIM115 - closed in stop()
        self.process = subprocess.Popen(
            [self.binary, "--config", str(self.home / "server.yaml"), "--port", str(self.port)],
            env=self.env,
            stdout=self.log,
            stderr=self.log,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                httpx.get(f"{self.url}/healthz", timeout=1)
                return
            except httpx.TransportError:
                time.sleep(0.1)
        raise RuntimeError(f"submilli-server did not start; see {self.home / 'server.log'}")

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def crash(self) -> None:
        """Stop without draining or committing pending work."""
        assert self.process is not None
        self.process.kill()
        self.process.wait(timeout=30)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def apply(self, name: str, yaml: str) -> None:
        response = httpx.put(
            f"{self.url}/v1/blueprints/{name}",
            json={"yaml": yaml},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        response.raise_for_status()


@pytest.fixture
def managed(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[ManagedServer]:
    binary = _binary()
    if binary is None:
        pytest.skip("submilli-server not found (set SUBMILLI_BIN)")
    server = ManagedServer(tmp_path, binary, durable=getattr(request, "param", False))
    server.start()
    try:
        yield server
    finally:
        server.stop()
