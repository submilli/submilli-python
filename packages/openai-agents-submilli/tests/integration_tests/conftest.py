"""A submilli-server for the integration tests.

`managed` is a server these tests start, stop and restart themselves. It needs
the `submilli-server` binary, from `SUBMILLI_BIN` or `PATH`, and is skipped
without it. Blueprints are registered over the admin API.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import httpx
import pytest


def _binary(name: str) -> str | None:
    directory = os.environ.get("SUBMILLI_BIN")
    if directory:
        path = Path(directory) / name
        return str(path) if path.exists() else None
    return shutil.which(name)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast("int", sock.getsockname()[1])


class ManagedServer:
    """A submilli-server in a temporary home, which a test may restart."""

    def __init__(self, home: Path) -> None:
        server_bin = _binary("submilli-server")
        if not server_bin:
            pytest.skip("submilli-server not found (set SUBMILLI_BIN)")
        self.server_bin = server_bin
        self.home = home
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.token = secrets.token_hex(32)
        self.process: subprocess.Popen[bytes] | None = None
        (home / "store.key").write_text(__import__("base64").b64encode(os.urandom(32)).decode())
        config = f"secret_store:\n  key_file: {home / 'store.key'}\n"
        (home / "server.yaml").write_text(config)
        self.env = {
            **os.environ,
            "SUBMILLI_HOME": str(home),
            "SUBMILLI_SERVER_TOKEN": self.token,
            "SUBMILLI_SERVER_URL": self.url,
        }

    def start(self) -> None:
        log = open(self.home / "server.log", "ab")  # noqa: SIM115
        self.process = subprocess.Popen(
            [self.server_bin, "--config", str(self.home / "server.yaml"), "--port", str(self.port)],
            env=self.env,
            stdout=log,
            stderr=log,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                httpx.get(f"{self.url}/v1/blueprints/none/builtins", timeout=1)
                return
            except httpx.TransportError:
                time.sleep(0.1)
        raise RuntimeError(f"submilli-server did not start; see {self.home / 'server.log'}")

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def apply(self, blueprint: str) -> None:
        name = re.search(r"^name: (\S+)$", blueprint, re.MULTILINE)
        assert name, "the Blueprint has no name"
        response = httpx.put(
            f"{self.url}/v1/blueprints/{name[1]}",
            json={"yaml": blueprint},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        response.raise_for_status()


@pytest.fixture
def managed(tmp_path: Path) -> Iterator[ManagedServer]:
    server = ManagedServer(tmp_path)
    server.start()
    try:
        yield server
    finally:
        server.stop()
