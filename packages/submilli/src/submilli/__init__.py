"""The Python SDK for submilli-server's HTTP API.

```python
from submilli import AsyncSubmilli

async with AsyncSubmilli() as client:
    session = await client.sessions.create("research", variables={"userId": "u_ada"})
    result = await client.sessions.run(session.id, code, idempotency_key="call-1")
```
"""

from submilli._async_client import AsyncBlueprints, AsyncRuns, AsyncSessions, AsyncSubmilli, AsyncToolHandler
from submilli._base import DEFAULT_BASE_URL, idempotency_key
from submilli._client import Blueprints, Runs, Sessions, Submilli, ToolHandler
from submilli._errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ForbiddenError,
    IdempotencyError,
    InternalServerError,
    NotFoundError,
    RunNotSuspendedError,
    RunStateError,
    ServiceUnavailableError,
    SessionRequiresSecretsError,
    SubmilliError,
    UnknownBlueprintError,
    UnknownCallError,
    UnknownRunError,
    UnknownSessionError,
)
from submilli._types import (
    BlueprintPrompt,
    CreatedSession,
    ExecuteResult,
    LastRun,
    PackageDocs,
    PendingCall,
    RunError,
    RunningRun,
    RunState,
    SuspendedRun,
    ToolDeclaration,
    ToolResult,
)
from submilli._version import __version__

__all__ = [
    "DEFAULT_BASE_URL",
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "AsyncBlueprints",
    "AsyncRuns",
    "AsyncSessions",
    "AsyncSubmilli",
    "AsyncToolHandler",
    "AuthenticationError",
    "BadRequestError",
    "BlueprintPrompt",
    "Blueprints",
    "ConflictError",
    "CreatedSession",
    "ExecuteResult",
    "ForbiddenError",
    "IdempotencyError",
    "InternalServerError",
    "LastRun",
    "NotFoundError",
    "PackageDocs",
    "PendingCall",
    "RunError",
    "RunNotSuspendedError",
    "RunState",
    "RunStateError",
    "RunningRun",
    "Runs",
    "ServiceUnavailableError",
    "SessionRequiresSecretsError",
    "Sessions",
    "Submilli",
    "SubmilliError",
    "SuspendedRun",
    "ToolDeclaration",
    "ToolHandler",
    "ToolResult",
    "UnknownBlueprintError",
    "UnknownCallError",
    "UnknownRunError",
    "UnknownSessionError",
    "__version__",
    "idempotency_key",
]
