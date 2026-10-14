"""A program that calls one of the application's tools.

Needs a submilli-server (`$SUBMILLI_SERVER`, `$SUBMILLI_SERVER_TOKEN`) with a
Blueprint named `shop` that grants `tools.call`:

    kind: blueprint
    name: shop
    variables:
      userId:
        required: true
    default: deny
    permissions:
      main:
      - capability: tools.call
        filter: tool == "query_orders"
        action: allow
"""

import asyncio

from submilli import AsyncSubmilli, PendingCall, ToolDeclaration, ToolResult

QUERY_ORDERS: ToolDeclaration = {
    "name": "query_orders",
    "description": "Orders of one customer.",
    "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]},
    "output_schema": {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]},
}

PROGRAM = """
import tools from "@tools";

function main(): string {
    const orders = tools.query_orders({ customer: "c_42" });
    return `c_42 has ${orders.count} orders`;
}
"""


def query_orders(call: PendingCall) -> ToolResult:
    # Your application's own code: a database query, an API call, a person's approval.
    return ToolResult.success(call.call_id, {"count": len(call.args["customer"])})


async def main() -> None:
    async with AsyncSubmilli() as client:
        session = await client.sessions.create("shop", variables={"userId": "u_ada"}, tools=[QUERY_ORDERS])
        for warning in session.warnings:
            print("warning:", warning)
        result = await client.sessions.run(session.id, PROGRAM, idempotency_key="example-1", on_tool_call=query_orders)
        print(result.error.message if result.error else result.result)
        await client.sessions.delete(session.id)


if __name__ == "__main__":
    asyncio.run(main())
