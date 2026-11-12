"""Thin transports; all tool schemas, state transitions and metering live in the env."""
import asyncio
import json
import math
import sys

from .schemas import TOOL_SCHEMAS


def _reject_constant(value):
    raise ValueError(f"Nonstandard JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite float range")
    return number


def _error(code, message):
    return {"error": {"code": code, "message": message, "details": {}}}


def serve_json_lines(env, input_stream=None, output_stream=None):
    source, sink = input_stream or sys.stdin, output_stream or sys.stdout
    for line in source:
        request_id = None
        try:
            request = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            candidate_id = request.get("id")
            if candidate_id is not None and type(candidate_id) not in (str, int):
                raise ValueError("Request id must be a string, integer, or null")
            request_id = candidate_id
            if set(request) - {"id", "tool", "args"} or not {"id", "tool"} <= request.keys():
                raise ValueError("Request must contain id, tool, and optional args")
            if not isinstance(request["tool"], str):
                raise ValueError("Request tool must be a string")
            response = env.call(request["tool"], request.get("args", {}))
        except (ValueError, TypeError, RecursionError) as error:
            response = _error("invalid_request", str(error))
        try:
            encoded = json.dumps({**response, "id": request_id}, allow_nan=False)
        except (ValueError, TypeError, OverflowError, RecursionError):
            # Encode completely before writing, so a bad result cannot emit a
            # partial frame or discard the next request. The call may have run.
            encoded = json.dumps({"id": request_id, **_error(
                "invalid_response", "Tool response is not finite JSON; inspect history before retrying")})
        sink.write(encoded + "\n")
        sink.flush()


def serve_mcp(env):
    try:
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as error:
        raise NotImplementedError("MCP requires optional dependency: pip install 'mcp>=1.30,<2'") from error

    server = Server("archexplorer")

    @server.list_tools()
    async def list_tools():
        return [Tool(**schema) for schema in TOOL_SCHEMAS]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        result = env.call(name, arguments)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, allow_nan=False))],
                              isError="error" in result)

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(run())
