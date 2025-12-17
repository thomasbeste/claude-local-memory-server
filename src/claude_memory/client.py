"""MCP stdio client that proxies to remote HTTP memory server."""

import asyncio
import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


def create_client_server(server_url: str, api_key: str | None = None, client_id: str | None = None) -> Server:
    """Create MCP server that proxies to HTTP backend."""
    
    server = Server("claude-memory-client")
    
    def get_headers() -> dict:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        if client_id:
            headers["X-Client-ID"] = client_id
        return headers

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available memory tools."""
        return [
            Tool(
                name="memory_add",
                description="Store a new memory. Use this to remember facts, decisions, preferences, or any information that should persist across conversations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation"],
                            "description": "Type of memory",
                            "default": "observation",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization (e.g., ['project:gpi', 'tech:python'])",
                        },
                        "source": {
                            "type": "string",
                            "description": "Optional source/context for this memory",
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="memory_search",
                description="Search through stored memories. Use this to recall information from previous conversations. Supports semantic search for finding conceptually related memories.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (searches in content)",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation"],
                            "description": "Filter by memory type",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags (matches any)",
                        },
                        "client_id": {
                            "type": "string",
                            "description": "Filter by client ID (e.g., 'laptop', 'desktop')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return",
                            "default": 20,
                        },
                        "search_mode": {
                            "type": "string",
                            "enum": ["keyword", "semantic", "hybrid"],
                            "description": "Search mode: 'keyword' for exact matching, 'semantic' for meaning-based search, 'hybrid' combines both (default)",
                            "default": "hybrid",
                        },
                    },
                },
            ),
            Tool(
                name="memory_delete",
                description="Delete a specific memory by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_update",
                description="Update an existing memory's content or tags.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to update",
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (optional)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New tags (optional, replaces existing)",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_stats",
                description="Get statistics about stored memories.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls by proxying to HTTP server."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if name == "memory_add":
                    resp = await client.post(
                        f"{server_url}/memories",
                        headers=get_headers(),
                        json={
                            "content": arguments["content"],
                            "memory_type": arguments.get("memory_type", "observation"),
                            "tags": arguments.get("tags"),
                            "source": arguments.get("source"),
                        },
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return [TextContent(type="text", text=f"Memory stored: {json.dumps(result, default=str)}")]

                elif name == "memory_search":
                    resp = await client.post(
                        f"{server_url}/memories/search",
                        headers=get_headers(),
                        json={
                            "query": arguments.get("query"),
                            "memory_type": arguments.get("memory_type"),
                            "tags": arguments.get("tags"),
                            "client_id": arguments.get("client_id"),
                            "limit": arguments.get("limit", 20),
                            "search_mode": arguments.get("search_mode", "hybrid"),
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if not data["results"]:
                        return [TextContent(type="text", text="No memories found matching the criteria.")]
                    return [TextContent(type="text", text=json.dumps(data["results"], default=str, indent=2))]

                elif name == "memory_delete":
                    resp = await client.delete(
                        f"{server_url}/memories/{arguments['memory_id']}",
                        headers=get_headers(),
                    )
                    if resp.status_code == 404:
                        return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]
                    resp.raise_for_status()
                    return [TextContent(type="text", text=f"Memory {arguments['memory_id']} deleted.")]

                elif name == "memory_update":
                    resp = await client.patch(
                        f"{server_url}/memories/{arguments['memory_id']}",
                        headers=get_headers(),
                        json={
                            "content": arguments.get("content"),
                            "tags": arguments.get("tags"),
                        },
                    )
                    if resp.status_code == 404:
                        return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]
                    resp.raise_for_status()
                    result = resp.json()
                    return [TextContent(type="text", text=f"Memory updated: {json.dumps(result, default=str)}")]

                elif name == "memory_stats":
                    resp = await client.get(
                        f"{server_url}/stats",
                        headers=get_headers(),
                    )
                    resp.raise_for_status()
                    stats = resp.json()
                    return [TextContent(type="text", text=json.dumps(stats, default=str, indent=2))]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except httpx.ConnectError:
            return [TextContent(type="text", text=f"Error: Cannot connect to memory server at {server_url}")]
        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=f"Error: HTTP {e.response.status_code} - {e.response.text}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def main() -> None:
    """Run the MCP client server."""
    server_url = os.environ.get("MEMORY_SERVER", "http://localhost:8420")
    api_key = os.environ.get("MEMORY_API_KEY")
    client_id = os.environ.get("MEMORY_CLIENT_ID")
    
    server = create_client_server(server_url, api_key, client_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run() -> None:
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
