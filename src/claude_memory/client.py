"""MCP stdio client that proxies to remote HTTP memory server."""

import asyncio
import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, ResourceContents, TextResourceContents


def create_client_server(
    server_url: str,
    api_key: str | None = None,
    client_id: str | None = None,
    project_id: str | None = None,
) -> Server:
    """Create MCP server that proxies to HTTP backend."""

    server = Server("claude-memory-client")
    _project_id = project_id

    def get_headers() -> dict:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        if client_id:
            headers["X-Client-ID"] = client_id
        if _project_id:
            headers["X-Project-ID"] = _project_id
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
                        "project_id": {
                            "type": "string",
                            "description": "Project to associate with (uses auto-detected project if not specified)",
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
                        "project_id": {
                            "type": "string",
                            "description": "Filter by project ID (uses current project if not specified)",
                        },
                        "global_search": {
                            "type": "boolean",
                            "description": "Search across all projects (ignores project_id filter)",
                            "default": False,
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
            Tool(
                name="memory_context",
                description="Get a curated context summary for the current project. Use this at the start of a session to understand what you already know about this project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project to get context for (uses current project if not specified)",
                        },
                        "max_memories": {
                            "type": "integer",
                            "description": "Maximum number of memories to include",
                            "default": 10,
                        },
                        "days": {
                            "type": "integer",
                            "description": "Only include memories from the last N days",
                            "default": 30,
                        },
                    },
                },
            ),
        ]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available memory resources."""
        resources = []

        if _project_id:
            resources.append(
                Resource(
                    uri=f"memory://context/{_project_id}",
                    name=f"Memory Context: {_project_id}",
                    description=f"Curated context summary for project '{_project_id}'.",
                    mimeType="text/markdown",
                )
            )

        resources.append(
            Resource(
                uri="memory://context",
                name="Memory Context (All Projects)",
                description="Context summary across all projects.",
                mimeType="text/markdown",
            )
        )

        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> ResourceContents:
        """Read a memory resource by proxying to HTTP server."""
        if uri.startswith("memory://context"):
            parts = uri.split("/")
            project = parts[3] if len(parts) > 3 else _project_id

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(
                        f"{server_url}/context",
                        headers=get_headers(),
                        params={"project_id": project} if project else {},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return TextResourceContents(
                        uri=uri,
                        mimeType="text/markdown",
                        text=data.get("summary", "No context available."),
                    )
            except Exception as e:
                return TextResourceContents(
                    uri=uri,
                    mimeType="text/markdown",
                    text=f"Error fetching context: {str(e)}",
                )

        raise ValueError(f"Unknown resource: {uri}")

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
                            "project_id": arguments.get("project_id"),
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
                            "project_id": arguments.get("project_id"),
                            "global_search": arguments.get("global_search", False),
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

                elif name == "memory_context":
                    params = {}
                    if arguments.get("project_id"):
                        params["project_id"] = arguments["project_id"]
                    if arguments.get("max_memories"):
                        params["max_memories"] = arguments["max_memories"]
                    if arguments.get("days"):
                        params["days"] = arguments["days"]

                    resp = await client.get(
                        f"{server_url}/context",
                        headers=get_headers(),
                        params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return [TextContent(type="text", text=data.get("summary", "No context available."))]

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
    from .storage import get_current_project

    server_url = os.environ.get("MEMORY_SERVER", "http://localhost:8420")
    api_key = os.environ.get("MEMORY_API_KEY")
    client_id = os.environ.get("MEMORY_CLIENT_ID")
    # Auto-detect project if not specified via env
    project_id = os.environ.get("MEMORY_PROJECT") or get_current_project()

    server = create_client_server(server_url, api_key, client_id, project_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run() -> None:
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
