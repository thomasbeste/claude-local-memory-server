"""MCP server exposing memory tools to Claude."""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .storage import MemoryStorage


def create_server(data_dir: str | None = None, client_id: str | None = None) -> Server:
    """Create and configure the MCP server."""
    storage = MemoryStorage(data_dir or "~/.claude-memory")
    server = Server("claude-memory")
    # Store client_id for use in tools
    _client_id = client_id

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
        """Handle tool calls."""
        try:
            if name == "memory_add":
                result = storage.add(
                    content=arguments["content"],
                    memory_type=arguments.get("memory_type", "observation"),
                    tags=arguments.get("tags"),
                    source=arguments.get("source"),
                    client_id=_client_id,
                )
                return [TextContent(type="text", text=f"Memory stored: {json.dumps(result, default=str)}")]

            elif name == "memory_search":
                results = storage.search(
                    query=arguments.get("query"),
                    memory_type=arguments.get("memory_type"),
                    tags=arguments.get("tags"),
                    client_id=arguments.get("client_id"),
                    limit=arguments.get("limit", 20),
                    search_mode=arguments.get("search_mode", "hybrid"),
                )
                if not results:
                    return [TextContent(type="text", text="No memories found matching the criteria.")]
                return [TextContent(type="text", text=json.dumps(results, default=str, indent=2))]

            elif name == "memory_delete":
                success = storage.delete(arguments["memory_id"])
                if success:
                    return [TextContent(type="text", text=f"Memory {arguments['memory_id']} deleted.")]
                return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]

            elif name == "memory_update":
                result = storage.update(
                    memory_id=arguments["memory_id"],
                    content=arguments.get("content"),
                    tags=arguments.get("tags"),
                )
                if result:
                    return [TextContent(type="text", text=f"Memory updated: {json.dumps(result, default=str)}")]
                return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]

            elif name == "memory_stats":
                stats = storage.stats()
                return [TextContent(type="text", text=json.dumps(stats, default=str, indent=2))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def main(data_dir: str | None = None, client_id: str | None = None) -> None:
    """Run the MCP server."""
    server = create_server(data_dir, client_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run(data_dir: str | None = None, client_id: str | None = None) -> None:
    """Entry point for running the server."""
    import os
    # Allow env override
    client_id = client_id or os.environ.get("MEMORY_CLIENT_ID")
    asyncio.run(main(data_dir, client_id))
