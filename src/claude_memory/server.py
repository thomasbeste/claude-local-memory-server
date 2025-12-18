"""MCP server exposing memory tools to Claude."""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, ResourceContents, TextResourceContents

from .storage import MemoryStorage, get_current_project


def create_server(data_dir: str | None = None, client_id: str | None = None, project_id: str | None = None) -> Server:
    """Create and configure the MCP server."""
    storage = MemoryStorage(data_dir or "~/.claude-memory")
    server = Server("claude-memory")
    # Store client_id and project_id for use in tools
    _client_id = client_id
    _project_id = project_id

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
            Tool(
                name="memory_purge",
                description="Delete multiple memories matching filter criteria. Requires at least one filter to prevent accidental deletion.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Delete only memories from this project",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation", "session"],
                            "description": "Delete only memories of this type",
                        },
                        "older_than_days": {
                            "type": "integer",
                            "description": "Delete memories older than N days",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Delete memories that have any of these tags",
                        },
                        "content_contains": {
                            "type": "string",
                            "description": "Delete memories where content contains this string (case-insensitive)",
                        },
                    },
                },
            ),
        ]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available memory resources."""
        resources = []

        # Always provide a context resource for the current project
        if _project_id:
            resources.append(
                Resource(
                    uri=f"memory://context/{_project_id}",
                    name=f"Memory Context: {_project_id}",
                    description=f"Curated context summary for project '{_project_id}'. Contains recent decisions, preferences, and key facts.",
                    mimeType="text/markdown",
                )
            )

        # Also provide a global context resource
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
        """Read a memory resource."""
        if uri.startswith("memory://context"):
            # Extract project from URI if specified
            parts = uri.split("/")
            project = parts[3] if len(parts) > 3 else _project_id

            context = storage.get_context_summary(
                project_id=project,
                max_memories=10,
                days=30,
            )

            return TextResourceContents(
                uri=uri,
                mimeType="text/markdown",
                text=context["summary"],
            )

        raise ValueError(f"Unknown resource: {uri}")

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "memory_add":
                # Use provided project_id, fall back to server's default project
                project = arguments.get("project_id") or _project_id
                result = storage.add(
                    content=arguments["content"],
                    memory_type=arguments.get("memory_type", "observation"),
                    tags=arguments.get("tags"),
                    source=arguments.get("source"),
                    client_id=_client_id,
                    project_id=project,
                )
                return [TextContent(type="text", text=f"Memory stored: {json.dumps(result, default=str)}")]

            elif name == "memory_search":
                # Use provided project_id, fall back to server's default project
                project = arguments.get("project_id") or _project_id
                results = storage.search(
                    query=arguments.get("query"),
                    memory_type=arguments.get("memory_type"),
                    tags=arguments.get("tags"),
                    client_id=arguments.get("client_id"),
                    project_id=project,
                    global_search=arguments.get("global_search", False),
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

            elif name == "memory_context":
                project = arguments.get("project_id") or _project_id
                context = storage.get_context_summary(
                    project_id=project,
                    max_memories=arguments.get("max_memories", 10),
                    days=arguments.get("days", 30),
                )
                return [TextContent(type="text", text=context["summary"])]

            elif name == "memory_purge":
                result = storage.purge(
                    project_id=arguments.get("project_id"),
                    memory_type=arguments.get("memory_type"),
                    older_than_days=arguments.get("older_than_days"),
                    tags=arguments.get("tags"),
                    content_contains=arguments.get("content_contains"),
                )
                return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def main(data_dir: str | None = None, client_id: str | None = None, project_id: str | None = None) -> None:
    """Run the MCP server."""
    server = create_server(data_dir, client_id, project_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run(data_dir: str | None = None, client_id: str | None = None, project_id: str | None = None) -> None:
    """Entry point for running the server."""
    import os
    # Allow env override
    client_id = client_id or os.environ.get("MEMORY_CLIENT_ID")
    # Auto-detect project if not specified
    project_id = project_id or os.environ.get("MEMORY_PROJECT") or get_current_project()
    asyncio.run(main(data_dir, client_id, project_id))
