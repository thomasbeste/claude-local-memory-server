"""HTTP API server for central memory deployment."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent

from .storage import MemoryStorage

# --- Logging setup ---

logger = logging.getLogger("claude_memory.api")


# --- Rate limiting ---

def get_api_key_or_ip(request: Request) -> str:
    """Rate limit by API key if present, otherwise by IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[:8]}"  # Use first 8 chars as identifier
    return get_remote_address(request)


limiter = Limiter(key_func=get_api_key_or_ip)


# --- Models ---

class MemoryCreate(BaseModel):
    content: str
    memory_type: str = "observation"
    tags: list[str] | None = None
    source: str | None = None
    client_id: str | None = None  # Can also be set via X-Client-ID header
    project_id: str | None = None  # Can also be set via X-Project-ID header


class MemoryUpdate(BaseModel):
    content: str | None = None
    tags: list[str] | None = None


class SearchQuery(BaseModel):
    query: str | None = None
    memory_type: str | None = None
    tags: list[str] | None = None
    client_id: str | None = None
    project_id: str | None = None  # Filter by project
    global_search: bool = False  # Search across all projects
    limit: int = 20
    search_mode: str = "hybrid"  # "keyword", "semantic", "hybrid"


# --- App setup ---

storage: MemoryStorage | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage and MCP server on startup."""
    global storage, mcp_session_manager
    data_dir = Path(app.state.data_dir)
    storage = MemoryStorage(data_dir)

    # Create MCP server and session manager
    mcp_server = create_mcp_server(storage)
    mcp_session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=False,  # Use SSE streaming
        stateless=True,  # Each request is independent (no persistent sessions)
    )

    # Run the session manager's lifecycle
    async with mcp_session_manager.run():
        yield


app = FastAPI(
    title="Claude Memory Server",
    description="Central persistent memory for Claude Code instances",
    version="0.1.0",
)

# --- Middleware ---

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - configure allowed origins via environment
cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
cors_origins = [o.strip() for o in cors_origins if o.strip()]

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["X-API-Key", "X-Client-ID", "X-Project-ID", "Content-Type"],
    )


class LoggingMiddleware:
    """ASGI middleware for logging requests (skips /mcp to support streaming)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip logging for MCP endpoint (uses streaming)
        path = scope.get("path", "")
        if path == "/mcp":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        request = Request(scope, receive, send)

        # Extract identifying info
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", "")
        key_hint = f"{api_key[:8]}..." if api_key else "none"

        # Capture response status
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # Calculate duration and log
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "%s %s | %d | %.1fms | ip=%s key=%s",
            request.method,
            path,
            status_code,
            duration_ms,
            client_ip,
            key_hint,
        )


# Add middleware as ASGI middleware (wraps after FastAPI's internal middleware)
app.add_middleware(LoggingMiddleware)


def get_storage() -> MemoryStorage:
    if storage is None:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return storage


# Optional: Simple API key auth
def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str | None:
    """Optional API key verification. Set MEMORY_API_KEY env to enable."""
    expected = os.environ.get("MEMORY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Routes ---

@app.get("/health")
async def health():
    """Health check endpoint (no rate limit - nginx probes this)."""
    return {"status": "ok"}


@app.post("/memories")
@limiter.limit("30/minute")
async def create_memory(
    request: Request,
    memory: MemoryCreate,
    x_client_id: Annotated[str | None, Header()] = None,
    x_project_id: Annotated[str | None, Header()] = None,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Create a new memory."""
    # Prefer body values, fall back to headers
    client_id = memory.client_id or x_client_id
    project_id = memory.project_id or x_project_id
    result = store.add(
        content=memory.content,
        memory_type=memory.memory_type,
        tags=memory.tags,
        source=memory.source,
        client_id=client_id,
        project_id=project_id,
    )
    return result


@app.post("/memories/search")
@limiter.limit("60/minute")
async def search_memories(
    request: Request,
    query: SearchQuery,
    x_project_id: Annotated[str | None, Header()] = None,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Search memories. Supports keyword, semantic, and hybrid search modes."""
    # Prefer body project_id, fall back to header
    project_id = query.project_id or x_project_id
    results = store.search(
        query=query.query,
        memory_type=query.memory_type,
        tags=query.tags,
        client_id=query.client_id,
        project_id=project_id,
        global_search=query.global_search,
        limit=query.limit,
        search_mode=query.search_mode,
    )
    return {"results": results, "count": len(results)}


@app.get("/memories/{memory_id}")
@limiter.limit("60/minute")
async def get_memory(
    request: Request,
    memory_id: str,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Get a specific memory."""
    result = store.get(memory_id)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@app.patch("/memories/{memory_id}")
@limiter.limit("30/minute")
async def update_memory(
    request: Request,
    memory_id: str,
    update: MemoryUpdate,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Update a memory."""
    result = store.update(memory_id, content=update.content, tags=update.tags)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@app.delete("/memories/{memory_id}")
@limiter.limit("30/minute")
async def delete_memory(
    request: Request,
    memory_id: str,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Delete a memory."""
    success = store.delete(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": memory_id}


@app.get("/stats")
@limiter.limit("60/minute")
async def get_stats(
    request: Request,
    project_id: str | None = None,
    x_project_id: Annotated[str | None, Header()] = None,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Get memory statistics, optionally filtered by project."""
    effective_project = project_id or x_project_id
    return store.stats(project_id=effective_project)


@app.get("/context")
@limiter.limit("60/minute")
async def get_context(
    request: Request,
    project_id: str | None = None,
    max_memories: int = 10,
    days: int = 30,
    x_project_id: Annotated[str | None, Header()] = None,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """
    Get curated context summary for a project.

    Returns prioritized memories (decisions, preferences, facts) suitable
    for injection at session start.
    """
    effective_project = project_id or x_project_id
    return store.get_context_summary(
        project_id=effective_project,
        max_memories=max_memories,
        days=days,
    )


# --- MCP Server Setup ---

def create_mcp_server(store: MemoryStorage) -> Server:
    """Create MCP server with memory tools."""
    server = Server("claude-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory_add",
                description="Store a new memory. Use this to remember facts, decisions, preferences, or any information that should persist across conversations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The content to remember"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation"],
                            "description": "Type of memory",
                            "default": "observation",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                        "source": {"type": "string", "description": "Optional source/context"},
                        "project_id": {"type": "string", "description": "Project to associate with"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="memory_search",
                description="Search through stored memories. Supports semantic search for finding conceptually related memories.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation"],
                            "description": "Filter by memory type",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                        "project_id": {"type": "string", "description": "Filter by project ID"},
                        "global_search": {"type": "boolean", "description": "Search across all projects", "default": False},
                        "limit": {"type": "integer", "description": "Max results", "default": 20},
                        "search_mode": {
                            "type": "string",
                            "enum": ["keyword", "semantic", "hybrid"],
                            "description": "Search mode",
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
                    "properties": {"memory_id": {"type": "string", "description": "The ID of the memory to delete"}},
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_update",
                description="Update an existing memory's content or tags.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "The ID of the memory to update"},
                        "content": {"type": "string", "description": "New content (optional)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (optional)"},
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="memory_stats",
                description="Get statistics about stored memories.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_context",
                description="Get a curated context summary for a project.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Project to get context for"},
                        "max_memories": {"type": "integer", "description": "Maximum memories to include", "default": 10},
                        "days": {"type": "integer", "description": "Only include memories from last N days", "default": 30},
                    },
                },
            ),
            Tool(
                name="memory_purge",
                description="Delete multiple memories matching filter criteria. Requires at least one filter.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Delete only memories from this project"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "decision", "preference", "observation", "entity", "relation", "session"],
                            "description": "Delete only memories of this type",
                        },
                        "older_than_days": {"type": "integer", "description": "Delete memories older than N days"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Delete memories with these tags"},
                        "content_contains": {"type": "string", "description": "Delete memories containing this string"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "memory_add":
                result = store.add(
                    content=arguments["content"],
                    memory_type=arguments.get("memory_type", "observation"),
                    tags=arguments.get("tags"),
                    source=arguments.get("source"),
                    project_id=arguments.get("project_id"),
                )
                return [TextContent(type="text", text=f"Memory stored: {json.dumps(result, default=str)}")]

            elif name == "memory_search":
                results = store.search(
                    query=arguments.get("query"),
                    memory_type=arguments.get("memory_type"),
                    tags=arguments.get("tags"),
                    project_id=arguments.get("project_id"),
                    global_search=arguments.get("global_search", False),
                    limit=arguments.get("limit", 20),
                    search_mode=arguments.get("search_mode", "hybrid"),
                )
                if not results:
                    return [TextContent(type="text", text="No memories found matching the criteria.")]
                return [TextContent(type="text", text=json.dumps(results, default=str, indent=2))]

            elif name == "memory_delete":
                success = store.delete(arguments["memory_id"])
                if success:
                    return [TextContent(type="text", text=f"Memory {arguments['memory_id']} deleted.")]
                return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]

            elif name == "memory_update":
                result = store.update(
                    memory_id=arguments["memory_id"],
                    content=arguments.get("content"),
                    tags=arguments.get("tags"),
                )
                if result:
                    return [TextContent(type="text", text=f"Memory updated: {json.dumps(result, default=str)}")]
                return [TextContent(type="text", text=f"Memory {arguments['memory_id']} not found.")]

            elif name == "memory_stats":
                stats = store.stats()
                return [TextContent(type="text", text=json.dumps(stats, default=str, indent=2))]

            elif name == "memory_context":
                context = store.get_context_summary(
                    project_id=arguments.get("project_id"),
                    max_memories=arguments.get("max_memories", 10),
                    days=arguments.get("days", 30),
                )
                return [TextContent(type="text", text=context["summary"])]

            elif name == "memory_purge":
                result = store.purge(
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


# --- MCP HTTP Transport ---

# Global session manager - will be initialized during lifespan
mcp_session_manager: StreamableHTTPSessionManager | None = None


class MCPEndpoint:
    """ASGI app that handles MCP requests directly."""

    async def __call__(self, scope, receive, send):
        if mcp_session_manager is None:
            response = Response(content='{"error": "MCP server not initialized"}', status_code=500)
            await response(scope, receive, send)
            return
        await mcp_session_manager.handle_request(scope, receive, send)


# Mount MCP endpoint directly as ASGI app to bypass FastAPI response handling
from starlette.routing import Route
app.routes.append(Route("/mcp", MCPEndpoint(), methods=["GET", "POST", "DELETE"]))


# --- Entrypoint ---

def create_app(data_dir: str = "/var/lib/claude-memory") -> FastAPI:
    """Create app with custom data directory."""
    app.state.data_dir = data_dir
    app.router.lifespan_context = lifespan
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8420,
    data_dir: str = "/var/lib/claude-memory",
    log_level: str = "info",
):
    """Run the server directly."""
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure app with data dir and lifespan
    configured_app = create_app(data_dir)
    uvicorn.run(configured_app, host=host, port=port, lifespan="on", log_level=log_level)
