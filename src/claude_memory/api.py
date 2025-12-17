"""HTTP API server for central memory deployment."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from .storage import MemoryStorage


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
    """Initialize storage on startup."""
    global storage
    data_dir = Path(app.state.data_dir)
    storage = MemoryStorage(data_dir)
    yield


app = FastAPI(
    title="Claude Memory Server",
    description="Central persistent memory for Claude Code instances",
    version="0.1.0",
)


def get_storage() -> MemoryStorage:
    if storage is None:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return storage


# Optional: Simple API key auth
def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str | None:
    """Optional API key verification. Set MEMORY_API_KEY env to enable."""
    import os
    expected = os.environ.get("MEMORY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Routes ---

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/memories")
async def create_memory(
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
async def search_memories(
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
async def get_memory(
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
async def update_memory(
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
async def delete_memory(
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
async def get_stats(
    project_id: str | None = None,
    x_project_id: Annotated[str | None, Header()] = None,
    store: MemoryStorage = Depends(get_storage),
    _: str | None = Depends(verify_api_key),
):
    """Get memory statistics, optionally filtered by project."""
    effective_project = project_id or x_project_id
    return store.stats(project_id=effective_project)


# --- Entrypoint ---

def create_app(data_dir: str = "/var/lib/claude-memory") -> FastAPI:
    """Create app with custom data directory."""
    app.state.data_dir = data_dir
    app.router.lifespan_context = lifespan
    return app


def run_server(host: str = "0.0.0.0", port: int = 8420, data_dir: str = "/var/lib/claude-memory"):
    """Run the server directly."""
    import uvicorn
    app.state.data_dir = data_dir
    uvicorn.run(app, host=host, port=port, lifespan="on")
