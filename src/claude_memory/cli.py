"""CLI for testing and running the memory server."""

import json
import click

from .storage import MemoryStorage, get_current_project, detect_project


@click.group()
@click.option("--data-dir", default="~/.claude-memory", help="Data directory for memories")
@click.option("--project", "-p", default=None, help="Project ID (auto-detected from git if not specified)")
@click.pass_context
def main(ctx: click.Context, data_dir: str, project: str | None) -> None:
    """Claude Memory - Local persistent memory for Claude."""
    ctx.ensure_object(dict)
    ctx.obj["storage"] = MemoryStorage(data_dir)
    # Auto-detect project if not specified
    ctx.obj["project"] = project or get_current_project()


@main.command()
@click.argument("content")
@click.option("--type", "memory_type", default="observation", help="Memory type")
@click.option("--tags", "-t", multiple=True, help="Tags (can be repeated)")
@click.option("--source", "-s", help="Source/context")
@click.option("--no-project", is_flag=True, help="Don't associate with current project")
@click.pass_context
def add(ctx: click.Context, content: str, memory_type: str, tags: tuple, source: str | None, no_project: bool) -> None:
    """Add a new memory."""
    storage: MemoryStorage = ctx.obj["storage"]
    project_id = None if no_project else ctx.obj.get("project")

    result = storage.add(
        content=content,
        memory_type=memory_type,
        tags=list(tags) if tags else None,
        source=source,
        project_id=project_id,
    )
    click.echo(f"Added memory: {result['id']}")
    if project_id:
        click.echo(f"Project: {project_id}")
    click.echo(json.dumps(result, default=str, indent=2))


@main.command()
@click.option("--query", "-q", help="Search query")
@click.option("--type", "memory_type", help="Filter by type")
@click.option("--tags", "-t", multiple=True, help="Filter by tags")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--mode", "-m", default="hybrid", type=click.Choice(["keyword", "semantic", "hybrid"]),
              help="Search mode: keyword (exact), semantic (meaning), hybrid (both)")
@click.option("--global", "-g", "global_search", is_flag=True, help="Search across all projects")
@click.pass_context
def search(ctx: click.Context, query: str | None, memory_type: str | None, tags: tuple, limit: int, mode: str, global_search: bool) -> None:
    """Search memories with keyword, semantic, or hybrid search."""
    storage: MemoryStorage = ctx.obj["storage"]
    project_id = ctx.obj.get("project")

    results = storage.search(
        query=query,
        memory_type=memory_type,
        tags=list(tags) if tags else None,
        project_id=project_id,
        global_search=global_search,
        limit=limit,
        search_mode=mode,
    )
    if not results:
        scope = "all projects" if global_search else f"project '{project_id}'" if project_id else "all memories"
        click.echo(f"No memories found in {scope}.")
        return

    scope_str = "all projects" if global_search else f"project '{project_id}'" if project_id else "all"
    click.echo(f"Found {len(results)} memories (mode: {mode}, scope: {scope_str}):\n")
    for mem in results:
        tags_str = ", ".join(mem["tags"]) if mem["tags"] else "none"
        project_str = f" @{mem['project_id']}" if mem.get("project_id") else ""
        score_str = ""
        if "score" in mem:
            score_str = f" [score: {mem['score']:.3f}]"
        elif "hybrid_score" in mem:
            score_str = f" [hybrid: {mem['hybrid_score']:.4f}]"
        click.echo(f"[{mem['id']}] ({mem['memory_type']}){project_str}{score_str} {mem['content'][:70]}")
        click.echo(f"    tags: {tags_str} | {mem['created_at']}")
        click.echo()


@main.command()
@click.argument("memory_id")
@click.pass_context
def delete(ctx: click.Context, memory_id: str) -> None:
    """Delete a memory by ID."""
    storage: MemoryStorage = ctx.obj["storage"]
    if storage.delete(memory_id):
        click.echo(f"Deleted memory: {memory_id}")
    else:
        click.echo(f"Memory not found: {memory_id}")


@main.command()
@click.option("--global", "-g", "global_stats", is_flag=True, help="Show stats for all projects")
@click.pass_context
def stats(ctx: click.Context, global_stats: bool) -> None:
    """Show memory statistics."""
    storage: MemoryStorage = ctx.obj["storage"]
    project_id = None if global_stats else ctx.obj.get("project")

    s = storage.stats(project_id=project_id)

    if project_id:
        click.echo(f"Project: {project_id}")
    click.echo(f"Total memories: {s['total_memories']}")
    click.echo(f"With embeddings: {s.get('memories_with_embeddings', 'N/A')}")
    click.echo(f"Embeddings available: {s.get('embeddings_available', False)}")
    click.echo(f"Storage: {s['storage_path']}")

    if s["by_type"]:
        click.echo("\nBy type:")
        for t, count in s["by_type"].items():
            click.echo(f"  {t}: {count}")

    if s.get("by_client"):
        click.echo("\nBy client:")
        for c, count in s["by_client"].items():
            click.echo(f"  {c}: {count}")

    if s.get("by_project") and (global_stats or not project_id):
        click.echo("\nBy project:")
        for p, count in s["by_project"].items():
            click.echo(f"  {p}: {count}")


@main.command("project")
@click.pass_context
def show_project(ctx: click.Context) -> None:
    """Show the current project (auto-detected or specified)."""
    project = ctx.obj.get("project")
    if project:
        click.echo(f"Current project: {project}")
        detected = detect_project()
        if detected and detected != project:
            click.echo(f"(Auto-detected: {detected}, overridden by --project or MEMORY_PROJECT)")
    else:
        click.echo("No project detected.")
        click.echo("Run from within a git repository to auto-detect, or use --project/-p flag.")


@main.command()
@click.option("--max", "-n", "max_memories", default=10, help="Maximum memories to include")
@click.option("--days", "-d", default=30, help="Only include memories from last N days")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON")
@click.option("--global", "-g", "global_context", is_flag=True, help="Get context across all projects")
@click.pass_context
def context(ctx: click.Context, max_memories: int, days: int, as_json: bool, global_context: bool) -> None:
    """
    Get curated context summary for the current project.

    Shows prioritized memories (decisions, preferences, facts) suitable
    for understanding the current state of a project.
    """
    storage: MemoryStorage = ctx.obj["storage"]
    project_id = None if global_context else ctx.obj.get("project")

    result = storage.get_context_summary(
        project_id=project_id,
        max_memories=max_memories,
        days=days,
    )

    if as_json:
        click.echo(json.dumps(result, default=str, indent=2))
    else:
        if not result.get("has_context"):
            scope = "all projects" if global_context else f"project '{project_id}'" if project_id else "any project"
            click.echo(f"No memories found for {scope}.")
            return

        click.echo(result["summary"])

        if result.get("last_activity"):
            last = result["last_activity"]
            click.echo(f"\n---\nLast activity: {last['timestamp']}")
            click.echo(f"Last memory: [{last['type']}] {last['last_memory'][:60]}...")


@main.command("backfill-embeddings")
@click.option("--batch-size", "-b", default=100, help="Batch size for processing")
@click.pass_context
def backfill_embeddings(ctx: click.Context, batch_size: int) -> None:
    """Generate embeddings for memories that don't have them."""
    storage: MemoryStorage = ctx.obj["storage"]

    # Check if embeddings are available
    from .storage import Embedder
    if not Embedder.is_available():
        click.echo("Error: sentence-transformers not installed.", err=True)
        click.echo("Install with: pip install claude-memory[embeddings]", err=True)
        raise SystemExit(1)

    click.echo("Backfilling embeddings for memories without them...")
    result = storage.backfill_embeddings(batch_size=batch_size)

    click.echo(f"Processed: {result['processed']}")
    click.echo(f"Failed: {result['failed']}")
    if result.get("error"):
        click.echo(f"Error: {result['error']}", err=True)


@main.command()
@click.option("--data-dir", default="~/.claude-memory", help="Data directory")
def serve(data_dir: str) -> None:
    """Run the MCP stdio server (for local Claude integration)."""
    from .server import run
    click.echo(f"Starting MCP stdio server with data dir: {data_dir}", err=True)
    run(data_dir)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8420, help="Port to bind to")
@click.option("--data-dir", default="/var/lib/claude-memory", help="Data directory")
def server(host: str, port: int, data_dir: str) -> None:
    """Run the HTTP API server (for central deployment)."""
    try:
        from .api import run_server
    except ImportError:
        click.echo("Error: Server dependencies not installed. Run: pip install claude-memory[server]", err=True)
        raise SystemExit(1)
    
    click.echo(f"Starting HTTP server on {host}:{port}", err=True)
    click.echo(f"Data directory: {data_dir}", err=True)
    run_server(host=host, port=port, data_dir=data_dir)


@main.command()
def client() -> None:
    """Run the MCP client (connects to remote HTTP server)."""
    import os
    server_url = os.environ.get("MEMORY_SERVER", "http://localhost:8420")
    click.echo(f"Starting MCP client, connecting to: {server_url}", err=True)
    
    from .client import run
    run()


@main.command()
@click.option("--remote", is_flag=True, help="Show config for remote server connection")
@click.option("--server-url", default="http://your-server:8420", help="Remote server URL")
@click.option("--client-id", default=None, help="Client identifier (e.g., 'laptop', 'desktop')")
def config(remote: bool, server_url: str, client_id: str | None) -> None:
    """Print MCP configuration for Claude."""
    import sys
    from pathlib import Path
    
    python_path = sys.executable
    venv_path = Path(python_path).parent.parent
    
    if remote:
        # Remote client config
        env = {
            "MEMORY_SERVER": server_url,
            # "MEMORY_API_KEY": "your-api-key",  # uncomment if using auth
        }
        if client_id:
            env["MEMORY_CLIENT_ID"] = client_id
        
        config = {
            "mcpServers": {
                "memory": {
                    "command": python_path,
                    "args": ["-m", "claude_memory.cli", "client"],
                    "env": env,
                }
            }
        }
        click.echo("Remote client configuration:\n")
    else:
        # Local server config
        env = {}
        if client_id:
            env["MEMORY_CLIENT_ID"] = client_id
        
        config = {
            "mcpServers": {
                "memory": {
                    "command": python_path,
                    "args": ["-m", "claude_memory.cli", "serve"],
                }
            }
        }
        if env:
            config["mcpServers"]["memory"]["env"] = env
        
        click.echo("Local server configuration:\n")
    
    click.echo(json.dumps(config, indent=2))
    click.echo("\nOr run:")
    click.echo(f'claude mcp add-json "memory" \'{json.dumps(config["mcpServers"]["memory"])}\'')
    
    if "venv" in python_path or ".venv" in python_path:
        click.echo(f"\n✓ Using venv: {venv_path}")
    
    if remote:
        click.echo(f"\nServer URL: {server_url}")
        click.echo("Set MEMORY_API_KEY env var if server requires authentication.")
    
    if client_id:
        click.echo(f"Client ID: {client_id}")
    else:
        click.echo("\nTip: Add --client-id 'laptop' to identify this machine's memories.")


if __name__ == "__main__":
    main()
