"""CLI for testing and running the memory server."""

import json
import click

from .storage import MemoryStorage


@click.group()
@click.option("--data-dir", default="~/.claude-memory", help="Data directory for memories")
@click.pass_context
def main(ctx: click.Context, data_dir: str) -> None:
    """Claude Memory - Local persistent memory for Claude."""
    ctx.ensure_object(dict)
    ctx.obj["storage"] = MemoryStorage(data_dir)


@main.command()
@click.argument("content")
@click.option("--type", "memory_type", default="observation", help="Memory type")
@click.option("--tags", "-t", multiple=True, help="Tags (can be repeated)")
@click.option("--source", "-s", help="Source/context")
@click.pass_context
def add(ctx: click.Context, content: str, memory_type: str, tags: tuple, source: str | None) -> None:
    """Add a new memory."""
    storage: MemoryStorage = ctx.obj["storage"]
    result = storage.add(
        content=content,
        memory_type=memory_type,
        tags=list(tags) if tags else None,
        source=source,
    )
    click.echo(f"Added memory: {result['id']}")
    click.echo(json.dumps(result, default=str, indent=2))


@main.command()
@click.option("--query", "-q", help="Search query")
@click.option("--type", "memory_type", help="Filter by type")
@click.option("--tags", "-t", multiple=True, help="Filter by tags")
@click.option("--limit", "-n", default=20, help="Max results")
@click.pass_context
def search(ctx: click.Context, query: str | None, memory_type: str | None, tags: tuple, limit: int) -> None:
    """Search memories."""
    storage: MemoryStorage = ctx.obj["storage"]
    results = storage.search(
        query=query,
        memory_type=memory_type,
        tags=list(tags) if tags else None,
        limit=limit,
    )
    if not results:
        click.echo("No memories found.")
        return

    click.echo(f"Found {len(results)} memories:\n")
    for mem in results:
        tags_str = ", ".join(mem["tags"]) if mem["tags"] else "none"
        click.echo(f"[{mem['id']}] ({mem['memory_type']}) {mem['content'][:80]}")
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
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory statistics."""
    storage: MemoryStorage = ctx.obj["storage"]
    s = storage.stats()
    click.echo(f"Total memories: {s['total_memories']}")
    click.echo(f"Storage: {s['storage_path']}")
    if s["by_type"]:
        click.echo("\nBy type:")
        for t, count in s["by_type"].items():
            click.echo(f"  {t}: {count}")
    if s.get("by_client"):
        click.echo("\nBy client:")
        for c, count in s["by_client"].items():
            click.echo(f"  {c}: {count}")


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
