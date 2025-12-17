# Claude Memory

**Persistent, shared memory for Claude Code via MCP**

A lightweight memory server that gives Claude Code durable, searchable memory across machines. Built with DuckDB and Parquet for efficient storage and fast queries.

```
┌─────────────────┐     ┌─────────────────┐
│ Claude Code     │     │ Claude Code     │
│ (laptop)        │     │ (desktop)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────────┐
          │  claude-memory      │
          │  server             │
          │  (your LXC/server)  │
          └─────────────────────┘
```

## Features

- **Persistent storage** – Memories survive restarts, stored in Parquet format
- **Shared across machines** – Run a central server, connect from anywhere
- **Fast queries** – DuckDB-powered search, not full-scan
- **Client tracking** – Know which machine created each memory
- **Tag-based organization** – Filter by project, tech stack, or custom tags
- **Simple setup** – One script installs everything on Debian/Ubuntu

## Quick Start

### Option 1: Local Mode (single machine)

```bash
# Install
pip install claude-memory

# Configure Claude Code
claude mcp add-json "memory" '{
  "command": "python",
  "args": ["-m", "claude_memory.cli", "serve"],
  "env": {"MEMORY_CLIENT_ID": "my-laptop"}
}'
```

### Option 2: Server Mode (shared across machines)

**On your server (Debian/Ubuntu LXC, VM, etc.):**

```bash
git clone https://github.com/YOUR_USERNAME/claude-memory.git
cd claude-memory
sudo ./scripts/install.sh
```

The script installs Python, creates a systemd service, and prints your API key.

**On each client machine:**

```bash
pip install claude-memory

claude mcp add-json "memory" '{
  "command": "python",
  "args": ["-m", "claude_memory.cli", "client"],
  "env": {
    "MEMORY_SERVER": "http://your-server:8420",
    "MEMORY_API_KEY": "your-key-here",
    "MEMORY_CLIENT_ID": "laptop"
  }
}'
```

## Usage

Once configured, Claude can store and retrieve memories:

**Store a memory:**
> "Remember that we decided to use FastAPI for the GPI project"

**Recall memories:**
> "What decisions have we made about GPI?"

**Search by client:**
> "What did I work on from my laptop last week?"

### CLI Commands

```bash
# Add a memory
claude-memory add "Project X uses Python 3.12" --type fact --tags project:x

# Search memories
claude-memory search --query "Python"
claude-memory search --type decision --tags project:x

# View statistics
claude-memory stats

# Delete a memory
claude-memory delete <memory-id>
```

## Memory Types

| Type | Use for |
|------|---------|
| `fact` | Concrete facts (names, versions, configs) |
| `decision` | Decisions made during conversations |
| `preference` | User preferences (tools, style, language) |
| `observation` | General observations |
| `entity` | People, projects, organizations |
| `relation` | Relationships between entities |

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MEMORY_SERVER` | HTTP server URL (client mode) |
| `MEMORY_API_KEY` | API key for authentication |
| `MEMORY_CLIENT_ID` | Identifier for this client (e.g., "laptop", "work-pc") |

### Server Options

```bash
claude-memory server --help

Options:
  --host TEXT      Host to bind to [default: 0.0.0.0]
  --port INTEGER   Port to bind to [default: 8420]
  --data-dir TEXT  Data directory [default: /var/lib/claude-memory]
```

### Install Script Options

```bash
./scripts/install.sh --help

Options:
  --port PORT         HTTP port [default: 8420]
  --api-key KEY       Set API key (generated if omitted)
  --data-dir DIR      Data directory [default: /var/lib/claude-memory]
  --no-service        Don't install systemd service
```

## Architecture

```
src/claude_memory/
├── storage.py   # DuckDB/Parquet storage layer
├── server.py    # MCP stdio server (local mode)
├── api.py       # FastAPI HTTP server (server mode)
├── client.py    # MCP stdio client → HTTP proxy
└── cli.py       # Command-line interface
```

**Local mode:** Claude Code ↔ stdio ↔ `server.py` ↔ DuckDB ↔ Parquet file

**Server mode:** Claude Code ↔ stdio ↔ `client.py` ↔ HTTP ↔ `api.py` ↔ DuckDB ↔ Parquet file

## Data Storage

Memories are stored in a single Parquet file:
- **Server:** `/var/lib/claude-memory/memories.parquet`
- **Local:** `~/.claude-memory/memories.parquet`

Inspect with DuckDB:
```bash
duckdb -c "SELECT * FROM '~/.claude-memory/memories.parquet' LIMIT 10"
```

Backup:
```bash
./scripts/backup.sh  # Creates timestamped backup in /var/backups/claude-memory/
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/YOUR_USERNAME/claude-memory.git
cd claude-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,server]"

# Run tests
pytest

# Run linter
ruff check src/
```

## Roadmap

- [ ] Semantic search with embeddings
- [ ] Web UI for browsing memories
- [ ] Memory expiration/TTL
- [ ] User/project scoping
- [ ] Prometheus metrics endpoint

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
