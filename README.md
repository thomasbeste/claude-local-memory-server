# Claude Memory

**Persistent, shared memory for Claude Code via MCP**

A lightweight memory server that gives Claude Code durable, searchable memory across machines. Built with DuckDB for efficient storage with semantic search powered by sentence-transformers.

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

- **Persistent storage** – Memories survive restarts, stored in DuckDB
- **Semantic search** – Find memories by meaning, not just keywords
- **Hybrid search** – Combines keyword matching with vector similarity (RRF)
- **Project scoping** – Auto-detect project from git, isolate memories per project
- **Shared across machines** – Run a central server, connect from anywhere
- **Fast queries** – DuckDB-powered with automatic indexing
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

### Option 2: Server Mode with Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/claude-memory.git
cd claude-memory
./scripts/install-docker.sh
```

The script builds the image, starts the container, and prints your API key.

Or use docker compose directly:

```bash
# Generate an API key
export MEMORY_API_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)

# Start the server
docker compose up -d

# Check status
docker compose logs -f
```

### Option 3: Server Mode with Systemd (LXC/VM)

**On your server (Debian/Ubuntu LXC, VM, etc.):**

```bash
git clone https://github.com/YOUR_USERNAME/claude-memory.git
cd claude-memory
sudo ./scripts/install-systemd.sh
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
# Show current project (auto-detected from git)
claude-memory project

# Add a memory (auto-associates with current project)
claude-memory add "Project X uses Python 3.12" --type fact --tags project:x

# Add memory to a specific project
claude-memory -p my-project add "Memory content"

# Add memory without project association
claude-memory add "Global memory" --no-project

# Search memories (default: hybrid mode, scoped to current project)
claude-memory search --query "Python"

# Search with specific mode
claude-memory search -q "software development" -m semantic  # meaning-based
claude-memory search -q "Python" -m keyword                  # exact matching
claude-memory search -q "API framework" -m hybrid            # combined (default)

# Search across ALL projects
claude-memory search -q "Python" --global

# Filter by type and tags
claude-memory search --type decision --tags project:x

# View statistics (for current project)
claude-memory stats

# View statistics across all projects
claude-memory stats --global

# Backfill embeddings for existing memories
claude-memory backfill-embeddings

# Delete a memory
claude-memory delete <memory-id>
```

### Search Modes

| Mode | Description |
|------|-------------|
| `keyword` | Traditional substring matching (ILIKE) |
| `semantic` | Vector similarity using sentence-transformers |
| `hybrid` | Combines both with Reciprocal Rank Fusion (default) |

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
| `MEMORY_PROJECT` | Override auto-detected project ID |

### Server Options

```bash
claude-memory server --help

Options:
  --host TEXT      Host to bind to [default: 0.0.0.0]
  --port INTEGER   Port to bind to [default: 8420]
  --data-dir TEXT  Data directory [default: /var/lib/claude-memory]
```

### Install Script Options

**Docker (install-docker.sh):**
```bash
./scripts/install-docker.sh --help

Options:
  --port PORT         HTTP port [default: 8420]
  --api-key KEY       Set API key (generated if omitted)
  --no-start          Build only, don't start the container
```

**Systemd (install-systemd.sh):**
```bash
./scripts/install-systemd.sh --help

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

**Local mode:** Claude Code ↔ stdio ↔ `server.py` ↔ DuckDB

**Server mode:** Claude Code ↔ stdio ↔ `client.py` ↔ HTTP ↔ `api.py` ↔ DuckDB

## Data Storage

Memories are stored in a persistent DuckDB database:
- **Server:** `/var/lib/claude-memory/memories.duckdb`
- **Local:** `~/.claude-memory/memories.duckdb`

The database includes embeddings for semantic search (384-dimensional vectors from `all-MiniLM-L6-v2`).

Inspect with DuckDB:
```bash
duckdb ~/.claude-memory/memories.duckdb -c "SELECT id, content, memory_type FROM memories LIMIT 10"
```

Export to Parquet (for backup):
```bash
# From Python
from claude_memory.storage import MemoryStorage
storage = MemoryStorage()
storage.export_parquet("/path/to/backup.parquet")
```

Backup script:
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
pip install -e ".[dev,server,embeddings]"

# Run tests
pytest

# Run linter
ruff check src/
```

### Optional Dependencies

| Extra | Description |
|-------|-------------|
| `server` | FastAPI + uvicorn for HTTP server mode |
| `embeddings` | sentence-transformers for semantic search |
| `dev` | pytest + ruff for development |
| `all` | Everything above |

## Roadmap

- [x] Semantic search with embeddings
- [x] Hybrid search (keyword + semantic)
- [x] DuckDB persistent mode
- [ ] Web UI for browsing memories
- [ ] Memory expiration/TTL
- [ ] User/project scoping
- [ ] Knowledge graph relationships
- [ ] Auto-summarization
- [ ] Prometheus metrics endpoint

See [TODO.md](TODO.md) for the full roadmap.

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
