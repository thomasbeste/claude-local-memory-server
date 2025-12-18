# Claude Local Memory Server

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
          │  claude-local-      │
          │  memory-server      │
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
- **Production ready** – Rate limiting, request logging, CORS support
- **Simple setup** – One script installs everything on Debian/Ubuntu

## Quick Start

### Option 1: Local Mode (single machine)

```bash
# Install
pip install claude-local-memory-server

# Configure Claude Code
claude mcp add-json "memory" '{
  "command": "python",
  "args": ["-m", "claude_memory.cli", "serve"],
  "env": {"MEMORY_CLIENT_ID": "my-laptop"}
}'
```

### Option 2: Server Mode with Docker (recommended)

```bash
git clone https://github.com/thomasbeste/claude-local-memory-server.git
cd claude-local-memory-server
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

**HTTP MCP Mode (direct connection):**

The server exposes an MCP-over-HTTP endpoint at `/mcp`. Configure Claude Code to connect directly:

```bash
claude mcp add memory --transport http --url http://your-server:8420/mcp
```

This eliminates the need for a local client process — Claude Code connects directly to the server via HTTP.

### Option 3: Server Mode with Systemd (LXC/VM)

**On your server (Debian/Ubuntu LXC, VM, etc.):**

```bash
git clone https://github.com/thomasbeste/claude-local-memory-server.git
cd claude-local-memory-server
sudo ./scripts/install-systemd.sh
```

The script installs Python, creates a systemd service, and prints your API key.

**On each client machine:**

```bash
pip install claude-local-memory-server

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

# Get context summary for current project (prioritized decisions, preferences, facts)
claude-memory context

# Get context as JSON
claude-memory context --json

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

**Client variables:**

| Variable | Description |
|----------|-------------|
| `MEMORY_SERVER` | HTTP server URL (client mode) |
| `MEMORY_API_KEY` | API key for authentication |
| `MEMORY_CLIENT_ID` | Identifier for this client (e.g., "laptop", "work-pc") |
| `MEMORY_PROJECT` | Override auto-detected project ID |

**Server variables:**

| Variable | Description |
|----------|-------------|
| `MEMORY_API_KEY` | API key for authentication (required for production) |
| `CORS_ORIGINS` | Comma-separated allowed origins (e.g., `https://ui.example.com`) |

### Server Options

```bash
claude-memory server --help

Options:
  --host TEXT      Host to bind to [default: 0.0.0.0]
  --port INTEGER   Port to bind to [default: 8420]
  --data-dir TEXT  Data directory [default: /var/lib/claude-local-memory-server]
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
  --data-dir DIR      Data directory [default: /var/lib/claude-local-memory-server]
  --no-service        Don't install systemd service
```

## Deploying Behind a Reverse Proxy

The server is designed to be exposed over the internet behind nginx or another reverse proxy. It includes:

- **API key authentication** – All routes (except `/health`) require `X-API-Key` header
- **Rate limiting** – 30/min for writes, 60/min for reads (per API key)
- **Request logging** – Every request logged with timing, IP, and API key hint
- **CORS support** – Configurable via `CORS_ORIGINS` environment variable

### Nginx Configuration

```nginx
# Rate limiting zone (optional, server has built-in limits)
limit_req_zone $binary_remote_addr zone=memory_api:10m rate=10r/s;

server {
    listen 443 ssl http2;
    server_name memory.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/memory.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;

    location / {
        proxy_pass http://127.0.0.1:8420;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Optional: nginx-level rate limiting
        limit_req zone=memory_api burst=20 nodelay;

        # Request size limit (prevent memory bombs)
        client_max_body_size 1m;
    }

    # Health check (no auth required)
    location /health {
        proxy_pass http://127.0.0.1:8420/health;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name memory.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Rate Limits

Built-in rate limits per API key (or IP if no key):

| Operation | Limit |
|-----------|-------|
| Create memory | 30/minute |
| Update memory | 30/minute |
| Delete memory | 30/minute |
| Search | 60/minute |
| Get memory | 60/minute |
| Stats/Context | 60/minute |

When rate limited, the server returns `429 Too Many Requests`.

### Request Logging

All requests are logged in this format:

```
2025-12-18 12:34:56 | claude_memory.api | INFO | POST /memories | 201 | 45.2ms | ip=192.168.1.5 key=abc12345...
```

Logs include:
- Timestamp
- HTTP method and path
- Response status code
- Request duration
- Client IP address
- First 8 characters of API key (for debugging without exposing full key)

## Architecture

```
src/claude_memory/
├── storage.py   # DuckDB/Parquet storage layer
├── server.py    # MCP stdio server (local mode)
├── api.py       # FastAPI HTTP server + MCP-over-HTTP endpoint
├── client.py    # MCP stdio client → HTTP proxy
└── cli.py       # Command-line interface
```

**Local mode:** Claude Code ↔ stdio ↔ `server.py` ↔ DuckDB

**Client mode:** Claude Code ↔ stdio ↔ `client.py` ↔ HTTP REST ↔ `api.py` ↔ DuckDB

**HTTP MCP mode:** Claude Code ↔ HTTP MCP ↔ `api.py` `/mcp` endpoint ↔ DuckDB

## Data Storage

Memories are stored in a persistent DuckDB database:
- **Server:** `/var/lib/claude-local-memory-server/memories.duckdb`
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
./scripts/backup.sh  # Creates timestamped backup in /var/backups/claude-local-memory-server/
```

## Using Hooks for Automatic Context

Claude Code supports [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) — shell commands that run in response to events. You can use hooks to automatically inject memory context at the start of every conversation.

### Auto-inject Project Context

Create `.claude/settings.json` in your project (or `~/.claude/settings.json` globally):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "claude-memory context --json 2>/dev/null | jq -r '\"[Memory Context for \" + .project_id + \"]\\n\" + (.memories | map(\"- [\" + .memory_type + \"] \" + .content) | join(\"\\n\"))' 2>/dev/null || echo ''",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

This runs on every prompt submission and prepends relevant memories to your conversation. The empty `matcher` means it runs on all prompts.

### Simpler Hook (No jq Required)

If you don't have `jq` installed:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "claude-memory context 2>/dev/null || echo ''",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

### Hook Tips

- **Timeout**: Set a reasonable timeout (5000ms) so slow network doesn't block your conversation
- **Error handling**: The `|| echo ''` ensures Claude Code doesn't choke on errors
- **Stderr redirect**: `2>/dev/null` suppresses connection warnings
- **Project scoping**: `claude-memory context` automatically detects your git project and returns only relevant memories

### Other Hook Ideas

| Event | Use Case |
|-------|----------|
| `PreToolUse` | Log when Claude is about to store a memory |
| `PostToolUse` | Trigger notifications after memory operations |
| `Stop` | Summarize what was remembered in the session |

See the [Claude Code hooks documentation](https://docs.anthropic.com/en/docs/claude-code/hooks) for the full event reference.

## Development

```bash
# Clone and install in development mode
git clone https://github.com/thomasbeste/claude-local-memory-server.git
cd claude-local-memory-server
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
- [x] Rate limiting and request logging
- [x] Production-ready for reverse proxy deployment
- [x] MCP-over-HTTP endpoint (direct Claude Code connection)
- [ ] Web UI for browsing memories
- [ ] Memory expiration/TTL
- [ ] Knowledge graph relationships
- [ ] Auto-summarization
- [ ] Prometheus metrics endpoint

See [TODO.md](TODO.md) for the full roadmap.

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
