# Contributing to Claude Local Memory Server

Thanks for your interest in contributing! This project is open to contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/thomasbeste/claude-local-memory-server.git
   cd claude-local-memory-server
   ```

3. Set up development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   pip install -e ".[dev,server]"
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
pytest
```

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

### Testing Your Changes

**Local mode:**
```bash
claude-memory add "test memory" --type fact
claude-memory search --query "test"
claude-memory stats
```

**Server mode:**
```bash
# Terminal 1: Start server
claude-memory server --port 8420

# Terminal 2: Test API
curl http://localhost:8420/health
curl http://localhost:8420/stats
```

## Pull Request Process

1. Ensure tests pass and code is formatted
2. Update documentation if needed
3. Write a clear PR description explaining your changes
4. Link any related issues

## What to Contribute

### Good First Issues

- Improve error messages
- Add more CLI options
- Write tests
- Documentation improvements

### Feature Ideas

- Semantic search with embeddings
- Web UI for browsing memories
- Memory expiration/TTL
- Export/import functionality
- Additional storage backends

### Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs/error messages

## Project Structure

```
claude-local-memory-server/
├── src/claude_memory/
│   ├── __init__.py     # Package exports
│   ├── storage.py      # DuckDB/Parquet storage
│   ├── server.py       # MCP stdio server (local)
│   ├── api.py          # FastAPI HTTP server
│   ├── client.py       # HTTP client wrapper
│   └── cli.py          # Click CLI
├── scripts/
│   ├── install.sh      # Server installation
│   ├── uninstall.sh    # Cleanup script
│   └── backup.sh       # Backup utility
├── tests/              # Test files
└── pyproject.toml      # Project config
```

## Code Guidelines

- Type hints for all function signatures
- Docstrings for public functions
- Keep functions focused and small
- Prefer clarity over cleverness

## Questions?

Open an issue or start a discussion. We're happy to help!
