# GitHub Repository Setup

## Repository Name
claude-local-memory-server

## Description (About)
Persistent, shared memory for Claude Code via MCP. DuckDB/Parquet storage with multi-machine support.

## Topics/Tags
- claude
- mcp
- model-context-protocol
- memory
- duckdb
- parquet
- ai
- llm
- claude-code

## Website
(leave blank or add docs URL later)

## Settings Recommendations

### General
- [x] Issues enabled
- [x] Discussions enabled (for Q&A and ideas)
- [ ] Projects disabled (unless you want kanban)
- [ ] Wiki disabled (use docs/ folder instead)

### Branches
- Default branch: `main`
- Branch protection on `main`:
  - [x] Require pull request before merging
  - [x] Require status checks (CI)
  - [x] Require branches to be up to date

### Actions
- Allow all actions

---

## Initial Commit Message

```
Initial release: MCP memory server for Claude Code

Features:
- Local and server modes
- DuckDB/Parquet storage
- Client identification
- Tag-based organization
- CLI for testing
- Systemd service installer
```
