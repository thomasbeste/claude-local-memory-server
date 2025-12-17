# Claude Memory Server - Development Roadmap

> A comprehensive list of improvements to transform this from "it works on my machine" to "enterprise-grade memory persistence."

---

## Table of Contents

1. [Critical: Storage Architecture](#1-critical-storage-architecture)
2. [High Priority: Semantic Search](#2-high-priority-semantic-search)
3. [High Priority: Scoping & Multi-tenancy](#3-high-priority-scoping--multi-tenancy)
4. [Medium Priority: Knowledge Graphs](#4-medium-priority-knowledge-graphs)
5. [Medium Priority: Auto-Summarization](#5-medium-priority-auto-summarization)
6. [Low Priority: Quality of Life](#6-low-priority-quality-of-life)
7. [Future Considerations](#7-future-considerations)

---

## 1. Critical: Storage Architecture

The current approach regenerates the entire Parquet file on every write. That's fine for your diary, not for a production system.

### 1.1 DuckDB Persistent Mode

**Current Problem:** ~~In-memory DuckDB + full Parquet export on every mutation. O(n) writes.~~ SOLVED

**Solution:** Use DuckDB's native persistent mode instead of Parquet export.

- [x] Replace in-memory DuckDB with persistent database file
- [x] Remove manual Parquet export/import logic
- [x] Add automatic migration from legacy parquet files
- [x] Keep Parquet export as optional backup/export feature
- [ ] Implement proper connection pooling for concurrent access
- [ ] Add WAL (Write-Ahead Logging) mode for crash recovery
- [ ] Implement periodic VACUUM for space reclamation
- [ ] Add database migration system for schema changes

### 1.2 Event Sourcing Architecture

**Rationale:** Memory systems benefit from immutability. Know what changed, when, and why.

- [ ] Design event schema:
  ```sql
  CREATE TABLE memory_events (
      event_id VARCHAR PRIMARY KEY,
      memory_id VARCHAR NOT NULL,
      event_type VARCHAR NOT NULL,  -- 'created', 'updated', 'deleted', 'merged'
      payload JSON NOT NULL,
      actor_id VARCHAR,             -- client_id or 'system'
      timestamp TIMESTAMPTZ DEFAULT now(),
      sequence_number BIGINT        -- for ordering guarantees
  );
  ```
- [ ] Implement event handlers:
  - [ ] `MemoryCreated` - stores full memory content
  - [ ] `MemoryUpdated` - stores delta/patch
  - [ ] `MemoryDeleted` - soft delete marker
  - [ ] `MemoryMerged` - when duplicates are consolidated
  - [ ] `MemorySummarized` - when auto-summary runs
- [ ] Build materialized view (current state) from events
- [ ] Add event replay capability for debugging/recovery
- [ ] Implement snapshots for faster reconstruction (every N events)
- [ ] Add event streaming endpoint for real-time sync

### 1.3 Write-Ahead Log (Alternative to Full Event Sourcing)

**Lighter-weight option if full event sourcing is overkill:**

- [ ] Implement simple WAL for pending writes
  ```
  ~/.claude-memory/wal/
    ├── 000001.wal
    ├── 000002.wal
    └── checkpoint.meta
  ```
- [ ] Batch writes and flush periodically (configurable interval)
- [ ] Implement checkpoint mechanism to merge WAL into main store
- [ ] Add crash recovery from WAL on startup
- [ ] Support concurrent readers during WAL flush

---

## 2. High Priority: Semantic Search

Current search is `ILIKE '%query%'`. That's keyword matching from 2005.

### 2.1 Embedding Infrastructure

- [x] Add sentence-transformers integration (already in optional deps)
- [x] Design embedding storage schema (FLOAT[] column in memories table)
- [x] Implement embedding generation on memory creation
- [x] Add backfill command for embedding existing memories (`claude-memory backfill-embeddings`)
- [x] Cache model in memory (singleton Embedder class)
- [ ] Support multiple embedding models (swap without data loss)
- [ ] Add background job for embedding existing memories (async)

### 2.2 Vector Search Implementation

- [x] Implement cosine similarity search using `list_cosine_similarity()`
- [x] Implement hybrid search with Reciprocal Rank Fusion (RRF)
- [x] Add relevance threshold filtering (`min_score` parameter)
- [ ] Add DuckDB VSS extension when stable (or use pgvector if migrating)
- [ ] Implement query expansion using embeddings

### 2.3 Search API Enhancements

- [x] Add `search_mode` parameter: `keyword`, `semantic`, `hybrid`
- [x] Return relevance scores in results (`score` and `hybrid_score`)
- [ ] Add `min_score` threshold parameter to API
- [ ] Support "more like this" queries (find similar to memory ID)
- [ ] Add faceted search (aggregate by type, tags, client)

---

## 3. High Priority: Scoping & Multi-tenancy

Currently everything is one big bucket. That doesn't scale socially.

### 3.1 Project Scoping

- [ ] Add `project_id` column to memories:
  ```sql
  ALTER TABLE memories ADD COLUMN project_id VARCHAR;
  CREATE INDEX idx_memories_project ON memories(project_id);
  ```
- [ ] Auto-detect project from git repo root or config file
- [ ] Support explicit project assignment via CLI/API
- [ ] Add project-level statistics
- [ ] Implement project isolation (search defaults to current project)
- [ ] Add cross-project search with explicit flag

### 3.2 User/Workspace Scoping

- [ ] Add `user_id` column for multi-user deployments
- [ ] Implement workspace concept (group of projects):
  ```sql
  CREATE TABLE workspaces (
      id VARCHAR PRIMARY KEY,
      name VARCHAR NOT NULL,
      owner_id VARCHAR NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
  );
  CREATE TABLE workspace_members (
      workspace_id VARCHAR REFERENCES workspaces(id),
      user_id VARCHAR NOT NULL,
      role VARCHAR DEFAULT 'member',
      PRIMARY KEY (workspace_id, user_id)
  );
  ```
- [ ] Add access control (read/write/admin per workspace)
- [ ] Support personal vs shared memories
- [ ] Implement memory visibility levels: `private`, `project`, `workspace`, `public`

### 3.3 Context Inheritance

- [ ] Design scope hierarchy: `global -> workspace -> project -> session`
- [ ] Memories inherit scope from context by default
- [ ] Search bubbles up through scope hierarchy
- [ ] Add scope override capability for cross-cutting concerns

---

## 4. Medium Priority: Knowledge Graphs

The `relation` memory type exists but isn't leveraged. Time to build an actual graph.

### 4.1 Entity-Relationship Model

- [ ] Create explicit entity table:
  ```sql
  CREATE TABLE entities (
      id VARCHAR PRIMARY KEY,
      name VARCHAR NOT NULL,
      entity_type VARCHAR NOT NULL,  -- person, project, technology, concept
      properties JSON,
      memory_id VARCHAR REFERENCES memories(id),  -- link to original memory
      created_at TIMESTAMPTZ DEFAULT now()
  );
  ```
- [ ] Create relations table:
  ```sql
  CREATE TABLE relations (
      id VARCHAR PRIMARY KEY,
      source_entity_id VARCHAR REFERENCES entities(id),
      relation_type VARCHAR NOT NULL,  -- 'works_on', 'depends_on', 'prefers', 'decided'
      target_entity_id VARCHAR REFERENCES entities(id),
      properties JSON,
      confidence FLOAT DEFAULT 1.0,
      memory_id VARCHAR REFERENCES memories(id),
      created_at TIMESTAMPTZ DEFAULT now()
  );
  ```
- [ ] Implement entity extraction from memory content (NER)
- [ ] Auto-generate relations from entity memories

### 4.2 Graph Query Capabilities

- [ ] Add graph traversal queries:
  - [ ] "What do I know about [entity]?"
  - [ ] "How are [entity A] and [entity B] related?"
  - [ ] "What technologies are used in [project]?"
  - [ ] "Who decided [decision]?"
- [ ] Implement path finding between entities
- [ ] Add neighborhood queries (entities within N hops)
- [ ] Support Cypher-like query syntax or GraphQL

### 4.3 Graph Visualization

- [ ] Export graph to common formats (GraphML, GEXF, JSON)
- [ ] Add API endpoint for graph data
- [ ] Build simple web visualization (D3.js force graph)
- [ ] Support filtering graph by time range, entity type, relation type

### 4.4 Graph Maintenance

- [ ] Implement entity deduplication/merging
- [ ] Add confidence scoring for auto-extracted relations
- [ ] Support manual relation curation via CLI/API
- [ ] Implement entity resolution (same entity, different names)

---

## 5. Medium Priority: Auto-Summarization

Memories accumulate. Eventually you need synthesis, not just storage.

### 5.1 Memory Consolidation

- [ ] Implement duplicate detection:
  ```python
  def find_duplicates(threshold=0.9):
      # Use embedding similarity to find near-duplicates
      # Return clusters of similar memories
  ```
- [ ] Add merge capability for duplicate memories
- [ ] Track merge history (which memories combined into which)
- [ ] Implement "canonical" memory selection

### 5.2 Automatic Summarization

- [ ] Design summarization triggers:
  - [ ] Time-based: summarize memories older than X days
  - [ ] Count-based: summarize when > N memories on topic
  - [ ] Tag-based: summarize all memories with specific tag
  - [ ] Manual: on-demand summarization command
- [ ] Implement summarization pipeline:
  ```python
  async def summarize_memories(memory_ids: list[str]) -> Memory:
      memories = [get_memory(id) for id in memory_ids]
      prompt = build_summarization_prompt(memories)
      summary = await llm.complete(prompt)
      # Create new summary memory, link to originals
      return create_summary_memory(summary, memory_ids)
  ```
- [ ] Add summary memory type with source references
- [ ] Implement hierarchical summarization (summaries of summaries)
- [ ] Support customizable summarization prompts

### 5.3 Memory Lifecycle Management

- [ ] Implement TTL (time-to-live) for ephemeral memories
- [ ] Add importance scoring:
  - Access frequency
  - Reference count
  - Explicit user rating
  - Relation density
- [ ] Auto-archive low-importance old memories
- [ ] Implement "forgetting" policy (configurable retention)
- [ ] Add memory resurrection from archives

### 5.4 Topic Clustering

- [ ] Implement automatic topic detection using embeddings
- [ ] Cluster memories by semantic similarity
- [ ] Generate topic labels automatically
- [ ] Add topic-based navigation/filtering
- [ ] Track topic evolution over time

---

## 6. Low Priority: Quality of Life

### 6.1 CLI Improvements

- [ ] Add interactive mode with REPL
- [ ] Implement `memory import` from various formats (JSON, CSV, Markdown)
- [ ] Add `memory export` with format options
- [ ] Implement `memory doctor` for health checks and repairs
- [ ] Add shell completions (bash, zsh, fish)
- [ ] Support piping/stdin for bulk operations

### 6.2 Web UI

- [ ] Build simple admin dashboard:
  - [ ] Memory browser with search/filter
  - [ ] Statistics and charts
  - [ ] Entity/relation graph view
  - [ ] Backup/restore controls
- [ ] Add memory editor (create, update, delete)
- [ ] Implement bulk operations UI
- [ ] Add dark mode (obviously)

### 6.3 Observability

- [ ] Add structured logging (JSON format option)
- [ ] Implement metrics endpoint (Prometheus format):
  - Memory count by type/tag/client
  - Query latency percentiles
  - Storage size over time
  - Error rates
- [ ] Add distributed tracing support (OpenTelemetry)
- [ ] Implement audit log for compliance

### 6.4 Performance

- [ ] Add query result caching (LRU with TTL)
- [ ] Implement connection pooling properly
- [ ] Add database query analysis/EXPLAIN logging
- [ ] Optimize embedding generation (batch processing)
- [ ] Add async everywhere (currently mixed)

---

## 7. Future Considerations

### 7.1 Distributed Architecture

- [ ] Evaluate CRDTs for conflict-free multi-master sync
- [ ] Consider PostgreSQL + pgvector for serious deployments
- [ ] Implement pub/sub for real-time sync between clients
- [ ] Add conflict resolution strategies

### 7.2 Advanced Features

- [ ] Memory chains (linked sequences of memories)
- [ ] Temporal queries ("what did I know on date X?")
- [ ] Memory version control (full history, not just current state)
- [ ] Plugin system for custom memory processors
- [ ] Integration with external knowledge bases (Wikipedia, docs)

### 7.3 Security

- [ ] Implement encryption at rest
- [ ] Add field-level encryption for sensitive memories
- [ ] Support external secret management (Vault, AWS Secrets Manager)
- [ ] Add rate limiting per client
- [ ] Implement proper RBAC

### 7.4 AI Integration

- [ ] Proactive memory suggestions ("you might want to remember this")
- [ ] Contradiction detection ("this conflicts with memory X")
- [ ] Context-aware retrieval (automatically inject relevant memories)
- [ ] Memory quality scoring (is this worth remembering?)

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Priority | Status |
|---------|--------|--------|----------|--------|
| DuckDB Persistent Mode | High | Low | P0 | DONE |
| Semantic Search (basic) | High | Medium | P0 | DONE |
| Project Scoping | High | Low | P1 | |
| WAL/Event Sourcing | High | High | P1 | |
| Knowledge Graph (basic) | Medium | Medium | P2 | |
| Auto-Summarization | Medium | High | P2 | |
| Web UI | Low | Medium | P3 | |
| Distributed Sync | Medium | Very High | P3 | |

---

## Quick Wins (< 1 day each)

1. ~~Switch to DuckDB persistent mode (remove Parquet dance)~~ DONE
2. Add `project_id` column and basic filtering
3. ~~Integrate sentence-transformers for embedding generation~~ DONE
4. Add `--project` flag to CLI commands
5. Implement basic duplicate detection
6. Add Prometheus metrics endpoint
7. Create backup rotation in backup.sh

---

## Notes

- sentence-transformers is already listed as an optional dependency
- DuckDB supports persistent mode natively - no migration needed
- Event sourcing can be added incrementally (start with audit log)
- Knowledge graph can bootstrap from existing `relation` type memories

*"The best memory system is one that forgets the right things." - Nobody, but they should have*
