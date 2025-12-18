"""DuckDB persistent storage for memories with semantic search support."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
import logging
import subprocess
import os

import duckdb

logger = logging.getLogger(__name__)


def detect_project(path: Path | str | None = None) -> str | None:
    """
    Auto-detect project ID from git repository root.

    Returns the git repo name (directory name of repo root) or None if not in a git repo.
    """
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path).expanduser().resolve()

    try:
        # Get git repo root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            repo_root = Path(result.stdout.strip())
            return repo_root.name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


def get_current_project() -> str | None:
    """Get current project from environment or auto-detect."""
    # Check environment variable first
    project = os.environ.get("MEMORY_PROJECT")
    if project:
        return project

    # Auto-detect from git
    return detect_project()

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, good balance of speed/quality
EMBEDDING_DIM = 384


class Embedder:
    """Lazy-loaded sentence transformer for generating embeddings."""

    _instance = None
    _model = None

    @classmethod
    def get_instance(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("Embedding model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic search disabled. Install with: pip install claude-memory[embeddings]"
                )
                return None
        return self._model

    def embed(self, text: str) -> list[float] | None:
        """Generate embedding for text. Returns None if embeddings unavailable."""
        if self.model is None:
            return None
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            return None
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @classmethod
    def is_available(cls) -> bool:
        """Check if embedding support is available."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False


class MemoryStorage:
    """Persistent memory storage using DuckDB with semantic search support."""

    def __init__(self, data_dir: Path | str = "~/.claude-memory"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memories.duckdb"

        # Legacy parquet path for migration
        self.legacy_parquet_path = self.data_dir / "memories.parquet"

        # Use persistent DuckDB database
        self.db = duckdb.connect(str(self.db_path))
        self._init_schema()
        self._migrate_from_parquet()

        # Embedder instance (lazy loaded)
        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        """Get or create embedder instance."""
        if self._embedder is None:
            self._embedder = Embedder.get_instance()
        return self._embedder

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Create memories table if it doesn't exist
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR PRIMARY KEY,
                content VARCHAR NOT NULL,
                memory_type VARCHAR DEFAULT 'observation',
                tags VARCHAR[],
                source VARCHAR,
                client_id VARCHAR,
                project_id VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE,
                updated_at TIMESTAMP WITH TIME ZONE,
                embedding FLOAT[]
            )
        """)

        # Create indexes for common queries
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_client ON memories(client_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id)
        """)

        # Schema migrations for existing databases
        columns = self.db.execute("DESCRIBE memories").fetchall()
        column_names = [col[0] for col in columns]

        if "embedding" not in column_names:
            logger.info("Adding embedding column to existing database")
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding FLOAT[]")

        if "project_id" not in column_names:
            logger.info("Adding project_id column to existing database")
            self.db.execute("ALTER TABLE memories ADD COLUMN project_id VARCHAR")

        if "session_id" not in column_names:
            logger.info("Adding session_id column to existing database")
            self.db.execute("ALTER TABLE memories ADD COLUMN session_id VARCHAR")

        # Create sessions table for conversation tracking
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR PRIMARY KEY,
                project_id VARCHAR,
                client_id VARCHAR,
                started_at TIMESTAMP WITH TIME ZONE,
                ended_at TIMESTAMP WITH TIME ZONE,
                summary VARCHAR,
                topics VARCHAR[],
                decisions VARCHAR[],
                open_questions VARCHAR[],
                status VARCHAR DEFAULT 'active'
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)
        """)

    def _migrate_from_parquet(self) -> None:
        """Migrate data from legacy parquet file if it exists."""
        if not self.legacy_parquet_path.exists():
            return

        # Check if we already have data
        count = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > 0:
            logger.info(f"Database already has {count} memories, skipping parquet migration")
            return

        logger.info(f"Migrating from legacy parquet file: {self.legacy_parquet_path}")
        try:
            # Load from parquet into temp table
            self.db.execute(f"""
                CREATE TEMP TABLE legacy_memories AS
                SELECT * FROM read_parquet('{self.legacy_parquet_path}')
            """)

            # Check what columns exist in legacy data
            legacy_cols = self.db.execute("DESCRIBE legacy_memories").fetchall()
            legacy_col_names = [col[0] for col in legacy_cols]

            # Build insert statement based on available columns
            base_cols = ["id", "content", "memory_type", "tags", "source", "client_id", "created_at", "updated_at"]
            available_cols = [c for c in base_cols if c in legacy_col_names]

            col_list = ", ".join(available_cols)
            self.db.execute(f"""
                INSERT INTO memories ({col_list})
                SELECT {col_list} FROM legacy_memories
            """)

            migrated = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            logger.info(f"Migrated {migrated} memories from parquet")

            # Rename legacy file as backup
            backup_path = self.legacy_parquet_path.with_suffix(".parquet.migrated")
            self.legacy_parquet_path.rename(backup_path)
            logger.info(f"Legacy parquet backed up to: {backup_path}")

        except Exception as e:
            logger.error(f"Failed to migrate from parquet: {e}")

    def find_similar(
        self,
        content: str,
        project_id: str | None = None,
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find memories similar to the given content using embedding similarity.

        Args:
            content: Text to find similar memories for
            project_id: Limit search to specific project
            threshold: Minimum similarity score (0-1, default 0.85)
            limit: Maximum results to return

        Returns:
            List of similar memories with similarity scores
        """
        if not Embedder.is_available():
            return []

        embedding = self.embedder.embed(content)
        if embedding is None:
            return []

        conditions = ["embedding IS NOT NULL"]
        params: list = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where_clause = " AND ".join(conditions)

        result = self.db.execute(
            f"""
            SELECT
                id, content, memory_type, tags, project_id, created_at,
                list_cosine_similarity(embedding, ?::FLOAT[]) as similarity
            FROM memories
            WHERE {where_clause}
            AND list_cosine_similarity(embedding, ?::FLOAT[]) > ?
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [embedding] + params + [embedding, threshold, limit],
        ).fetchall()

        columns = ["id", "content", "memory_type", "tags", "project_id", "created_at", "similarity"]
        return [dict(zip(columns, row)) for row in result]

    def add(
        self,
        content: str,
        memory_type: str = "observation",
        tags: list[str] | None = None,
        source: str | None = None,
        client_id: str | None = None,
        project_id: str | None = None,
        check_duplicates: bool = True,
        duplicate_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """
        Add a new memory with optional embedding and duplicate detection.

        Args:
            content: Memory content
            memory_type: Type of memory
            tags: Optional tags
            source: Optional source
            client_id: Client identifier
            project_id: Project identifier
            check_duplicates: Whether to check for similar existing memories
            duplicate_threshold: Similarity threshold for duplicate detection (0-1)

        Returns:
            Dict with memory info and any duplicate warnings
        """
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        tags = tags or []

        # Check for duplicates before adding
        duplicates = []
        if check_duplicates:
            duplicates = self.find_similar(content, project_id, duplicate_threshold, limit=3)

        # Generate embedding if available
        embedding = self.embedder.embed(content)

        self.db.execute(
            """
            INSERT INTO memories (id, content, memory_type, tags, source, client_id, project_id, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [memory_id, content, memory_type, tags, source, client_id, project_id, now, now, embedding],
        )

        result = {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "source": source,
            "client_id": client_id,
            "project_id": project_id,
            "created_at": now.isoformat(),
            "has_embedding": embedding is not None,
        }

        if duplicates:
            result["similar_memories"] = duplicates
            result["duplicate_warning"] = f"Found {len(duplicates)} similar memories"

        return result

    def search(
        self,
        query: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        client_id: str | None = None,
        project_id: str | None = None,
        global_search: bool = False,
        limit: int = 20,
        search_mode: str = "hybrid",  # "keyword", "semantic", "hybrid"
    ) -> list[dict[str, Any]]:
        """
        Search memories with optional filters.

        Args:
            query: Search text
            memory_type: Filter by memory type
            tags: Filter by tags (matches any)
            client_id: Filter by client ID
            project_id: Filter by project ID. If None and global_search=False,
                       no project filter is applied (for backwards compatibility).
            global_search: If True, search across all projects (ignore project_id)
            limit: Max results to return
            search_mode:
                - "keyword": Traditional ILIKE substring matching
                - "semantic": Vector similarity search using embeddings
                - "hybrid": Combines keyword and semantic (default)
        """
        # If no query provided, fall back to keyword mode (just filters)
        if not query:
            search_mode = "keyword"

        # Check if semantic search is possible
        can_semantic = Embedder.is_available() and query is not None
        if search_mode in ("semantic", "hybrid") and not can_semantic:
            logger.debug("Semantic search unavailable, falling back to keyword")
            search_mode = "keyword"

        # Determine effective project filter
        effective_project = None if global_search else project_id

        if search_mode == "keyword":
            return self._keyword_search(query, memory_type, tags, client_id, effective_project, limit)
        elif search_mode == "semantic":
            return self._semantic_search(query, memory_type, tags, client_id, effective_project, limit)
        else:  # hybrid
            return self._hybrid_search(query, memory_type, tags, client_id, effective_project, limit)

    def _keyword_search(
        self,
        query: str | None,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
        project_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Traditional keyword-based search."""
        conditions = []
        params = []

        if query:
            conditions.append("content ILIKE ?")
            params.append(f"%{query}%")

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if tags:
            conditions.append("len(list_intersect(tags, ?)) > 0")
            params.append(tags)

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        result = self.db.execute(
            f"""
            SELECT id, content, memory_type, tags, source, client_id, project_id, created_at, updated_at
            FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "project_id", "created_at", "updated_at"]
        return [dict(zip(columns, row)) for row in result]

    def _semantic_search(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
        project_id: str | None,
        limit: int,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Vector similarity search using embeddings."""
        query_embedding = self.embedder.embed(query)
        if query_embedding is None:
            return self._keyword_search(query, memory_type, tags, client_id, project_id, limit)

        conditions = ["embedding IS NOT NULL"]
        params = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if tags:
            conditions.append("len(list_intersect(tags, ?)) > 0")
            params.append(tags)

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where_clause = " AND ".join(conditions)

        # Add query embedding and limit at the end
        params.extend([query_embedding, min_score, limit])

        result = self.db.execute(
            f"""
            SELECT
                id, content, memory_type, tags, source, client_id, project_id, created_at, updated_at,
                list_cosine_similarity(embedding, ?::FLOAT[]) as score
            FROM memories
            WHERE {where_clause}
            AND list_cosine_similarity(embedding, ?::FLOAT[]) > ?
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_embedding] + params,
        ).fetchall()

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "project_id", "created_at", "updated_at", "score"]
        return [dict(zip(columns, row)) for row in result]

    def _hybrid_search(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
        project_id: str | None,
        limit: int,
        alpha: float = 0.5,  # Weight for semantic vs keyword (0=keyword only, 1=semantic only)
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining keyword and semantic results.
        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get results from both methods
        keyword_results = self._keyword_search(query, memory_type, tags, client_id, project_id, limit * 2)
        semantic_results = self._semantic_search(query, memory_type, tags, client_id, project_id, limit * 2)

        # If semantic failed, just return keyword
        if not semantic_results or not any(r.get("score") for r in semantic_results):
            return keyword_results[:limit]

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        results_by_id: dict[str, dict] = {}

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            mem_id = result["id"]
            results_by_id[mem_id] = result
            scores[mem_id] = scores.get(mem_id, 0) + (1 - alpha) * (1 / (k + rank + 1))

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            mem_id = result["id"]
            results_by_id[mem_id] = result
            scores[mem_id] = scores.get(mem_id, 0) + alpha * (1 / (k + rank + 1))

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        final_results = []
        for mem_id in sorted_ids[:limit]:
            result = results_by_id[mem_id]
            result["hybrid_score"] = scores[mem_id]
            final_results.append(result)

        return final_results

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID."""
        result = self.db.execute(
            "SELECT id, content, memory_type, tags, source, client_id, project_id, created_at, updated_at FROM memories WHERE id = ?",
            [memory_id]
        ).fetchone()

        if not result:
            return None

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "project_id", "created_at", "updated_at"]
        return dict(zip(columns, result))

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        before = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.db.execute("DELETE FROM memories WHERE id = ?", [memory_id])
        after = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return before > after

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Update an existing memory."""
        existing = self.get(memory_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc)
        new_content = content if content is not None else existing["content"]
        new_tags = tags if tags is not None else existing["tags"]

        # Regenerate embedding if content changed
        new_embedding = None
        if content is not None:
            new_embedding = self.embedder.embed(new_content)

        if content is not None:
            self.db.execute(
                """
                UPDATE memories
                SET content = ?, tags = ?, updated_at = ?, embedding = ?
                WHERE id = ?
                """,
                [new_content, new_tags, now, new_embedding, memory_id],
            )
        else:
            self.db.execute(
                """
                UPDATE memories
                SET tags = ?, updated_at = ?
                WHERE id = ?
                """,
                [new_tags, now, memory_id],
            )

        return self.get(memory_id)

    def stats(self, project_id: str | None = None) -> dict[str, Any]:
        """Get storage statistics, optionally filtered by project."""
        project_filter = ""
        params: list = []
        if project_id:
            project_filter = "WHERE project_id = ?"
            params = [project_id]

        total = self.db.execute(
            f"SELECT COUNT(*) FROM memories {project_filter}", params
        ).fetchone()[0]

        with_embeddings = self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL {'AND project_id = ?' if project_id else ''}",
            params
        ).fetchone()[0]

        by_type = self.db.execute(f"""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            {project_filter}
            GROUP BY memory_type
        """, params).fetchall()

        by_client = self.db.execute(f"""
            SELECT COALESCE(client_id, 'unknown') as client, COUNT(*) as count
            FROM memories
            {project_filter}
            GROUP BY client_id
        """, params).fetchall()

        by_project = self.db.execute("""
            SELECT COALESCE(project_id, 'unassigned') as project, COUNT(*) as count
            FROM memories
            GROUP BY project_id
            ORDER BY count DESC
        """).fetchall()

        result = {
            "total_memories": total,
            "memories_with_embeddings": with_embeddings,
            "embeddings_available": Embedder.is_available(),
            "by_type": {row[0]: row[1] for row in by_type},
            "by_client": {row[0]: row[1] for row in by_client},
            "by_project": {row[0]: row[1] for row in by_project},
            "storage_path": str(self.db_path),
        }

        if project_id:
            result["filtered_by_project"] = project_id

        return result

    def backfill_embeddings(self, batch_size: int = 100) -> dict[str, int]:
        """
        Generate embeddings for memories that don't have them.
        Returns count of processed and failed memories.
        """
        if not Embedder.is_available():
            return {"processed": 0, "failed": 0, "error": "embeddings not available"}

        # Get memories without embeddings
        result = self.db.execute("""
            SELECT id, content FROM memories WHERE embedding IS NULL
        """).fetchall()

        if not result:
            return {"processed": 0, "failed": 0}

        processed = 0
        failed = 0

        # Process in batches
        for i in range(0, len(result), batch_size):
            batch = result[i:i + batch_size]
            ids = [row[0] for row in batch]
            contents = [row[1] for row in batch]

            try:
                embeddings = self.embedder.embed_batch(contents)
                if embeddings:
                    for mem_id, embedding in zip(ids, embeddings):
                        self.db.execute(
                            "UPDATE memories SET embedding = ? WHERE id = ?",
                            [embedding, mem_id]
                        )
                    processed += len(batch)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                failed += len(batch)

        return {"processed": processed, "failed": failed}

    def export_parquet(self, path: Path | str | None = None) -> str:
        """Export memories to parquet file (for backup/migration)."""
        if path is None:
            path = self.data_dir / f"memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path = Path(path)

        self.db.execute(f"""
            COPY (SELECT id, content, memory_type, tags, source, client_id, project_id, created_at, updated_at
                  FROM memories)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        return str(path)

    def get_context_summary(
        self,
        project_id: str | None = None,
        max_memories: int = 10,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get a curated context summary for a project.

        Returns prioritized memories suitable for injection at session start.
        Priority order: decisions > preferences > facts > observations

        Args:
            project_id: Project to get context for
            max_memories: Maximum number of memories to include
            days: Only include memories from the last N days

        Returns:
            Dict with summary text and metadata
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Build project filter
        project_filter = "project_id = ?" if project_id else "project_id IS NOT NULL"
        params: list = [project_id] if project_id else []

        # Query with priority ordering
        # Decisions and preferences are most important, then facts, then observations
        result = self.db.execute(
            f"""
            SELECT
                id, content, memory_type, tags, created_at,
                CASE memory_type
                    WHEN 'decision' THEN 1
                    WHEN 'preference' THEN 2
                    WHEN 'fact' THEN 3
                    WHEN 'entity' THEN 4
                    WHEN 'relation' THEN 5
                    WHEN 'observation' THEN 6
                    ELSE 7
                END as priority
            FROM memories
            WHERE {project_filter}
            AND created_at > ?
            ORDER BY priority ASC, created_at DESC
            LIMIT ?
            """,
            params + [cutoff, max_memories * 2],  # Get extra for grouping
        ).fetchall()

        if not result:
            return {
                "project_id": project_id,
                "has_context": False,
                "summary": "No memories found for this project.",
                "memories": [],
            }

        # Group by type for structured output
        memories_by_type: dict[str, list] = {}
        for row in result[:max_memories]:
            mem_type = row[2]
            if mem_type not in memories_by_type:
                memories_by_type[mem_type] = []
            memories_by_type[mem_type].append({
                "id": row[0],
                "content": row[1],
                "tags": row[3],
                "created_at": row[4],
            })

        # Build summary text
        lines = []

        if project_id:
            lines.append(f"## Context for project: {project_id}\n")

        # Format by priority
        type_labels = {
            "decision": "Decisions",
            "preference": "Preferences",
            "fact": "Key Facts",
            "entity": "Entities",
            "relation": "Relationships",
            "observation": "Observations",
        }

        for mem_type in ["decision", "preference", "fact", "entity", "relation", "observation"]:
            if mem_type in memories_by_type:
                lines.append(f"### {type_labels.get(mem_type, mem_type.title())}")
                for mem in memories_by_type[mem_type][:3]:  # Max 3 per type
                    lines.append(f"- {mem['content']}")
                lines.append("")

        # Get last activity info
        last_activity = self.db.execute(
            f"""
            SELECT created_at, content, memory_type
            FROM memories
            WHERE {project_filter}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [project_id] if project_id else [],
        ).fetchone()

        last_activity_info = None
        if last_activity:
            last_activity_info = {
                "timestamp": last_activity[0],
                "last_memory": last_activity[1],
                "type": last_activity[2],
            }

        # Get last completed session for "continue where we left off"
        last_session = self.get_last_session(project_id)
        if last_session and last_session.get("summary"):
            lines.append("### Last Session")
            lines.append(f"_{last_session['ended_at']}_")
            lines.append(f"{last_session['summary']}")
            if last_session.get("open_questions"):
                lines.append("\n**Open questions:**")
                for q in last_session["open_questions"][:3]:
                    lines.append(f"- {q}")
            lines.append("")

        return {
            "project_id": project_id,
            "has_context": True,
            "memory_count": len(result),
            "summary": "\n".join(lines),
            "memories_by_type": memories_by_type,
            "last_activity": last_activity_info,
            "last_session": last_session,
        }

    # --- Session Management ---

    def start_session(
        self,
        project_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Start a new conversation session.

        Args:
            project_id: Project this session belongs to
            client_id: Client starting the session

        Returns:
            Session info dict
        """
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)

        self.db.execute(
            """
            INSERT INTO sessions (id, project_id, client_id, started_at, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            [session_id, project_id, client_id, now],
        )

        return {
            "id": session_id,
            "project_id": project_id,
            "client_id": client_id,
            "started_at": now.isoformat(),
            "status": "active",
        }

    def end_session(
        self,
        session_id: str,
        summary: str | None = None,
        topics: list[str] | None = None,
        decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        End a session with optional summary.

        Args:
            session_id: ID of the session to end
            summary: Free-form summary of the session
            topics: List of topics discussed
            decisions: List of decisions made
            open_questions: List of unresolved questions/TODOs

        Returns:
            Updated session info or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        now = datetime.now(timezone.utc)

        self.db.execute(
            """
            UPDATE sessions
            SET ended_at = ?, summary = ?, topics = ?, decisions = ?,
                open_questions = ?, status = 'completed'
            WHERE id = ?
            """,
            [now, summary, topics or [], decisions or [], open_questions or [], session_id],
        )

        # Also create a session-type memory for easy searching
        if summary:
            self.add(
                content=summary,
                memory_type="session",
                tags=["session:" + session_id] + (topics or []),
                project_id=session.get("project_id"),
                client_id=session.get("client_id"),
                check_duplicates=False,
            )

        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a specific session by ID."""
        result = self.db.execute(
            """
            SELECT id, project_id, client_id, started_at, ended_at,
                   summary, topics, decisions, open_questions, status
            FROM sessions WHERE id = ?
            """,
            [session_id],
        ).fetchone()

        if not result:
            return None

        columns = ["id", "project_id", "client_id", "started_at", "ended_at",
                   "summary", "topics", "decisions", "open_questions", "status"]
        return dict(zip(columns, result))

    def get_active_session(self, project_id: str | None = None) -> dict[str, Any] | None:
        """Get the current active session for a project."""
        if project_id:
            result = self.db.execute(
                """
                SELECT id, project_id, client_id, started_at, ended_at,
                       summary, topics, decisions, open_questions, status
                FROM sessions
                WHERE project_id = ? AND status = 'active'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                [project_id],
            ).fetchone()
        else:
            result = self.db.execute(
                """
                SELECT id, project_id, client_id, started_at, ended_at,
                       summary, topics, decisions, open_questions, status
                FROM sessions
                WHERE status = 'active'
                ORDER BY started_at DESC
                LIMIT 1
                """,
            ).fetchone()

        if not result:
            return None

        columns = ["id", "project_id", "client_id", "started_at", "ended_at",
                   "summary", "topics", "decisions", "open_questions", "status"]
        return dict(zip(columns, result))

    def get_last_session(self, project_id: str | None = None) -> dict[str, Any] | None:
        """Get the most recent completed session for a project."""
        if project_id:
            result = self.db.execute(
                """
                SELECT id, project_id, client_id, started_at, ended_at,
                       summary, topics, decisions, open_questions, status
                FROM sessions
                WHERE project_id = ? AND status = 'completed'
                ORDER BY ended_at DESC
                LIMIT 1
                """,
                [project_id],
            ).fetchone()
        else:
            result = self.db.execute(
                """
                SELECT id, project_id, client_id, started_at, ended_at,
                       summary, topics, decisions, open_questions, status
                FROM sessions
                WHERE status = 'completed'
                ORDER BY ended_at DESC
                LIMIT 1
                """,
            ).fetchone()

        if not result:
            return None

        columns = ["id", "project_id", "client_id", "started_at", "ended_at",
                   "summary", "topics", "decisions", "open_questions", "status"]
        return dict(zip(columns, result))

    def list_sessions(
        self,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List sessions with optional filters."""
        conditions = []
        params: list = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        result = self.db.execute(
            f"""
            SELECT id, project_id, client_id, started_at, ended_at,
                   summary, topics, decisions, open_questions, status
            FROM sessions
            WHERE {where_clause}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        columns = ["id", "project_id", "client_id", "started_at", "ended_at",
                   "summary", "topics", "decisions", "open_questions", "status"]
        return [dict(zip(columns, row)) for row in result]

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
