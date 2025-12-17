"""DuckDB persistent storage for memories with semantic search support."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
import logging

import duckdb

logger = logging.getLogger(__name__)

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

        # Check if embedding column exists (for upgrades from older versions)
        columns = self.db.execute("DESCRIBE memories").fetchall()
        column_names = [col[0] for col in columns]
        if "embedding" not in column_names:
            logger.info("Adding embedding column to existing database")
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding FLOAT[]")

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

    def add(
        self,
        content: str,
        memory_type: str = "observation",
        tags: list[str] | None = None,
        source: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a new memory with optional embedding."""
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        tags = tags or []

        # Generate embedding if available
        embedding = self.embedder.embed(content)

        self.db.execute(
            """
            INSERT INTO memories (id, content, memory_type, tags, source, client_id, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [memory_id, content, memory_type, tags, source, client_id, now, now, embedding],
        )

        return {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "source": source,
            "client_id": client_id,
            "created_at": now.isoformat(),
            "has_embedding": embedding is not None,
        }

    def search(
        self,
        query: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        client_id: str | None = None,
        limit: int = 20,
        search_mode: str = "hybrid",  # "keyword", "semantic", "hybrid"
    ) -> list[dict[str, Any]]:
        """
        Search memories with optional filters.

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

        if search_mode == "keyword":
            return self._keyword_search(query, memory_type, tags, client_id, limit)
        elif search_mode == "semantic":
            return self._semantic_search(query, memory_type, tags, client_id, limit)
        else:  # hybrid
            return self._hybrid_search(query, memory_type, tags, client_id, limit)

    def _keyword_search(
        self,
        query: str | None,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
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

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        result = self.db.execute(
            f"""
            SELECT id, content, memory_type, tags, source, client_id, created_at, updated_at
            FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "created_at", "updated_at"]
        return [dict(zip(columns, row)) for row in result]

    def _semantic_search(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
        limit: int,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Vector similarity search using embeddings."""
        query_embedding = self.embedder.embed(query)
        if query_embedding is None:
            return self._keyword_search(query, memory_type, tags, client_id, limit)

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

        where_clause = " AND ".join(conditions)

        # Add query embedding and limit at the end
        params.extend([query_embedding, min_score, limit])

        result = self.db.execute(
            f"""
            SELECT
                id, content, memory_type, tags, source, client_id, created_at, updated_at,
                list_cosine_similarity(embedding, ?::FLOAT[]) as score
            FROM memories
            WHERE {where_clause}
            AND list_cosine_similarity(embedding, ?::FLOAT[]) > ?
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_embedding] + params,
        ).fetchall()

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "created_at", "updated_at", "score"]
        return [dict(zip(columns, row)) for row in result]

    def _hybrid_search(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        client_id: str | None,
        limit: int,
        alpha: float = 0.5,  # Weight for semantic vs keyword (0=keyword only, 1=semantic only)
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining keyword and semantic results.
        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get results from both methods
        keyword_results = self._keyword_search(query, memory_type, tags, client_id, limit * 2)
        semantic_results = self._semantic_search(query, memory_type, tags, client_id, limit * 2)

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
            "SELECT id, content, memory_type, tags, source, client_id, created_at, updated_at FROM memories WHERE id = ?",
            [memory_id]
        ).fetchone()

        if not result:
            return None

        columns = ["id", "content", "memory_type", "tags", "source", "client_id", "created_at", "updated_at"]
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

    def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        with_embeddings = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        by_type = self.db.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """).fetchall()

        by_client = self.db.execute("""
            SELECT COALESCE(client_id, 'unknown') as client, COUNT(*) as count
            FROM memories
            GROUP BY client_id
        """).fetchall()

        return {
            "total_memories": total,
            "memories_with_embeddings": with_embeddings,
            "embeddings_available": Embedder.is_available(),
            "by_type": {row[0]: row[1] for row in by_type},
            "by_client": {row[0]: row[1] for row in by_client},
            "storage_path": str(self.db_path),
        }

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
            COPY (SELECT id, content, memory_type, tags, source, client_id, created_at, updated_at
                  FROM memories)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        return str(path)

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
