"""DuckDB/Parquet storage for memories."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import duckdb


class MemoryStorage:
    """Persistent memory storage using DuckDB with Parquet backend."""

    def __init__(self, data_dir: Path | str = "~/.claude-memory"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.data_dir / "memories.parquet"
        self.db = duckdb.connect()
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize or load existing parquet file."""
        if self.parquet_path.exists():
            self.db.execute(f"""
                CREATE TABLE memories AS 
                SELECT * FROM read_parquet('{self.parquet_path}')
            """)
        else:
            self.db.execute("""
                CREATE TABLE memories (
                    id VARCHAR PRIMARY KEY,
                    content VARCHAR NOT NULL,
                    memory_type VARCHAR DEFAULT 'observation',
                    tags VARCHAR[],
                    source VARCHAR,
                    client_id VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
            """)
            self._persist()

    def _persist(self) -> None:
        """Write current state to parquet."""
        self.db.execute(f"""
            COPY memories TO '{self.parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    def add(
        self,
        content: str,
        memory_type: str = "observation",
        tags: list[str] | None = None,
        source: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a new memory."""
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        tags = tags or []

        self.db.execute(
            """
            INSERT INTO memories (id, content, memory_type, tags, source, client_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [memory_id, content, memory_type, tags, source, client_id, now, now],
        )
        self._persist()

        return {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "source": source,
            "client_id": client_id,
            "created_at": now.isoformat(),
        }

    def search(
        self,
        query: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        client_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search memories with optional filters."""
        conditions = []
        params = []

        if query:
            conditions.append("content ILIKE ?")
            params.append(f"%{query}%")

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if tags:
            # Match any of the provided tags
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

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID."""
        result = self.db.execute(
            "SELECT id, content, memory_type, tags, source, client_id, created_at, updated_at FROM memories WHERE id = ?", [memory_id]
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

        if before > after:
            self._persist()
            return True
        return False

    def update(self, memory_id: str, content: str | None = None, tags: list[str] | None = None) -> dict[str, Any] | None:
        """Update an existing memory."""
        existing = self.get(memory_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc)
        new_content = content if content is not None else existing["content"]
        new_tags = tags if tags is not None else existing["tags"]

        self.db.execute(
            """
            UPDATE memories 
            SET content = ?, tags = ?, updated_at = ?
            WHERE id = ?
            """,
            [new_content, new_tags, now, memory_id],
        )
        self._persist()

        return self.get(memory_id)

    def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
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
            "by_type": {row[0]: row[1] for row in by_type},
            "by_client": {row[0]: row[1] for row in by_client},
            "storage_path": str(self.parquet_path),
        }
