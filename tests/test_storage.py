"""Tests for claude-memory storage."""

import tempfile
from pathlib import Path

import pytest

from claude_memory.storage import MemoryStorage


@pytest.fixture
def storage():
    """Create a temporary storage instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield MemoryStorage(tmpdir)


class TestMemoryStorage:
    """Tests for MemoryStorage class."""

    def test_add_memory(self, storage):
        """Test adding a memory."""
        result = storage.add(
            content="Test memory content",
            memory_type="fact",
            tags=["test", "example"],
            client_id="test-client",
        )
        
        assert "id" in result
        assert result["content"] == "Test memory content"
        assert result["memory_type"] == "fact"
        assert result["tags"] == ["test", "example"]
        assert result["client_id"] == "test-client"

    def test_search_by_query_keyword(self, storage):
        """Test searching memories by query with keyword mode."""
        storage.add(content="Python is great", memory_type="fact")
        storage.add(content="JavaScript is also good", memory_type="fact")

        # Keyword mode does exact substring matching
        results = storage.search(query="Python", search_mode="keyword")

        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_search_semantic(self, storage):
        """Test semantic search finds conceptually related content."""
        storage.add(content="I love programming in Python", memory_type="fact")
        storage.add(content="The weather is nice today", memory_type="fact")

        # Semantic search should rank programming-related content higher
        # when searching for coding-related terms
        results = storage.search(query="software development", search_mode="semantic")

        assert len(results) >= 1
        # The programming memory should be scored higher than weather
        if len(results) > 1 and "score" in results[0]:
            programming_result = next((r for r in results if "Python" in r["content"]), None)
            weather_result = next((r for r in results if "weather" in r["content"]), None)
            if programming_result and weather_result:
                assert programming_result.get("score", 0) > weather_result.get("score", 0)

    def test_search_hybrid(self, storage):
        """Test hybrid search combines keyword and semantic."""
        storage.add(content="Python programming language", memory_type="fact")
        storage.add(content="Software engineering best practices", memory_type="fact")

        # Hybrid search should return results from both methods
        results = storage.search(query="Python", search_mode="hybrid")

        assert len(results) >= 1
        # Should have hybrid_score when in hybrid mode
        assert "hybrid_score" in results[0]

    def test_search_by_type(self, storage):
        """Test searching memories by type."""
        storage.add(content="A fact", memory_type="fact")
        storage.add(content="A decision", memory_type="decision")
        
        results = storage.search(memory_type="decision")
        
        assert len(results) == 1
        assert results[0]["memory_type"] == "decision"

    def test_search_by_tags(self, storage):
        """Test searching memories by tags."""
        storage.add(content="Project A stuff", tags=["project:a"])
        storage.add(content="Project B stuff", tags=["project:b"])
        
        results = storage.search(tags=["project:a"])
        
        assert len(results) == 1
        assert "Project A" in results[0]["content"]

    def test_search_by_client_id(self, storage):
        """Test searching memories by client_id."""
        storage.add(content="From laptop", client_id="laptop")
        storage.add(content="From desktop", client_id="desktop")
        
        results = storage.search(client_id="laptop")
        
        assert len(results) == 1
        assert results[0]["client_id"] == "laptop"

    def test_get_memory(self, storage):
        """Test retrieving a specific memory."""
        added = storage.add(content="Test content")
        
        result = storage.get(added["id"])
        
        assert result is not None
        assert result["content"] == "Test content"

    def test_get_nonexistent_memory(self, storage):
        """Test retrieving a memory that doesn't exist."""
        result = storage.get("nonexistent-id")
        
        assert result is None

    def test_delete_memory(self, storage):
        """Test deleting a memory."""
        added = storage.add(content="To be deleted")
        
        success = storage.delete(added["id"])
        
        assert success is True
        assert storage.get(added["id"]) is None

    def test_delete_nonexistent_memory(self, storage):
        """Test deleting a memory that doesn't exist."""
        success = storage.delete("nonexistent-id")
        
        assert success is False

    def test_update_memory(self, storage):
        """Test updating a memory."""
        added = storage.add(content="Original content", tags=["old"])
        
        result = storage.update(
            added["id"],
            content="Updated content",
            tags=["new"],
        )
        
        assert result is not None
        assert result["content"] == "Updated content"
        assert result["tags"] == ["new"]

    def test_update_nonexistent_memory(self, storage):
        """Test updating a memory that doesn't exist."""
        result = storage.update("nonexistent-id", content="New content")
        
        assert result is None

    def test_stats(self, storage):
        """Test getting storage statistics."""
        storage.add(content="Fact 1", memory_type="fact", client_id="laptop")
        storage.add(content="Fact 2", memory_type="fact", client_id="desktop")
        storage.add(content="Decision 1", memory_type="decision", client_id="laptop")
        
        stats = storage.stats()
        
        assert stats["total_memories"] == 3
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["decision"] == 1
        assert stats["by_client"]["laptop"] == 2
        assert stats["by_client"]["desktop"] == 1

    def test_persistence(self):
        """Test that memories persist to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and add memory
            storage1 = MemoryStorage(tmpdir)
            storage1.add(content="Persistent memory")
            storage1.close()

            # Create new instance pointing to same directory
            storage2 = MemoryStorage(tmpdir)
            results = storage2.search(query="Persistent", search_mode="keyword")

            assert len(results) == 1
            assert results[0]["content"] == "Persistent memory"
            storage2.close()

    def test_backfill_embeddings(self, storage):
        """Test backfilling embeddings for existing memories."""
        # Add memories (they'll get embeddings automatically if available)
        storage.add(content="Test memory one")
        storage.add(content="Test memory two")

        stats = storage.stats()
        # If embeddings are available, they should be generated
        from claude_memory.storage import Embedder
        if Embedder.is_available():
            assert stats["memories_with_embeddings"] == 2
        else:
            assert stats["memories_with_embeddings"] == 0

    def test_add_memory_with_project(self, storage):
        """Test adding a memory with project_id."""
        result = storage.add(
            content="Project-specific memory",
            memory_type="fact",
            project_id="my-project",
        )

        assert result["project_id"] == "my-project"

        # Verify it's stored correctly
        retrieved = storage.get(result["id"])
        assert retrieved["project_id"] == "my-project"

    def test_search_by_project(self, storage):
        """Test searching memories filtered by project."""
        storage.add(content="Project A memory", project_id="project-a")
        storage.add(content="Project B memory", project_id="project-b")
        storage.add(content="No project memory")

        # Search within project A
        results = storage.search(project_id="project-a", search_mode="keyword")
        assert len(results) == 1
        assert results[0]["project_id"] == "project-a"

        # Search within project B
        results = storage.search(project_id="project-b", search_mode="keyword")
        assert len(results) == 1
        assert results[0]["project_id"] == "project-b"

    def test_global_search_across_projects(self, storage):
        """Test global search ignores project filter."""
        storage.add(content="Project A memory", project_id="project-a")
        storage.add(content="Project B memory", project_id="project-b")
        storage.add(content="No project memory")

        # Global search should return all
        results = storage.search(global_search=True, search_mode="keyword")
        assert len(results) == 3

        # Global search with query
        results = storage.search(query="memory", global_search=True, search_mode="keyword")
        assert len(results) == 3

    def test_stats_by_project(self, storage):
        """Test statistics filtered by project."""
        storage.add(content="A fact", memory_type="fact", project_id="project-a")
        storage.add(content="A decision", memory_type="decision", project_id="project-a")
        storage.add(content="B fact", memory_type="fact", project_id="project-b")

        # Stats for project A
        stats_a = storage.stats(project_id="project-a")
        assert stats_a["total_memories"] == 2
        assert stats_a["by_type"]["fact"] == 1
        assert stats_a["by_type"]["decision"] == 1

        # Global stats
        stats_all = storage.stats()
        assert stats_all["total_memories"] == 3
        assert "by_project" in stats_all
        assert stats_all["by_project"]["project-a"] == 2
        assert stats_all["by_project"]["project-b"] == 1

    def test_context_summary(self, storage):
        """Test getting curated context summary."""
        # Add various types of memories
        storage.add(content="We decided to use PostgreSQL", memory_type="decision", project_id="my-project")
        storage.add(content="User prefers dark mode", memory_type="preference", project_id="my-project")
        storage.add(content="API runs on port 8080", memory_type="fact", project_id="my-project")
        storage.add(content="Noticed high latency", memory_type="observation", project_id="my-project")
        storage.add(content="Other project fact", memory_type="fact", project_id="other-project")

        # Get context for my-project
        result = storage.get_context_summary(project_id="my-project")

        assert result["has_context"] is True
        assert result["project_id"] == "my-project"
        assert result["memory_count"] >= 4
        assert "summary" in result
        assert "PostgreSQL" in result["summary"]
        assert "dark mode" in result["summary"]
        assert "Other project" not in result["summary"]  # Should not include other project

        # Verify priority ordering in summary (decisions should come before observations)
        summary = result["summary"]
        decision_pos = summary.find("PostgreSQL")
        observation_pos = summary.find("latency")
        assert decision_pos < observation_pos  # Decisions should appear before observations

    def test_context_summary_empty(self, storage):
        """Test context summary when no memories exist."""
        result = storage.get_context_summary(project_id="nonexistent")

        assert result["has_context"] is False
        assert "No memories found" in result["summary"]

    def test_find_similar(self, storage):
        """Test finding similar memories."""
        # Add some memories
        storage.add(content="Python is a great programming language", project_id="test")
        storage.add(content="JavaScript is used for web development", project_id="test")
        storage.add(content="Python is an excellent language for coding", project_id="test")

        # Find similar to Python content
        results = storage.find_similar(
            content="Python programming is awesome",
            project_id="test",
            threshold=0.5,
        )

        # Should find the Python-related memories with higher similarity
        assert len(results) >= 1
        # The most similar should be about Python
        assert "Python" in results[0]["content"]

    def test_duplicate_detection_on_add(self, storage):
        """Test that adding similar memories triggers duplicate warning."""
        # Add initial memory
        storage.add(content="We use PostgreSQL for the database", project_id="test")

        # Add very similar memory
        result = storage.add(
            content="We use PostgreSQL as our database",
            project_id="test",
            check_duplicates=True,
            duplicate_threshold=0.8,
        )

        # Should have found the similar memory
        assert "similar_memories" in result or "duplicate_warning" in result

    def test_skip_duplicate_check(self, storage):
        """Test that duplicate check can be skipped."""
        storage.add(content="Test memory one", project_id="test")

        # Add with duplicate check disabled
        result = storage.add(
            content="Test memory one",  # Exact same content
            project_id="test",
            check_duplicates=False,
        )

        # Should not have duplicate warning
        assert "similar_memories" not in result
        assert "duplicate_warning" not in result
