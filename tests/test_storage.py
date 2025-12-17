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

    def test_search_by_query(self, storage):
        """Test searching memories by query."""
        storage.add(content="Python is great", memory_type="fact")
        storage.add(content="JavaScript is also good", memory_type="fact")
        
        results = storage.search(query="Python")
        
        assert len(results) == 1
        assert "Python" in results[0]["content"]

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
            
            # Create new instance pointing to same directory
            storage2 = MemoryStorage(tmpdir)
            results = storage2.search(query="Persistent")
            
            assert len(results) == 1
            assert results[0]["content"] == "Persistent memory"
