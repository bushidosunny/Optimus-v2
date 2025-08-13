"""
Test suite for memory manager
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
import streamlit as st
from memory_manager import MemoryManager
from models import Memory, MemoryType, MemorySearchResult


class TestMemoryManager:
    """Test memory management system"""
    
    @pytest.fixture
    def mock_secrets(self):
        """Mock Streamlit secrets"""
        return {
            'QDRANT_URL': 'http://localhost:6333',
            'QDRANT_API_KEY': 'test_key',
            'QDRANT_USE_GRPC': False,
            'OPENAI_API_KEY': 'test_openai_key'
        }
    
    @pytest.fixture
    def memory_manager(self, mock_secrets):
        """Create memory manager with mocked dependencies"""
        with patch.object(st, 'secrets', mock_secrets):
            with patch('memory_manager.QdrantClient'):
                with patch('memory_manager.Memory'):
                    with patch('memory_manager.SentenceTransformer'):
                        with patch.object(MemoryManager, '_setup_monitoring'):
                            manager = MemoryManager()
                            # Mock the internal clients
                            manager.qdrant = MagicMock()
                            manager.mem0 = MagicMock()
                            manager.embedder = MagicMock()
                            # Mock monitoring attributes with all required metrics
                            manager.metrics = {
                                'memory_operations': MagicMock(),
                                'memory_count': MagicMock(),
                                'search_latency': MagicMock(),
                                'embedding_time': MagicMock(),
                                'search_duration': MagicMock()
                            }
                            return manager
    
    @pytest.mark.asyncio
    async def test_add_memory(self, memory_manager):
        """Test adding a new memory"""
        user_id = "test_user"
        content = "This is a test memory"
        
        # Mock Mem0 response
        memory_manager.mem0.add.return_value = {
            'id': 'mem_123',
            'result': 'success'
        }
        
        # Mock extraction methods
        memory_manager._extract_memory_components = AsyncMock(return_value={
            'content': content,
            'entities': ['test'],
            'relationships': [],
            'importance': 0.7,
            'metadata': {}
        })
        
        memory_manager._update_or_create_memory = AsyncMock(return_value={
            'id': 'mem_123',
            'content': content,
            'metadata': {'importance': 0.7},
            'entities': ['test'],
            'relationships': [],
            'importance': 0.7
        })
        
        memory = await memory_manager.add_memory(
            content=content,
            user_id=user_id,
            memory_type=MemoryType.FACT
        )
        
        assert memory is not None
        assert memory.content == content
        assert memory.user_id == user_id
        assert memory.memory_type == MemoryType.FACT
        assert memory.importance == 0.7
    
    @pytest.mark.asyncio
    async def test_search_memories(self, memory_manager):
        """Test searching memories"""
        query = "test query"
        user_id = "test_user"
        
        # Mock Mem0 search response
        memory_manager.mem0.search.return_value = [
            {
                'id': 'mem_1',
                'text': 'Memory about testing',
                'metadata': {
                    'importance': 0.8,
                    'entities': ['testing'],
                    'memory_type': 'fact'
                },
                'score': 0.9
            }
        ]
        
        results = await memory_manager.search_memories(query, user_id)
        
        assert len(results) == 1
        assert results[0].memory.content == 'Memory about testing'
        assert results[0].score == 0.9
        assert results[0].memory.importance == 0.8
    
    @pytest.mark.asyncio
    async def test_get_all_memories(self, memory_manager):
        """Test getting all memories for a user"""
        user_id = "test_user"
        
        # Mock Mem0 get_all response
        memory_manager.mem0.get_all.return_value = [
            {
                'id': 'mem_1',
                'text': 'First memory',
                'metadata': {'importance': 0.5}
            },
            {
                'id': 'mem_2',
                'text': 'Second memory',
                'metadata': {'importance': 0.7}
            }
        ]
        
        memories = await memory_manager.get_all_memories(user_id)
        
        assert len(memories) == 2
        assert memories[0].content == 'First memory'
        assert memories[1].content == 'Second memory'
    
    @pytest.mark.asyncio
    async def test_update_memory(self, memory_manager):
        """Test updating an existing memory"""
        memory_id = "mem_123"
        user_id = "test_user"
        new_content = "Updated content"
        
        # Mock Mem0 update
        memory_manager.mem0.update.return_value = {'result': 'success'}
        
        # Mock get to return existing memory
        memory_manager.mem0.get.return_value = {
            'id': memory_id,
            'text': new_content,
            'metadata': {'importance': 0.9}
        }
        
        updated = await memory_manager.update_memory(
            memory_id=memory_id,
            content=new_content,
            user_id=user_id
        )
        
        assert updated is not None
        assert updated.content == new_content
        assert updated.id == memory_id
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager):
        """Test deleting a memory"""
        memory_id = "mem_123"
        user_id = "test_user"
        
        # Mock Mem0 delete
        memory_manager.mem0.delete.return_value = None
        
        result = await memory_manager.delete_memory(memory_id, user_id)
        
        assert result is True
        memory_manager.mem0.delete.assert_called_once_with(memory_id, user_id=user_id)
    
    def test_calculate_importance(self, memory_manager):
        """Test importance calculation"""
        # Test with important keywords
        important_content = "This is important to remember"
        importance = memory_manager._calculate_importance(important_content, None)
        assert importance > 0.5
        
        # Test with metadata importance
        normal_content = "Regular content"
        high_importance_meta = {'importance': 'high'}
        importance = memory_manager._calculate_importance(normal_content, high_importance_meta)
        assert importance > 0.7
        
        # Test low importance
        low_importance_meta = {'importance': 'low'}
        importance = memory_manager._calculate_importance(normal_content, low_importance_meta)
        assert importance < 0.5
    
    def test_extract_entities(self, memory_manager):
        """Test entity extraction"""
        content = "John Smith works at Microsoft in Seattle"
        entities = memory_manager._extract_entities(content)
        
        assert "John" in entities
        assert "Smith" in entities
        assert "Microsoft" in entities
        assert "Seattle" in entities
    
    def test_extract_highlights(self, memory_manager):
        """Test highlight extraction"""
        query = "Python programming"
        content = "I love Python programming. Python is great for data science and web development."
        
        highlights = memory_manager._extract_highlights(query, content)
        
        assert len(highlights) > 0
        assert any("Python" in h for h in highlights)
    
    @pytest.mark.asyncio
    async def test_consolidate_memories(self, memory_manager):
        """Test memory consolidation"""
        user_id = "test_user"
        
        # Create test memories
        memories = [
            Memory(
                id="1",
                user_id=user_id,
                content="Python is a programming language",
                importance=0.6,
                created_at=datetime.now(timezone.utc)
            ),
            Memory(
                id="2",
                user_id=user_id,
                content="Python is a programming language used for AI",  # Similar to first
                importance=0.7,
                created_at=datetime.now(timezone.utc)
            )
        ]
        
        # Mock get_all_memories
        memory_manager.get_all_memories = AsyncMock(return_value=memories)
        
        # Mock embedder
        memory_manager.embedder.encode = MagicMock(return_value=[
            [0.1, 0.2, 0.3],  # Mock embeddings
            [0.1, 0.2, 0.35]  # Very similar
        ])
        
        # Mock delete
        memory_manager.delete_memory = AsyncMock(return_value=True)
        
        stats = await memory_manager.consolidate_memories(user_id)
        
        assert 'removed' in stats
        assert 'merged' in stats
    
    def test_get_statistics(self, memory_manager):
        """Test getting memory statistics"""
        user_id = "test_user"
        
        # Mock Mem0 get_all
        memory_manager.mem0.get_all.return_value = [
            {
                'id': '1',
                'text': 'Memory 1',
                'metadata': {
                    'memory_type': 'fact',
                    'importance': 0.8,
                    'entities': ['Python', 'AI'],
                    'created_at': '2025-01-01T00:00:00'
                }
            },
            {
                'id': '2',
                'text': 'Memory 2',
                'metadata': {
                    'memory_type': 'preference',
                    'importance': 0.6,
                    'entities': ['coffee'],
                    'created_at': '2025-01-02T00:00:00'
                }
            }
        ]
        
        stats = memory_manager.get_statistics(user_id)
        
        assert stats['total_memories'] == 2
        assert 'fact' in stats['memory_types']
        assert 'preference' in stats['memory_types']
        assert stats['memory_types']['fact'] == 1
        assert stats['memory_types']['preference'] == 1
        assert stats['avg_importance'] == 0.7
        assert stats['total_entities'] == 3
    
    @pytest.mark.asyncio
    async def test_export_memories(self, memory_manager):
        """Test memory export"""
        user_id = "test_user"
        
        # Mock memories
        test_memories = [
            Memory(
                id="1",
                user_id=user_id,
                content="Test memory 1",
                importance=0.5
            ),
            Memory(
                id="2",
                user_id=user_id,
                content="Test memory 2",
                importance=0.7
            )
        ]
        
        memory_manager.get_all_memories = AsyncMock(return_value=test_memories)
        
        export_data = await memory_manager.export_memories(user_id, format='json')
        
        assert export_data is not None
        assert 'user_id' in export_data
        assert 'memory_count' in export_data
        assert 'memories' in export_data
        assert export_data['memory_count'] == 2
    
    @pytest.mark.asyncio
    async def test_delete_all_memories(self, memory_manager):
        """Test deleting all memories for a user"""
        user_id = "test_user"
        
        # Mock memories
        test_memories = [
            Memory(id="1", user_id=user_id, content="Memory 1"),
            Memory(id="2", user_id=user_id, content="Memory 2")
        ]
        
        memory_manager.get_all_memories = AsyncMock(return_value=test_memories)
        memory_manager.delete_memory = AsyncMock(return_value=True)
        
        count = await memory_manager.delete_all_memories(user_id)
        
        assert count == 2
        assert memory_manager.delete_memory.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])