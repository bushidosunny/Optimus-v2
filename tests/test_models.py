"""
Test suite for data models
"""

import pytest
from datetime import datetime, timezone
from models import (
    Memory, MemoryType, User, UserRole, AuthSession, 
    ChatMessage, Conversation, utc_now
)


class TestMemoryModel:
    """Test Memory model"""
    
    def test_memory_creation(self):
        """Test creating a memory instance"""
        memory = Memory(
            user_id="test_user",
            content="Test memory content",
            memory_type=MemoryType.FACT,
            importance=0.8
        )
        
        assert memory.user_id == "test_user"
        assert memory.content == "Test memory content"
        assert memory.memory_type == MemoryType.FACT
        assert memory.importance == 0.8
        assert memory.confidence == 1.0  # Default value
        assert memory.access_count == 0
        assert isinstance(memory.created_at, datetime)
        assert memory.created_at.tzinfo is not None  # Should be timezone-aware
    
    def test_memory_with_entities(self):
        """Test memory with entities and relationships"""
        memory = Memory(
            user_id="test_user",
            content="John works at TechCorp",
            entities=["John", "TechCorp"],
            relationships=[{"subject": "John", "relation": "works_at", "object": "TechCorp"}]
        )
        
        assert len(memory.entities) == 2
        assert "John" in memory.entities
        assert len(memory.relationships) == 1
    
    def test_memory_importance_bounds(self):
        """Test importance score bounds"""
        with pytest.raises(ValueError):
            Memory(
                user_id="test",
                content="test",
                importance=1.5  # Should fail - max is 1.0
            )
        
        with pytest.raises(ValueError):
            Memory(
                user_id="test",
                content="test",
                importance=-0.1  # Should fail - min is 0.0
            )


class TestUserModel:
    """Test User model"""
    
    def test_user_creation(self):
        """Test creating a user"""
        user = User(
            username="alice",
            email="alice@example.com",
            password_hash="$2b$12$hashedpassword",
            role=UserRole.ADMIN
        )
        
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.role == UserRole.ADMIN
        assert user.is_active is True
        assert user.failed_login_attempts == 0
    
    def test_user_role_enum(self):
        """Test user role enumeration"""
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"
        # Note: MODERATOR might not exist in the enum


class TestAuthSession:
    """Test AuthSession model"""
    
    def test_session_creation(self):
        """Test creating an auth session"""
        expires_at = utc_now()
        session = AuthSession(
            token="test_token_123",
            username="alice",
            user_id="user_123",
            role=UserRole.USER,
            expires_at=expires_at
        )
        
        assert session.token == "test_token_123"
        assert session.username == "alice"
        assert session.is_active is True
        assert session.expires_at == expires_at
        assert session.expires_at.tzinfo is not None


class TestChatMessage:
    """Test ChatMessage model"""
    
    def test_message_creation(self):
        """Test creating a chat message"""
        message = ChatMessage(
            content="Hello, world!",
            role="user"
        )
        
        assert message.content == "Hello, world!"
        assert message.role == "user"
        assert hasattr(message, 'id')
    
    def test_message_roles(self):
        """Test different message roles"""
        user_msg = ChatMessage(content="Question?", role="user")
        assistant_msg = ChatMessage(content="Answer.", role="assistant")
        system_msg = ChatMessage(content="System prompt", role="system")
        
        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"
        assert system_msg.role == "system"


class TestConversation:
    """Test Conversation model"""
    
    def test_conversation_creation(self):
        """Test creating a conversation"""
        messages = [
            ChatMessage(content="Hello", role="user"),
            ChatMessage(content="Hi there!", role="assistant")
        ]
        
        conversation = Conversation(
            user_id="user_123",
            title="Test Chat",
            messages=messages
        )
        
        assert conversation.user_id == "user_123"
        assert conversation.title == "Test Chat"
        assert len(conversation.messages) == 2
        assert conversation.is_active is True


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_utc_now(self):
        """Test utc_now returns timezone-aware datetime"""
        now = utc_now()
        
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])