"""
Data models for the Optimus
Updated for 2025 with enhanced features
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
from enum import Enum
import uuid


def utc_now() -> datetime:
    """Get current UTC datetime with timezone info"""
    return datetime.now(timezone.utc)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    TOGETHER = "together"
    GOOGLE = "google"
    GROQ = "groq"
    MISTRAL = "mistral"


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class MemoryType(str, Enum):
    """Types of memories"""
    CONVERSATION = "conversation"
    NOTE = "note"
    DOCUMENT = "document"
    PREFERENCE = "preference"
    FACT = "fact"
    RELATIONSHIP = "relationship"


class Memory(BaseModel):
    """Enhanced memory model with 2025 features"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType = MemoryType.CONVERSATION
    
    # Enhanced fields for 2025
    entities: List[str] = Field(default_factory=list)  # Extracted entities
    relationships: List[Dict[str, str]] = Field(default_factory=list)  # Entity relationships
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0, le=1)  # Memory confidence score
    
    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # For temporary memories
    
    # Usage tracking
    access_count: int = 0
    relevance_scores: List[float] = Field(default_factory=list)  # Historical relevance
    
    # Source tracking
    source: Optional[str] = None  # Where the memory came from
    source_message_id: Optional[str] = None  # Link to original message


class Conversation(BaseModel):
    """Enhanced conversation model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    messages: List['ChatMessage'] = Field(default_factory=list)
    
    # Enhanced metadata
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_active: datetime = Field(default_factory=utc_now)
    
    # Status
    is_active: bool = True
    is_archived: bool = False
    
    # Metrics
    total_tokens: int = 0
    total_cost: float = 0.0
    model_usage: Dict[str, int] = Field(default_factory=dict)  # Model -> token count


class ChatMessage(BaseModel):
    """Enhanced chat message model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "assistant", "system", "function"]
    content: str
    
    # Enhanced metadata
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None  # Seconds
    
    # Memory tracking
    memories_used: List[str] = Field(default_factory=list)  # Memory IDs
    memories_created: List[str] = Field(default_factory=list)  # New memory IDs
    
    # Function calling (for PydanticAI tools)
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None
    function_result: Optional[Any] = None
    
    # Timestamps
    timestamp: datetime = Field(default_factory=utc_now)
    edited_at: Optional[datetime] = None
    
    # User feedback
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None


class User(BaseModel):
    """User model with enhanced security features"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    
    # Profile
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Security
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    
    # Sessions
    active_sessions: List[str] = Field(default_factory=list)  # JWT tokens
    last_login: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    password_changed_at: datetime = Field(default_factory=utc_now)  # For password expiry
    
    # Usage limits
    daily_token_limit: Optional[int] = None
    monthly_token_limit: Optional[int] = None
    tokens_used_today: int = 0
    tokens_used_month: int = 0


class AuthSession(BaseModel):
    """Authentication session model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    token: str
    username: str
    user_id: str
    role: UserRole
    
    # Session metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    last_activity: datetime = Field(default_factory=utc_now)
    
    # Security
    is_active: bool = True
    refresh_token: Optional[str] = None


class MemorySearchResult(BaseModel):
    """Search result model with relevance information"""
    memory: Memory
    score: float = Field(ge=0, le=1)
    highlights: List[str] = Field(default_factory=list)
    matched_entities: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None  # Why this memory was relevant


class SystemMetrics(BaseModel):
    """System metrics for monitoring"""
    timestamp: datetime = Field(default_factory=utc_now)
    
    # Performance metrics
    active_users: int = 0
    total_memories: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    vector_db_size_mb: float = 0.0
    
    # Business metrics
    queries_per_minute: float = 0.0
    tokens_per_minute: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0


class ExportData(BaseModel):
    """Data export model for GDPR compliance"""
    user: User
    memories: List[Memory]
    conversations: List[Conversation]
    export_timestamp: datetime = Field(default_factory=utc_now)
    format_version: str = "2.0"


# Update forward references
Conversation.model_rebuild()