"""
Utility functions and helpers for the Optimus
Includes monitoring, validation, and common operations
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Union
import hashlib
import secrets
import re
from datetime import datetime, timedelta
import json
import logging
from functools import wraps
import time
import asyncio
from prometheus_client import Counter, Histogram, Gauge, Summary
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Metrics collectors
REQUEST_COUNT = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency', ['method', 'endpoint'])
ACTIVE_USERS = Gauge('app_active_users', 'Number of active users')
MEMORY_OPERATIONS = Counter('memory_operations_total', 'Memory operations', ['operation', 'status'])
LLM_ERRORS = Counter('llm_errors_total', 'LLM errors', ['provider', 'error_type'])


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper


def async_timing_decorator(func):
    """Decorator to measure async function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper


class SecurityUtils:
    """Security-related utility functions"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_data(data: str, salt: Optional[str] = None) -> str:
        """Hash data with optional salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return f"{salt}${hash_obj.hex()}"
    
    @staticmethod
    def verify_hash(data: str, hashed: str) -> bool:
        """Verify data against hash"""
        try:
            salt, hash_value = hashed.split('$')
            test_hash = SecurityUtils.hash_data(data, salt)
            return test_hash == hashed
        except:
            return False
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Limit length
        text = text[:max_length]
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))


class ValidationUtils:
    """Input validation utilities"""
    
    @staticmethod
    def validate_memory_content(content: str) -> tuple[bool, str]:
        """Validate memory content"""
        if not content or not content.strip():
            return False, "Content cannot be empty"
        
        if len(content) > 10000:
            return False, "Content too long (max 10,000 characters)"
        
        if len(content.strip()) < 10:
            return False, "Content too short (min 10 characters)"
        
        return True, "Valid"
    
    @staticmethod
    def validate_search_query(query: str) -> tuple[bool, str]:
        """Validate search query"""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 500:
            return False, "Query too long (max 500 characters)"
        
        # Check for injection attempts
        dangerous_patterns = [
            r'<script', r'javascript:', r'onload=', r'onerror=',
            r'DROP TABLE', r'DELETE FROM', r'UPDATE SET'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Invalid characters in query"
        
        return True, "Valid"


class MemoryUtils:
    """Memory-related utility functions"""
    
    @staticmethod
    def calculate_memory_score(
        importance: float,
        access_count: int,
        age_days: int,
        entity_count: int
    ) -> float:
        """Calculate overall memory quality score"""
        # Importance weight: 40%
        importance_score = importance * 0.4
        
        # Access frequency weight: 30%
        access_score = min(access_count / 10, 1.0) * 0.3
        
        # Recency weight: 20%
        recency_score = max(0, 1 - (age_days / 365)) * 0.2
        
        # Entity richness weight: 10%
        entity_score = min(entity_count / 5, 1.0) * 0.1
        
        return importance_score + access_score + recency_score + entity_score
    
    @staticmethod
    def format_memory_summary(content: str, max_length: int = 100) -> str:
        """Format memory content for display"""
        if len(content) <= max_length:
            return content
        
        # Try to break at sentence boundary
        sentences = re.split(r'[.!?]+', content[:max_length + 50])
        if len(sentences) > 1:
            summary = sentences[0] + '.'
            if len(summary) <= max_length:
                return summary
        
        # Fallback to word boundary
        words = content[:max_length].rsplit(' ', 1)
        if len(words) > 1:
            return words[0] + '...'
        
        return content[:max_length] + '...'
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text (simple version)"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and count
        word_counts = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in keywords[:max_keywords]]


class FormatUtils:
    """Formatting utilities"""
    
    @staticmethod
    def format_timestamp(dt: datetime, format: str = "relative") -> str:
        """Format timestamp for display"""
        if format == "relative":
            delta = datetime.now() - dt
            
            if delta.days > 365:
                return f"{delta.days // 365} year{'s' if delta.days // 365 > 1 else ''} ago"
            elif delta.days > 30:
                return f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
            elif delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
            elif delta.seconds > 60:
                return f"{delta.seconds // 60} minute{'s' if delta.seconds // 60 > 1 else ''} ago"
            else:
                return "just now"
        
        elif format == "short":
            return dt.strftime("%b %d, %H:%M")
        
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def format_number(num: Union[int, float], precision: int = 2) -> str:
        """Format number with appropriate units"""
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.{precision}f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.{precision}f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.{precision}f}K"
        else:
            return str(int(num) if isinstance(num, float) and num.is_integer() else round(num, precision))
    
    @staticmethod
    def format_bytes(bytes: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"


class CacheUtils:
    """Caching utilities"""
    
    @staticmethod
    def get_cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    @st.cache_data(ttl=300)
    def cached_search(query: str, user_id: str, limit: int) -> List[Dict]:
        """Cached memory search (5 min TTL)"""
        # This would be implemented with actual search
        return []
    
    @staticmethod
    def clear_user_cache(user_id: str):
        """Clear cache for specific user"""
        # Clear relevant cached data
        st.cache_data.clear()
        logger.info(f"Cleared cache for user {user_id}")


class MonitoringUtils:
    """Monitoring and metrics utilities"""
    
    @staticmethod
    def track_request(method: str, endpoint: str, duration: float, status: str = "success"):
        """Track API request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        
        if status != "success":
            logger.warning(f"Request failed: {method} {endpoint} - {status}")
    
    @staticmethod
    def track_memory_operation(operation: str, status: str = "success"):
        """Track memory operation metrics"""
        MEMORY_OPERATIONS.labels(operation=operation, status=status).inc()
    
    @staticmethod
    def track_llm_error(provider: str, error_type: str):
        """Track LLM errors"""
        LLM_ERRORS.labels(provider=provider, error_type=error_type).inc()
        logger.error(f"LLM error: {provider} - {error_type}")
    
    @staticmethod
    def update_active_users(count: int):
        """Update active users gauge"""
        ACTIVE_USERS.set(count)
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get current system metrics"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "timestamp": datetime.now().isoformat()
        }


class ExportUtils:
    """Data export utilities"""
    
    @staticmethod
    def export_to_json(data: Any, filename: str) -> bytes:
        """Export data to JSON format"""
        return json.dumps(data, indent=2, default=str).encode('utf-8')
    
    @staticmethod
    def export_to_csv(data: List[Dict], filename: str) -> bytes:
        """Export data to CSV format"""
        if not data:
            return b""
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def export_to_markdown(memories: List[Dict]) -> str:
        """Export memories to Markdown format"""
        md_content = "# Memory Export\n\n"
        md_content += f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += f"**Total Memories:** {len(memories)}\n\n"
        md_content += "---\n\n"
        
        for i, memory in enumerate(memories, 1):
            md_content += f"## Memory {i}\n\n"
            md_content += f"**Content:** {memory.get('content', '')}\n\n"
            md_content += f"**Type:** {memory.get('type', 'Unknown')}\n\n"
            md_content += f"**Created:** {memory.get('created_at', '')}\n\n"
            
            if memory.get('entities'):
                md_content += f"**Entities:** {', '.join(memory['entities'])}\n\n"
            
            md_content += "---\n\n"
        
        return md_content


class DebugUtils:
    """Debugging utilities"""
    
    @staticmethod
    def log_session_state():
        """Log current session state (for debugging)"""
        logger.debug("Session State:")
        for key, value in st.session_state.items():
            if key in ['auth_token', 'password']:
                logger.debug(f"  {key}: [REDACTED]")
            else:
                logger.debug(f"  {key}: {type(value).__name__}")
    
    @staticmethod
    def create_test_memories(user_id: str, count: int = 10) -> List[Dict]:
        """Create test memories for development"""
        memories = []
        
        topics = [
            "Python programming tips",
            "Machine learning concepts",
            "Data science best practices",
            "Web development trends",
            "AI ethics discussions"
        ]
        
        for i in range(count):
            memory = {
                "id": SecurityUtils.generate_secure_token(16),
                "user_id": user_id,
                "content": f"Test memory {i+1}: {np.random.choice(topics)}",
                "type": np.random.choice(["conversation", "note", "fact"]),
                "importance": np.random.uniform(0.3, 1.0),
                "created_at": datetime.now() - timedelta(days=np.random.randint(0, 30)),
                "entities": np.random.choice(["Python", "AI", "ML", "Data"], size=np.random.randint(1, 4), replace=False).tolist()
            }
            memories.append(memory)
        
        return memories
    
    @staticmethod
    def benchmark_operation(operation_name: str, func, *args, **kwargs):
        """Benchmark an operation"""
        import timeit
        
        def wrapper():
            return func(*args, **kwargs)
        
        times = []
        for _ in range(5):
            start = timeit.default_timer()
            wrapper()
            end = timeit.default_timer()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        logger.info(f"Benchmark {operation_name}: {avg_time:.4f}s average ({len(times)} runs)")
        
        return {
            "operation": operation_name,
            "average_time": avg_time,
            "min_time": min(times),
            "max_time": max(times),
            "runs": len(times)
        }


# Streamlit-specific utilities
def display_metric_card(title: str, value: Any, delta: Optional[Any] = None, color: str = "#FF6B6B"):
    """Display a custom metric card"""
    st.markdown(
        f"""
        <div style="
            background-color: #1E222A;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid {color};
            margin-bottom: 10px;
        ">
            <h4 style="margin: 0; color: #FAFAFA;">{title}</h4>
            <h2 style="margin: 10px 0; color: {color};">{value}</h2>
            {f'<p style="margin: 0; color: #888;">Change: {delta}</p>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def create_download_link(data: bytes, filename: str, mime_type: str = "application/octet-stream") -> str:
    """Create a download link for data"""
    import base64
    
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'


def display_progress_ring(progress: float, label: str = ""):
    """Display a circular progress indicator"""
    angle = int(progress * 360)
    
    st.markdown(
        f"""
        <div style="position: relative; width: 100px; height: 100px; margin: auto;">
            <svg width="100" height="100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#262730" stroke-width="10"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="#FF6B6B" stroke-width="10"
                        stroke-dasharray="{angle * 0.785} 1000"
                        transform="rotate(-90 50 50)"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        text-align: center; color: #FAFAFA;">
                <div style="font-size: 24px; font-weight: bold;">{int(progress * 100)}%</div>
                {f'<div style="font-size: 12px;">{label}</div>' if label else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Initialize logging
def setup_logging(log_level: str = "INFO"):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='a')
        ]
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)