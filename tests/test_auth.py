"""
Test suite for authentication system
"""

import pytest
import streamlit as st
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import jwt
from auth import EnhancedAuth
from models import UserRole, User


class TestEnhancedAuth:
    """Test authentication system"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create auth manager with test configuration"""
        # Mock Streamlit secrets
        with patch.object(st, 'secrets', {
            'auth': {
                'type': 'basic',
                'session_expiry_hours': 24,
                'max_login_attempts': 5,
                'require_captcha_after_failures': 3,
                'users': [
                    {
                        'username': 'test_user',
                        'email': 'test@example.com',
                        'password_hash': '$2b$12$YXNkZmFzZGZhc2RmYXNkZu/Ou5U2Xbz3fJT3lO3kE6T9H5KxF7iVm',  # "password123"
                        'role': 'user'
                    }
                ]
            },
            'security': {
                'jwt_secret': 'test-secret-key-at-least-32-characters-long!!',
                'encryption_key': '32-byte-key-for-encryption-here!!'
            }
        }):
            # Mock session state
            if not hasattr(st, 'session_state'):
                st.session_state = MagicMock()
            st.session_state.failed_attempts = {}
            st.session_state.active_sessions = {}
            
            return EnhancedAuth()
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = auth_manager.hash_password(password)
        
        assert hashed != password
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrong_password", hashed)
    
    def test_password_strength_validation(self, auth_manager):
        """Test password strength requirements"""
        # Weak passwords
        weak_passwords = [
            "short",  # Too short
            "alllowercase123!",  # No uppercase
            "ALLUPPERCASE123!",  # No lowercase
            "NoNumbers!",  # No numbers
            "NoSpecialChars123",  # No special characters
        ]
        
        for password in weak_passwords:
            valid, message = auth_manager.validate_password_strength(password)
            assert not valid
            assert message != "Password is strong"
        
        # Strong password
        strong_password = "StrongPass123!"
        valid, message = auth_manager.validate_password_strength(strong_password)
        assert valid
        assert message == "Password is strong"
    
    def test_jwt_token_generation(self, auth_manager):
        """Test JWT token generation and validation"""
        username = "test_user"
        role = "admin"
        user_id = "user_123"
        
        token = auth_manager.generate_jwt_token(username, role, user_id)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify the token was stored in session
        assert token in st.session_state.active_sessions
        session = st.session_state.active_sessions[token]
        assert session.username == username
        assert session.role == UserRole.ADMIN
    
    def test_jwt_token_verification(self, auth_manager):
        """Test JWT token verification"""
        username = "test_user"
        token = auth_manager.generate_jwt_token(username, "user", "123")
        
        # Verify valid token
        payload = auth_manager.verify_jwt_token(token)
        assert payload is not None
        assert payload['username'] == username
        assert payload['role'] == "user"
        
        # Test invalid token
        invalid_payload = auth_manager.verify_jwt_token("invalid_token")
        assert invalid_payload is None
    
    def test_jwt_token_expiration(self, auth_manager):
        """Test JWT token expiration"""
        # Create an expired token
        past_time = datetime.now(timezone.utc) - timedelta(hours=25)
        expired_payload = {
            "username": "test",
            "user_id": "123",
            "role": "user",
            "exp": int(past_time.timestamp()),
            "iat": int(past_time.timestamp())
        }
        
        expired_token = jwt.encode(
            expired_payload,
            auth_manager.jwt_secret,
            algorithm="HS256"
        )
        
        # Should return None for expired token
        result = auth_manager.verify_jwt_token(expired_token)
        assert result is None
    
    def test_rate_limiting(self, auth_manager):
        """Test login rate limiting"""
        username = "test_user"
        
        # Should not be rate limited initially
        assert not auth_manager.check_rate_limit(username)
        
        # Record multiple failed attempts
        for _ in range(5):
            auth_manager.record_failed_attempt(username)
        
        # Should now be rate limited
        assert auth_manager.check_rate_limit(username)
    
    def test_captcha_requirement(self, auth_manager):
        """Test CAPTCHA requirement after failures"""
        username = "test_user"
        
        # No CAPTCHA needed initially
        assert not auth_manager.needs_captcha(username)
        
        # Record failures up to threshold
        for _ in range(3):
            auth_manager.record_failed_attempt(username)
        
        # Should need CAPTCHA now
        assert auth_manager.needs_captcha(username)
    
    def test_captcha_generation_and_verification(self, auth_manager):
        """Test CAPTCHA generation and verification"""
        # Mock random for predictable CAPTCHA
        import random
        random.seed(42)
        
        question, answer_hash = auth_manager.generate_captcha()
        
        assert question is not None
        assert answer_hash is not None
        
        # The actual answer would be calculated based on the question
        # For testing, we'll verify the hash was stored correctly
        assert st.session_state.captcha_answer == answer_hash
        
        # Since we can't know the actual answer without parsing the question,
        # we'll test that wrong answers fail
        assert not auth_manager.verify_captcha("wrong_answer")
        
        # Test that the verification process works (hash comparison)
        import hashlib
        # If question is "7 + 3 = ?", answer would be "10"
        # We can extract numbers from question for testing
        import re
        numbers = re.findall(r'\d+', question)
        if '+' in question:
            actual_answer = str(int(numbers[0]) + int(numbers[1]))
        elif '-' in question:
            actual_answer = str(int(numbers[0]) - int(numbers[1]))
        else:  # multiplication
            actual_answer = str(int(numbers[0]) * int(numbers[1]))
        
        # Verify with the correct answer
        assert auth_manager.verify_captcha(actual_answer)
    
    def test_session_management(self, auth_manager):
        """Test session creation and management"""
        username = "test_user"
        token = auth_manager.generate_jwt_token(username, "user", "123")
        
        # Get active sessions
        sessions = auth_manager.get_active_sessions(username)
        assert len(sessions) == 1
        assert sessions[0].username == username
        
        # Revoke token
        auth_manager.revoke_token(token)
        session = st.session_state.active_sessions[token]
        assert not session.is_active
    
    def test_revoke_all_sessions(self, auth_manager):
        """Test revoking all user sessions"""
        username = "test_user"
        
        # Create multiple sessions
        tokens = []
        for i in range(3):
            token = auth_manager.generate_jwt_token(username, "user", f"123_{i}")
            tokens.append(token)
        
        # Revoke all sessions
        count = auth_manager.revoke_all_sessions(username)
        assert count == 3
        
        # Check all sessions are inactive
        for token in tokens:
            assert not st.session_state.active_sessions[token].is_active
    
    @patch('bcrypt.checkpw')
    def test_authenticate_basic_success(self, mock_checkpw, auth_manager):
        """Test successful basic authentication"""
        mock_checkpw.return_value = True
        
        # Mock the users
        auth_manager.users = {
            'test_user': User(
                username='test_user',
                email='test@example.com',
                password_hash='hashed_password',
                role=UserRole.USER
            )
        }
        
        result = auth_manager.authenticate_basic('test_user', 'password123')
        
        assert result is not None
        assert result['username'] == 'test_user'
        assert result['email'] == 'test@example.com'
        assert result['role'] == 'user'
        assert 'token' in result
    
    @patch('bcrypt.checkpw')
    def test_authenticate_basic_failure(self, mock_checkpw, auth_manager):
        """Test failed basic authentication"""
        mock_checkpw.return_value = False
        
        auth_manager.users = {
            'test_user': User(
                username='test_user',
                email='test@example.com',
                password_hash='hashed_password',
                role=UserRole.USER
            )
        }
        
        result = auth_manager.authenticate_basic('test_user', 'wrong_password')
        assert result is None
        
        # Check failed attempt was recorded
        assert 'test_user' in st.session_state.failed_attempts
    
    def test_password_expiry_check(self, auth_manager):
        """Test password expiry checking"""
        # Create user with old password
        old_user = User(
            username='old_user',
            email='old@example.com',
            password_hash='hash',
            role=UserRole.USER
        )
        old_user.password_changed_at = datetime.now(timezone.utc) - timedelta(days=91)
        
        # Should need password change
        assert auth_manager.check_password_expiry(old_user)
        
        # Create user with recent password
        new_user = User(
            username='new_user',
            email='new@example.com',
            password_hash='hash',
            role=UserRole.USER
        )
        new_user.password_changed_at = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Should not need password change
        assert not auth_manager.check_password_expiry(new_user)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])