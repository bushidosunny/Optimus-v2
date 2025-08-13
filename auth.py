"""
Enhanced Authentication Module (2025)
Supports basic auth, OIDC/SSO, JWT tokens, and security features
Cookie-based session persistence across page refreshes
"""

import streamlit as st
import bcrypt
import jwt
import streamlit_authenticator as stauth
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta, timezone
import time
import secrets
import logging
from models import User, UserRole, AuthSession
import re
import hashlib
import json
import os
from pathlib import Path
import base64

logger = logging.getLogger(__name__)


class EnhancedAuth:
    """Enhanced authentication with 2025 security features"""
    
    def __init__(self):
        """Initialize authentication system"""
        auth_config = st.secrets.get("auth", {})
        self.auth_type = auth_config.get("type", "basic")
        self.session_expiry_hours = auth_config.get("session_expiry_hours", 24)
        self.max_login_attempts = auth_config.get("max_login_attempts", 5)
        self.require_captcha_after = auth_config.get("require_captcha_after_failures", 3)
        
        # Security settings
        security_config = st.secrets.get("security", {})
        self.jwt_secret = security_config.get("jwt_secret", self._generate_secret())
        self.encryption_key = security_config.get("encryption_key", self._generate_secret()[:32])
        
        # User storage file
        self.users_file = Path("users.json")
        
        # Use query parameters for session persistence
        self.use_query_params = True
        
        # Initialize session state
        if 'failed_attempts' not in st.session_state:
            st.session_state.failed_attempts = {}
        if 'active_sessions' not in st.session_state:
            st.session_state.active_sessions = {}
        if 'show_registration' not in st.session_state:
            st.session_state.show_registration = False
        
        # Initialize based on auth type
        if self.auth_type == "oidc":
            self._init_oidc()
        else:
            self._init_basic_auth()
        
        logger.info(f"Initialized {self.auth_type} authentication")
    
    def _generate_secret(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    
    def _init_basic_auth(self):
        """Initialize basic authentication with bcrypt"""
        self.users = {}
        
        # Load users from secrets first (admin users)
        for user_data in st.secrets["auth"].get("users", []):
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                password_hash=user_data["password_hash"],
                role=UserRole(user_data.get("role", "user"))
            )
            self.users[user.username] = user
        
        # Load users from file (registered users)
        self._load_users_from_file()
        
        logger.info(f"Loaded {len(self.users)} users")
    
    def _init_oidc(self):
        """Initialize OIDC authentication"""
        oidc_config = st.secrets["auth"]["oidc"]
        self.oidc_config = {
            "provider_url": oidc_config["provider_url"],
            "client_id": oidc_config["client_id"],
            "client_secret": oidc_config["client_secret"],
            "redirect_uri": oidc_config["redirect_uri"],
            "scope": oidc_config.get("scope", "openid profile email")
        }
        logger.info("Initialized OIDC configuration")
    
    def _load_users_from_file(self):
        """Load registered users from JSON file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for username, user_data in users_data.items():
                        # Don't override admin users from secrets
                        if username not in self.users:
                            user = User(
                                username=user_data["username"],
                                email=user_data["email"],
                                password_hash=user_data["password_hash"],
                                role=UserRole(user_data.get("role", "user")),
                                is_active=user_data.get("is_active", True),
                                is_verified=user_data.get("is_verified", False)
                            )
                            self.users[username] = user
                logger.info(f"Loaded {len(users_data)} users from file")
            except Exception as e:
                logger.error(f"Failed to load users from file: {e}")
    
    def _save_users_to_file(self):
        """Save registered users to JSON file"""
        try:
            # Only save non-admin users (admin users are in secrets)
            users_to_save = {}
            for username, user in self.users.items():
                # Check if user is from secrets
                is_from_secrets = False
                for secret_user in st.secrets["auth"].get("users", []):
                    if secret_user["username"] == username:
                        is_from_secrets = True
                        break
                
                if not is_from_secrets:
                    users_to_save[username] = {
                        "username": user.username,
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "role": user.role.value,
                        "is_active": user.is_active,
                        "is_verified": user.is_verified
                    }
            
            with open(self.users_file, 'w') as f:
                json.dump(users_to_save, f, indent=2)
            logger.info(f"Saved {len(users_to_save)} users to file")
        except Exception as e:
            logger.error(f"Failed to save users to file: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password using bcrypt"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'), 
                password_hash.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is strong"
    
    def _save_persistent_session(self, token: str, user_data: Dict):
        """Save session to query parameters for persistence across refreshes"""
        try:
            # Store session token in query parameters
            session_token = base64.urlsafe_b64encode(token.encode()).decode()
            
            logger.info(f"🔄 Attempting to set query parameter with token length: {len(session_token)}")
            
            # Try different methods to set query parameters
            try:
                # Method 1: Direct assignment (newer Streamlit)
                st.query_params["session"] = session_token
                logger.info("✅ Method 1: Direct assignment successful")
            except Exception as e1:
                logger.warning(f"Method 1 failed: {e1}")
                try:
                    # Method 2: experimental_set_query_params (older Streamlit)
                    st.experimental_set_query_params(session=session_token)
                    logger.info("✅ Method 2: experimental_set_query_params successful")
                except Exception as e2:
                    logger.warning(f"Method 2 failed: {e2}")
                    try:
                        # Method 3: Use st.rerun with query params
                        st.query_params.update({"session": session_token})
                        logger.info("✅ Method 3: query_params.update successful")
                    except Exception as e3:
                        logger.error(f"All methods failed: {e1}, {e2}, {e3}")
                        raise
            
            logger.info(f"✅ Saved session token to query params for user: {user_data.get('username')}")
            
            # Verify the parameter was set
            current_params = dict(st.query_params)
            logger.info(f"🔍 Current query params: {list(current_params.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to save persistent session: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _restore_persistent_session(self):
        """Restore session from query parameters"""
        try:
            logger.info("🔍 Attempting to restore session from query params")
            
            # Get session token from query parameters
            session_param = st.query_params.get("session")
            
            if session_param:
                logger.info(f"📥 Found session parameter")
                
                # Decode the token
                try:
                    token = base64.urlsafe_b64decode(session_param.encode()).decode()
                    
                    # Verify the token is still valid
                    user_data = self.verify_jwt_token(token)
                    if user_data:
                        st.session_state.auth_token = token
                        st.session_state.user_data = user_data
                        logger.info(f"✅ Restored session for user: {user_data.get('username')}")
                        return True
                    else:
                        # Token invalid, clear query param
                        self._clear_persistent_session()
                        logger.info("🚫 Invalid token, cleared session")
                        
                except Exception as decode_error:
                    logger.error(f"Failed to decode session token: {decode_error}")
                    self._clear_persistent_session()
            else:
                logger.info("🚫 No session parameter found")
            return False
        except Exception as e:
            logger.error(f"Failed to restore persistent session: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _clear_persistent_session(self):
        """Clear persistent session query parameter"""
        try:
            logger.info("🧹 Attempting to clear session parameter")
            
            # Try different methods to clear query parameters
            try:
                if "session" in st.query_params:
                    del st.query_params["session"]
                    logger.info("✅ Method 1: Direct deletion successful")
            except Exception as e1:
                logger.warning(f"Method 1 failed: {e1}")
                try:
                    # Clear all query params
                    st.experimental_set_query_params()
                    logger.info("✅ Method 2: experimental_set_query_params clear successful")
                except Exception as e2:
                    logger.warning(f"Method 2 failed: {e2}")
                    try:
                        st.query_params.clear()
                        logger.info("✅ Method 3: query_params.clear successful")
                    except Exception as e3:
                        logger.error(f"All clear methods failed: {e1}, {e2}, {e3}")
            
            logger.info("Cleared persistent session")
        except Exception as e:
            logger.error(f"Failed to clear persistent session: {e}")
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def validate_username(self, username: str) -> tuple[bool, str]:
        """Validate username"""
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 20:
            return False, "Username must be 20 characters or less"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        # Check if username already exists
        if username in self.users:
            return False, "Username already exists"
        
        return True, "Username is valid"
    
    def register_user(self, username: str, email: str, password: str) -> tuple[bool, str]:
        """Register a new user"""
        # Validate username
        is_valid, msg = self.validate_username(username)
        if not is_valid:
            return False, msg
        
        # Validate email
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        # Check if email already exists
        for user in self.users.values():
            if user.email == email:
                return False, "Email already registered"
        
        # Validate password
        is_valid, msg = self.validate_password_strength(password)
        if not is_valid:
            return False, msg
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role=UserRole.USER,
            is_active=True,
            is_verified=False
        )
        
        # Add to users dictionary
        self.users[username] = new_user
        
        # Save to file
        self._save_users_to_file()
        
        logger.info(f"New user registered: {username}")
        
        return True, "Registration successful!"
    
    def generate_jwt_token(self, username: str, role: str, user_id: str) -> str:
        """Generate JWT token for session management"""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.session_expiry_hours)
        
        payload = {
            "username": username,
            "user_id": user_id,
            "role": role,
            "exp": int(expires_at.timestamp()),  # Convert to Unix timestamp
            "iat": int(now.timestamp()),  # Convert to Unix timestamp
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        token = jwt.encode(
            payload, 
            self.jwt_secret, 
            algorithm="HS256"
        )
        
        # Store session
        session = AuthSession(
            token=token,
            username=username,
            user_id=user_id,
            role=UserRole(role),
            expires_at=expires_at  # Use the datetime object for session
        )
        
        st.session_state.active_sessions[token] = session
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            # Check if session exists and is active
            session = st.session_state.active_sessions.get(token)
            if not session or not session.is_active:
                return None
            
            # Verify token
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=["HS256"]
            )
            
            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            if token in st.session_state.active_sessions:
                del st.session_state.active_sessions[token]
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke a JWT token"""
        if token in st.session_state.active_sessions:
            st.session_state.active_sessions[token].is_active = False
            logger.info(f"Revoked token for session")
    
    def check_rate_limit(self, username: str) -> bool:
        """Check if user is rate limited (True = rate limited, False = OK to proceed)"""
        attempts = st.session_state.failed_attempts.get(username, {})
        count = attempts.get("count", 0)
        last_attempt = attempts.get("last_attempt", 0)
        
        # Reset counter after 15 minutes
        if time.time() - last_attempt > 900:
            st.session_state.failed_attempts[username] = {"count": 0, "last_attempt": 0}
            return False  # Not rate limited after reset
        
        return count >= self.max_login_attempts  # Rate limited if at or above max attempts
    
    def record_failed_attempt(self, username: str):
        """Record a failed login attempt"""
        if username not in st.session_state.failed_attempts:
            st.session_state.failed_attempts[username] = {"count": 0, "last_attempt": 0}
        
        st.session_state.failed_attempts[username]["count"] += 1
        st.session_state.failed_attempts[username]["last_attempt"] = time.time()
        
        logger.warning(f"Failed login attempt for {username}")
    
    def needs_captcha(self, username: str) -> bool:
        """Check if captcha is required"""
        attempts = st.session_state.failed_attempts.get(username, {})
        return attempts.get("count", 0) >= self.require_captcha_after
    
    def generate_captcha(self) -> tuple[str, str]:
        """Generate a simple math captcha"""
        import random
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operation = random.choice(["+", "-", "*"])
        
        if operation == "+":
            answer = str(a + b)
            question = f"{a} + {b} = ?"
        elif operation == "-":
            answer = str(a - b)
            question = f"{a} - {b} = ?"
        else:
            answer = str(a * b)
            question = f"{a} × {b} = ?"
        
        # Store answer in session with hash
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        st.session_state.captcha_answer = answer_hash
        
        return question, answer_hash
    
    def verify_captcha(self, user_answer: str) -> bool:
        """Verify captcha answer"""
        if 'captcha_answer' not in st.session_state:
            return False
        
        answer_hash = hashlib.sha256(user_answer.encode()).hexdigest()
        return answer_hash == st.session_state.captcha_answer
    
    def authenticate_basic(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate using basic auth"""
        if self.check_rate_limit(username):  # True means rate limited
            logger.warning(f"Rate limit exceeded for {username}")
            return None
        
        if username in self.users:
            user = self.users[username]
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"Inactive user attempted login: {username}")
                return None
            
            if self.verify_password(password, user.password_hash):
                # Reset failed attempts on success
                st.session_state.failed_attempts.pop(username, None)
                
                # Generate token
                token = self.generate_jwt_token(
                    username, 
                    user.role.value,
                    str(user.created_at.timestamp())  # Use timestamp as user ID
                )
                
                # Update user login info
                user.last_login = datetime.now(timezone.utc)
                user.failed_login_attempts = 0
                
                logger.info(f"Successful login for {username}")
                
                # Get the session we just created to get the expiry time
                session = st.session_state.active_sessions[token]
                
                return {
                    "username": username,
                    "email": user.email,
                    "role": user.role.value,
                    "token": token,
                    "user_id": str(user.created_at.timestamp()),
                    "exp": session.expires_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                }
            else:
                self.record_failed_attempt(username)
                user.failed_login_attempts += 1
                user.last_failed_login = datetime.now(timezone.utc)
        else:
            # Record failed attempt even for non-existent users
            # to prevent username enumeration
            self.record_failed_attempt(username)
        
        return None
    
    def get_login_interface(self) -> Optional[Dict]:
        """Enhanced login interface with 2025 features and persistent sessions"""
        # Check for existing session in st.session_state first
        if 'auth_token' in st.session_state:
            user_data = self.verify_jwt_token(st.session_state.auth_token)
            if user_data:
                return user_data
            else:
                # Token expired or invalid
                if 'auth_token' in st.session_state:
                    del st.session_state.auth_token
                if 'user_data' in st.session_state:
                    del st.session_state.user_data
                self._clear_persistent_session()
        
        # If no session state, try to restore from persistent storage (cookies)
        if 'auth_token' not in st.session_state:
            if self._restore_persistent_session():
                # Session restored, verify token
                if 'auth_token' in st.session_state:
                    user_data = self.verify_jwt_token(st.session_state.auth_token)
                    if user_data:
                        # Update user_data in session state if it was restored
                        if 'user_data' not in st.session_state and user_data:
                            st.session_state.user_data = user_data
                        return user_data
                    else:
                        # Token expired or invalid
                        if 'auth_token' in st.session_state:
                            del st.session_state.auth_token
                        if 'user_data' in st.session_state:
                            del st.session_state.user_data
                        self._clear_persistent_session()
        
        # Display login form
        if self.auth_type == "oidc":
            return self._oidc_login_interface()
        else:
            return self._basic_login_interface()
    
    def _basic_login_interface(self) -> Optional[Dict]:
        """Basic authentication login interface"""
        # Initialize session state if needed
        if 'failed_attempts' not in st.session_state:
            st.session_state.failed_attempts = {}
        if 'active_sessions' not in st.session_state:
            st.session_state.active_sessions = {}
        if 'show_registration' not in st.session_state:
            st.session_state.show_registration = False
        
        # Show registration form if toggled
        if st.session_state.show_registration:
            return self._show_registration_form()
        
        # Show login form
        with st.form("enhanced_login_form", clear_on_submit=False):
            st.markdown("### 🔐 Secure Login")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input(
                    "Username", 
                    placeholder="Enter your username",
                    key="login_username"
                )
                password = st.text_input(
                    "Password", 
                    type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )
            
            with col2:
                st.markdown("#### Security Status")
                if username:
                    attempts = st.session_state.failed_attempts.get(username, {})
                    failed_count = attempts.get("count", 0)
                    
                    if failed_count == 0:
                        st.success("✅ No failed attempts")
                    elif failed_count < self.require_captcha_after:
                        st.warning(f"⚠️ {failed_count} failed attempts")
                    else:
                        st.error(f"🚫 {failed_count} failed attempts")
            
            # Show captcha if needed
            captcha_valid = True
            if username and self.needs_captcha(username):
                st.warning("⚠️ Please complete the security check")
                
                captcha_question, _ = self.generate_captcha()
                captcha_answer = st.text_input(
                    f"Security Question: {captcha_question}",
                    key="captcha_answer"
                )
                
                if captcha_answer:
                    captcha_valid = self.verify_captcha(captcha_answer)
                    if not captcha_valid:
                        st.error("❌ Incorrect answer")
                else:
                    captcha_valid = False
            
            # Login buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                submitted = st.form_submit_button(
                    "🔐 Login", 
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                if st.form_submit_button("🔑 Forgot Password?", use_container_width=True):
                    st.info("Password reset would be implemented here")
            
            with col3:
                if st.form_submit_button("📝 Register", use_container_width=True):
                    st.session_state.show_registration = True
                    st.rerun()
            
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                elif not captcha_valid:
                    st.error("Please complete the security check")
                else:
                    # Check rate limit (True = rate limited, False = OK)
                    if self.check_rate_limit(username):
                        attempts = st.session_state.failed_attempts.get(username, {})
                        last_attempt = attempts.get("last_attempt", time.time())
                        remaining_time = 900 - (time.time() - last_attempt)
                        minutes = max(1, int(remaining_time / 60))
                        st.error(f"Too many failed attempts. Please try again in {minutes} minutes.")
                    else:
                        # Attempt authentication
                        user_data = self.authenticate_basic(username, password)
                        
                        if user_data:
                            st.session_state.auth_token = user_data["token"]
                            st.session_state.user_data = user_data
                            
                            logger.info(f"🔐 Login successful for {username}, saving persistent session...")
                            # Save session for persistence across refreshes
                            self._save_persistent_session(user_data["token"], user_data)
                            
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                            
                            # Show remaining attempts
                            attempts = st.session_state.failed_attempts.get(username, {})
                            failed_count = attempts.get("count", 0)
                            remaining = self.max_login_attempts - failed_count
                            
                            if remaining > 0:
                                st.warning(f"⚠️ {remaining} attempts remaining")
        
        # Show security tips
        with st.expander("🔒 Security Tips"):
            st.markdown("""
            - Use a strong password with mixed case, numbers, and symbols
            - Never share your login credentials
            - Enable two-factor authentication when available
            - Report suspicious activity to administrators
            - Session expires after {} hours of inactivity
            """.format(self.session_expiry_hours))
        
        return None
    
    def _show_registration_form(self) -> Optional[Dict]:
        """Show registration form"""
        st.markdown("### 📝 User Registration")
        
        # Add back to login button
        if st.button("← Back to Login"):
            st.session_state.show_registration = False
            st.rerun()
        
        with st.form("registration_form", clear_on_submit=False):
            st.markdown("#### Create New Account")
            
            # Registration fields
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input(
                    "Username",
                    placeholder="Choose a username (3-20 chars)",
                    key="reg_username",
                    help="Letters, numbers, and underscores only"
                )
                
                reg_email = st.text_input(
                    "Email",
                    placeholder="your.email@example.com",
                    key="reg_email"
                )
                
                reg_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Choose a strong password",
                    key="reg_password",
                    help="Min 8 chars, must include uppercase, lowercase, number, and special character"
                )
                
                reg_password_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password",
                    key="reg_password_confirm"
                )
            
            with col2:
                # Real-time validation feedback
                st.markdown("#### Validation Status")
                
                if reg_username:
                    is_valid, msg = self.validate_username(reg_username)
                    if is_valid:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
                
                if reg_email:
                    if self.validate_email(reg_email):
                        st.success("✅ Valid email format")
                    else:
                        st.error("❌ Invalid email format")
                
                if reg_password:
                    is_valid, msg = self.validate_password_strength(reg_password)
                    if is_valid:
                        st.success(f"✅ {msg}")
                    else:
                        st.warning(f"⚠️ {msg}")
                
                if reg_password and reg_password_confirm:
                    if reg_password == reg_password_confirm:
                        st.success("✅ Passwords match")
                    else:
                        st.error("❌ Passwords do not match")
            
            # Terms and conditions
            agree_terms = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy",
                key="agree_terms"
            )
            
            # CAPTCHA for registration
            st.markdown("#### Security Verification")
            
            # Generate CAPTCHA only once and store in session
            if 'reg_captcha_question' not in st.session_state:
                captcha_question, answer_hash = self.generate_captcha()
                st.session_state.reg_captcha_question = captcha_question
                st.session_state.reg_captcha_answer = answer_hash
            
            col1, col2 = st.columns([3, 1])
            with col1:
                reg_captcha = st.text_input(
                    f"Security Question: {st.session_state.reg_captcha_question}",
                    key="reg_captcha"
                )
            with col2:
                if st.form_submit_button("🔄 New Question", use_container_width=True):
                    # Generate new CAPTCHA
                    captcha_question, answer_hash = self.generate_captcha()
                    st.session_state.reg_captcha_question = captcha_question
                    st.session_state.reg_captcha_answer = answer_hash
                    st.rerun()
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Create Account",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                # Validate all fields
                errors = []
                
                if not reg_username or not reg_email or not reg_password or not reg_password_confirm:
                    errors.append("All fields are required")
                
                if reg_password != reg_password_confirm:
                    errors.append("Passwords do not match")
                
                if not agree_terms:
                    errors.append("You must agree to the terms")
                
                # Verify CAPTCHA using the stored answer
                if 'reg_captcha_answer' in st.session_state:
                    answer_hash = hashlib.sha256(reg_captcha.strip().encode()).hexdigest()
                    if answer_hash != st.session_state.reg_captcha_answer:
                        errors.append("Incorrect security answer")
                else:
                    errors.append("Security verification required")
                
                # If no basic errors, try to register
                if not errors:
                    success, message = self.register_user(
                        reg_username,
                        reg_email,
                        reg_password
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        
                        # Clear CAPTCHA from session
                        if 'reg_captcha_question' in st.session_state:
                            del st.session_state.reg_captcha_question
                        if 'reg_captcha_answer' in st.session_state:
                            del st.session_state.reg_captcha_answer
                        
                        # Clear any failed attempts for this username
                        if reg_username in st.session_state.failed_attempts:
                            del st.session_state.failed_attempts[reg_username]
                        
                        # Auto-login after registration
                        st.info("Logging you in...")
                        time.sleep(1)
                        
                        # Authenticate the new user
                        user_data = self.authenticate_basic(reg_username, reg_password)
                        if user_data:
                            st.session_state.auth_token = user_data["token"]
                            st.session_state.user_data = user_data
                            # Save session for persistence across refreshes
                            self._save_persistent_session(user_data["token"], user_data)
                            st.session_state.show_registration = False
                            st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    # Show all errors
                    for error in errors:
                        st.error(f"❌ {error}")
        
        # Registration benefits
        with st.expander("🎁 Why Register?"):
            st.markdown("""
            **Benefits of creating an account:**
            - 🧠 Persistent memory storage across sessions
            - 💬 Personalized AI conversations
            - 📊 Usage analytics and insights
            - 🔒 Secure data management
            - ⚡ Access to all AI models
            - 📥 Export your data anytime
            """)
        
        return None
    
    def _oidc_login_interface(self) -> Optional[Dict]:
        """OIDC/SSO login interface"""
        st.markdown("### 🔐 Single Sign-On")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Click below to login with your organization's identity provider")
            
            if st.button("🚀 Login with SSO", type="primary", use_container_width=True):
                # In production, this would redirect to OIDC provider
                st.info("Redirecting to identity provider...")
                
                # For demo purposes, show what would happen
                with st.expander("OIDC Flow (Demo)"):
                    st.code(f"""
                    1. Redirect to: {self.oidc_config['provider_url']}/authorize
                    2. Parameters:
                       - client_id: {self.oidc_config['client_id']}
                       - redirect_uri: {self.oidc_config['redirect_uri']}
                       - scope: {self.oidc_config['scope']}
                       - response_type: code
                    3. User authenticates with provider
                    4. Provider redirects back with authorization code
                    5. Exchange code for tokens
                    6. Validate tokens and create session
                    """)
        
        with col2:
            st.markdown("#### Benefits")
            st.markdown("""
            - ✅ No password to remember
            - ✅ Enterprise security
            - ✅ Automatic provisioning
            - ✅ Centralized access control
            """)
        
        return None
    
    def logout(self):
        """Enhanced logout with session cleanup"""
        # Revoke current token
        if 'auth_token' in st.session_state:
            self.revoke_token(st.session_state.auth_token)
            del st.session_state.auth_token
        
        # Clear user data
        if 'user_data' in st.session_state:
            del st.session_state.user_data
        
        # Clear persistent session
        self._clear_persistent_session()
        
        # Clear all session data except failed attempts
        keys_to_keep = ['failed_attempts', 'active_sessions', 'browser_session_id']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        logger.info("User logged out")
        st.success("✅ Logged out successfully")
        time.sleep(1)
        st.rerun()
    
    def get_active_sessions(self, username: str) -> List[AuthSession]:
        """Get all active sessions for a user"""
        active_sessions = []
        
        for token, session in list(st.session_state.active_sessions.items()):
            if session.username == username and session.is_active:
                try:
                    # Handle different types of expires_at
                    if isinstance(session.expires_at, (int, float)):
                        # Convert Unix timestamp to datetime
                        session.expires_at = datetime.fromtimestamp(session.expires_at, tz=timezone.utc)
                    elif isinstance(session.expires_at, datetime) and session.expires_at.tzinfo is None:
                        # Convert naive datetime to UTC
                        session.expires_at = session.expires_at.replace(tzinfo=timezone.utc)
                    
                    if session.expires_at > datetime.now(timezone.utc):
                        active_sessions.append(session)
                    else:
                        # Clean up expired session
                        session.is_active = False
                except Exception as e:
                    # Remove invalid session
                    logger.warning(f"Removing invalid session: {e}")
                    del st.session_state.active_sessions[token]
        
        return active_sessions
    
    def revoke_all_sessions(self, username: str):
        """Revoke all sessions for a user"""
        revoked_count = 0
        
        for token, session in st.session_state.active_sessions.items():
            if session.username == username and session.is_active:
                session.is_active = False
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} sessions for {username}")
        return revoked_count
    
    def enforce_session_limit(self, username: str, limit: int = 3):
        """Enforce maximum concurrent sessions per user"""
        active_sessions = self.get_active_sessions(username)
        
        if len(active_sessions) > limit:
            # Revoke oldest sessions
            sessions_to_revoke = sorted(
                active_sessions, 
                key=lambda s: s.created_at
            )[:-limit]
            
            for session in sessions_to_revoke:
                session.is_active = False
            
            logger.info(f"Revoked {len(sessions_to_revoke)} excess sessions for {username}")
    
    def check_password_expiry(self, user: User) -> bool:
        """Check if user's password has expired (90 days)"""
        if not hasattr(user, 'password_changed_at'):
            return False
        
        expiry_days = 90
        password_age = datetime.now(timezone.utc) - user.password_changed_at
        
        return password_age.days > expiry_days