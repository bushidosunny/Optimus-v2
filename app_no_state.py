"""
Streamlit app without session state dependency
"""
import streamlit as st
from datetime import datetime
import hashlib

st.set_page_config(
    page_title="Optimus v2",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - AI Memory System")

# Use query params for auth state (persists across reloads)
query_params = st.query_params

# Simple auth token check
def check_auth():
    auth_token = query_params.get("auth", None)
    expected_token = hashlib.md5(f"{st.secrets.get('ADMIN_USERNAME', 'sunny')}:{st.secrets.get('ADMIN_PASSWORD', 'loveMOM1!')}".encode()).hexdigest()
    return auth_token == expected_token

def set_auth(username, password):
    if username == st.secrets.get("ADMIN_USERNAME", "sunny") and password == st.secrets.get("ADMIN_PASSWORD", "loveMOM1!"):
        token = hashlib.md5(f"{username}:{password}".encode()).hexdigest()
        st.query_params["auth"] = token
        return True
    return False

# Main interface
is_authenticated = check_auth()

if not is_authenticated:
    # Login page
    st.markdown("### 🔐 Please Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if set_auth(username, password):
                    st.success("✅ Login successful!")
                    st.balloons()
                    # Force reload with auth token
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.info("Use your admin credentials to login")
else:
    # Authenticated interface
    st.success(f"✅ Welcome {st.secrets.get('ADMIN_USERNAME', 'User')}!")
    
    # Logout button
    if st.button("🚪 Logout", key="logout"):
        st.query_params.clear()
        st.rerun()
    
    # Simple tabbed interface
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memories", "⚙️ Settings"])
    
    with tab1:
        st.markdown("### Chat Interface")
        
        # Simple chat (no history for now)
        user_input = st.text_input("Type a message:", key="chat_input")
        if st.button("Send", key="send_btn"):
            if user_input:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.info(f"You: {user_input}")
                with col2:
                    st.success(f"AI: I received: '{user_input}'")
    
    with tab2:
        st.markdown("### Memory Management")
        
        # Simple memory form
        with st.form("memory_form"):
            memory = st.text_area("Enter a memory:")
            importance = st.slider("Importance", 0.0, 1.0, 0.5)
            if st.form_submit_button("Save Memory"):
                st.success(f"✅ Memory saved: {memory[:50]}...")
    
    with tab3:
        st.markdown("### Settings")
        
        # Show API status
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### API Keys")
            if st.secrets.get("OPENAI_API_KEY"):
                st.success("✅ OpenAI")
            if st.secrets.get("ANTHROPIC_API_KEY"):
                st.success("✅ Anthropic")
            if st.secrets.get("QDRANT_API_KEY"):
                st.success("✅ Qdrant")
        
        with col2:
            st.markdown("#### System")
            st.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            st.info(f"Auth Token: {query_params.get('auth', 'None')[:8]}...")

st.markdown("---")
st.caption("© 2025 Optimus v2 - Foundation Care")