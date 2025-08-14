"""
Minimal app for Streamlit Cloud - Fixed version
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Optimus v2 - AI Memory System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - AI Memory System")

# Simple login interface
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            # Check against secrets
            if username == st.secrets.get("ADMIN_USERNAME") and password == st.secrets.get("ADMIN_PASSWORD"):
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
else:
    # Main interface - authenticated user
    st.sidebar.success(f"✅ Logged in as {st.secrets.get('ADMIN_USERNAME')}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Create three columns for a simple layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💬 Quick Chat")
        user_input = st.text_input("Ask me anything:")
        if st.button("Send") and user_input:
            st.write(f"You said: {user_input}")
            st.info("Chat response would appear here")
    
    with col2:
        st.markdown("### 🧠 Quick Memory")
        memory = st.text_area("Save a memory:")
        if st.button("Save Memory") and memory:
            if 'saved_memories' not in st.session_state:
                st.session_state.saved_memories = []
            st.session_state.saved_memories.append(memory)
            st.success("Memory saved!")
    
    with col3:
        st.markdown("### ⚙️ Status")
        st.success("✅ App is running")
        st.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Show API status
        if st.secrets.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI connected")
        else:
            st.warning("⚠️ No OpenAI key")
    
    # Show saved memories if any
    if 'saved_memories' in st.session_state and st.session_state.saved_memories:
        st.markdown("---")
        st.markdown("### 📚 Your Memories")
        for i, mem in enumerate(st.session_state.saved_memories[-5:], 1):  # Show last 5
            st.write(f"{i}. {mem}")

st.caption("© 2025 Optimus v2 - Foundation Care")