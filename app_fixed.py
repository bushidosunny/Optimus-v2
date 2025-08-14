"""
Fixed Streamlit app - No conditional rendering issues
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Optimus v2",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - AI Memory System")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Create tabs regardless of login state
tab1, tab2, tab3 = st.tabs(["🔐 Login/Chat", "🧠 Memories", "⚙️ Settings"])

with tab1:
    if not st.session_state.logged_in:
        # Login form
        st.markdown("### Please Login")
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username == st.secrets.get("ADMIN_USERNAME", "sunny") and password == st.secrets.get("ADMIN_PASSWORD", "loveMOM1!"):
                    st.session_state.logged_in = True
                    st.success("✅ Login successful! Chat is now available.")
                else:
                    st.error("❌ Invalid credentials")
    else:
        # Chat interface
        st.markdown(f"### Welcome {st.secrets.get('ADMIN_USERNAME', 'User')}! 👋")
        
        # Logout button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.messages = []
                st.experimental_rerun()
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Simple AI response
            response = f"Echo: {prompt}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

with tab2:
    if st.session_state.logged_in:
        st.markdown("### Memory Storage")
        
        # Memory input
        with st.form("memory_form", clear_on_submit=True):
            memory_text = st.text_area("Add a memory:")
            importance = st.slider("Importance", 0.0, 1.0, 0.5)
            if st.form_submit_button("💾 Save Memory"):
                if 'memories' not in st.session_state:
                    st.session_state.memories = []
                st.session_state.memories.append({
                    "text": memory_text,
                    "importance": importance,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                st.success("Memory saved!")
        
        # Display memories
        if 'memories' in st.session_state and st.session_state.memories:
            st.markdown("#### Saved Memories")
            for i, mem in enumerate(st.session_state.memories[-5:]):  # Last 5
                st.write(f"{i+1}. [{mem['time']}] {mem['text']} (Importance: {mem['importance']})")
    else:
        st.info("Please login to access memories")

with tab3:
    st.markdown("### Settings & Status")
    
    # Login status
    if st.session_state.logged_in:
        st.success(f"✅ Logged in as: {st.secrets.get('ADMIN_USERNAME', 'User')}")
    else:
        st.warning("❌ Not logged in")
    
    # API Keys status
    st.markdown("#### API Keys")
    col1, col2 = st.columns(2)
    with col1:
        if st.secrets.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI configured")
        else:
            st.error("❌ OpenAI not configured")
        
        if st.secrets.get("QDRANT_API_KEY"):
            st.success("✅ Qdrant configured")
        else:
            st.error("❌ Qdrant not configured")
    
    with col2:
        if st.secrets.get("ANTHROPIC_API_KEY"):
            st.success("✅ Anthropic configured")
        else:
            st.warning("⚠️ Anthropic not configured")
        
        st.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")

# Footer
st.markdown("---")
st.caption("© 2025 Optimus v2 - Foundation Care")