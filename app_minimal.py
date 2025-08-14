"""
Minimal app for Streamlit Cloud - bypasses heavy initialization
"""
import streamlit as st
import asyncio
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
    # Main interface
    try:
        st.sidebar.success(f"✅ Logged in as {st.secrets.get('ADMIN_USERNAME')}")
        
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.write("Debug: Creating tabs...")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memories", "⚙️ Settings"])
        
        st.write("Debug: Tabs created successfully")
    
        with tab1:
            st.markdown("## 💬 Chat Interface")
        
        # Initialize session state for messages
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Simple response using OpenAI
            with st.chat_message("assistant"):
                try:
                    import openai
                    
                    client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
                    
                    # Create response
                    with st.spinner("Thinking..."):
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are Optimus Prime, an AI assistant for Foundation Care."},
                                {"role": "user", "content": prompt}
                            ],
                            stream=True
                        )
                    
                    # Stream response
                    full_response = ""
                    placeholder = st.empty()
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            placeholder.markdown(full_response)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    fallback = "I'm having trouble connecting to the AI service. Please check your API keys."
                    st.write(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})
    
    with tab2:
        st.markdown("## 🧠 Memory Management")
        
        # Simple memory storage using session state
        if 'memories' not in st.session_state:
            st.session_state.memories = []
        
        # Add memory form
        with st.form("add_memory"):
            memory_content = st.text_area("Add a memory")
            importance = st.slider("Importance", 0.0, 1.0, 0.5)
            if st.form_submit_button("💾 Save Memory"):
                st.session_state.memories.append({
                    "content": memory_content,
                    "importance": importance,
                    "timestamp": datetime.now().isoformat()
                })
                st.success("Memory saved!")
                st.rerun()
        
        # Display memories
        if st.session_state.memories:
            st.markdown("### 📚 Stored Memories")
            for i, mem in enumerate(reversed(st.session_state.memories)):
                with st.expander(f"Memory {len(st.session_state.memories) - i}"):
                    st.write(mem["content"])
                    st.caption(f"Importance: {mem['importance']:.1f}")
                    st.caption(f"Created: {mem['timestamp']}")
        else:
            st.info("No memories stored yet")
    
    with tab3:
        st.markdown("## ⚙️ Settings")
        
        # Show configuration
        st.markdown("### 🔑 API Keys Status")
        
        api_keys = [
            ("OpenAI", "OPENAI_API_KEY"),
            ("Anthropic", "ANTHROPIC_API_KEY"),
            ("Qdrant", "QDRANT_API_KEY"),
            ("X.AI", "XAI_API_KEY"),
            ("Groq", "GROQ_API_KEY")
        ]
        
        for name, key in api_keys:
            if st.secrets.get(key):
                st.success(f"✅ {name} configured")
            else:
                st.warning(f"⚠️ {name} not configured")
        
        st.markdown("### 📊 System Info")
        import sys
        st.write(f"Python version: {sys.version}")
        st.write(f"Streamlit version: {st.__version__}")

st.caption("© 2025 Optimus v2 - Foundation Care")