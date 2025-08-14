"""
Simplest possible Streamlit app - no auth, no session state
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Optimus v2",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - AI Memory System")
st.caption(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Simple tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memories", "⚙️ Settings"])

with tab1:
    st.markdown("### Chat Interface")
    
    # Simple chat input/output
    user_input = st.text_input("Type your message:", key="chat")
    if st.button("Send Message"):
        if user_input:
            st.success(f"You said: {user_input}")
            st.info(f"AI response: Echo - {user_input}")

with tab2:
    st.markdown("### Memory Storage")
    
    # Simple memory input
    memory_text = st.text_area("Enter a memory:", key="memory")
    importance = st.slider("Importance", 0.0, 1.0, 0.5, key="importance")
    
    if st.button("Save Memory"):
        if memory_text:
            st.success(f"✅ Saved: {memory_text[:100]}...")

with tab3:
    st.markdown("### System Status")
    
    # Check API keys
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### API Keys")
        
        # Check each API key
        keys_to_check = [
            ("OpenAI", "OPENAI_API_KEY"),
            ("Anthropic", "ANTHROPIC_API_KEY"),
            ("Qdrant", "QDRANT_API_KEY"),
            ("X.AI", "XAI_API_KEY")
        ]
        
        for name, key in keys_to_check:
            if st.secrets.get(key):
                st.success(f"✅ {name}")
            else:
                st.warning(f"⚠️ {name}")
    
    with col2:
        st.markdown("#### Info")
        st.info(f"Python: {st.__version__}")
        st.info(f"Streamlit: {st.__version__}")
        
        if st.button("Test Button"):
            st.balloons()

# Footer
st.markdown("---")
st.caption("© 2025 Optimus v2 - Open Access Version")