"""
Working Streamlit app - Progressive build
"""
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Optimus v2",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - AI Memory System")

# Test 1: Basic session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Test Counter"):
    st.session_state.counter += 1
    st.write(f"Counter: {st.session_state.counter}")

st.markdown("---")

# Test 2: Simple authentication without rerun
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("### Login")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")
    with col2:
        password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == st.secrets.get("ADMIN_USERNAME", "admin") and password == st.secrets.get("ADMIN_PASSWORD", "password"):
            st.session_state.logged_in = True
            st.success("✅ Logged in!")
        else:
            st.error("Invalid credentials")

# Test 3: Show content after login (no rerun)
if st.session_state.logged_in:
    st.markdown("### Welcome! You are logged in.")
    
    # Simple chat interface
    st.markdown("#### Chat")
    user_input = st.text_input("Message:")
    if st.button("Send") and user_input:
        st.write(f"You: {user_input}")
        st.info("AI: I received your message!")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.info("Logged out. Refresh the page.")

st.markdown("---")
st.caption(f"Status: {'Logged In ✅' if st.session_state.get('logged_in') else 'Not Logged In ❌'}")