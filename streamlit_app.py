import streamlit as st

st.write("App is loading...")

try:
    st.write("Step 1: Basic imports")
    import sys
    st.write(f"Python version: {sys.version}")
    
    st.write("Step 2: Checking streamlit")
    st.write(f"Streamlit version: {st.__version__}")
    
    st.write("Step 3: Checking secrets")
    st.write(f"Has secrets: {bool(st.secrets)}")
    
    st.write("Step 4: Trying to import app_minimal")
    import app_minimal
    st.write("✅ Successfully imported app_minimal")
    
except Exception as e:
    st.error(f"Error during import: {e}")
    import traceback
    st.code(traceback.format_exc())