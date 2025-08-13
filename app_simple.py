"""
Simplified app for debugging Streamlit Cloud deployment
"""
import streamlit as st

st.set_page_config(
    page_title="Optimus v2 - Debug",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Optimus v2 - Deployment Debug")

# Check secrets
st.header("1. Checking Secrets")
secrets_to_check = [
    'QDRANT_URL',
    'QDRANT_API_KEY', 
    'OPENAI_API_KEY',
    'JWT_SECRET_KEY',
    'ADMIN_USERNAME',
    'ADMIN_PASSWORD'
]

for secret in secrets_to_check:
    if st.secrets.get(secret):
        st.success(f"✅ {secret} is set")
    else:
        st.error(f"❌ {secret} is missing")

# Test Qdrant connection
st.header("2. Testing Qdrant Connection")
try:
    from qdrant_client import QdrantClient
    
    if st.secrets.get('QDRANT_URL') and st.secrets.get('QDRANT_API_KEY'):
        with st.spinner("Connecting to Qdrant..."):
            client = QdrantClient(
                url=st.secrets['QDRANT_URL'],
                api_key=st.secrets['QDRANT_API_KEY'],
                timeout=10
            )
            collections = client.get_collections()
            st.success(f"✅ Connected to Qdrant! Found {len(collections.collections)} collections")
            for col in collections.collections:
                st.write(f"  - {col.name}")
    else:
        st.error("Missing Qdrant credentials")
except Exception as e:
    st.error(f"❌ Qdrant connection failed: {e}")

# Test OpenAI
st.header("3. Testing OpenAI Connection")
try:
    if st.secrets.get('OPENAI_API_KEY'):
        import openai
        client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
        models = client.models.list()
        st.success("✅ OpenAI API connected")
    else:
        st.warning("No OpenAI API key found")
except Exception as e:
    st.error(f"❌ OpenAI connection failed: {e}")

# Test Mem0
st.header("4. Testing Mem0 Initialization")
try:
    from mem0 import Memory
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": st.secrets.get('QDRANT_URL', ''),
                "api_key": st.secrets.get('QDRANT_API_KEY', ''),
                "collection_name": "memories"
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-3.5-turbo",
                "api_key": st.secrets.get('OPENAI_API_KEY', '')
            }
        }
    }
    
    with st.spinner("Initializing Mem0..."):
        mem0 = Memory.from_config(config)
        st.success("✅ Mem0 initialized successfully")
except Exception as e:
    st.error(f"❌ Mem0 initialization failed: {e}")

st.header("5. System Info")
import sys
import platform
st.write(f"Python version: {sys.version}")
st.write(f"Platform: {platform.platform()}")

# Package versions
st.header("6. Package Versions")
try:
    import streamlit
    import mem0
    import qdrant_client
    import pydantic_ai
    
    st.write(f"- Streamlit: {streamlit.__version__}")
    st.write(f"- Mem0: {mem0.__version__ if hasattr(mem0, '__version__') else 'unknown'}")
    st.write(f"- Qdrant Client: {qdrant_client.__version__ if hasattr(qdrant_client, '__version__') else 'unknown'}")
    st.write(f"- PydanticAI: {pydantic_ai.__version__ if hasattr(pydantic_ai, '__version__') else 'unknown'}")
except ImportError as e:
    st.error(f"Missing package: {e}")

st.success("Debug complete! If all checks pass, the main app should work.")