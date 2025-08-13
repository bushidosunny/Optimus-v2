#!/usr/bin/env python3
"""
Test script to verify Optimus setup
"""

import sys
import importlib
from typing import Dict, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(module_name, package_name)
        else:
            importlib.import_module(module_name)
        return True, "✅ Installed"
    except ImportError as e:
        return False, f"❌ Not installed: {str(e)}"
    except Exception as e:
        return False, f"⚠️  Error: {str(e)}"

def check_dependencies():
    """Check all required dependencies"""
    print("🧠 Optimus - Setup Verification")
    print("=" * 50)
    
    # Core dependencies
    core_deps = {
        "streamlit": "Streamlit (UI Framework)",
        "pydantic": "Pydantic (Data Validation)",
        "pydantic_ai": "PydanticAI (LLM Framework)",
        "mem0": "Mem0 (Memory Management)",
        "qdrant_client": "Qdrant (Vector Database)",
        "sentence_transformers": "Sentence Transformers (Embeddings)",
        "jwt": "PyJWT (Authentication)",
        "bcrypt": "Bcrypt (Password Hashing)",
        "pandas": "Pandas (Data Processing)",
        "plotly": "Plotly (Visualizations)",
        "httpx": "HTTPX (HTTP Client)",
        "numpy": "NumPy (Numerical Computing)"
    }
    
    print("\n📦 Core Dependencies:")
    all_good = True
    for module, name in core_deps.items():
        success, message = test_import(module)
        print(f"  {name}: {message}")
        if not success:
            all_good = False
    
    # Optional LLM providers
    llm_providers = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google.generativeai": "Google AI",
        "groq": "Groq",
        "mistralai": "Mistral AI"
    }
    
    print("\n🤖 LLM Providers (at least one required):")
    llm_available = False
    for module, name in llm_providers.items():
        success, message = test_import(module)
        print(f"  {name}: {message}")
        if success:
            llm_available = True
    
    # Check local modules
    print("\n📁 Local Modules:")
    local_modules = ["models", "memory_manager", "llm_handler", "auth", "utils"]
    for module in local_modules:
        success, message = test_import(module)
        print(f"  {module}.py: {message}")
        if not success:
            all_good = False
    
    # Check configuration
    print("\n⚙️  Configuration:")
    import os
    
    config_checks = {
        ".streamlit/config.toml": os.path.exists(".streamlit/config.toml"),
        ".streamlit/secrets.toml": os.path.exists(".streamlit/secrets.toml"),
        "requirements.txt": os.path.exists("requirements.txt")
    }
    
    for file, exists in config_checks.items():
        status = "✅ Found" if exists else "❌ Not found"
        print(f"  {file}: {status}")
        if not exists and file == ".streamlit/secrets.toml":
            print("    💡 Copy from .streamlit/secrets.toml.example")
    
    # Summary
    print("\n" + "=" * 50)
    if all_good and llm_available:
        print("✅ All core dependencies are installed!")
        print("✅ At least one LLM provider is available!")
        print("\n🚀 Ready to run: streamlit run app.py")
    else:
        print("❌ Some dependencies are missing.")
        print("\n💡 Try running: pip install -r requirements.txt")
        if not llm_available:
            print("⚠️  No LLM providers found. Install at least one:")
            print("   pip install openai  # If you have OpenAI API key")
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("⚠️  Python 3.10+ is recommended")

if __name__ == "__main__":
    check_dependencies()