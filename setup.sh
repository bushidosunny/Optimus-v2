#!/bin/bash

# Optimus - Local Setup Script
# This script sets up the virtual environment and installs dependencies

echo "🧠 Optimus - Setup Script"
echo "=================================="

# Check if Python 3.10+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if (( $(echo "$python_version < $required_version" |bc -l) )); then
    echo "❌ Error: Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
echo "This may take a few minutes..."

# Install core dependencies first
pip install streamlit==1.38.0
pip install pydantic==2.7.1
pip install numpy==1.26.4

# Then install the rest
pip install -r requirements.txt

# Create .streamlit/secrets.toml if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "📝 Creating secrets.toml from template..."
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    echo "⚠️  Please edit .streamlit/secrets.toml with your API keys"
else
    echo "✅ secrets.toml already exists"
fi

# Create necessary directories
mkdir -p logs
mkdir -p exports
mkdir -p qdrant_storage

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .streamlit/secrets.toml with your API keys"
echo "2. Start Qdrant (optional): docker-compose up -d"
echo "3. Run the app: streamlit run app.py"
echo ""
echo "For more information, see README.md"