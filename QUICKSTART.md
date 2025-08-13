# 🚀 Quick Start Guide - Local Testing

This guide will help you get the Optimus running locally in under 5 minutes.

## Prerequisites

- Python 3.10 or higher
- Git
- At least one API key (OpenAI, Anthropic, or free providers like Groq)

## Step 1: Clone and Navigate

```bash
cd ai-memory-system
```

## Step 2: Run Setup Script

### macOS/Linux:
```bash
./setup.sh
```

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Configure API Keys

Edit `.streamlit/secrets.toml` with your API keys:

```toml
# Minimal configuration for local testing

# Option 1: Use Qdrant Cloud (free tier)
QDRANT_URL = "https://your-cluster.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"

# Option 2: Use local Qdrant (requires Docker)
# QDRANT_URL = "http://localhost:6333"
# QDRANT_API_KEY = ""  # Empty for local

# Add at least one LLM provider
OPENAI_API_KEY = "sk-..."  # OpenAI
# ANTHROPIC_API_KEY = "sk-ant-..."  # Anthropic
# GROQ_API_KEY = "gsk_..."  # Groq (free)

# Basic auth setup (for testing)
[auth]
type = "basic"
[[auth.users]]
username = "test"
email = "test@example.com"
password_hash = "$2b$12$iWkiJBdHWCrXmP15CyKKBuvMKvzR8XqKxRbnmRBzMRXhPpMxhW1om"  # password: test123
role = "admin"

[security]
jwt_secret = "your-secret-key-at-least-32-characters-long!!"
encryption_key = "32-byte-key-for-encryption-here!!"
```

## Step 4: Start Local Qdrant (Optional)

If you don't have a Qdrant Cloud account, run locally with Docker:

```bash
docker-compose up -d qdrant
```

Then update `.streamlit/secrets.toml`:
```toml
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""  # Empty for local
```

## Step 5: Verify Setup

```bash
python test_setup.py
```

## Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## Default Login Credentials

- Username: `test`
- Password: `test123`

## Quick Test Workflow

1. **Login** with the test credentials
2. **Select an AI Model** from the sidebar
3. **Start Chatting** - the system will automatically create memories
4. **Check Memories Tab** to see stored information
5. **Try Analytics Tab** to visualize your usage

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements-minimal.txt
```

### Qdrant connection failed
- Make sure Qdrant is running: `docker ps`
- Or use Qdrant Cloud (free tier)

### No LLM providers available
- Ensure you've added at least one API key in secrets.toml
- Free option: Get a Groq API key from https://console.groq.com

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## Generate Password Hash

To create your own user:

```python
import bcrypt
password = "your_password"
hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
print(hash)
```

## Minimal Test Mode

For quick testing without all features:

1. Use `requirements-minimal.txt` instead
2. Use local Qdrant with Docker
3. Use free LLM providers (Groq, Google)

## Next Steps

- Add more users in secrets.toml
- Configure additional LLM providers
- Explore memory management features
- Check analytics dashboard
- Export your data

## Need Help?

- Check `test_setup.py` output for missing dependencies
- Review logs in the terminal
- See full README.md for detailed documentation