# 🧠 Optimus v2.0

A production-ready, AI-agnostic memory system built with Streamlit, Mem0, Qdrant, and PydanticAI. This system provides persistent memory capabilities for AI applications with support for multiple LLM providers, advanced search, and comprehensive analytics.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.38.0-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

## 🌟 Features

### Core Capabilities
- **🤖 Multi-LLM Support**: Seamlessly switch between OpenAI, Anthropic, Google, Groq, Mistral, and X.AI
- **💾 Intelligent Memory Management**: Two-phase pipeline for extraction and deduplication
- **🔍 Advanced Search**: Semantic search with entity extraction and relevance scoring
- **📊 Comprehensive Analytics**: Visualize memory usage, patterns, and insights
- **🔐 Enterprise Security**: JWT authentication, rate limiting, and session management
- **🚀 Performance Optimized**: Binary quantization, gRPC support, and caching

### 2025 Enhancements
- **OpenMemory MCP**: Local-first memory portability
- **Type-Safe Agents**: PydanticAI integration for reliable tool calling
- **GDPR Compliance**: Data export and deletion capabilities
- **Real-time Monitoring**: Prometheus metrics and health checks
- **Cost Optimization**: Smart model routing based on query complexity

## 📋 Prerequisites

- Python 3.10 or higher
- Git
- Free accounts for:
  - [Qdrant Cloud](https://cloud.qdrant.io) (vector database)
  - [Streamlit Community](https://streamlit.io/cloud) (optional, for deployment)
  - At least one LLM provider API key

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-memory-system.git
cd ai-memory-system
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets

Copy the secrets template and fill in your API keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your credentials:

```toml
# Qdrant Configuration
QDRANT_URL = "https://your-cluster.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"

# LLM API Keys (add at least one)
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
# ... add other providers as needed

# Generate password hashes for users
# python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   LLM Handler   │────▶│ Multiple LLMs   │
│   (app.py)      │     │ (llm_handler.py)│     │ (OpenAI, etc)  │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Auth Manager   │     │ Memory Manager  │────▶│     Qdrant      │
│   (auth.py)     │     │(memory_manager) │     │ Vector Database │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │      Mem0       │
                        │ Memory Pipeline │
                        └─────────────────┘
```

## 📁 Project Structure

```
ai-memory-system/
├── app.py                 # Main Streamlit application
├── memory_manager.py      # Mem0 + Qdrant integration
├── llm_handler.py        # PydanticAI multi-LLM handler
├── models.py             # Pydantic data models
├── auth.py               # Enhanced authentication
├── utils.py              # Helper functions and monitoring
├── requirements.txt      # Python dependencies
├── .streamlit/
│   ├── config.toml      # Streamlit configuration
│   └── secrets.toml     # API keys (create from template)
├── tests/               # Unit tests
├── scripts/             # Deployment and utility scripts
└── docs/                # Additional documentation
```

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

The app comes with optimized settings for performance and UI:
- Dark theme with custom colors
- WebSocket optimization
- Enhanced security settings

### Environment Variables

You can also use environment variables instead of `secrets.toml`:

```bash
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"
export OPENAI_API_KEY="sk-..."
```

## 💻 Usage

### Basic Chat with Memory

1. Login with your credentials
2. Select an AI model from the sidebar
3. Start chatting - the system automatically stores and retrieves relevant memories

### Memory Management

- **Add Memory**: Manually add important information
- **Search**: Find specific memories with semantic search
- **Consolidate**: Remove duplicates and merge similar memories
- **Export**: Download your data in JSON format

### Analytics

View insights about your memory usage:
- Memory creation timeline
- Category distribution
- Most accessed memories
- Entity relationships

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in the deployment settings
5. Deploy!

### Docker Deployment

```bash
# Build image
docker build -t ai-memory-system .

# Run container
docker run -p 8501:8501 \
  -e QDRANT_URL=$QDRANT_URL \
  -e QDRANT_API_KEY=$QDRANT_API_KEY \
  ai-memory-system
```

### Local Qdrant Instance

For development, you can run Qdrant locally:

```bash
docker-compose up -d
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_memory_manager.py
```

## 📊 Performance Optimization

### Memory Usage
- Binary quantization reduces memory by 75%
- Embedding model uses ~200MB RAM
- Streamlit Community Cloud limit: 1GB

### Latency Targets
- Memory search: < 100ms
- LLM response: < 2s
- UI updates: < 50ms

### Cost Management
- Use free tier models for development
- Route simple queries to cheaper models
- Monitor token usage per user

## 🔒 Security Best Practices

1. **Authentication**
   - Use strong passwords (bcrypt hash)
   - Enable session timeout
   - Implement rate limiting

2. **Data Protection**
   - Never commit secrets to Git
   - Use HTTPS in production
   - Encrypt sensitive data

3. **Access Control**
   - Role-based permissions
   - Audit logging
   - Session management

## 🐛 Troubleshooting

### Common Issues

**"Module not found" error**
```bash
pip install -r requirements.txt --upgrade
```

**Qdrant connection failed**
- Check API key and URL
- Ensure cluster is active (free tier sleeps after 7 days)

**High memory usage**
```python
# Reduce batch size in memory_manager.py
embeddings = self.embedder.encode(contents, batch_size=16)
```

**LLM timeout errors**
- Increase timeout in llm_handler.py
- Use retry logic with exponential backoff

### Debug Mode

Enable debug logging:

```python
# In app.py
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Monitoring

### Metrics Collected
- Request latency
- Memory operations
- LLM token usage
- Error rates

### Prometheus Integration

Metrics are exposed at `/metrics` endpoint:

```bash
# Example Prometheus config
scrape_configs:
  - job_name: 'ai-memory-system'
    static_configs:
      - targets: ['localhost:8501']
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Code Style

- Use Black for formatting
- Follow PEP 8 guidelines
- Add type hints
- Write docstrings

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the amazing framework
- [Mem0](https://mem0.ai) for intelligent memory management
- [Qdrant](https://qdrant.tech) for vector search
- [PydanticAI](https://ai.pydantic.dev) for type-safe agents

## 📚 Resources

- [Documentation](docs/)
- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)

## 🗺️ Roadmap

- [ ] Multi-modal memory support (images, audio)
- [ ] Real-time collaboration features
- [ ] Mobile app
- [ ] Voice interface
- [ ] Advanced graph visualizations

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ai-memory-system/issues)

---

Built with ❤️ for the AI community