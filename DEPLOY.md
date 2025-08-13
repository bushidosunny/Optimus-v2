# Deployment Guide for Streamlit Cloud

## Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. Qdrant Cloud account (free tier available at [cloud.qdrant.io](https://cloud.qdrant.io))

## Step 1: Prepare Your Repository

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Optimus v2"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-memory-system.git
   git push -u origin main
   ```

2. **Required Files** (already created):
   - `app.py` or `streamlit_app.py` - Main application
   - `requirements.txt` - Python dependencies
   - `.streamlit/config.toml` - App configuration
   - `.streamlit/secrets.toml.example` - Secrets template

## Step 2: Set Up Qdrant Cloud

1. **Create Qdrant Cloud Cluster**:
   - Go to [cloud.qdrant.io](https://cloud.qdrant.io)
   - Create a free cluster
   - Note your cluster URL and API key

## Step 3: Deploy to Streamlit Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with GitHub

3. **Click "New app"** button

4. **Configure Deployment**:
   - Repository: `YOUR_USERNAME/ai-memory-system`
   - Branch: `main`
   - Main file path: `app.py` or `streamlit_app.py`
   - App URL: Choose a subdomain (e.g., `optimus-v2`)

5. **Add Secrets**:
   - Click on "Advanced settings" before deploying
   - OR after deployment: App settings → Secrets
   - Copy contents from `.streamlit/secrets.toml.example`
   - Fill in your actual API keys:
     ```toml
     # Required
     QDRANT_URL = "https://your-cluster.qdrant.io"
     QDRANT_API_KEY = "your-api-key"
     OPENAI_API_KEY = "sk-..."
     
     # Authentication
     JWT_SECRET_KEY = "generate-random-key"
     ADMIN_USERNAME = "admin"
     ADMIN_PASSWORD = "strong-password"
     ```

6. **Click "Deploy"**

## Step 4: Post-Deployment

1. **Monitor Logs**: Watch the deployment logs for any errors

2. **Test the App**:
   - Visit your app at `https://YOUR_APP_NAME.streamlit.app`
   - Log in with admin credentials
   - Test chat and memory features

3. **Update App** (automatic):
   - Push changes to GitHub
   - Streamlit Cloud auto-deploys on push

## Environment Variables Reference

### Required:
- `QDRANT_URL`: Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Qdrant Cloud API key
- `OPENAI_API_KEY`: For embeddings and Mem0 extraction

### Optional LLM Providers:
- `ANTHROPIC_API_KEY`: Claude models
- `GOOGLE_API_KEY`: Gemini models
- `GROQ_API_KEY`: Fast inference models
- `MISTRAL_API_KEY`: Mistral models
- `XAI_API_KEY`: Grok models

### Security:
- `JWT_SECRET_KEY`: Random secret for JWT tokens
- `ADMIN_USERNAME`: Admin login username
- `ADMIN_PASSWORD`: Admin login password

## Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check for typos in import statements

2. **Qdrant connection errors**:
   - Verify QDRANT_URL format (include https://)
   - Check API key is correct
   - Ensure cluster is running

3. **Memory/Resource limits**:
   - Streamlit Cloud free tier has resource limits
   - Optimize memory usage with caching
   - Consider upgrading for production use

4. **Secrets not loading**:
   - Ensure secrets are properly formatted (TOML)
   - No quotes around boolean values
   - Check for special characters in strings

## Local Development vs Cloud

### Differences:
- **Secrets**: Local uses `.streamlit/secrets.toml`, Cloud uses web interface
- **Resources**: Cloud has memory/CPU limits on free tier
- **Persistence**: Cloud apps may restart, use external storage (Qdrant)
- **URLs**: Cloud provides public HTTPS URLs

### Best Practices:
1. Test locally first
2. Use environment detection for cloud-specific features
3. Implement proper error handling
4. Use st.cache_resource for expensive operations
5. Monitor resource usage in Cloud dashboard

## Support

- **Streamlit Cloud Docs**: [docs.streamlit.io/deploy](https://docs.streamlit.io/deploy)
- **Qdrant Docs**: [qdrant.tech/documentation](https://qdrant.tech/documentation)
- **Issues**: Report at GitHub repository

## Next Steps

After deployment:
1. Share your app URL with team members
2. Set up monitoring (optional)
3. Configure custom domain (Streamlit Cloud Teams/Enterprise)
4. Implement backup strategy for memories
5. Set up CI/CD pipeline (GitHub Actions)