#!/bin/bash
# Script to switch back to main app after debugging

echo "Switching back to main app..."
cat > streamlit_app.py << 'EOF'
# Streamlit Cloud entry point
# This file is used by Streamlit Cloud to identify the main app file
import app

# The app.py file runs main() automatically at the bottom
# So we just need to import it
EOF

git add streamlit_app.py
git commit -m "Switch back to main app after debugging"
git push

echo "✅ Switched back to main app and pushed to GitHub"