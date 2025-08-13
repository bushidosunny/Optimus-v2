#!/usr/bin/env python3
"""
Utility to clear rate limiting for a user
"""

import streamlit as st
import sys

if len(sys.argv) > 1:
    username = sys.argv[1]
    print(f"Clearing rate limit for user: {username}")
    
    # Clear from session state if running in Streamlit context
    if 'failed_attempts' in st.session_state:
        if username in st.session_state.failed_attempts:
            del st.session_state.failed_attempts[username]
            print(f"✅ Rate limit cleared for {username}")
        else:
            print(f"No rate limit found for {username}")
    else:
        print("Note: This should be run while the Streamlit app is running")
        print("Alternatively, restart the Streamlit app to clear all rate limits")
else:
    print("Usage: python clear_rate_limit.py <username>")
    print("\nTo clear all rate limits, simply restart the Streamlit app")