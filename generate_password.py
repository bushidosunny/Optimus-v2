#!/usr/bin/env python3
"""
Generate bcrypt password hashes for Optimus users
"""

import bcrypt
import sys

def generate_password_hash(password: str) -> str:
    """Generate a bcrypt hash for a password"""
    salt = bcrypt.gensalt(rounds=12)
    hash_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hash_bytes.decode('utf-8')

def main():
    print("🔐 Optimus - Password Hash Generator")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Use command line argument
        password = sys.argv[1]
    else:
        # Interactive mode
        password = input("Enter password to hash: ")
        if not password:
            print("❌ Password cannot be empty")
            return
    
    # Generate hash
    password_hash = generate_password_hash(password)
    
    print("\n✅ Password hash generated successfully!")
    print("\nAdd this to your .streamlit/secrets.toml:")
    print("-" * 50)
    print(f'password_hash = "{password_hash}"')
    print("-" * 50)
    
    # Example configuration
    print("\nExample user configuration:")
    print("""
[[auth.users]]
username = "your_username"
email = "your_email@example.com"
password_hash = "{}"
role = "user"  # or "admin"
""".format(password_hash))

if __name__ == "__main__":
    main()