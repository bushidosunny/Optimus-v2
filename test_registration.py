#!/usr/bin/env python3
"""
Test script for registration functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import EnhancedAuth
from pathlib import Path
import json

def test_registration():
    """Test the registration functionality"""
    print("Testing registration functionality...")
    
    # Initialize auth system
    auth = EnhancedAuth()
    
    # Test user data
    test_username = "testuser123"
    test_email = "test@example.com"
    test_password = "TestPass123!"
    
    # Test validation functions
    print("\n1. Testing username validation...")
    is_valid, msg = auth.validate_username(test_username)
    print(f"   Username '{test_username}': {msg}")
    
    print("\n2. Testing email validation...")
    is_valid = auth.validate_email(test_email)
    print(f"   Email '{test_email}': {'Valid' if is_valid else 'Invalid'}")
    
    print("\n3. Testing password strength...")
    is_valid, msg = auth.validate_password_strength(test_password)
    print(f"   Password: {msg}")
    
    # Test registration
    print("\n4. Testing user registration...")
    success, message = auth.register_user(test_username, test_email, test_password)
    print(f"   Registration result: {message}")
    
    if success:
        # Check if user was saved
        users_file = Path("users.json")
        if users_file.exists():
            with open(users_file, 'r') as f:
                users = json.load(f)
                if test_username in users:
                    print(f"   ✅ User '{test_username}' successfully saved to file")
                    print(f"   User data: {json.dumps(users[test_username], indent=2)}")
                else:
                    print(f"   ❌ User not found in file")
        
        # Test authentication with new user
        print("\n5. Testing authentication with new user...")
        user_data = auth.authenticate_basic(test_username, test_password)
        if user_data:
            print(f"   ✅ Authentication successful!")
            print(f"   User role: {user_data['role']}")
            print(f"   Token expires: {user_data['exp']}")
        else:
            print(f"   ❌ Authentication failed")
        
        # Test duplicate registration
        print("\n6. Testing duplicate registration prevention...")
        success, message = auth.register_user(test_username, test_email, test_password)
        print(f"   Duplicate registration result: {message}")
        
        # Clean up test user
        print("\n7. Cleaning up test data...")
        if users_file.exists():
            with open(users_file, 'r') as f:
                users = json.load(f)
            
            if test_username in users:
                del users[test_username]
                
            if users:
                with open(users_file, 'w') as f:
                    json.dump(users, f, indent=2)
            else:
                users_file.unlink()
            
            print("   ✅ Test user cleaned up")
    
    print("\n✅ Registration test complete!")

if __name__ == "__main__":
    test_registration()