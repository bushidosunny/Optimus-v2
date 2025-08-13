#!/usr/bin/env python
"""
Test runner for Optimus
Run all tests and generate coverage report
"""

import sys
import os
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Run all tests with coverage"""
    print("🧪 Running Optimus Tests")
    print("=" * 50)
    
    # Test arguments
    args = [
        'tests/',  # Test directory
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--color=yes',  # Colored output
        '-W', 'ignore::DeprecationWarning',  # Ignore deprecation warnings
    ]
    
    # Run tests
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"🧪 Running test: {test_file}")
    print("=" * 50)
    
    args = [
        test_file,
        '-v',
        '--tb=short',
        '--color=yes',
    ]
    
    return pytest.main(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        exit_code = run_specific_test(sys.argv[1])
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)