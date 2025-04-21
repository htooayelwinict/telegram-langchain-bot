#!/usr/bin/env python3
"""
Test runner for the Telegram-Langchain-RAG Bot.
"""
import sys
import os
import pytest

if __name__ == "__main__":
    # Add the project root to the path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run tests with pytest
    exit_code = pytest.main(['-xvs', 'tests'])
    
    # Exit with non-zero code if tests failed
    sys.exit(exit_code)
