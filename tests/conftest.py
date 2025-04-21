"""
Pytest configuration for the Telegram-Langchain-RAG Bot tests.
"""
import pytest

# Register asyncio marker
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")
