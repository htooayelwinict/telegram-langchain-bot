"""
Module to fix the tiktoken encoding warning.
This module is imported at application startup to ensure proper encoding setup.
"""
import logging
import tiktoken
import importlib.util

logger = logging.getLogger(__name__)

def fix_tiktoken_warning():
    """
    Pre-load the cl100k_base encoding to avoid the warning:
    'Warning: model not found. Using cl100k_base encoding.'
    
    This is needed for text-embedding-3-small and text-embedding-3-large models.
    """
    try:
        # First, ensure tiktoken is properly installed
        if importlib.util.find_spec("tiktoken") is None:
            logger.warning("tiktoken package not found. Please install it with: pip install tiktoken")
            return False
            
        # Pre-register the cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(f"Successfully pre-loaded tiktoken encoding: cl100k_base")
        return True
    except Exception as e:
        logger.error(f"Error initializing tiktoken encoding: {e}")
        return False

# Initialize when module is imported
fix_tiktoken_warning()
