"""
Directory manager utility to ensure all necessary directories exist.
"""
import os
import logging
from typing import List, Optional

from ..config import config

logger = logging.getLogger(__name__)

def ensure_directories_exist(additional_dirs: Optional[List[str]] = None) -> bool:
    """
    Ensure all necessary directories exist.
    
    Args:
        additional_dirs: Optional list of additional directories to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Default directories to ensure
        default_dirs = [
            config.CHROMA_PERSIST_DIRECTORY,  # Vector store directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "langchain", "prompts", "templates"),  # Prompt templates
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),  # General data directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"),  # Logs directory
        ]
        
        # Combine with additional directories
        dirs_to_create = default_dirs
        if additional_dirs:
            dirs_to_create.extend(additional_dirs)
        
        # Create each directory
        for directory in dirs_to_create:
            if directory:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring directories exist: {e}")
        return False
