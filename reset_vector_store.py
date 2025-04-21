#!/usr/bin/env python3
"""
Script to reset the vector store when changing embedding models.
This is necessary when switching between models with different dimensions.
"""
import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_vector_store(vector_store_path: str) -> bool:
    """
    Reset the vector store by removing all files in the directory.
    
    Args:
        vector_store_path: Path to the vector store directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(vector_store_path)
        
        # Check if the directory exists
        if not path.exists():
            logger.info(f"Vector store directory {vector_store_path} does not exist. Creating it.")
            path.mkdir(parents=True, exist_ok=True)
            return True
            
        # Remove all files in the directory
        logger.info(f"Removing all files in {vector_store_path}")
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
        # Remove the SQLite database if it exists
        sqlite_db = path / "chroma.sqlite3"
        if sqlite_db.exists():
            logger.info(f"Removing SQLite database {sqlite_db}")
            sqlite_db.unlink()
                
        logger.info(f"Vector store at {vector_store_path} has been reset")
        return True
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        return False

if __name__ == "__main__":
    # Get the vector store path from environment or use default
    vector_store_path = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./data/vector_store")
    
    # Reset the vector store
    if reset_vector_store(vector_store_path):
        logger.info("Vector store reset successful")
    else:
        logger.error("Failed to reset vector store")
