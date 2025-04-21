"""
Base prompt management system for loading and managing prompt templates.
Implements file-based prompt template storage and retrieval.
"""
import logging
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.config import config

logger = logging.getLogger(__name__)

# Prompt directories
SYSTEM_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "system")
USER_QUERY_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "user_query")

# Default prompts
DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant. Answer the user's questions accurately, helpfully, and responsibly.
If you don't know the answer, say so rather than making up information.
"""

DEFAULT_USER_QUERY_PROMPT = """
{query}
"""


def load_prompt_template(prompt_type: str, prompt_name: Optional[str] = "default") -> str:
    """
    Load a prompt template from file.
    
    Args:
        prompt_type: Type of prompt (system, user_query)
        prompt_name: Name of prompt template
        
    Returns:
        Prompt template string
    """
    try:
        # Determine prompt directory
        if prompt_type == "system":
            prompt_dir = SYSTEM_PROMPT_DIR
            default_prompt = DEFAULT_SYSTEM_PROMPT
        elif prompt_type == "user_query":
            prompt_dir = USER_QUERY_PROMPT_DIR
            default_prompt = DEFAULT_USER_QUERY_PROMPT
        else:
            logger.error(f"Unknown prompt type: {prompt_type}")
            return default_prompt
        
        # Create prompt directory if it doesn't exist
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Build prompt file path
        prompt_file = os.path.join(prompt_dir, f"{prompt_name}.txt")
        
        # Check if prompt file exists
        if not os.path.exists(prompt_file):
            # Create default prompt file if it doesn't exist
            with open(prompt_file, "w") as f:
                f.write(default_prompt)
            
            logger.info(f"Created default {prompt_type} prompt file: {prompt_file}")
        
        # Load prompt from file
        with open(prompt_file, "r") as f:
            prompt = f.read()
        
        return prompt
    except Exception as e:
        logger.error(f"Error loading {prompt_type} prompt template: {e}")
        
        # Return default prompt
        if prompt_type == "system":
            return DEFAULT_SYSTEM_PROMPT
        elif prompt_type == "user_query":
            return DEFAULT_USER_QUERY_PROMPT
        else:
            return ""


def save_prompt_template(prompt_type: str, prompt_content: str, prompt_name: str = "default") -> bool:
    """
    Save a prompt template to file.
    
    Args:
        prompt_type: Type of prompt (system, user_query)
        prompt_content: Prompt template content
        prompt_name: Name of prompt template
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine prompt directory
        if prompt_type == "system":
            prompt_dir = SYSTEM_PROMPT_DIR
        elif prompt_type == "user_query":
            prompt_dir = USER_QUERY_PROMPT_DIR
        else:
            logger.error(f"Unknown prompt type: {prompt_type}")
            return False
        
        # Create prompt directory if it doesn't exist
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Build prompt file path
        prompt_file = os.path.join(prompt_dir, f"{prompt_name}.txt")
        
        # Save prompt to file
        with open(prompt_file, "w") as f:
            f.write(prompt_content)
        
        logger.info(f"Saved {prompt_type} prompt template: {prompt_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving {prompt_type} prompt template: {e}")
        return False


def list_prompt_templates(prompt_type: str) -> List[str]:
    """
    List available prompt templates.
    
    Args:
        prompt_type: Type of prompt (system, user_query)
        
    Returns:
        List of prompt template names
    """
    try:
        # Determine prompt directory
        if prompt_type == "system":
            prompt_dir = SYSTEM_PROMPT_DIR
        elif prompt_type == "user_query":
            prompt_dir = USER_QUERY_PROMPT_DIR
        else:
            logger.error(f"Unknown prompt type: {prompt_type}")
            return []
        
        # Create prompt directory if it doesn't exist
        os.makedirs(prompt_dir, exist_ok=True)
        
        # List prompt files
        prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith(".txt")]
        
        # Extract prompt names
        prompt_names = [os.path.splitext(f)[0] for f in prompt_files]
        
        return prompt_names
    except Exception as e:
        logger.error(f"Error listing {prompt_type} prompt templates: {e}")
        return []


def delete_prompt_template(prompt_type: str, prompt_name: str) -> bool:
    """
    Delete a prompt template.
    
    Args:
        prompt_type: Type of prompt (system, user_query)
        prompt_name: Name of prompt template
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Don't allow deleting default prompt
        if prompt_name == "default":
            logger.error("Cannot delete default prompt template")
            return False
        
        # Determine prompt directory
        if prompt_type == "system":
            prompt_dir = SYSTEM_PROMPT_DIR
        elif prompt_type == "user_query":
            prompt_dir = USER_QUERY_PROMPT_DIR
        else:
            logger.error(f"Unknown prompt type: {prompt_type}")
            return False
        
        # Build prompt file path
        prompt_file = os.path.join(prompt_dir, f"{prompt_name}.txt")
        
        # Check if prompt file exists
        if not os.path.exists(prompt_file):
            logger.error(f"Prompt template not found: {prompt_file}")
            return False
        
        # Delete prompt file
        os.remove(prompt_file)
        
        logger.info(f"Deleted {prompt_type} prompt template: {prompt_file}")
        return True
    except Exception as e:
        logger.error(f"Error deleting {prompt_type} prompt template: {e}")
        return False
