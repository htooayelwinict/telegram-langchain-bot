"""
Patch for tiktoken to support newer OpenAI models.
This module adds newer models to tiktoken's MODEL_TO_ENCODING mapping.
"""
import logging
import importlib.util

logger = logging.getLogger(__name__)

def patch_tiktoken():
    """
    Patch tiktoken to support newer OpenAI models like text-embedding-3-small.
    This avoids the warning: 'Warning: model not found. Using cl100k_base encoding.'
    """
    try:
        # Check if tiktoken is installed
        if importlib.util.find_spec("tiktoken") is None:
            logger.warning("tiktoken package not found. Please install it with: pip install tiktoken")
            return False
            
        # Import tiktoken and patch the MODEL_TO_ENCODING dictionary
        import tiktoken
        
        # Add newer models to the MODEL_TO_ENCODING dictionary
        new_models = {
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-3-large": "cl100k_base",
            "gpt-4.1-mini": "cl100k_base",
            "gpt-4o-mini-search-preview-2025-03-11": "cl100k_base"
        }
        
        # Check if models are already in the dictionary
        for model, encoding in new_models.items():
            if model not in tiktoken.model.MODEL_TO_ENCODING:
                tiktoken.model.MODEL_TO_ENCODING[model] = encoding
                logger.info(f"Added model {model} to tiktoken MODEL_TO_ENCODING with encoding {encoding}")
        
        # Verify the patch worked
        for model in new_models:
            if model in tiktoken.model.MODEL_TO_ENCODING:
                logger.info(f"Model {model} is now in tiktoken MODEL_TO_ENCODING")
            else:
                logger.warning(f"Failed to add model {model} to tiktoken MODEL_TO_ENCODING")
                
        return True
    except Exception as e:
        logger.error(f"Error patching tiktoken: {e}")
        return False
