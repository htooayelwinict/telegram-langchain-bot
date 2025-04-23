"""
Embeddings utility for OpenAI embeddings.
Provides a wrapper for OpenAI embeddings with enhanced error handling.
"""
import logging
from typing import List, Optional

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

from ..config import config

logger = logging.getLogger(__name__)


class EnhancedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Enhanced OpenAI embeddings with better error handling and retry logic.
    """
    
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize the enhanced OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model name
            **kwargs: Additional arguments to pass to OpenAIEmbeddings
        """
        model = model or config.OPENAI_EMBEDDING_MODEL
        openai_api_key = kwargs.pop("openai_api_key", config.OPENAI_API_KEY)
        
        # Explicitly set tiktoken_model for text-embedding-3 models
        if model.startswith("text-embedding-3"):
            # Use model_kwargs instead of direct tiktoken_model parameter
            if "model_kwargs" not in kwargs:
                kwargs["model_kwargs"] = {}
            kwargs["model_kwargs"]["tiktoken_model"] = "cl100k_base"
        
        super().__init__(
            model=model,
            openai_api_key=openai_api_key,
            **kwargs
        )
        
        logger.info(f"Initialized enhanced OpenAI embeddings with model '{model}'")

    def _get_len_safe_embeddings(self, texts, *, engine=None, chunk_size=0):
        """Override to handle tiktoken encoding better"""
        try:
            # Pre-load the cl100k_base encoding to avoid warnings
            import tiktoken
            tiktoken.get_encoding("cl100k_base")
            return super()._get_len_safe_embeddings(texts, engine=engine, chunk_size=chunk_size)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise


def get_embeddings() -> Embeddings:
    """
    Get embeddings model instance.
    
    Returns:
        Embeddings model
    """
    try:
        return EnhancedOpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Error creating embeddings model: {e}")
        raise
