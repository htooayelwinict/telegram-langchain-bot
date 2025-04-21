"""
Vector store manager with ChromaDB integration for enhanced RAG functionality.
Handles storing and retrieving vector embeddings for improved context retrieval.
"""
import logging
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from ..config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Vector store manager with ChromaDB integration for enhanced RAG functionality.
    Handles storing and retrieving vector embeddings for improved context retrieval.
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "chat_history",
        chroma_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embeddings: Embeddings model
            persist_directory: Directory to persist ChromaDB
            collection_name: Collection name in ChromaDB
            chroma_settings: ChromaDB settings
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name
        
        # Default ChromaDB settings
        self.chroma_settings = chroma_settings or {
            "anonymized_telemetry": False,
            "allow_reset": True,
            "is_persistent": True
        }
        
        self._client = None
        self._collection = None
        self._langchain_db = None
        
        logger.info(f"Initialized vector store manager with collection '{collection_name}'")
        
        # Create persist directory if it doesn't exist
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
    
    async def initialize(self) -> bool:
        """
        Initialize the vector store manager.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize ChromaDB client
            chroma_settings = Settings(
                anonymized_telemetry=self.chroma_settings.get("anonymized_telemetry", False),
                allow_reset=self.chroma_settings.get("allow_reset", True),
                is_persistent=self.chroma_settings.get("is_persistent", True),
                persist_directory=self.persist_directory
            )
            
            # Create client
            try:
                self._client = chromadb.Client(chroma_settings)
            except Exception as client_error:
                logger.error(f"Failed to create ChromaDB client: {client_error}")
                # Try with default settings as fallback
                logger.info("Attempting to create ChromaDB client with default settings")
                self._client = chromadb.Client()
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(name=self.collection_name)
                logger.info(f"Retrieved existing collection '{self.collection_name}'")
            except ValueError:
                # Collection doesn't exist, create it
                try:
                    self._collection = self._client.create_collection(name=self.collection_name)
                    logger.info(f"Created new collection '{self.collection_name}'")
                except Exception as collection_error:
                    logger.error(f"Failed to create collection '{self.collection_name}': {collection_error}")
                    return False
            except Exception as get_error:
                logger.error(f"Error accessing collection '{self.collection_name}': {get_error}")
                return False
            
            # Initialize LangChain ChromaDB
            try:
                self._langchain_db = Chroma(
                    client=self._client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                logger.info("Vector store manager initialized successfully")
                return True
            except Exception as langchain_error:
                logger.error(f"Failed to initialize LangChain ChromaDB: {langchain_error}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing vector store manager: {e}")
            return False
    
    @property
    def client(self) -> chromadb.Client:
        """Get ChromaDB client."""
        if not self._client:
            raise ValueError("ChromaDB client not initialized")
        return self._client
    
    @property
    def collection(self) -> Collection:
        """Get ChromaDB collection."""
        if not self._collection:
            raise ValueError("ChromaDB collection not initialized")
        return self._collection
    
    @property
    def langchain_db(self) -> Chroma:
        """Get LangChain ChromaDB instance."""
        if not self._langchain_db:
            raise ValueError("LangChain ChromaDB not initialized")
        return self._langchain_db
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure vector store is initialized
            if not self._langchain_db:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return False
            
            # Add documents to vector store
            try:
                # Use LangChain's add_documents method
                self._langchain_db.add_documents(documents)
                
                logger.info(f"Added {len(documents)} documents to vector store")
                return True
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                
                # Retry with smaller batches if there's an error
                try:
                    batch_size = max(1, len(documents) // 2)
                    logger.info(f"Retrying with smaller batch size: {batch_size}")
                    
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i+batch_size]
                        self._langchain_db.add_documents(batch)
                        logger.info(f"Added batch of {len(batch)} documents to vector store")
                    
                    return True
                except Exception as retry_error:
                    logger.error(f"Error retrying document addition: {retry_error}")
                    return False
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            score_threshold: Optional similarity score threshold
            
        Returns:
            List of documents
        """
        try:
            # Ensure vector store is initialized
            if not self._langchain_db:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return []
            
            # Perform similarity search
            if score_threshold is not None:
                docs = self._langchain_db.similarity_search_with_relevance_scores(
                    query=query,
                    k=k,
                    filter=filter,
                    score_threshold=score_threshold
                )
                # Extract just the documents
                return [doc for doc, score in docs]
            else:
                docs = self._langchain_db.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
                return docs
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores in the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            score_threshold: Optional similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Ensure vector store is initialized
            if not self._langchain_db:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return []
            
            # Perform similarity search with scores
            docs_with_scores = self._langchain_db.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
                score_threshold=score_threshold
            )
            
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            return []
    
    async def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure client is initialized
            if not self._client:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return False
            
            # Delete collection
            self._client.delete_collection(name=self.collection_name)
            
            # Reset instance variables
            self._collection = None
            self._langchain_db = None
            
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Ensure collection is initialized
            if not self._collection:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return {}
            
            # Get collection count
            count = self._collection.count()
            
            # Get collection metadata
            try:
                metadata = self._collection.get()
            except Exception:
                metadata = {}
            
            return {
                "name": self.collection_name,
                "count": count,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    async def delete_by_filter(self, filter: Dict[str, Any]) -> bool:
        """
        Delete documents by filter.
        
        Args:
            filter: Metadata filter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure collection is initialized
            if not self._collection:
                if not await self.initialize():
                    logger.error("Failed to initialize vector store")
                    return False
            
            # Get IDs matching filter
            results = self._collection.get(where=filter)
            ids = results.get("ids", [])
            
            if not ids:
                logger.info(f"No documents found matching filter: {filter}")
                return True
            
            # Delete documents
            self._collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} documents matching filter: {filter}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents by filter: {e}")
            return False
    
    async def delete_by_user_id(self, user_id: str) -> bool:
        """
        Delete documents by user ID.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create filter
            filter = {"user_id": str(user_id)}
            
            # Delete documents
            return await self.delete_by_filter(filter)
        except Exception as e:
            logger.error(f"Error deleting documents by user ID: {e}")
            return False
