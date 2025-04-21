"""
Enhanced RAG retriever with direct vectorization support.
Implements improved context handling and retrieval for better responses.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from langchain.schema import Document, BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import config
from ..db.vector_store import VectorStoreManager

settings = config
logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Enhanced RAG retriever with direct vectorization support.
    Implements improved context handling and retrieval for better responses.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        embeddings: Embeddings,
        collection_name: str = "chat_history",
        k: int = 5,
        score_threshold: float = 0.7,
        max_tokens_per_doc: int = 1000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store_manager: Vector store manager
            embeddings: Embeddings model
            collection_name: Collection name in the vector store
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            max_tokens_per_doc: Maximum tokens per document
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.vector_store_manager = vector_store_manager
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.k = k
        self.score_threshold = score_threshold
        self.max_tokens_per_doc = max_tokens_per_doc
        
        # Text splitter for chunking long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(f"Initialized RAG retriever with collection '{collection_name}'")
    
    async def initialize(self) -> bool:
        """
        Initialize the RAG retriever.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure vector store is initialized
            if not await self.vector_store_manager.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            logger.info("RAG retriever initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing RAG retriever: {e}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to langchain Document objects
            langchain_docs = []
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # Skip empty documents
                if not content.strip():
                    continue
                
                # Split long documents if needed
                if len(content) > self.max_tokens_per_doc * 4:  # Approximate chars per token
                    chunks = self.text_splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk"] = i
                        chunk_metadata["is_chunk"] = True
                        langchain_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
                else:
                    langchain_docs.append(Document(page_content=content, metadata=metadata))
            
            # Skip if no valid documents
            if not langchain_docs:
                logger.warning("No valid documents to add")
                return True
            
            # Add to vector store
            result = await self.vector_store_manager.add_documents(langchain_docs)
            
            if result:
                logger.info(f"Added {len(langchain_docs)} documents to vector store")
            else:
                logger.error("Failed to add documents to vector store")
            
            return result
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def retrieve_context(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query string
            user_id: Optional user ID to filter results
            filter_metadata: Optional metadata filters
            k: Optional number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Apply user_id filter if provided
            metadata_filter = filter_metadata or {}
            if user_id:
                metadata_filter["user_id"] = user_id
            
            # Use provided k or default
            k_value = k or self.k
            
            # Retrieve documents from vector store
            docs = await self.vector_store_manager.similarity_search(
                query=query,
                k=k_value,
                filter=metadata_filter if metadata_filter else None,
                score_threshold=self.score_threshold
            )
            
            if not docs:
                logger.info(f"No relevant documents found for query: {query}")
                return []
            
            logger.info(f"Retrieved {len(docs)} relevant documents for query")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def retrieve_context_with_scores(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant context with similarity scores.
        
        Args:
            query: Query string
            user_id: Optional user ID to filter results
            filter_metadata: Optional metadata filters
            k: Optional number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Apply user_id filter if provided
            metadata_filter = filter_metadata or {}
            if user_id:
                metadata_filter["user_id"] = user_id
            
            # Use provided k or default
            k_value = k or self.k
            
            # Retrieve documents with scores from vector store
            docs_with_scores = await self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=k_value,
                filter=metadata_filter if metadata_filter else None,
                score_threshold=self.score_threshold
            )
            
            if not docs_with_scores:
                logger.info(f"No relevant documents found for query: {query}")
                return []
            
            logger.info(f"Retrieved {len(docs_with_scores)} relevant documents with scores for query")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving context with scores: {e}")
            return []
    
    async def format_context_for_prompt(
        self, 
        documents: List[Document],
        max_tokens: int = 2000
    ) -> str:
        """
        Format retrieved documents into context for the prompt.
        
        Args:
            documents: List of documents
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        # Sort documents by metadata timestamp if available
        try:
            sorted_docs = sorted(
                documents,
                key=lambda d: d.metadata.get("timestamp", 0),
                reverse=False  # Oldest first
            )
        except Exception:
            sorted_docs = documents
        
        # Format each document
        context_parts = []
        total_length = 0
        
        for doc in sorted_docs:
            # Format metadata
            metadata = doc.metadata
            timestamp = metadata.get("timestamp", "")
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            
            role = metadata.get("role", "")
            user_id = metadata.get("user_id", "")
            
            # Format document
            header = ""
            if role and timestamp:
                header = f"[{timestamp}] {role.upper()}"
            elif role:
                header = f"{role.upper()}"
            elif timestamp:
                header = f"[{timestamp}]"
            
            # Add formatted document
            content = doc.page_content.strip()
            if header:
                formatted_doc = f"{header}:\n{content}\n"
            else:
                formatted_doc = f"{content}\n"
            
            # Check if adding this document would exceed max tokens
            # Approximate token count as 4 chars per token
            doc_length = len(formatted_doc) // 4
            if total_length + doc_length > max_tokens and context_parts:
                # Skip this document if it would exceed max tokens
                continue
            
            context_parts.append(formatted_doc)
            total_length += doc_length
        
        # Join all parts
        context = "\n".join(context_parts)
        
        return context
    
    async def vectorize_message(
        self,
        message: Dict[str, Any]
    ) -> bool:
        """
        Vectorize a message directly to the vector store.
        
        Args:
            message: Message dictionary with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract content and metadata
            content = message.get("content", "")
            if not content.strip():
                logger.warning("Empty message content, skipping vectorization")
                return True
            
            # Prepare metadata
            metadata = {
                "user_id": str(message.get("user_id", "")),
                "role": message.get("role", ""),
                "timestamp": message.get("created_at", datetime.now().timestamp()),
                "message_id": str(message.get("id", "")),
                "is_vectorized": True
            }
            
            # Add any additional metadata from message
            message_metadata = message.get("message_metadata", {})
            if message_metadata and isinstance(message_metadata, dict):
                metadata.update(message_metadata)
            
            # Create document
            document = {
                "content": content,
                "metadata": metadata
            }
            
            # Add to vector store
            result = await self.add_documents([document])
            
            if result:
                logger.info(f"Vectorized message with ID {metadata.get('message_id')}")
            else:
                logger.error(f"Failed to vectorize message with ID {metadata.get('message_id')}")
            
            return result
        except Exception as e:
            logger.error(f"Error vectorizing message: {e}")
            return False
    
    async def vectorize_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Vectorize multiple messages directly to the vector store.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert messages to documents
            documents = []
            for message in messages:
                content = message.get("content", "")
                if not content.strip():
                    continue
                
                # Prepare metadata
                metadata = {
                    "user_id": str(message.get("user_id", "")),
                    "role": message.get("role", ""),
                    "timestamp": message.get("created_at", datetime.now().timestamp()),
                    "message_id": str(message.get("id", "")),
                    "is_vectorized": True
                }
                
                # Add any additional metadata from message
                message_metadata = message.get("message_metadata", {})
                if message_metadata and isinstance(message_metadata, dict):
                    metadata.update(message_metadata)
                
                # Create document
                document = {
                    "content": content,
                    "metadata": metadata
                }
                
                documents.append(document)
            
            # Skip if no valid documents
            if not documents:
                logger.warning("No valid messages to vectorize")
                return True
            
            # Add to vector store
            result = await self.add_documents(documents)
            
            if result:
                logger.info(f"Vectorized {len(documents)} messages")
            else:
                logger.error(f"Failed to vectorize messages")
            
            return result
        except Exception as e:
            logger.error(f"Error vectorizing messages: {e}")
            return False
