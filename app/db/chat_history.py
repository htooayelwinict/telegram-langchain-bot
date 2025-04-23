"""
Enhanced chat history manager with direct vectorization support.
Handles storing and retrieving chat messages with PostgreSQL integration.
"""
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from sqlalchemy import select, desc, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from ..config import config
from .models import User, ChatHistory, get_async_session, get_or_create_user
from app.rag.retriever import RAGRetriever
logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Enhanced chat history manager with direct vectorization support.
    Handles storing and retrieving chat messages with PostgreSQL integration.
    """
    
    def __init__(
        self,
        session_factory: Optional[sessionmaker] = None,
        rag_retriever: Optional[RAGRetriever] = None
    ):
        """
        Initialize the chat history manager.
        
        Args:
            session_factory: SQLAlchemy session factory
            rag_retriever: Optional RAG retriever for direct vectorization
        """
        self.session_factory = session_factory
        self.rag_retriever = rag_retriever
        
        logger.info("Initialized chat history manager")
    
    async def initialize(self) -> bool:
        """
        Initialize the chat history manager.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create session factory if not provided
            if not self.session_factory:
                self.session_factory = get_async_session()
            
            # Test database connection
            async with self.session_factory() as session:
                await session.execute(select(func.now()))
            
            logger.info("Chat history manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing chat history manager: {e}")
            return False
    
    async def add_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> Optional[User]:
        """
        Add a new user or update existing user.
        
        Args:
            user_id: User ID
            username: Username
            first_name: First name
            last_name: Last name
            
        Returns:
            User object if successful, None otherwise
        """
        try:
            # Ensure session factory is initialized
            if not self.session_factory:
                if not await self.initialize():
                    logger.error("Failed to initialize session factory")
                    return None
            
            async with self.session_factory() as session:
                # Check if user exists
                result = await session.execute(
                    select(User).where(User.user_id == str(user_id))
                )
                user = result.scalars().first()
                
                if user:
                    # Update existing user
                    if username:
                        user.username = username
                    if first_name:
                        user.first_name = first_name
                    if last_name:
                        user.last_name = last_name
                    
                    user.updated_at = datetime.now()
                else:
                    # Create new user
                    user = User(
                        user_id=str(user_id),
                        username=username,
                        first_name=first_name,
                        last_name=last_name
                    )
                    session.add(user)
                
                await session.commit()
                
                logger.info(f"User {user_id} {'updated' if user else 'added'}")
                return user
        except Exception as e:
            logger.error(f"Error adding/updating user {user_id}: {e}")
            return None
    
    async def add_message(
        self,
        user_id: str,
        content: str,
        role: str,
        message_metadata: Optional[Dict[str, Any]] = None,
        vectorize: bool = True
    ) -> Optional[ChatHistory]:
        """
        Add a new message to chat history with direct vectorization.
        
        Args:
            user_id: User ID
            content: Message content
            role: Message role (user/assistant/system)
            message_metadata: Optional message metadata
            vectorize: Whether to vectorize the message
            
        Returns:
            ChatHistory object if successful, None otherwise
        """
        try:
            # Ensure session factory is initialized
            if not self.session_factory:
                if not await self.initialize():
                    logger.error("Failed to initialize session factory")
                    return None
            
            # Ensure user exists
            user = await self.add_user(user_id)
            if not user:
                logger.error(f"Failed to add/update user {user_id}")
                return None
            
            # Create message
            message = ChatHistory(
                user_id=str(user_id),
                content=content,
                role=role,
                message_metadata=message_metadata or {},
                created_at=datetime.now()
            )
            
            # Save to database
            async with self.session_factory() as session:
                session.add(message)
                await session.commit()
                await session.refresh(message)
            
            # Vectorize message if enabled
            if vectorize and self.rag_retriever:
                message_dict = {
                    "id": str(message.id),
                    "user_id": str(message.user_id),
                    "content": message.content,
                    "role": message.role,
                    "created_at": message.created_at.timestamp() if message.created_at else datetime.now().timestamp(),
                    "message_metadata": message.message_metadata
                }
                
                # Run vectorization in background to avoid blocking
                asyncio.create_task(self.rag_retriever.vectorize_message(message_dict))
            
            logger.info(f"Added message from user {user_id} with role {role}")
            return message
        except Exception as e:
            logger.error(f"Error adding message for user {user_id}: {e}")
            return None
    
    async def get_recent_history(
        self,
        user_id: str,
        limit: int = 50,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent chat history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of messages to retrieve
            hours: Optional time window in hours
            
        Returns:
            List of chat history messages
        """
        try:
            # Ensure session factory is initialized
            if not self.session_factory:
                if not await self.initialize():
                    logger.error("Failed to initialize session factory")
                    return []
            
            async with self.session_factory() as session:
                # Build query
                query = select(ChatHistory).where(ChatHistory.user_id == str(user_id))
                
                # Add time filter if specified
                if hours:
                    time_threshold = datetime.now() - timedelta(hours=hours)
                    query = query.where(ChatHistory.created_at >= time_threshold)
                
                # Order by creation time and limit results
                query = query.order_by(desc(ChatHistory.created_at)).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                messages = result.scalars().all()
                
                # Convert to dictionaries and reverse order (oldest first)
                history = []
                for message in reversed(messages):
                    history.append({
                        "id": str(message.id),
                        "user_id": str(message.user_id),
                        "content": message.content,
                        "role": message.role,
                        "created_at": message.created_at.timestamp() if message.created_at else None,
                        "message_metadata": message.message_metadata
                    })
                
                logger.info(f"Retrieved {len(history)} recent messages (requested limit: {limit}) for user {user_id}")
                return history
        except Exception as e:
            logger.error(f"Error getting recent history for user {user_id}: {e}")
            return []
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 20,
        include_system: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in a format suitable for LLM context.
        
        Args:
            user_id: User ID
            limit: Maximum number of messages to retrieve
            include_system: Whether to include system messages
            
        Returns:
            List of chat history messages in LLM format
        """
        try:
            # Get recent history
            history = await self.get_recent_history(user_id, limit=limit)
            
            # Convert to LLM format
            conversation = []
            for message in history:
                role = message.get("role", "")
                
                # Skip system messages if not included
                if role == "system" and not include_system:
                    continue
                
                # Add to conversation
                conversation.append({
                    "role": role,
                    "content": message.get("content", "")
                })
            
            return conversation
        except Exception as e:
            logger.error(f"Error getting conversation history for user {user_id}: {e}")
            return []
    
    async def get_relevant_history(
        self,
        user_id: str,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant chat history for a query using RAG.
        
        Args:
            user_id: User ID
            query: Query string
            k: Number of relevant messages to retrieve
            
        Returns:
            List of relevant chat history messages
        """
        try:
            # Ensure RAG retriever is available
            if not self.rag_retriever:
                logger.error("RAG retriever not available")
                return []
            
            # Retrieve relevant context
            filter_metadata = {"user_id": str(user_id)}
            docs = await self.rag_retriever.retrieve_context(
                query=query,
                user_id=str(user_id),
                filter_metadata=filter_metadata,
                k=k
            )
            
            # Convert to chat history format
            relevant_history = []
            for doc in docs:
                metadata = doc.metadata
                relevant_history.append({
                    "id": metadata.get("message_id", ""),
                    "user_id": metadata.get("user_id", ""),
                    "content": doc.page_content,
                    "role": metadata.get("role", ""),
                    "created_at": metadata.get("timestamp", 0),
                    "message_metadata": {k: v for k, v in metadata.items() 
                                       if k not in ["message_id", "user_id", "role", "timestamp"]}
                })
            
            # Sort by timestamp
            relevant_history.sort(key=lambda x: x.get("created_at", 0))
            
            logger.info(f"Retrieved {len(relevant_history)} relevant messages for user {user_id}")
            return relevant_history
        except Exception as e:
            logger.error(f"Error getting relevant history for user {user_id}: {e}")
            return []
    
    async def vectorize_user_history(
        self,
        user_id: str,
        limit: Optional[int] = None,
        hours: Optional[int] = None
    ) -> bool:
        """
        Vectorize chat history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of messages to vectorize
            hours: Optional time window in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure RAG retriever is available
            if not self.rag_retriever:
                logger.error("RAG retriever not available")
                return False
            
            # Ensure session factory is initialized
            if not self.session_factory:
                if not await self.initialize():
                    logger.error("Failed to initialize session factory")
                    return False
            
            async with self.session_factory() as session:
                # Build query
                query = select(ChatHistory).where(ChatHistory.user_id == str(user_id))
                
                # Add time filter if specified
                if hours:
                    time_threshold = datetime.now() - timedelta(hours=hours)
                    query = query.where(ChatHistory.created_at >= time_threshold)
                
                # Order by creation time and limit results
                query = query.order_by(desc(ChatHistory.created_at))
                if limit:
                    query = query.limit(limit)
                
                # Execute query
                result = await session.execute(query)
                messages = result.scalars().all()
                
                # Convert to dictionaries
                message_dicts = []
                for message in messages:
                    message_dicts.append({
                        "id": str(message.id),
                        "user_id": str(message.user_id),
                        "content": message.content,
                        "role": message.role,
                        "created_at": message.created_at.timestamp() if message.created_at else datetime.now().timestamp(),
                        "message_metadata": message.message_metadata
                    })
                
                # Vectorize messages
                if message_dicts:
                    result = await self.rag_retriever.vectorize_messages(message_dicts)
                    
                    if result:
                        logger.info(f"Vectorized {len(message_dicts)} messages for user {user_id}")
                    else:
                        logger.error(f"Failed to vectorize messages for user {user_id}")
                    
                    return result
                else:
                    logger.info(f"No messages to vectorize for user {user_id}")
                    return True
        except Exception as e:
            logger.error(f"Error vectorizing history for user {user_id}: {e}")
            return False
    
    async def delete_user_history(
        self,
        user_id: str,
        hours: Optional[int] = None
    ) -> bool:
        """
        Delete chat history for a user.
        
        Args:
            user_id: User ID
            hours: Optional time window in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure session factory is initialized
            if not self.session_factory:
                if not await self.initialize():
                    logger.error("Failed to initialize session factory")
                    return False
            
            async with self.session_factory() as session:
                # Build query
                query = select(ChatHistory).where(ChatHistory.user_id == str(user_id))
                
                # Add time filter if specified
                if hours:
                    time_threshold = datetime.now() - timedelta(hours=hours)
                    query = query.where(ChatHistory.created_at >= time_threshold)
                
                # Execute query
                result = await session.execute(query)
                messages = result.scalars().all()
                
                # Delete messages
                for message in messages:
                    await session.delete(message)
                
                await session.commit()
                
                logger.info(f"Deleted {len(messages)} messages for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting history for user {user_id}: {e}")
            return False
