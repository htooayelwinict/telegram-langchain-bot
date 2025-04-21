"""
Enhanced SQL models for PostgreSQL database.
Defines the database schema for users and chat history with direct vectorization support.
"""
import logging
from typing import List, Optional

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func, select
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import relationship

from ..config import config

logger = logging.getLogger(__name__)

# Create async engine with improved configuration
logger.info(f"Connecting to PostgreSQL database at: {config.DATABASE_URL}")
async_engine = create_async_engine(
    config.DATABASE_URL,
    echo=config.DEBUG,
    future=True,
    pool_size=5,
    max_overflow=10
)

# Create async session factory
async_session_factory = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_async_session():
    """Get the async session factory."""
    return async_session_factory
        
async def get_or_create_user(session: AsyncSession, user_id: str, username: Optional[str] = None,
                            first_name: Optional[str] = None, last_name: Optional[str] = None) -> 'User':
    """Get an existing user or create a new one."""
    result = await session.execute(select(User).where(User.user_id == user_id))
    user = result.scalars().first()

    if user is None:
        user = User(
            user_id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user

class User(AsyncAttrs, Base):
    """User model for storing Telegram user data."""
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    language_code = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"

class ChatHistory(AsyncAttrs, Base):
    """Chat history model for storing messages with enhanced metadata for vectorization."""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    message_metadata = Column(JSONB, default={})
    
    def __repr__(self):
        return f"<ChatHistory(id={self.id}, user_id='{self.user_id}', role='{self.role}')>"
