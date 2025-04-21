"""Database initialization utilities for async PostgreSQL operations."""

import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import config
from app.db.models import Base

logger = logging.getLogger(__name__)

# Create async engine
async_engine = create_async_engine(
    config.DATABASE_URL,
    echo=config.DEBUG,
    future=True,
    poolclass=NullPool  # Disable connection pooling for better async support
)

# Create async session factory
async_session_factory = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession
)

async def get_session():
    """Get a new async database session."""
    async with async_session_factory() as session:
        yield session

async def initialize():
    """Initialize the database with all tables.
    
    This will create tables if they don't exist, preserving existing data.
    """
    try:
        # Create all tables if they don't exist
        # This approach preserves existing data
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
