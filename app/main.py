"""
Main entry point for the Telegram bot application.
Implements proper component initialization and lifecycle management.
"""
import logging
import asyncio
import os
import sys
from typing import Dict, Any, Optional

from langchain.chat_models import ChatOpenAI
from redis.asyncio import Redis

from app.config import config
from app.db.init_db import initialize as initialize_db
from app.db.chat_history import ChatHistoryManager
from app.db.vector_store import VectorStoreManager
# Import tiktoken patch to fix encoding warnings
from app.rag.tiktoken_patch import patch_tiktoken
from app.rag.embeddings import get_embeddings
from app.rag.retriever import RAGRetriever
from app.langchain.chain import LangchainManager
from app.utils.rate_limiter import RedisRateLimiter, RateLimitConfig
from app.utils.directory_manager import ensure_directories_exist
from app.bot.telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("telegram_bot.log")
    ]
)

logger = logging.getLogger(__name__)

async def initialize_components():
    """
    Initialize all components of the application.
    
    Returns:
        Dictionary with initialized components
    """
    try:
        logger.info("Initializing application components")
        
        # Ensure all necessary directories exist
        logger.info("Ensuring all necessary directories exist")
        if not ensure_directories_exist():
            logger.error("Failed to ensure all directories exist")
            return None
            
        # Apply tiktoken patch to fix encoding warnings
        logger.info("Applying tiktoken patch for newer models")
        if patch_tiktoken():
            logger.info("Successfully patched tiktoken for newer models")
        else:
            logger.warning("Failed to patch tiktoken, encoding warnings may occur")
        
        # Initialize database
        logger.info("Initializing database")
        await initialize_db()
        
        # Initialize Redis client for rate limiting
        redis_client = None
        if config.REDIS_URL:
            logger.info(f"Connecting to Redis at {config.REDIS_URL}")
            redis_client = Redis.from_url(config.REDIS_URL, decode_responses=True)
            await redis_client.ping()  # Test connection
        
        # Initialize OpenAI client
        logger.info(f"Initializing OpenAI client with model {config.OPENAI_MODEL}")
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.7,
            streaming=False
        )
        
        # Initialize embeddings
        logger.info("Initializing embeddings")
        embeddings = get_embeddings()
        
        # Initialize vector store manager
        logger.info("Initializing vector store manager")
        vector_store_manager = VectorStoreManager(
            embeddings=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            collection_name=config.COLLECTION_NAME
        )
        await vector_store_manager.initialize()
        
        # Initialize RAG retriever
        logger.info("Initializing RAG retriever")
        rag_retriever = RAGRetriever(
            vector_store_manager=vector_store_manager,
            embeddings=embeddings,
            collection_name=config.COLLECTION_NAME
        )
        await rag_retriever.initialize()
        
        # Initialize chat history manager
        logger.info("Initializing chat history manager")
        chat_history_manager = ChatHistoryManager(rag_retriever=rag_retriever)
        await chat_history_manager.initialize()
        
        # Initialize rate limiter if Redis is available
        rate_limiter = None
        if redis_client:
            logger.info("Initializing rate limiter")
            rate_limit_config = RateLimitConfig(
                max_requests_per_minute=config.RATE_LIMIT_MESSAGES,
                max_tokens_per_day=10000,  # Adjust as needed
                enable_advanced_limiting=config.USE_ADVANCED_RATE_LIMITING,
                history_window_seconds=3600  # 1 hour
            )
            rate_limiter = RedisRateLimiter(
                redis_client=redis_client,
                config=rate_limit_config,
                chat_history_manager=chat_history_manager if config.USE_ADVANCED_RATE_LIMITING else None
            )
            await rate_limiter.initialize()
        
        # Initialize Langchain manager
        logger.info("Initializing Langchain manager")
        langchain_manager = LangchainManager(
            llm=llm,
            rag_retriever=rag_retriever,
            chat_history_manager=chat_history_manager
        )
        await langchain_manager.initialize()
        
        # Initialize Telegram bot
        logger.info("Initializing Telegram bot")
        telegram_bot = TelegramBot(
            token=config.TELEGRAM_TOKEN,
            chat_history_manager=chat_history_manager,
            langchain_manager=langchain_manager,
            rate_limiter=rate_limiter
        )
        await telegram_bot.initialize()
        
        return {
            "redis_client": redis_client,
            "llm": llm,
            "embeddings": embeddings,
            "vector_store_manager": vector_store_manager,
            "rag_retriever": rag_retriever,
            "chat_history_manager": chat_history_manager,
            "rate_limiter": rate_limiter,
            "langchain_manager": langchain_manager,
            "telegram_bot": telegram_bot
        }
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

async def start_application():
    """
    Start the application.
    """
    try:
        # Initialize components
        components = await initialize_components()
        
        # Get Telegram bot
        telegram_bot = components["telegram_bot"]
        
        # Start the bot based on configuration
        if config.WEBHOOK_URL:
            logger.info(f"Starting bot in webhook mode on port {config.PORT}")
            await telegram_bot.start_webhook(config.WEBHOOK_URL, config.PORT)
        else:
            logger.info("Starting bot in polling mode")
            await telegram_bot.start_polling()
    
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise

def main():
    """
    Main entry point.
    """
    try:
        # Run the application
        asyncio.run(start_application())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
