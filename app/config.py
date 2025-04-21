import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

class Config:
    # Telegram Bot Configuration
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    PORT = int(os.getenv("PORT", "8443"))
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@postgres:5432/telegram_bot")
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # Rate Limiting Configuration
    RATE_LIMIT_MESSAGES = int(os.getenv("RATE_LIMIT_MESSAGES", "5"))  # Messages per minute
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Window in seconds
    USE_ADVANCED_RATE_LIMITING = os.getenv("USE_ADVANCED_RATE_LIMITING", "False").lower() == "true"
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/vector_store")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chat_history")
    
    # RAG Configuration
    MAX_RELEVANT_CHUNKS = int(os.getenv("MAX_RELEVANT_CHUNKS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Application Configuration
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

config = Config()