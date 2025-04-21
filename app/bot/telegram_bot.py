"""
Telegram bot implementation with enhanced application lifecycle management.
Implements proper event loop handling and graceful shutdown.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union, Awaitable
import signal
import sys
import time
from urllib.parse import urlparse

from telegram import Bot, Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from ..config import config
from .handlers import BotHandlers
from ..db.chat_history import ChatHistoryManager
from ..langchain.chain import LangchainManager
from ..utils.rate_limiter import RedisRateLimiter

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot implementation with enhanced application lifecycle management.
    Implements proper event loop handling and graceful shutdown.
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        chat_history_manager: Optional[ChatHistoryManager] = None,
        langchain_manager: Optional[LangchainManager] = None,
        rate_limiter: Optional[RedisRateLimiter] = None
    ):
        """
        Initialize the Telegram bot.
        
        Args:
            token: Telegram bot token
            chat_history_manager: Chat history manager
            langchain_manager: Langchain manager
            rate_limiter: Rate limiter
        """
        self.token = token or config.TELEGRAM_TOKEN
        self.chat_history_manager = chat_history_manager
        self.langchain_manager = langchain_manager
        self.rate_limiter = rate_limiter
        
        self.application = None
        self.handlers = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        logger.info("Initialized Telegram bot")
    
    def _ensure_application(self) -> Application:
        """
        Ensure the application is created.
        
        Returns:
            Telegram bot application
        """
        if not self.application:
            # Create application
            self.application = ApplicationBuilder().token(self.token).build()
            
            # Create handlers if not already created
            if not self.handlers:
                self.handlers = BotHandlers(
                    chat_history_manager=self.chat_history_manager,
                    langchain_manager=self.langchain_manager,
                    rate_limiter=self.rate_limiter
                )
        
        return self.application
    
    async def initialize(self) -> bool:
        """
        Initialize the Telegram bot.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure application is created
            application = self._ensure_application()
            
            # Register handlers
            await self.handlers.register_handlers(application)
            
            logger.info("Telegram bot initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {e}")
            return False
    
    async def start_polling(self) -> None:
        """
        Start the bot in polling mode.
        """
        try:
            # Ensure application is created and initialized
            application = self._ensure_application()
            if not await self.initialize():
                logger.error("Failed to initialize Telegram bot")
                return
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start polling
            self.is_running = True
            logger.info("Starting Telegram bot in polling mode")
            
            # Start the application
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            # Wait for shutdown event
            await self.shutdown_event.wait()
            
            # Perform cleanup
            await self.stop()
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot in polling mode: {e}")
            self.is_running = False
    
    async def start_webhook(self, webhook_url: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Start the bot in webhook mode.
        
        Args:
            webhook_url: Webhook URL
            port: Port to listen on
        """
        try:
            # Ensure application is created and initialized
            application = self._ensure_application()
            if not await self.initialize():
                logger.error("Failed to initialize Telegram bot")
                return
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Get webhook URL and port
            webhook_url = webhook_url or config.WEBHOOK_URL
            port = port or config.PORT
            
            # Extract path from webhook URL using urllib for robust parsing
            try:
                # Parse the URL
                parsed_url = urlparse(webhook_url)
                
                # Extract the path, removing leading slash
                path = parsed_url.path.lstrip('/')
                
                # If path is empty, use a default path
                webhook_path = path if path else f"telegram_webhook_{int(time.time())}"
                
                # For ngrok URLs, ensure we have a proper path
                if 'ngrok' in webhook_url.lower() and not path:
                    webhook_path = f"telegram_webhook_{int(time.time())}"
                
                logger.info(f"Using webhook path: {webhook_path}")
            except Exception as e:
                # Fallback to simple extraction if parsing fails
                logger.warning(f"Error parsing webhook URL: {e}, falling back to simple extraction")
                webhook_path = webhook_url.split("/")[-1]
                if not webhook_path:
                    webhook_path = f"telegram_webhook_{int(time.time())}"
            
            # Start webhook
            self.is_running = True
            logger.info(f"Starting Telegram bot in webhook mode on port {port}")
            
            # Start the application with webhook
            await application.initialize()
            await application.start()
            await application.updater.start_webhook(
                listen="0.0.0.0",
                port=port,
                url_path=webhook_path,
                webhook_url=webhook_url
            )
            
            # Wait for shutdown event
            await self.shutdown_event.wait()
            
            # Perform cleanup
            await self.stop()
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot in webhook mode: {e}")
            self.is_running = False
    
    async def stop(self) -> None:
        """
        Stop the bot and perform cleanup.
        """
        try:
            if not self.application:
                return
            
            logger.info("Stopping Telegram bot")
            
            # Stop the updater first
            if self.application.updater and self.application.updater.running:
                await self.application.updater.stop()
            
            # Stop the application
            if self.application.running:
                await self.application.stop()
            
            # Shutdown the application
            await self.application.shutdown()
            
            self.is_running = False
            logger.info("Telegram bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.
        """
        loop = asyncio.get_event_loop()
        
        # Define signal handler
        def signal_handler():
            logger.info("Received shutdown signal")
            self.shutdown_event.set()
        
        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    
    def run_polling(self) -> None:
        """
        Run the bot in polling mode (blocking).
        """
        asyncio.run(self.start_polling())
    
    def run_webhook(self, webhook_url: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Run the bot in webhook mode (blocking).
        
        Args:
            webhook_url: Webhook URL
            port: Port to listen on
        """
        asyncio.run(self.start_webhook(webhook_url, port))
