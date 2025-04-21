"""
Telegram bot message handlers with comprehensive error handling.
Implements handlers for text messages, commands, and errors.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union, Awaitable
import traceback

from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters, Application
from telegram.constants import ParseMode

from ..config import config
from ..db.chat_history import ChatHistoryManager
from ..langchain.chain import LangchainManager
from ..utils.rate_limiter import RedisRateLimiter
from ..langchain.prompts.base import (
    load_prompt_template, save_prompt_template, 
    list_prompt_templates, delete_prompt_template
)

logger = logging.getLogger(__name__)

# Type alias for handler function
HandlerFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]

class BotHandlers:
    """
    Telegram bot message handlers with comprehensive error handling.
    Implements handlers for text messages, commands, and errors.
    """
    
    def __init__(
        self,
        chat_history_manager: ChatHistoryManager,
        langchain_manager: LangchainManager,
        rate_limiter: Optional[RedisRateLimiter] = None
    ):
        """
        Initialize the bot handlers.
        
        Args:
            chat_history_manager: Chat history manager
            langchain_manager: Langchain manager
            rate_limiter: Optional rate limiter
        """
        self.chat_history_manager = chat_history_manager
        self.langchain_manager = langchain_manager
        self.rate_limiter = rate_limiter
        
        logger.info("Initialized bot handlers")
    
    async def register_handlers(self, application: Application) -> None:
        """
        Register all handlers with the application.
        
        Args:
            application: Telegram bot application
        """
        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("prompt", self.prompt_command))
        application.add_handler(CommandHandler("reset", self.reset_command))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
        
        # Error handler
        application.add_error_handler(self.error_handler)
        
        logger.info("Registered all handlers")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            user = update.effective_user
            chat = update.effective_chat
            
            if not user or not chat:
                return
            
            # Add user to database
            await self.chat_history_manager.add_user(
                user_id=str(user.id),
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name
            )
            
            # Send welcome message
            welcome_message = (
                f"👋 Hello, {user.first_name}! I'm an AI assistant powered by advanced language models.\n\n"
                f"I can help answer your questions and have conversations with you. I also remember our "
                f"previous interactions to provide more relevant responses.\n\n"
                f"Just send me a message to get started, or use /help to see available commands."
            )
            
            await context.bot.send_message(
                chat_id=chat.id,
                text=welcome_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logger.info(f"User {user.id} started the bot")
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await self.send_error_message(update, context)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /help command.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            chat = update.effective_chat
            
            if not chat:
                return
            
            help_message = (
                "🤖 *Available Commands*\n\n"
                "/start - Start the bot and get a welcome message\n"
                "/help - Show this help message\n"
                "/stats - Show your usage statistics\n"
                "/prompt - Manage prompt templates (admin only)\n"
                "/reset - Reset your rate limits (admin only)\n\n"
                "Simply send me a message to start a conversation!"
            )
            
            await context.bot.send_message(
                chat_id=chat.id,
                text=help_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logger.info(f"Help command sent to chat {chat.id}")
        except Exception as e:
            logger.error(f"Error in help command: {e}")
            await self.send_error_message(update, context)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /stats command.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            user = update.effective_user
            chat = update.effective_chat
            
            if not user or not chat:
                return
            
            # Get rate limit stats if available
            rate_limit_stats = {}
            if self.rate_limiter:
                rate_limit_stats = await self.rate_limiter.get_user_limits(str(user.id))
            
            # Format stats message
            stats_message = f"📊 *Usage Statistics for {user.first_name}*\n\n"
            
            if rate_limit_stats:
                # Add rate limit stats
                stats_message += "*Rate Limits*:\n"
                stats_message += f"• Messages: {rate_limit_stats.get('minute_requests', 0)}/{rate_limit_stats.get('minute_requests_limit', 0)} per minute\n"
                stats_message += f"• Tokens: {rate_limit_stats.get('tokens_used_today', 0)}/{rate_limit_stats.get('tokens_limit', 0)} per day\n"
                
                # Add reset times
                minute_reset = rate_limit_stats.get('seconds_to_minute_reset', 0)
                day_reset = rate_limit_stats.get('seconds_to_day_reset', 0)
                
                stats_message += f"• Message limit resets in: {minute_reset//60} min {minute_reset%60} sec\n"
                stats_message += f"• Token limit resets in: {day_reset//3600} hours {(day_reset%3600)//60} min\n\n"
            
            # Send stats message
            await context.bot.send_message(
                chat_id=chat.id,
                text=stats_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logger.info(f"Stats command sent to user {user.id}")
        except Exception as e:
            logger.error(f"Error in stats command: {e}")
            await self.send_error_message(update, context)
    
    async def prompt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /prompt command for managing prompt templates.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            user = update.effective_user
            chat = update.effective_chat
            
            if not user or not chat:
                return
            
            # Check if user is admin
            if str(user.id) not in config.ADMIN_USER_IDS.split(','):
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="⚠️ Sorry, only admins can manage prompt templates."
                )
                return
            
            # Get command arguments
            args = context.args
            
            if not args or args[0] == "help":
                # Show help message
                help_message = (
                    "🔧 *Prompt Management Commands*\n\n"
                    "/prompt list [system|user_query] - List available prompt templates\n"
                    "/prompt show [system|user_query] [name] - Show a prompt template\n"
                    "/prompt save [system|user_query] [name] [content] - Save a prompt template\n"
                    "/prompt delete [system|user_query] [name] - Delete a prompt template\n"
                )
                
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=help_message,
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            action = args[0].lower()
            
            if action == "list" and len(args) >= 2:
                # List prompt templates
                prompt_type = args[1].lower()
                if prompt_type not in ["system", "user_query"]:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="⚠️ Invalid prompt type. Use 'system' or 'user_query'."
                    )
                    return
                
                templates = list_prompt_templates(prompt_type)
                
                if not templates:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"No {prompt_type} prompt templates found."
                    )
                    return
                
                templates_list = "\n".join([f"• {template}" for template in templates])
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=f"📝 *{prompt_type.capitalize()} Prompt Templates*:\n\n{templates_list}",
                    parse_mode=ParseMode.MARKDOWN
                )
                
            elif action == "show" and len(args) >= 3:
                # Show prompt template
                prompt_type = args[1].lower()
                prompt_name = args[2]
                
                if prompt_type not in ["system", "user_query"]:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="⚠️ Invalid prompt type. Use 'system' or 'user_query'."
                    )
                    return
                
                prompt_content = load_prompt_template(prompt_type, prompt_name)
                
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=f"📄 *{prompt_type.capitalize()} Prompt: {prompt_name}*\n\n```\n{prompt_content}\n```",
                    parse_mode=ParseMode.MARKDOWN
                )
                
            elif action == "save" and len(args) >= 4:
                # Save prompt template
                prompt_type = args[1].lower()
                prompt_name = args[2]
                prompt_content = " ".join(args[3:])
                
                if prompt_type not in ["system", "user_query"]:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="⚠️ Invalid prompt type. Use 'system' or 'user_query'."
                    )
                    return
                
                success = save_prompt_template(prompt_type, prompt_content, prompt_name)
                
                if success:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"✅ Saved {prompt_type} prompt template: {prompt_name}"
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"❌ Failed to save {prompt_type} prompt template: {prompt_name}"
                    )
                
            elif action == "delete" and len(args) >= 3:
                # Delete prompt template
                prompt_type = args[1].lower()
                prompt_name = args[2]
                
                if prompt_type not in ["system", "user_query"]:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="⚠️ Invalid prompt type. Use 'system' or 'user_query'."
                    )
                    return
                
                success = delete_prompt_template(prompt_type, prompt_name)
                
                if success:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"✅ Deleted {prompt_type} prompt template: {prompt_name}"
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"❌ Failed to delete {prompt_type} prompt template: {prompt_name}"
                    )
                
            else:
                # Invalid command
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="⚠️ Invalid prompt command. Use /prompt help for usage information."
                )
            
            logger.info(f"Prompt command executed by user {user.id}: {' '.join(args)}")
        except Exception as e:
            logger.error(f"Error in prompt command: {e}")
            await self.send_error_message(update, context)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /reset command for resetting rate limits.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            user = update.effective_user
            chat = update.effective_chat
            
            if not user or not chat:
                return
            
            # Check if user is admin or resetting their own limits
            target_user_id = str(user.id)
            is_admin = str(user.id) in config.ADMIN_USER_IDS.split(',')
            
            # Check for target user argument (admin only)
            if len(context.args) > 0 and is_admin:
                try:
                    target_user_id = context.args[0]
                except ValueError:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="⚠️ Invalid user ID format."
                    )
                    return
            
            # Check if rate limiter is available
            if not self.rate_limiter:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="⚠️ Rate limiting is not enabled."
                )
                return
            
            # Reset rate limits
            success = await self.rate_limiter.reset_user_limits(target_user_id)
            
            if success:
                if target_user_id == str(user.id):
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text="✅ Your rate limits have been reset."
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"✅ Rate limits for user {target_user_id} have been reset."
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="❌ Failed to reset rate limits."
                )
            
            logger.info(f"Reset command executed by user {user.id} for target {target_user_id}")
        except Exception as e:
            logger.error(f"Error in reset command: {e}")
            await self.send_error_message(update, context)
    
    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle text messages.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            user = update.effective_user
            chat = update.effective_chat
            message = update.effective_message
            
            if not user or not chat or not message or not message.text:
                return
            
            # Add user to database if not exists
            await self.chat_history_manager.add_user(
                user_id=str(user.id),
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name
            )
            
            # Check rate limit if enabled
            if self.rate_limiter:
                is_allowed, reason = await self.rate_limiter.check_rate_limit(str(user.id))
                
                if not is_allowed:
                    await context.bot.send_message(
                        chat_id=chat.id,
                        text=f"⚠️ {reason}"
                    )
                    return
            
            # Send typing action
            await context.bot.send_chat_action(chat_id=chat.id, action="typing")
            
            # Process the message
            user_query = message.text.strip()
            
            # Generate response
            response = await self.langchain_manager.generate_response(
                user_query=user_query,
                user_id=str(user.id),
                use_rag=True,
                include_chat_history=True
            )
            
            # Save conversation to chat history
            await self.langchain_manager.save_response(
                user_id=str(user.id),
                user_query=user_query,
                assistant_response=response,
                vectorize=True
            )
            
            # Send response
            await context.bot.send_message(
                chat_id=chat.id,
                text=response,
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Update token usage if rate limiter is enabled
            if self.rate_limiter:
                # Estimate token usage (very rough estimate)
                estimated_tokens = len(user_query.split()) + len(response.split())
                await self.rate_limiter.add_tokens_used(str(user.id), estimated_tokens)
            
            logger.info(f"Processed message from user {user.id}")
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await self.send_error_message(update, context)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle errors in the bot.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            # Log the error
            logger.error(f"Exception while handling an update: {context.error}")
            logger.error(traceback.format_exc())
            
            # Send error message to chat if possible
            await self.send_error_message(update, context)
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
    
    async def send_error_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Send an error message to the user.
        
        Args:
            update: Telegram update
            context: Callback context
        """
        try:
            if update and update.effective_chat:
                error_message = (
                    "❌ Sorry, something went wrong while processing your request.\n"
                    "Please try again later."
                )
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=error_message
                )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
