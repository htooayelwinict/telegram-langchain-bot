"""
Enhanced rate limiter with Redis integration and chat history support.
Implements both standard and advanced rate limiting options.
"""
import logging
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

import redis.asyncio as redis
from redis.exceptions import RedisError

from ..config import config

settings = config
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests_per_minute: int = 20
    max_tokens_per_day: int = 10000
    enable_advanced_limiting: bool = False
    history_window_seconds: int = 3600  # 1 hour
    cooldown_factor: float = 0.8  # Reduce limits to 80% after exceeding
    new_user_limit_factor: float = 0.5  # 50% of normal limits for new users
    burst_multiplier: int = 3  # Allow bursts of 3x normal rate


class RedisRateLimiter:
    """
    Enhanced rate limiter with Redis integration and chat history support.
    Implements both standard and advanced rate limiting with configurable options.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis = None,
        config: Optional[RateLimitConfig] = None,
        chat_history_manager = None
    ):
        """
        Initialize the enhanced rate limiter.
        
        Args:
            redis_client: Redis client
            config: Rate limit configuration
            chat_history_manager: Optional chat history manager for advanced limiting
        """
        self.redis = redis_client
        self.config = config or RateLimitConfig()
        self.chat_history_manager = chat_history_manager
        
        # Key prefixes for Redis
        self.minute_key_prefix = "rate:minute:"
        self.day_key_prefix = "rate:day:"
        self.tokens_key_prefix = "rate:tokens:"
        self.history_key_prefix = "rate:history:"
        self.metadata_key_prefix = "rate:metadata:"
        
        logger.info(f"Initialized enhanced rate limiter with Redis integration")
        logger.info(f"Rate limits: {self.config.max_requests_per_minute} req/min, "
                   f"{self.config.max_tokens_per_day} tokens/day, "
                   f"advanced_limiting={self.config.enable_advanced_limiting}")
    
    async def initialize(self) -> bool:
        """
        Initialize the rate limiter.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            try:
                # Connect to Redis
                self.redis = redis.Redis(
                    url=config.REDIS_URL,
                    decode_responses=True
                )
                
                # Test connection
                await self.redis.ping()
                logger.info("Connected to Redis successfully")
                return True
            except RedisError as e:
                logger.error(f"Error connecting to Redis: {e}")
                return False
        return True
    
    async def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """
        Check if a user has exceeded their rate limit.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        try:
            # Ensure Redis connection
            if not self.redis:
                if not await self.initialize():
                    logger.error("Failed to initialize Redis connection")
                    return True, ""  # Allow if Redis is not available
            
            # Get current time
            current_time = int(time.time())
            minute_window = current_time - (current_time % 60)  # Current minute
            day_window = current_time - (current_time % 86400)  # Current day
            
            # Keys for rate limiting
            minute_key = f"{self.minute_key_prefix}{user_id}:{minute_window}"
            day_key = f"{self.day_key_prefix}{user_id}:{day_window}"
            tokens_key = f"{self.tokens_key_prefix}{user_id}:{day_window}"
            
            # Standard rate limiting
            # Check minute limit
            minute_count = await self.redis.get(minute_key)
            minute_count = int(minute_count) if minute_count else 0
            
            if minute_count >= self.config.max_requests_per_minute:
                seconds_to_wait = 60 - (current_time % 60)
                return False, f"Rate limit exceeded. Please try again in {seconds_to_wait} seconds."
            
            # Check daily token limit
            tokens_used = await self.redis.get(tokens_key)
            tokens_used = int(tokens_used) if tokens_used else 0
            
            if tokens_used >= self.config.max_tokens_per_day:
                return False, "Daily token limit exceeded. Please try again tomorrow."
            
            # Advanced rate limiting if enabled
            if self.config.enable_advanced_limiting:
                # Get user metadata
                metadata_key = f"{self.metadata_key_prefix}{user_id}"
                metadata_json = await self.redis.get(metadata_key)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Check for burst patterns
                history_key = f"{self.history_key_prefix}{user_id}"
                await self.redis.zadd(history_key, {str(current_time): current_time})
                await self.redis.zremrangebyscore(
                    history_key,
                    0,
                    current_time - self.config.history_window_seconds
                )
                
                # Count requests in the window
                request_count = await self.redis.zcard(history_key)
                max_burst = self.config.max_requests_per_minute * self.config.burst_multiplier
                
                if request_count > max_burst:
                    # Apply cooldown if user is making too many requests in the window
                    effective_limit = int(self.config.max_requests_per_minute * self.config.cooldown_factor)
                    if minute_count >= effective_limit:
                        return False, "Advanced rate limit applied due to unusual activity pattern. Please slow down."
                
                # Check if user is new (less than 1 day old)
                created_at = metadata.get("created_at", 0)
                if created_at > 0 and (current_time - created_at) < 86400:
                    # Apply stricter limits for new users
                    stricter_limit = max(1, int(self.config.max_requests_per_minute * self.config.new_user_limit_factor))
                    if minute_count >= stricter_limit:
                        return False, "New user rate limit applied. Please try again later."
            
            # Increment counters
            pipeline = self.redis.pipeline()
            pipeline.incr(minute_key)
            pipeline.expire(minute_key, 60)  # Expire after 1 minute
            await pipeline.execute()
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True, ""  # Allow if there's an error
    
    async def add_tokens_used(self, user_id: str, token_count: int) -> bool:
        """
        Record tokens used by a user.
        
        Args:
            user_id: User ID
            token_count: Number of tokens used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure Redis connection
            if not self.redis:
                if not await self.initialize():
                    logger.error("Failed to initialize Redis connection")
                    return False
            
            # Get current time
            current_time = int(time.time())
            day_window = current_time - (current_time % 86400)  # Current day
            
            # Key for token usage
            tokens_key = f"{self.tokens_key_prefix}{user_id}:{day_window}"
            
            # Increment token usage
            pipeline = self.redis.pipeline()
            pipeline.incrby(tokens_key, token_count)
            pipeline.expire(tokens_key, 86400)  # Expire after 1 day
            await pipeline.execute()
            
            # Update metadata for advanced limiting
            if self.config.enable_advanced_limiting:
                metadata_key = f"{self.metadata_key_prefix}{user_id}"
                metadata_json = await self.redis.get(metadata_key)
                metadata = json.loads(metadata_json) if metadata_json else {
                    "created_at": current_time,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "avg_tokens_per_request": 0
                }
                
                # Update token usage statistics
                total_tokens = metadata.get("total_tokens", 0) + token_count
                total_requests = metadata.get("total_requests", 0) + 1
                avg_tokens = total_tokens / total_requests if total_requests > 0 else token_count
                
                metadata.update({
                    "total_tokens": total_tokens,
                    "total_requests": total_requests,
                    "avg_tokens_per_request": avg_tokens,
                    "last_updated": current_time
                })
                
                # Save metadata
                await self.redis.set(metadata_key, json.dumps(metadata))
                await self.redis.expire(metadata_key, 30 * 86400)  # Expire after 30 days
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding tokens used: {e}")
            return False
    
    async def get_user_limits(self, user_id: str) -> Dict[str, Any]:
        """
        Get the current rate limit status for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with rate limit information
        """
        try:
            # Ensure Redis connection
            if not self.redis:
                if not await self.initialize():
                    logger.error("Failed to initialize Redis connection")
                    return {}
            
            # Get current time
            current_time = int(time.time())
            minute_window = current_time - (current_time % 60)  # Current minute
            day_window = current_time - (current_time % 86400)  # Current day
            
            # Keys for rate limiting
            minute_key = f"{self.minute_key_prefix}{user_id}:{minute_window}"
            tokens_key = f"{self.tokens_key_prefix}{user_id}:{day_window}"
            
            # Get current usage
            minute_count = await self.redis.get(minute_key)
            minute_count = int(minute_count) if minute_count else 0
            
            tokens_used = await self.redis.get(tokens_key)
            tokens_used = int(tokens_used) if tokens_used else 0
            
            # Calculate remaining limits
            minute_remaining = max(0, self.config.max_requests_per_minute - minute_count)
            tokens_remaining = max(0, self.config.max_tokens_per_day - tokens_used)
            
            # Calculate reset times
            seconds_to_minute_reset = 60 - (current_time % 60)
            seconds_to_day_reset = 86400 - (current_time % 86400)
            
            # Get advanced limiting info if enabled
            advanced_info = {}
            if self.config.enable_advanced_limiting:
                metadata_key = f"{self.metadata_key_prefix}{user_id}"
                metadata_json = await self.redis.get(metadata_key)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                history_key = f"{self.history_key_prefix}{user_id}"
                request_count = await self.redis.zcard(history_key)
                
                advanced_info = {
                    "total_requests": metadata.get("total_requests", 0),
                    "total_tokens": metadata.get("total_tokens", 0),
                    "avg_tokens_per_request": round(metadata.get("avg_tokens_per_request", 0), 2),
                    "requests_last_hour": request_count,
                    "account_age_seconds": current_time - metadata.get("created_at", current_time),
                    "is_new_user": (current_time - metadata.get("created_at", 0)) < 86400
                }
            
            # Combine all info
            result = {
                "minute_requests": minute_count,
                "minute_requests_limit": self.config.max_requests_per_minute,
                "minute_requests_remaining": minute_remaining,
                "seconds_to_minute_reset": seconds_to_minute_reset,
                
                "tokens_used_today": tokens_used,
                "tokens_limit": self.config.max_tokens_per_day,
                "tokens_remaining": tokens_remaining,
                "seconds_to_day_reset": seconds_to_day_reset,
                
                "advanced_limiting_enabled": self.config.enable_advanced_limiting
            }
            
            if advanced_info:
                result["advanced"] = advanced_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting user limits: {e}")
            return {}
    
    async def reset_user_limits(self, user_id: str) -> bool:
        """
        Reset rate limits for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure Redis connection
            if not self.redis:
                if not await self.initialize():
                    logger.error("Failed to initialize Redis connection")
                    return False
            
            # Get current time
            current_time = int(time.time())
            minute_window = current_time - (current_time % 60)  # Current minute
            day_window = current_time - (current_time % 86400)  # Current day
            
            # Keys to delete
            minute_key = f"{self.minute_key_prefix}{user_id}:{minute_window}"
            day_key = f"{self.day_key_prefix}{user_id}:{day_window}"
            tokens_key = f"{self.tokens_key_prefix}{user_id}:{day_window}"
            history_key = f"{self.history_key_prefix}{user_id}"
            
            # Delete keys
            pipeline = self.redis.pipeline()
            pipeline.delete(minute_key)
            pipeline.delete(day_key)
            pipeline.delete(tokens_key)
            pipeline.delete(history_key)
            await pipeline.execute()
            
            logger.info(f"Reset rate limits for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting user limits: {e}")
            return False
