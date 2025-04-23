"""
Langchain Manager for integrating with RAG and handling prompt management.
Implements enhanced context handling and response generation.
"""
import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from ..config import config
from ..rag.retriever import RAGRetriever
from ..db.chat_history import ChatHistoryManager
from .prompts.base import load_prompt_template

logger = logging.getLogger(__name__)

class LangchainManager:
    """
    Langchain Manager for integrating with RAG and handling prompt management.
    Implements enhanced context handling and response generation.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        rag_retriever: Optional[RAGRetriever] = None,
        chat_history_manager: Optional[ChatHistoryManager] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Langchain Manager.
        
        Args:
            llm: Language model
            rag_retriever: Optional RAG retriever
            chat_history_manager: Optional chat history manager
            system_prompt: Optional system prompt
        """
        self.llm = llm
        self.rag_retriever = rag_retriever
        self.chat_history_manager = chat_history_manager
        
        # Load system prompt if not provided
        self.system_prompt = system_prompt
        if not self.system_prompt:
            try:
                self.system_prompt = load_prompt_template("system")
            except Exception as e:
                logger.error(f"Error loading system prompt: {e}")
                self.system_prompt = "You are a helpful AI assistant."
        
        logger.info("Initialized Langchain Manager")
    
    async def initialize(self) -> bool:
        """
        Initialize the Langchain Manager.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure RAG retriever is initialized if provided
            if self.rag_retriever and not await self.rag_retriever.initialize():
                logger.error("Failed to initialize RAG retriever")
                return False
            
            # Ensure chat history manager is initialized if provided
            if self.chat_history_manager and not await self.chat_history_manager.initialize():
                logger.error("Failed to initialize chat history manager")
                return False
            
            logger.info("Langchain Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Langchain Manager: {e}")
            return False
    
    async def generate_response(
        self,
        user_query: str,
        user_id: str,
        use_rag: bool = True,
        include_chat_history: bool = True,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response using Langchain.
        
        Args:
            user_query: User query
            user_id: User ID
            use_rag: Whether to use RAG
            include_chat_history: Whether to include chat history
            max_tokens: Maximum tokens for context
            
        Returns:
            Generated response
        """
        try:
            # Get chat history if available and requested
            chat_history = []
            if include_chat_history and self.chat_history_manager:
                try:
                    chat_history = await self.chat_history_manager.get_conversation_history(
                        user_id=user_id,
                        limit=50,
                        include_system=False
                    )
                except Exception as chat_error:
                    logger.error(f"Error retrieving chat history: {chat_error}")
                    # Continue without chat history
            
            # Get relevant context if RAG is enabled
            context = ""
            if use_rag and self.rag_retriever:
                try:
                    # Retrieve relevant documents
                    docs = await self.rag_retriever.retrieve_context(
                        query=user_query,
                        user_id=user_id,
                        k=5
                    )
                    
                    # Format context for prompt
                    if docs:
                        context = await self.rag_retriever.format_context_for_prompt(
                            documents=docs,
                            max_tokens=max_tokens
                        )
                except Exception as rag_error:
                    logger.error(f"Error retrieving RAG context: {rag_error}")
                    # Continue without RAG context
            
            # Create messages for the prompt
            messages = []
            
            # Add system message
            system_content = self.system_prompt
            if context:
                system_content += f"\n\nRelevant context:\n{context}"
            messages.append(SystemMessage(content=system_content))
            
            # Add chat history
            for message in chat_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            
            # Add current query
            messages.append(HumanMessage(content=user_query))
            
            # Generate response
            try:
                response = await self.llm.agenerate([messages])
                
                # Extract response text
                response_text = response.generations[0][0].text
                
                # Log token usage if available
                if hasattr(response, "llm_output") and response.llm_output:
                    token_usage = response.llm_output.get("token_usage", {})
                    if token_usage:
                        logger.info(f"Token usage: {token_usage}")
                
                return response_text
            except Exception as llm_error:
                logger.error(f"Error calling LLM API: {llm_error}")
                # Provide a more specific error message based on the error type
                if "rate limit" in str(llm_error).lower():
                    return "I'm sorry, the service is currently experiencing high demand. Please try again in a moment."
                elif "invalid api key" in str(llm_error).lower():
                    logger.critical("Invalid API key detected!")
                    return "I'm sorry, there's a configuration issue with the service. Please contact the administrator."
                elif "timeout" in str(llm_error).lower():
                    return "I'm sorry, the request timed out. Please try again with a shorter query."
                else:
                    return "I'm sorry, I encountered an error while generating a response. Please try again later."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again later."
    
    async def generate_rag_response(
        self,
        user_query: str,
        user_id: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_context_tokens: int = 2000
    ) -> str:
        """
        Generate a response using RAG.
        
        Args:
            user_query: User query
            user_id: User ID
            chat_history: Optional chat history
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Generated response
        """
        try:
            # Ensure RAG retriever is available
            if not self.rag_retriever:
                logger.error("RAG retriever not available")
                return await self.generate_response(user_query, user_id, use_rag=False)
            
            # Get chat history if not provided
            if chat_history is None and self.chat_history_manager:
                chat_history = await self.chat_history_manager.get_conversation_history(
                    user_id=user_id,
                    limit=50,
                    include_system=False
                )
            
            # Retrieve relevant context
            docs = await self.rag_retriever.retrieve_context(
                query=user_query,
                user_id=user_id,
                k=5
            )
            
            # Format context for prompt
            context = ""
            if docs:
                context = await self.rag_retriever.format_context_for_prompt(
                    documents=docs,
                    max_tokens=max_context_tokens
                )
            
            # Load prompt templates
            system_template = load_prompt_template("system")
            user_template = load_prompt_template("user_query")
            
            # Format system prompt with context
            system_prompt = system_template
            if context:
                system_prompt += f"\n\nRelevant context:\n{context}"
            
            # Format user prompt
            user_prompt = user_template.format(query=user_query)
            
            # Create messages for the prompt
            messages = [SystemMessage(content=system_prompt)]
            
            # Add chat history
            if chat_history:
                for message in chat_history:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            
            # Add current query
            messages.append(HumanMessage(content=user_prompt))
            
            # Generate response
            response = await self.llm.agenerate([messages])
            
            # Extract response text
            response_text = response.generations[0][0].text
            
            # Log token usage if available
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    logger.info(f"Token usage: {token_usage}")
            
            return response_text
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again later."
    
    async def save_response(
        self,
        user_id: str,
        user_query: str,
        assistant_response: str,
        vectorize: bool = True
    ) -> bool:
        """
        Save user query and assistant response to chat history.
        
        Args:
            user_id: User ID
            user_query: User query
            assistant_response: Assistant response
            vectorize: Whether to vectorize the messages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure chat history manager is available
            if not self.chat_history_manager:
                logger.error("Chat history manager not available")
                return False
            
            # Save user query
            await self.chat_history_manager.add_message(
                user_id=user_id,
                content=user_query,
                role="user",
                vectorize=vectorize
            )
            
            # Save assistant response
            await self.chat_history_manager.add_message(
                user_id=user_id,
                content=assistant_response,
                role="assistant",
                vectorize=vectorize
            )
            
            logger.info(f"Saved conversation for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
