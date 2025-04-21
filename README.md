# Telegram Langchain Bot

A sophisticated Telegram bot leveraging LangChain, RAG (Retrieval Augmented Generation), and PostgreSQL for enhanced conversational AI capabilities.

## Developer Portfolio

This project was developed by a cybersecurity and AI/ML specialist with extensive experience in system administration, infrastructure management, and AI integration. The developer brings expertise in:

- **AI/ML Technologies**: Expert implementation of Langchain, vector databases, and LLM integration
- **System Architecture**: Designing robust, modular systems with proper separation of concerns
- **Security-First Development**: Implementation of secure coding practices and system hardening
- **Python Development**: Strong object-oriented programming with asyncio for high-performance applications
- **DevOps Integration**: Containerized deployment with Docker and proper service orchestration

This Telegram bot showcases the integration of advanced AI capabilities with secure, scalable infrastructure - combining technical depth with practical implementation to deliver a robust solution for complex conversational AI challenges.

## Overview

This project implements a Telegram bot that uses OpenAI's language models with Retrieval Augmented Generation (RAG) to provide context-aware responses. The bot maintains conversation history in PostgreSQL and uses ChromaDB for vector storage and retrieval.

## System Architecture

The system is built with a modular architecture that separates concerns and allows for easy extension and maintenance.

### Key Components

1. **Telegram Bot Interface**
   - Handles user interactions via Telegram
   - Manages application lifecycle and event loops
   - Supports both polling and webhook modes

2. **Database Layer**
   - PostgreSQL for persistent storage
   - Models for users and chat history
   - Asynchronous database operations

3. **Vector Store**
   - ChromaDB for vector embeddings storage
   - Manages document retrieval and similarity search

4. **RAG System**
   - Embeddings generation using OpenAI models
   - Context retrieval based on user queries
   - Document chunking and processing

5. **LangChain Integration**
   - Manages prompt templates and LLM interactions
   - Integrates with RAG for context-aware responses
   - Handles conversation history

6. **Rate Limiting**
   - Redis-based rate limiting
   - Supports basic and advanced rate limiting strategies
   - Prevents abuse and manages resource usage

### System Architecture Diagram

See [System Architecture](docs/system_architecture.md) for a detailed diagram and explanation of how the components interact.

### Data Flow Diagram

See [Data Flow](docs/data_flow.md) for a detailed diagram and explanation of how data flows through the system.

## Project Structure

```
telegram-langchain-bot/
├── .env                      # Environment variables
├── docker-compose.yml        # Docker setup for all services
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py               # Entry point with enhanced DB initialization
│   ├── config.py             # Configuration
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── telegram_bot.py   # Telegram bot setup
│   │   └── handlers.py       # Message handlers
│   ├── langchain/
│   │   ├── __init__.py
│   │   ├── chain.py          # Langchain setup
│   │   ├── prompts/          # Modular prompt templates
│   │   │   ├── __init__.py
│   │   │   ├── base.py       # Base prompt handling
│   │   │   ├── user_query.py # User query prompts
│   │   │   ├── system.py     # System prompts
│   │   │   ├── system/       # Directory for system prompts
│   │   │   └── user_query/   # Directory for user query prompts
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py         # Enhanced SQL models with PostgreSQL
│   │   ├── init_db.py        # Database initialization utilities
│   │   ├── chat_history.py   # Enhanced chat history with vectorization
│   │   └── vector_store.py   # Vector store with ChromaDB integration
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Embedding utilities
│   │   ├── encoding_fix.py   # Fixes for encoding issues
│   │   ├── retriever.py      # RAG retrieval logic
│   │   └── tiktoken_patch.py # Patches for tiktoken library
│   └── utils/
│       ├── __init__.py
│       ├── rate_limiter.py   # Enhanced Redis rate limiting with chat history
│       ├── directory_manager.py # Directory management utilities
│       └── logger.py         # Logging utilities
├── data/
│   └── vector_store/         # Persistent storage for vector database
├── docs/
│   ├── system_architecture.md # System architecture documentation
│   └── data_flow.md          # Data flow documentation
└── tests/                    # Unit tests
```

## Technologies Used

- **Python**: Core programming language
- **Telegram API**: Bot interface
- **LangChain**: Framework for LLM applications
- **OpenAI API**: Language model provider
- **PostgreSQL**: Relational database
- **ChromaDB**: Vector database
- **Redis**: Caching and rate limiting
- **Docker**: Containerization
- **Async I/O**: Asynchronous programming

## Features

- Context-aware responses using RAG
- Persistent conversation history
- Vector-based similarity search
- Rate limiting to prevent abuse
- Support for both polling and webhook modes
- Modular prompt management
- Containerized deployment