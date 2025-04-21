# System Architecture

## System Components

The Telegram Langchain Bot consists of the following key components:

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

4. **RAG (Retrieval Augmented Generation) System**
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

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                           Telegram Bot Interface                        │
│                                                                         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                           LangChain Manager                             │
│                                                                         │
└───────────┬─────────────────────────┬──────────────────────┬────────────┘
            │                         │                      │
            ▼                         ▼                      ▼
┌───────────────────┐      ┌───────────────────┐   ┌──────────────────┐
│                   │      │                   │   │                  │
│  RAG Retriever    │◄────►│  Chat History     │   │   LLM (OpenAI)   │
│                   │      │  Manager          │   │                  │
└─────────┬─────────┘      └─────────┬─────────┘   └──────────────────┘
          │                          │
          ▼                          ▼
┌───────────────────┐      ┌───────────────────┐   ┌──────────────────┐
│                   │      │                   │   │                  │
│  Vector Store     │      │  PostgreSQL DB    │   │   Redis Cache    │
│  (ChromaDB)       │      │                   │   │                  │
└───────────────────┘      └───────────────────┘   └──────────────────┘
```

## Component Interactions

- **Telegram Bot Interface** receives user messages and forwards them to the LangChain Manager
- **LangChain Manager** coordinates between RAG, Chat History, and LLM to generate responses
- **RAG Retriever** uses the Vector Store to find relevant context for user queries
- **Chat History Manager** stores conversations in PostgreSQL and manages retrieval
- **Vector Store** maintains embeddings for efficient semantic search
- **Redis** provides caching and rate limiting capabilities
