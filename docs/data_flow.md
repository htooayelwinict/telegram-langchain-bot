# Data Flow Diagram

## Overview

This diagram illustrates how data flows through the Telegram Langchain Bot system, from user input to response generation and storage.

## Data Flow Diagram

```
┌──────────────┐          ┌───────────────┐          ┌───────────────┐
│              │          │               │          │               │
│   Telegram   │  Message │  Telegram Bot │ Message  │  Bot Handlers │
│    User      ├─────────►│  Interface    ├─────────►│               │
│              │          │               │          │               │
└──────────────┘          └───────┬───────┘          └───────┬───────┘
                                  │                          │
                                  │                          │ User Query
                                  │                          ▼
                                  │                  ┌───────────────┐
                                  │                  │               │
                                  │                  │  Rate Limiter │
                                  │                  │  (Redis)      │
                                  │                  │               │
                                  │                  └───────┬───────┘
                                  │                          │
                                  │                          │ Allowed Query
                                  │                          ▼
┌──────────────┐          ┌───────────────┐          ┌───────────────┐
│              │          │               │          │               │
│   Response   │  Response│  Telegram Bot │ Response │  LangChain    │
│   to User    │◄─────────┤  Interface    │◄─────────┤  Manager      │
│              │          │               │          │               │
└──────────────┘          └───────────────┘          └───────┬───────┘
                                                             │
                                                             │ Query
                                  ┌─────────────────────────┬┴┬─────────────────────────┐
                                  │                         │ │                         │
                                  ▼                         │ │                         ▼
                          ┌───────────────┐                 │ │                 ┌───────────────┐
                          │               │                 │ │                 │               │
                          │  RAG Retriever│◄────────────────┘ │                 │  Chat History │
                          │               │                   │                 │  Manager      │
                          └───────┬───────┘                   │                 └───────┬───────┘
                                  │                           │                         │
                                  │ Query                     │ Context + Query         │ Recent Messages
                                  ▼                           │                         ▼
                          ┌───────────────┐                   │                 ┌───────────────┐
                          │               │                   │                 │               │
                          │  Vector Store │                   │                 │  PostgreSQL   │
                          │  (ChromaDB)   │                   │                 │  Database     │
                          │               │                   │                 │               │
                          └───────┬───────┘                   │                 └───────────────┘
                                  │                           │
                                  │ Relevant Documents        │
                                  └───────────────────────────┘
                                                             │
                                                             ▼
                                                     ┌───────────────┐
                                                     │               │
                                                     │  OpenAI LLM   │
                                                     │               │
                                                     └───────┬───────┘
                                                             │
                                                             │ Generated Response
                                                             │
                                                     ┌───────▼───────┐
                                                     │               │
                                                     │  Save to      │
                                                     │  History      │
                                                     │               │
                                                     └───────────────┘
```

## Data Flow Process

1. **User Input**
   - User sends a message via Telegram
   - Telegram Bot Interface receives the message
   - Message is passed to Bot Handlers

2. **Rate Limiting**
   - Rate Limiter checks if the user is allowed to make a request
   - If allowed, the query proceeds; otherwise, an error is returned

3. **Context Retrieval**
   - RAG Retriever receives the user query
   - Vector Store is queried for relevant documents
   - Relevant documents are returned as context

4. **History Retrieval**
   - Chat History Manager retrieves recent conversation history
   - History is formatted for inclusion in the prompt

5. **Response Generation**
   - LangChain Manager combines context, history, and query
   - OpenAI LLM generates a response
   - Response is returned to the Telegram Bot Interface

6. **Storage**
   - User query and assistant response are saved to PostgreSQL
   - Messages are vectorized and stored in ChromaDB for future retrieval

7. **Response Delivery**
   - Response is sent back to the user via Telegram
