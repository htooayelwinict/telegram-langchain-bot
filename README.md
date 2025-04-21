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
│   │   └── retriever.py      # RAG retrieval logic
│   └── utils/
│       ├── __init__.py
│       ├── rate_limiter.py   # Enhanced Redis rate limiting with chat history
│       └── logger.py         # Logging utilities
├── data/
│   └── vector_store/         # Persistent storage for vector database
└── tests/                    # Unit tests