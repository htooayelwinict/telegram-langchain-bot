"""
Integration tests for the Telegram-Langchain-RAG Bot.
Tests core functionality to ensure components work together correctly.
"""
import os
import sys
import asyncio
import pytest
import logging
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Mock the database initialization to avoid real DB operations
with patch('app.db.init_db.initialize', return_value=None):
    with patch('sqlalchemy.ext.asyncio.create_async_engine'):
        with patch('sqlalchemy.ext.asyncio.async_sessionmaker'):
            from app.config import config
            from app.utils.directory_manager import ensure_directories_exist
            from app.rag.embeddings import get_embeddings
            from app.db.vector_store import VectorStoreManager
            from app.rag.retriever import RAGRetriever
            from app.langchain.chain import LangchainManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Fixtures for test setup and teardown
@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def mock_openai_key():
    """Mock OpenAI API key for testing."""
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
    yield
    # No cleanup needed as we're just setting an environment variable

@pytest.fixture(scope="module")
@patch('app.rag.embeddings.OpenAIEmbeddings')
@patch('chromadb.Client')
@patch('langchain.vectorstores.Chroma')
async def test_components(mock_chroma, mock_client, mock_embeddings):
    """Set up test components with mocks."""
    # Setup mock embeddings
    mock_embed_instance = MagicMock()
    mock_embed_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 10  # Mock embedding vectors
    mock_embed_instance.embed_query.return_value = [0.1, 0.2, 0.3]  # Mock query embedding
    mock_embeddings.return_value = mock_embed_instance
    
    # Setup mock ChromaDB client
    mock_collection = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.get_collection.return_value = mock_collection
    mock_client_instance.create_collection.return_value = mock_collection
    mock_client.return_value = mock_client_instance
    
    # Setup mock Chroma vectorstore
    mock_chroma_instance = MagicMock()
    mock_chroma_instance.similarity_search.return_value = [
        MagicMock(page_content="Test document content", metadata={"source": "test"})
    ]
    mock_chroma_instance.similarity_search_with_score.return_value = [
        (MagicMock(page_content="Test document content", metadata={"source": "test"}), 0.95)
    ]
    mock_chroma_instance.add_documents.return_value = None
    mock_chroma.return_value = mock_chroma_instance
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Initialize components with mocked embeddings
    embeddings = get_embeddings()
    
    # Initialize vector store with test collection
    vector_store = VectorStoreManager(
        embeddings=embeddings,
        collection_name="test_collection",
        persist_directory="./test_data/vector_store"
    )
    await vector_store.initialize()
    
    # Initialize RAG retriever
    rag_retriever = RAGRetriever(
        vector_store_manager=vector_store,
        embeddings=embeddings,
        collection_name="test_collection"
    )
    await rag_retriever.initialize()
    
    # Create mock LLM
    mock_llm = MagicMock()
    mock_llm.agenerate.return_value = MagicMock(
        generations=[[MagicMock(text="This is a test response.")]]
    )
    
    # Initialize LangchainManager with mock LLM
    langchain_manager = LangchainManager(
        llm=mock_llm,
        rag_retriever=rag_retriever
    )
    await langchain_manager.initialize()
    
    # Return all components for testing
    return {
        "embeddings": embeddings,
        "vector_store": vector_store,
        "rag_retriever": rag_retriever,
        "langchain_manager": langchain_manager,
        "mock_llm": mock_llm
    }

# Clean up test data after tests
@pytest.fixture(scope="module", autouse=True)
async def cleanup_test_data():
    """Clean up test data after all tests have run."""
    yield
    import shutil
    if os.path.exists("./test_data"):
        shutil.rmtree("./test_data")

# Tests
def test_ensure_directories_exist():
    """Test that ensure_directories_exist function works."""
    result = ensure_directories_exist(["./test_data/test_dir"])
    assert result
    assert os.path.exists("./test_data/test_dir")

@pytest.mark.asyncio
async def test_vector_store_initialization(test_components):
    """Test that vector store initializes correctly."""
    vector_store = test_components["vector_store"]
    assert vector_store._client is not None
    assert vector_store._collection is not None
    assert vector_store._langchain_db is not None

@pytest.mark.asyncio
async def test_rag_retriever_initialization(test_components):
    """Test that RAG retriever initializes correctly."""
    rag_retriever = test_components["rag_retriever"]
    assert rag_retriever.vector_store_manager is not None
    assert rag_retriever.embeddings is not None

@pytest.mark.asyncio
async def test_langchain_manager_initialization(test_components):
    """Test that LangchainManager initializes correctly."""
    langchain_manager = test_components["langchain_manager"]
    assert langchain_manager.llm is not None
    assert langchain_manager.rag_retriever is not None
    assert langchain_manager.system_prompt is not None

@pytest.mark.asyncio
async def test_add_and_retrieve_documents(test_components):
    """Test adding and retrieving documents."""
    rag_retriever = test_components["rag_retriever"]
    
    # Add test documents
    test_docs = [
        {
            "content": "This is a test document about artificial intelligence.",
            "metadata": {"user_id": "test_user", "source": "test"}
        },
        {
            "content": "Langchain is a framework for developing applications powered by language models.",
            "metadata": {"user_id": "test_user", "source": "test"}
        }
    ]
    
    # Add documents
    result = await rag_retriever.add_documents(test_docs)
    assert result
    
    # Retrieve documents
    docs = await rag_retriever.retrieve_context(
        query="What is Langchain?",
        user_id="test_user"
    )
    
    # Check that we got results
    assert len(docs) > 0

@pytest.mark.asyncio
async def test_generate_response(test_components):
    """Test generating a response."""
    langchain_manager = test_components["langchain_manager"]
    mock_llm = test_components["mock_llm"]
    
    # Generate response
    response = await langchain_manager.generate_response(
        user_query="What is Langchain?",
        user_id="test_user",
        use_rag=True
    )
    
    # Check response
    assert response == "This is a test response."
    
    # Verify that the LLM was called
    mock_llm.agenerate.assert_called()
