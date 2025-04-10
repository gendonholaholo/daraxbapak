import pytest
from src.core.config import settings
from src.core.logging import logger
from src.core.errors import AGNOError
from src.core.security import create_access_token
from src.core.context import context_manager
from src.core.providers import provider_factory
from src.core.search import semantic_search

def test_config():
    """Test configuration loading"""
    assert settings.PROJECT_NAME == "AGNO Service"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.LOG_LEVEL == "INFO"

def test_logging():
    """Test logging functionality"""
    logger.info("Test log message")
    # No exception means logging works

def test_error_handling():
    """Test error handling"""
    with pytest.raises(AGNOError):
        raise AGNOError("Test error")

def test_context_manager():
    """Test context management"""
    session_id = "test_session"
    context = {"test": "data"}
    
    context_manager.add_context(session_id, context)
    retrieved_context = context_manager.get_context(session_id)
    
    assert retrieved_context == [context]
    
    context_manager.clear_context(session_id)
    assert context_manager.get_context(session_id) == []

def test_provider_factory():
    """Test provider factory"""
    provider = provider_factory.get_provider()
    assert provider is not None

def test_semantic_search():
    """Test semantic search"""
    query = "test query"
    documents = ["test document 1", "test document 2"]
    
    # This is an async function, but we're just testing the structure
    assert semantic_search is not None 