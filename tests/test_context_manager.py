import pytest
from src.context.context_manager import ContextManager
import json
from datetime import datetime

@pytest.fixture
def context_manager():
    return ContextManager()

@pytest.fixture
def sample_context():
    return {
        "conversation": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "metadata": {
            "session_id": "test_session"
        }
    }

@pytest.mark.asyncio
async def test_store_and_retrieve_context(context_manager, sample_context):
    # Store context
    await context_manager.store_context("test_session", sample_context)
    
    # Retrieve context
    retrieved_contexts = await context_manager.retrieve_context("test_session")
    
    # Verify context
    assert len(retrieved_contexts) == 1
    retrieved_context = retrieved_contexts[0]
    assert retrieved_context["conversation"] == sample_context["conversation"]
    assert retrieved_context["metadata"] == sample_context["metadata"]
    assert "timestamp" in retrieved_context

@pytest.mark.asyncio
async def test_update_context(context_manager, sample_context):
    # Store initial context
    await context_manager.store_context("test_session", sample_context)
    
    # Update context
    new_context = {"additional_info": "test"}
    await context_manager.update_context("test_session", new_context)
    
    # Retrieve updated context
    retrieved_contexts = await context_manager.retrieve_context("test_session")
    
    # Verify update
    assert len(retrieved_contexts) == 1
    retrieved_context = retrieved_contexts[0]
    assert "additional_info" in retrieved_context
    assert retrieved_context["additional_info"] == "test"
    assert "conversation" in retrieved_context  # Original context preserved

@pytest.mark.asyncio
async def test_clear_context(context_manager, sample_context):
    # Store context
    await context_manager.store_context("test_session", sample_context)
    
    # Clear context
    await context_manager.clear_context("test_session")
    
    # Verify context is cleared
    retrieved_contexts = await context_manager.retrieve_context("test_session")
    assert len(retrieved_contexts) == 0

@pytest.mark.asyncio
async def test_context_compression(context_manager):
    # Create a large context
    large_context = {
        "text": " ".join(["test"] * 1000)  # Create a large text
    }
    
    # Store context
    await context_manager.store_context("large_session", large_context)
    
    # Retrieve context
    retrieved_contexts = await context_manager.retrieve_context("large_session")
    
    # Verify context was compressed
    assert len(retrieved_contexts) == 1
    retrieved_context = retrieved_contexts[0]
    assert "timestamp" in retrieved_context
    assert retrieved_context["text"] == large_context["text"]

@pytest.mark.asyncio
async def test_max_size_limit(context_manager):
    # Store more contexts than max_size
    for i in range(1500):  # More than default max_size of 1000
        await context_manager.store_context("test_session", {"index": i})
    
    # Retrieve context
    retrieved_contexts = await context_manager.retrieve_context("test_session")
    
    # Verify only max_size contexts are kept
    assert len(retrieved_contexts) == 1000
    assert retrieved_contexts[0]["index"] == 500  # First entry should be 500th
    assert retrieved_contexts[-1]["index"] == 1499  # Last entry should be 1499th 