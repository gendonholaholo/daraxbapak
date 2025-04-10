import pytest
from src.search.milvus_client import MilvusClient
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture
def milvus_client():
    with patch("pymilvus.connections.connect"), \
         patch("pymilvus.utility.has_collection"), \
         patch("pymilvus.Collection"):
        client = MilvusClient()
        yield client

@pytest.fixture
def sample_documents():
    return [
        "This is the first document",
        "This is the second document",
        "This is the third document"
    ]

def test_connect(milvus_client):
    # Verify connection was established
    assert milvus_client.collection is not None

def test_insert_documents(milvus_client, sample_documents):
    with patch("sentence_transformers.SentenceTransformer.encode") as mock_encode:
        # Setup mock embeddings
        mock_embeddings = np.random.rand(len(sample_documents), 384)
        mock_encode.return_value = mock_embeddings
        
        # Mock collection insert
        mock_insert = MagicMock()
        mock_insert.primary_keys = [1, 2, 3]
        milvus_client.collection.insert.return_value = mock_insert
        
        # Insert documents
        ids = milvus_client.insert_documents(sample_documents)
        
        # Verify results
        assert len(ids) == len(sample_documents)
        assert all(isinstance(id, int) for id in ids)
        
        # Verify embeddings were generated
        mock_encode.assert_called_once_with(sample_documents)
        
        # Verify collection insert
        milvus_client.collection.insert.assert_called_once()
        call_args = milvus_client.collection.insert.call_args[0]
        assert len(call_args) == 2  # text and embeddings
        assert call_args[0] == sample_documents

def test_search(milvus_client):
    with patch("sentence_transformers.SentenceTransformer.encode") as mock_encode:
        # Setup mock query embedding
        mock_embedding = np.random.rand(384)
        mock_encode.return_value = [mock_embedding]
        
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.distance = 0.1
        mock_hit.entity = {"text": "Test document"}
        
        mock_hits = MagicMock()
        mock_hits.__iter__.return_value = [mock_hit]
        
        mock_search = MagicMock()
        mock_search.__iter__.return_value = [mock_hits]
        milvus_client.collection.search.return_value = mock_search
        
        # Perform search
        results = milvus_client.search("test query", top_k=5)
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["text"] == "Test document"
        assert results[0]["distance"] == 0.1
        
        # Verify search was called with correct parameters
        milvus_client.collection.search.assert_called_once()
        call_args = milvus_client.collection.search.call_args
        assert call_args[1]["anns_field"] == "embedding"
        assert call_args[1]["limit"] == 5

def test_delete_documents(milvus_client):
    # Mock collection delete
    milvus_client.collection.delete = MagicMock()
    
    # Delete documents
    ids = [1, 2, 3]
    milvus_client.delete_documents(ids)
    
    # Verify delete was called
    milvus_client.collection.delete.assert_called_once_with(f"id in {ids}")

def test_error_handling(milvus_client):
    # Test error during document insertion
    milvus_client.collection.insert.side_effect = Exception("Insert error")
    
    with pytest.raises(Exception) as exc_info:
        milvus_client.insert_documents(["test document"])
    assert "Insert error" in str(exc_info.value)
    
    # Test error during search
    milvus_client.collection.search.side_effect = Exception("Search error")
    
    with pytest.raises(Exception) as exc_info:
        milvus_client.search("test query")
    assert "Search error" in str(exc_info.value)
    
    # Test error during deletion
    milvus_client.collection.delete.side_effect = Exception("Delete error")
    
    with pytest.raises(Exception) as exc_info:
        milvus_client.delete_documents([1])
    assert "Delete error" in str(exc_info.value) 