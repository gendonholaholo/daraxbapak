"""
Milvus client for semantic search functionality.
"""

import os
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import logging
import numpy as np

class MilvusClient:
    """Client for interacting with Milvus vector database."""
    
    def __init__(self, host: str = "localhost", port: str = "19530"):
        """Initialize Milvus client."""
        self.host = host
        self.port = port
        self.collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.connect()
    
    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(host=self.host, port=self.port)
            if not utility.has_collection("documents"):
                self._create_collection()
            self.collection = Collection("documents")
            self.collection.load()
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _create_collection(self):
        """Set up collection if it doesn't exist."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            schema = CollectionSchema(fields=fields, description="Document collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()
    
    def insert_documents(self, documents: List[str]) -> List[int]:
        """
        Insert documents into the collection.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Generate embeddings
        embeddings = self.model.encode(documents)
        
        # Prepare data for insertion
        entities = [
            documents,  # text field
            embeddings.tolist()  # embedding field
        ]
        
        # Insert data
        mr = self.collection.insert(entities)
        return mr.primary_keys
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "text": hit.entity.get('text'),
                    "distance": hit.distance
                })
        
        return formatted_results
    
    def delete_documents(self, ids: List[int]) -> None:
        """
        Delete documents from the collection.
        
        Args:
            ids: List of document IDs to delete
        """
        expr = f"id in {ids}"
        self.collection.delete(expr)
    
    def __del__(self):
        """Clean up connections."""
        connections.disconnect("default") 