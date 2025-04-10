from typing import List, Dict, Optional
import numpy as np
from .config import settings
from .logging import logger
from .errors import AGNOError
from .providers import provider_factory

class SemanticSearch:
    def __init__(self):
        self.provider = provider_factory.get_provider()
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = await self.provider.get_embeddings(text)
        self.embeddings_cache[text] = np.array(embedding)
        return self.embeddings_cache[text]

    async def similarity_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """Perform similarity search on documents"""
        try:
            query_embedding = await self.get_embedding(query)
            doc_embeddings = [await self.get_embedding(doc) for doc in documents]
            
            # Calculate cosine similarity
            similarities = []
            for doc, doc_embedding in zip(documents, doc_embeddings):
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append({"document": doc, "similarity": float(similarity)})
            
            # Sort by similarity and return top_k results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            raise AGNOError(f"Semantic search failed: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the embeddings cache"""
        self.embeddings_cache.clear()

semantic_search = SemanticSearch() 