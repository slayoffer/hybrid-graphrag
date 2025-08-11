"""Reranking and result optimization for hybrid retrieval."""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class RetrievedChunk:
    """Container for retrieved chunk with metadata."""
    chunk_id: str
    content: str
    similarity: float
    source: str  # 'vector', 'graph', or 'both'
    embedding: List[float] = None
    
class Reranker:
    """Reranks and optimizes retrieval results."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize reranker.
        
        Args:
            similarity_threshold: Minimum similarity to keep chunks
        """
        self.similarity_threshold = similarity_threshold
        
    def rerank_results(
        self,
        query_embedding: List[float],
        chunks: List[Dict[str, Any]],
        query_type: str = "analytical"
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using multiple signals.
        
        Args:
            query_embedding: Query embedding vector
            chunks: List of retrieved chunks
            query_type: Type of query for adaptive reranking
            
        Returns:
            Reranked and filtered chunks
        """
        # First, deduplicate chunks
        unique_chunks = self._deduplicate_chunks(chunks)
        
        # Calculate enhanced similarity scores
        scored_chunks = []
        for chunk in unique_chunks:
            # Get base similarity
            base_score = chunk.get("similarity", 0.0)
            
            # Calculate cosine similarity if embedding available
            if "embedding" in chunk and chunk["embedding"]:
                cosine_sim = self._cosine_similarity(
                    query_embedding, 
                    chunk["embedding"]
                )
            else:
                cosine_sim = base_score
                
            # Apply source boost
            source_boost = self._get_source_boost(chunk.get("source", ""), query_type)
            
            # Calculate final score
            final_score = (cosine_sim * 0.7) + (base_score * 0.3) + source_boost
            
            # Add to scored chunks if above threshold
            if final_score >= self.similarity_threshold:
                chunk["reranked_score"] = final_score
                scored_chunks.append(chunk)
        
        # Sort by reranked score
        scored_chunks.sort(key=lambda x: x["reranked_score"], reverse=True)
        
        # Apply result compression
        compressed_chunks = self._compress_results(scored_chunks)
        
        return compressed_chunks
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks based on content similarity.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated chunks
        """
        seen = {}
        unique = []
        
        for chunk in chunks:
            # Create content hash for exact duplicates
            content_hash = hashlib.md5(
                chunk["content"].encode()
            ).hexdigest()
            
            # Check for exact duplicates
            if content_hash not in seen:
                seen[content_hash] = True
                unique.append(chunk)
            else:
                # If duplicate, keep the one with higher similarity
                for i, existing in enumerate(unique):
                    existing_hash = hashlib.md5(
                        existing["content"].encode()
                    ).hexdigest()
                    if existing_hash == content_hash:
                        if chunk.get("similarity", 0) > existing.get("similarity", 0):
                            unique[i] = chunk
                        break
        
        return unique
    
    def _compress_results(
        self, 
        chunks: List[Dict[str, Any]], 
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Compress results by removing redundant information.
        
        Args:
            chunks: Reranked chunks
            max_chunks: Maximum number of chunks to return
            
        Returns:
            Compressed chunk list
        """
        if len(chunks) <= max_chunks:
            return chunks
        
        # Keep top chunks and apply diversity
        compressed = []
        content_coverage = set()
        
        for chunk in chunks:
            # Extract key terms from chunk
            terms = set(chunk["content"].lower().split()[:20])
            
            # Check coverage overlap
            overlap = len(terms & content_coverage) / max(len(terms), 1)
            
            # Add chunk if it provides new information
            if overlap < 0.5 or len(compressed) < 3:
                compressed.append(chunk)
                content_coverage.update(terms)
                
            if len(compressed) >= max_chunks:
                break
        
        return compressed
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _get_source_boost(self, source: str, query_type: str) -> float:
        """
        Get source-based score boost.
        
        Args:
            source: Source of the chunk ('vector', 'graph', 'both')
            query_type: Type of query
            
        Returns:
            Score boost value
        """
        boosts = {
            "factual": {"vector": 0.1, "graph": 0.0, "both": 0.15},
            "analytical": {"vector": 0.05, "graph": 0.05, "both": 0.1},
            "complex": {"vector": 0.0, "graph": 0.1, "both": 0.15}
        }
        
        query_type_lower = query_type.lower()
        if query_type_lower not in boosts:
            query_type_lower = "analytical"
            
        return boosts[query_type_lower].get(source, 0.0)