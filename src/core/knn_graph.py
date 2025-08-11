#!/usr/bin/env python3
"""KNN Graph implementation for building chunk similarity relationships."""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
import time


class KNNGraph:
    """Manages K-Nearest Neighbors graph for chunk similarity relationships.
    
    This creates SIMILAR relationships between content chunks based on
    embedding similarity, enabling graph traversal for information discovery.
    """
    
    def __init__(
        self,
        min_similarity: float = 0.7,
        max_neighbors: int = 15,
        batch_size: int = 50
    ):
        """Initialize KNN graph manager.
        
        Args:
            min_similarity: Minimum similarity score for creating SIMILAR relationships
            max_neighbors: Maximum number of similar neighbors per chunk
            batch_size: Batch size for processing
        """
        self.min_similarity = min_similarity
        self.max_neighbors = max_neighbors
        self.batch_size = batch_size
        
    async def build_knn_relationships(
        self,
        session,
        workspace: str,
        rebuild: bool = False
    ) -> int:
        """Build KNN SIMILAR relationships between chunks.
        
        Args:
            session: Neo4j session
            workspace: Workspace to process
            rebuild: Whether to rebuild all relationships
            
        Returns:
            Number of relationships created
        """
        start_time = time.time()
        
        # Clear existing relationships if rebuilding
        if rebuild:
            logger.info("Clearing existing SIMILAR relationships...")
            await session.run("""
                MATCH (c1:Chunk {workspace: $workspace})-[r:SIMILAR]-(c2:Chunk {workspace: $workspace})
                DELETE r
            """, workspace=workspace)
        
        # Get all chunks with embeddings
        result = await session.run("""
            MATCH (c:Chunk {workspace: $workspace})
            WHERE c.embedding IS NOT NULL
            RETURN c.id as id, c.embedding as embedding
            ORDER BY c.id
        """, workspace=workspace)
        
        chunks = []
        async for record in result:
            chunks.append({
                'id': record['id'],
                'embedding': record['embedding']
            })
        
        if not chunks:
            logger.warning("No chunks with embeddings found")
            return 0
            
        logger.info(f"Processing {len(chunks)} chunks for KNN relationships...")
        
        relationships_created = 0
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            
            chunk_id = chunk['id']
            chunk_embedding = np.array(chunk['embedding'])
            
            # Find similar chunks
            similarities = []
            for j, other in enumerate(chunks):
                if i == j:  # Skip self
                    continue
                    
                other_embedding = np.array(other['embedding'])
                similarity = self._cosine_similarity(chunk_embedding, other_embedding)
                
                if similarity >= self.min_similarity:
                    similarities.append((other['id'], similarity))
            
            # Sort by similarity and take top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:self.max_neighbors]
            
            # Create relationships
            for target_id, score in top_similar:
                await session.run("""
                    MATCH (c1:Chunk {workspace: $workspace, id: $source_id})
                    MATCH (c2:Chunk {workspace: $workspace, id: $target_id})
                    MERGE (c1)-[r:SIMILAR]-(c2)
                    SET r.score = $score,
                        r.created_at = timestamp()
                """, 
                    workspace=workspace,
                    source_id=chunk_id,
                    target_id=target_id,
                    score=float(score)
                )
                relationships_created += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Created {relationships_created} SIMILAR relationships in {elapsed:.2f}s")
        
        return relationships_created
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    async def find_similar_chunks(
        self,
        session,
        workspace: str,
        chunk_id: str,
        limit: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to a given chunk using SIMILAR relationships.
        
        Args:
            session: Neo4j session
            workspace: Workspace
            chunk_id: ID of the source chunk
            limit: Maximum number of similar chunks to return
            min_score: Minimum similarity score
            
        Returns:
            List of similar chunks with scores
        """
        min_score = min_score or self.min_similarity
        
        result = await session.run("""
            MATCH (source:Chunk {workspace: $workspace, id: $chunk_id})-[r:SIMILAR]-(target:Chunk)
            WHERE r.score >= $min_score
            RETURN target.id as id,
                   target.content as content,
                   r.score as score
            ORDER BY r.score DESC
            LIMIT $limit
        """,
            workspace=workspace,
            chunk_id=chunk_id,
            min_score=min_score,
            limit=limit
        )
        
        chunks = []
        async for record in result:
            chunks.append({
                'id': record['id'],
                'content': record['content'],
                'score': record['score']
            })
        
        return chunks
    
    async def get_graph_stats(self, session, workspace: str) -> Dict[str, Any]:
        """Get statistics about the KNN graph.
        
        Args:
            session: Neo4j session
            workspace: Workspace
            
        Returns:
            Dictionary with graph statistics
        """
        result = await session.run("""
            MATCH (c:Chunk {workspace: $workspace})
            WITH count(c) as total_chunks
            OPTIONAL MATCH (c1:Chunk {workspace: $workspace})-[r:SIMILAR]-(c2:Chunk {workspace: $workspace})
            WITH total_chunks, count(DISTINCT r) as total_relationships, avg(r.score) as avg_score
            RETURN total_chunks, total_relationships/2 as total_relationships, avg_score
        """, workspace=workspace)
        
        record = await result.single()
        
        if record:
            total_chunks = record["total_chunks"]
            total_relationships = record["total_relationships"] or 0
            avg_score = record["avg_score"] or 0
            
            return {
                "total_chunks": total_chunks,
                "total_relationships": total_relationships,
                "avg_similarity_score": avg_score,
                "avg_neighbors_per_chunk": (
                    (2 * total_relationships) / total_chunks 
                    if total_chunks > 0 else 0
                )
            }
        else:
            return {
                "total_chunks": 0,
                "total_relationships": 0,
                "avg_similarity_score": 0,
                "avg_neighbors_per_chunk": 0
            }