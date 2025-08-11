#!/usr/bin/env python3
"""Production RAG system interface."""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
import openai
from neo4j import AsyncGraphDatabase
from src.config import Settings
from src.core.embeddings import OpenAIEmbeddings
from src.core.knn_graph import KNNGraph
from src.core.query_classifier import QueryClassifier, QueryType
from src.core.reranker import Reranker


class RAGSystem:
    """Production RAG system using Neo4j and OpenAI."""
    
    def __init__(self, workspace: str = "sast_dast_enhanced", settings: Optional[Settings] = None):
        """Initialize RAG system."""
        self.settings = settings or Settings()
        self.workspace = workspace  # Configurable workspace
        
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        
        # Initialize embedding generator
        self.embedding_gen = OpenAIEmbeddings(
            api_key=self.settings.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize Neo4j driver
        self.driver = AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )
        
        # Initialize KNN graph for graph traversal (tighter parameters)
        self.knn_graph = KNNGraph(
            min_similarity=0.85,  # Much higher threshold for quality
            max_neighbors=5  # Fewer but higher quality neighbors
        )
        
        # Initialize query classifier and reranker
        self.query_classifier = QueryClassifier()
        self.reranker = Reranker(similarity_threshold=0.7)
    
    async def vector_search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure vector search on document chunks (98% accuracy for fact-based queries)."""
        
        # Generate query embedding
        query_embedding = await self.embedding_gen([query])
        query_embedding = query_embedding[0].tolist()
        
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.embedding IS NOT NULL
                WITH c, 
                     reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) | 
                            dot + (c.embedding[i] * $query_embedding[i])) as similarity
                WHERE similarity > 0.2
                RETURN c.content as content, 
                       c.id as chunk_id,
                       similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, 
                workspace=self.workspace, 
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            chunks = []
            async for record in result:
                chunks.append({
                    "content": record["content"],
                    "chunk_id": record["chunk_id"],
                    "similarity": record["similarity"]
                })
            
            return chunks
    
    async def graph_search_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Graph traversal search using KNN SIMILAR relationships.
        
        This method:
        1. Finds initial chunks via vector search
        2. Traverses SIMILAR relationships to gather related chunks
        3. Returns expanded context from multiple connected chunks
        """
        
        # Generate query embedding
        query_embedding = await self.embedding_gen([query])
        query_embedding = query_embedding[0].tolist()
        
        async with self.driver.session() as session:
            # Find initial entry points (top 3-5 chunks)
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.embedding IS NOT NULL
                WITH c, 
                     reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) | 
                            dot + (c.embedding[i] * $query_embedding[i])) as similarity
                WHERE similarity > 0.2
                RETURN c.id as chunk_id, c.content as content, similarity
                ORDER BY similarity DESC
                LIMIT 7
            """, 
                workspace=self.workspace, 
                query_embedding=query_embedding
            )
            
            initial_chunks = []
            chunk_ids = set()
            async for record in result:
                initial_chunks.append({
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "similarity": record["similarity"]
                })
                chunk_ids.add(record["chunk_id"])
            
            if not initial_chunks:
                return []
            
            # Traverse graph to find related chunks
            all_chunks = list(initial_chunks)
            
            for chunk in initial_chunks[:5]:  # Traverse from top 5 chunks
                # Find similar chunks via SIMILAR relationships
                similar_chunks = await self.knn_graph.find_similar_chunks(
                    session,
                    self.workspace,
                    chunk["chunk_id"],
                    limit=7,
                    min_score=0.5
                )
                
                for sim_chunk in similar_chunks:
                    if sim_chunk["id"] not in chunk_ids:
                        chunk_ids.add(sim_chunk["id"])
                        all_chunks.append({
                            "chunk_id": sim_chunk["id"],
                            "content": sim_chunk["content"],
                            "similarity": sim_chunk["score"] * 0.8  # Reduce score for indirect matches
                        })
            
            # Sort by similarity and return top K
            all_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            return all_chunks[:top_k]
    
    async def graph_search_chunks_optimized(
        self, 
        query: str, 
        initial_chunks: int = 3,
        traversal_depth: int = 2,
        similarity_threshold: float = 0.8,
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """Optimized graph traversal with depth limiting and similarity decay.
        
        Args:
            query: Search query
            initial_chunks: Number of seed chunks for traversal
            traversal_depth: Maximum traversal depth
            similarity_threshold: Minimum similarity for traversal
            max_chunks: Maximum chunks to return
        """
        
        # Generate query embedding
        query_embedding = await self.embedding_gen([query])
        query_embedding = query_embedding[0].tolist()
        
        async with self.driver.session() as session:
            # Find initial entry points (fewer, higher quality)
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.embedding IS NOT NULL
                WITH c, 
                     reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) | 
                            dot + (c.embedding[i] * $query_embedding[i])) as similarity
                WHERE similarity > $sim_threshold
                RETURN c.id as chunk_id, c.content as content, similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """, 
                workspace=self.workspace, 
                query_embedding=query_embedding,
                sim_threshold=similarity_threshold * 0.8,  # Slightly lower for initial
                limit=initial_chunks
            )
            
            chunks_by_depth = {0: []}
            chunk_ids = set()
            
            async for record in result:
                chunks_by_depth[0].append({
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "similarity": record["similarity"],
                    "depth": 0
                })
                chunk_ids.add(record["chunk_id"])
            
            if not chunks_by_depth[0]:
                return []
            
            # Depth-limited traversal with similarity decay
            for depth in range(1, traversal_depth + 1):
                chunks_by_depth[depth] = []
                decay_factor = 0.8 ** depth  # Exponential decay
                
                # Only traverse from high-quality chunks
                source_chunks = [
                    c for c in chunks_by_depth[depth - 1]
                    if c["similarity"] > similarity_threshold
                ][:2]  # Limit sources per depth
                
                for chunk in source_chunks:
                    # Find similar chunks via SIMILAR relationships
                    similar_chunks = await self.knn_graph.find_similar_chunks(
                        session,
                        self.workspace,
                        chunk["chunk_id"],
                        limit=3,  # Fewer per hop
                        min_score=similarity_threshold
                    )
                    
                    for sim_chunk in similar_chunks:
                        if sim_chunk["id"] not in chunk_ids:
                            chunk_ids.add(sim_chunk["id"])
                            chunks_by_depth[depth].append({
                                "chunk_id": sim_chunk["id"],
                                "content": sim_chunk["content"],
                                "similarity": sim_chunk["score"] * decay_factor,
                                "depth": depth
                            })
                    
                    # Stop if we have enough chunks
                    if len(chunk_ids) >= max_chunks * 2:
                        break
            
            # Combine all chunks
            all_chunks = []
            for depth_chunks in chunks_by_depth.values():
                all_chunks.extend(depth_chunks)
            
            # Sort by similarity and return top chunks
            all_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            return all_chunks[:max_chunks]
    
    async def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and graph approaches.
        
        Runs both vector and graph searches, then merges results with weighted scoring.
        """
        
        # Run both searches in parallel
        vector_results = await self.vector_search_chunks(query, top_k=top_k)
        graph_results = await self.graph_search_chunks(query, top_k=top_k)
        
        # Merge results with weighted scoring
        chunk_scores = {}
        
        # Add vector results (weight 0.6)
        for result in vector_results:
            chunk_id = result["chunk_id"]
            chunk_scores[chunk_id] = {
                "content": result["content"],
                "vector_score": result["similarity"] * 0.6,
                "graph_score": 0
            }
        
        # Add graph results (weight 0.4)
        for result in graph_results:
            chunk_id = result["chunk_id"]
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]["graph_score"] = result["similarity"] * 0.4
            else:
                chunk_scores[chunk_id] = {
                    "content": result["content"],
                    "vector_score": 0,
                    "graph_score": result["similarity"] * 0.4
                }
        
        # Calculate combined scores
        merged_results = []
        for chunk_id, scores in chunk_scores.items():
            merged_results.append({
                "chunk_id": chunk_id,
                "content": scores["content"],
                "similarity": scores["vector_score"] + scores["graph_score"]
            })
        
        # Sort by combined score and return top K
        merged_results.sort(key=lambda x: x["similarity"], reverse=True)
        return merged_results[:top_k]
    
    async def hybrid_search_advanced(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Advanced hybrid search with query classification and reranking.
        
        This method:
        1. Classifies the query to determine optimal retrieval strategy
        2. Runs adaptive retrieval with dynamic weights
        3. Applies reranking and result optimization
        """
        
        # Classify query
        query_type, weights = self.query_classifier.classify(query)
        retrieval_params = self.query_classifier.get_retrieval_params(query_type)
        
        logger.debug(f"Query type: {query_type.value}, Weights: {weights}")
        
        # Get query embedding for reranking
        query_embedding = await self.embedding_gen([query])
        query_embedding = query_embedding[0].tolist()
        
        # Run searches with adaptive parameters
        vector_results = await self.vector_search_chunks(
            query, 
            top_k=retrieval_params["initial_chunks"] * 2
        )
        
        # For graph search, use optimized parameters
        graph_results = await self.graph_search_chunks_optimized(
            query,
            initial_chunks=retrieval_params["initial_chunks"],
            traversal_depth=retrieval_params["traversal_depth"],
            similarity_threshold=retrieval_params["similarity_threshold"],
            max_chunks=retrieval_params["max_graph_chunks"]
        )
        
        # Merge results with adaptive weights
        all_chunks = []
        chunk_map = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "similarity": result["similarity"] * weights["vector"],
                    "source": "vector",
                    "embedding": None  # Would need to fetch if needed
                }
            else:
                chunk_map[chunk_id]["similarity"] += result["similarity"] * weights["vector"]
                chunk_map[chunk_id]["source"] = "both"
        
        # Add graph results
        for result in graph_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": result["content"],
                    "similarity": result["similarity"] * weights["graph"],
                    "source": "graph",
                    "embedding": None
                }
            else:
                chunk_map[chunk_id]["similarity"] += result["similarity"] * weights["graph"]
                chunk_map[chunk_id]["source"] = "both"
        
        # Convert to list
        all_chunks = list(chunk_map.values())
        
        # Apply reranking
        reranked_chunks = self.reranker.rerank_results(
            query_embedding,
            all_chunks,
            query_type.value
        )
        
        return reranked_chunks[:top_k]
    
    async def query(self, question: str, search_mode: str = "graph", top_k: int = 5) -> str:
        """
        Query the RAG system with multiple search modes.
        
        Args:
            question: The user's question
            search_mode: Search strategy to use:
                - "vector": Pure vector search (baseline)
                - "graph": Graph traversal using SIMILAR relationships (best for split info)
                - "hybrid": Combination of vector and graph (balanced)
                - "hybrid_advanced": Advanced hybrid with query classification and reranking
            top_k: Number of results to retrieve
        
        Returns:
            Generated answer based on retrieved context
        """
        
        # Retrieve relevant context based on mode
        if search_mode == "vector" or search_mode == "vector_chunks":
            search_results = await self.vector_search_chunks(question, top_k)
        elif search_mode == "graph":
            search_results = await self.graph_search_chunks(question, top_k)
        elif search_mode == "hybrid":
            search_results = await self.hybrid_search(question, top_k)
        elif search_mode == "hybrid_advanced":
            search_results = await self.hybrid_search_advanced(question, top_k)
        else:
            # Default to graph mode for best accuracy
            search_results = await self.graph_search_chunks(question, top_k)
        
        # Build context
        context_parts = []
        for result in search_results:
            context_parts.append(result["content"])
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on the provided context. Be accurate and specific."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ],
            temperature=0.0,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        
        async with self.driver.session() as session:
            # Count entities
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                RETURN count(e) as count
            """, workspace=self.workspace)
            record = await result.single()
            entity_count = record["count"] if record else 0
            
            # Count chunks
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                RETURN count(c) as count
            """, workspace=self.workspace)
            record = await result.single()
            chunk_count = record["count"] if record else 0
            
            # Count relationships
            result = await session.run("""
                MATCH (e1:Entity {workspace: $workspace})-[r]-(e2:Entity {workspace: $workspace})
                RETURN count(DISTINCT r) as count
            """, workspace=self.workspace)
            record = await result.single()
            rel_count = record["count"] if record else 0
            
            return {
                "entities": entity_count,
                "chunks": chunk_count,
                "relationships": rel_count
            }
    
    async def close(self):
        """Close database connection."""
        await self.driver.close()


async def main():
    """Example usage of RAG system."""
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Get statistics
    stats = await rag.get_statistics()
    logger.info(f"Database stats: {stats}")
    
    # Example queries
    test_questions = [
        "What type of testing does Aikido provide?",
        "What is the pricing model for OX Security?",
        "Which tools support broker deployment?"
    ]
    
    for question in test_questions:
        logger.info(f"\nQuestion: {question}")
        
        # Use vector search on chunks (recommended for production)
        answer = await rag.query(question, search_mode="vector_chunks")
        logger.info(f"Answer: {answer[:200]}...")
    
    # Close connection
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())