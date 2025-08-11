"""Unified Neo4j storage for both graph and vector operations.

This module implements a unified storage backend that uses Neo4j 5.13+
for both graph structures and vector embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from pathlib import Path
from loguru import logger
from neo4j import AsyncGraphDatabase
import numpy as np

# Try to import LightRAG base classes
try:
    from lightrag.kg.base_storage import BaseGraphStorage, BaseVectorStorage
except ImportError:
    # Fallback for development
    logger.warning("LightRAG base storage not available, using mock interfaces")
    
    class BaseGraphStorage:
        async def initialize(self):
            pass
            
        async def close(self):
            pass
    
    class BaseVectorStorage:
        async def initialize(self):
            pass
            
        async def close(self):
            pass


class Neo4JUnifiedStorage(BaseGraphStorage, BaseVectorStorage):
    """Unified Neo4j storage for both graph and vectors.
    
    This implementation uses Neo4j 5.13+ to store both graph structures
    and vector embeddings as node properties, enabling hybrid queries
    that combine graph traversal with vector similarity search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified storage with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - uri: Neo4j connection URI
                - username: Neo4j username
                - password: Neo4j password
                - database: Database name (default: neo4j)
                - workspace: Workspace for isolation (default: base)
                - vector_dimensions: Vector dimensions (default: 1536)
                - max_connection_pool_size: Max connections (default: 100)
                - vector_similarity: Similarity function (default: cosine)
                - vector_quantization: Enable quantization (default: True)
                - hnsw_m: HNSW m parameter (default: 16)
                - hnsw_ef_construction: HNSW efConstruction (default: 200)
        """
        self.config = config
        self.uri = config["uri"]
        self.username = config["username"]
        self.password = config["password"]
        self.database = config.get("database", "neo4j")
        self.workspace = config.get("workspace", "base")
        self.vector_dimensions = config.get("vector_dimensions", 1536)
        self.max_pool_size = config.get("max_connection_pool_size", 100)
        
        # NEW: Vector index configuration
        self.vector_config = self._prepare_vector_config()
        
        # Initialize driver
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_pool_size=self.max_pool_size
        )
        
        logger.info(f"Initialized Neo4JUnifiedStorage for workspace: {self.workspace}")
        logger.info(f"Vector dimensions: {self.vector_dimensions}")
        logger.info(f"Vector quantization: {self.vector_config['vector.quantization.enabled']}")
        logger.info(f"HNSW parameters: m={self.vector_config['vector.hnsw.m']}, efConstruction={self.vector_config['vector.hnsw.ef_construction']}")
        
    def _prepare_vector_config(self) -> Dict[str, Any]:
        """Prepare HNSW vector index configuration with optimal parameters.
        
        Returns:
            Dictionary with vector index configuration
        """
        return {
            'vector.dimensions': self.config.get('vector_dimensions', 1536),
            'vector.similarity_function': self.config.get('vector_similarity', 'cosine'),
            'vector.quantization.enabled': self.config.get('vector_quantization', True),
            'vector.hnsw.m': self.config.get('hnsw_m', 16),
            'vector.hnsw.ef_construction': self.config.get('hnsw_ef_construction', 200)
        }
        
    async def initialize(self):
        """Initialize storage and create indexes."""
        logger.info("Initializing unified Neo4j storage...")
        
        # Create vector indexes
        await self._create_vector_indexes()
        
        # Create graph indexes
        await self._create_graph_indexes()
        
        logger.info("Unified storage initialized successfully")
        
    async def _create_vector_indexes(self):
        """Create optimized vector indexes for entities, chunks, and relationships."""
        logger.info("Creating vector indexes with HNSW optimization...")
        
        # Use research-based optimal HNSW parameters from vector_config
        indexes = [
            ("entity_embeddings", "Entity", "embedding"),
            ("chunk_embeddings", "Chunk", "embedding"), 
            ("relationship_embeddings", "RELATED", "embedding")
        ]
        
        async with self.driver.session(database=self.database) as session:
            for index_name, label, property_name in indexes:
                if label == "RELATED":
                    # Relationship vector index
                    cypher = f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR ()-[r:{label}]-() ON (r.{property_name})
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.vector_config['vector.dimensions']},
                                `vector.similarity_function`: '{self.vector_config['vector.similarity_function']}',
                                `vector.quantization.enabled`: {str(self.vector_config['vector.quantization.enabled']).lower()},
                                `vector.hnsw.m`: {self.vector_config['vector.hnsw.m']},
                                `vector.hnsw.ef_construction`: {self.vector_config['vector.hnsw.ef_construction']}
                            }}
                        }}
                    """
                else:
                    # Node vector index
                    cypher = f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR ({label.lower()[0]}:{label}) ON ({label.lower()[0]}.{property_name})
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.vector_config['vector.dimensions']},
                                `vector.similarity_function`: '{self.vector_config['vector.similarity_function']}',
                                `vector.quantization.enabled`: {str(self.vector_config['vector.quantization.enabled']).lower()},
                                `vector.hnsw.m`: {self.vector_config['vector.hnsw.m']},
                                `vector.hnsw.ef_construction`: {self.vector_config['vector.hnsw.ef_construction']}
                            }}
                        }}
                    """
                
                try:
                    await session.run(cypher)
                    logger.info(f"Vector index {index_name} created/verified with HNSW optimization")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Vector index {index_name} creation warning: {e}")
        
    async def get_vector_index_stats(self) -> Dict[str, Any]:
        """Get statistics about vector indexes.
        
        Returns:
            Dictionary with vector index statistics
        """
        stats = {}
        
        query = """
        SHOW VECTOR INDEXES
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query)
                
                async for record in result:
                    index_name = record.get("name", "unknown")
                    stats[index_name] = {
                        "state": record.get("state", "unknown"),
                        "type": record.get("type", "unknown"),
                        "options": record.get("options", {})
                    }
                
                logger.debug(f"Retrieved vector index statistics for {len(stats)} indexes")
                return stats
        except Exception as e:
            logger.warning(f"Could not get vector index stats: {e}")
            return {}
    
    async def _create_graph_indexes(self):
        """Create graph indexes for better performance."""
        logger.info("Creating graph indexes...")
        
        queries = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_workspace_index IF NOT EXISTS FOR (e:Entity) ON (e.workspace)",
            "CREATE INDEX chunk_workspace_index IF NOT EXISTS FOR (c:Chunk) ON (c.workspace)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Graph index creation warning: {e}")
    
    # Vector Storage Operations
    
    async def upsert_vector(
        self,
        collection: str,
        entity_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector as node property.
        
        Args:
            collection: Collection name (Entity, Chunk, etc.)
            entity_id: Unique identifier for the entity
            embedding: Vector embedding (list of floats)
            metadata: Additional metadata to store
            
        Returns:
            Success status
        """
        if metadata is None:
            metadata = {}
            
        query = f"""
        MERGE (n:{collection} {{id: $entity_id, workspace: $workspace}})
        SET n.embedding = $embedding
        SET n += $metadata
        RETURN n
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    entity_id=entity_id,
                    embedding=embedding,
                    metadata=metadata,
                    workspace=self.workspace
                )
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            return False
    
    async def search_similar(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Vector similarity search using Neo4j vector index.
        
        Args:
            collection: Collection to search (Entity, Chunk, etc.)
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional filters to apply
            
        Returns:
            List of similar nodes with scores
        """
        index_name = f"{collection.lower()}_embedding_index"
        
        # Build filter clause
        filter_clause = "WHERE node.workspace = $workspace"
        if filter_dict:
            for key, value in filter_dict.items():
                filter_clause += f" AND node.{key} = ${key}"
        
        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k,
            $query_embedding
        ) YIELD node, score
        {filter_clause}
        RETURN node, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        try:
            params = {
                'top_k': top_k,
                'query_embedding': query_embedding,
                'workspace': self.workspace
            }
            if filter_dict:
                params.update(filter_dict)
                
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, **params)
                
                results = []
                async for record in result:
                    node = dict(record["node"])
                    results.append({
                        "node": node,
                        "score": float(record["score"]),
                        "id": node.get("id"),
                        "content": node.get("content", ""),
                        "metadata": {k: v for k, v in node.items() 
                                   if k not in ["embedding", "id", "content"]}
                    })
                return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_vector(self, collection: str, entity_id: str) -> bool:
        """Delete a vector from storage.
        
        Args:
            collection: Collection name
            entity_id: Entity identifier
            
        Returns:
            Success status
        """
        query = f"""
        MATCH (n:{collection} {{id: $entity_id, workspace: $workspace}})
        REMOVE n.embedding
        RETURN n
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    entity_id=entity_id,
                    workspace=self.workspace
                )
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            return False
    
    # Graph Storage Operations
    
    async def create_node(
        self,
        node_type: str,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Create a node in the graph.
        
        Args:
            node_type: Type/label of the node
            node_id: Unique identifier
            properties: Node properties
            
        Returns:
            Success status
        """
        query = f"""
        MERGE (n:{node_type} {{id: $node_id, workspace: $workspace}})
        SET n += $properties
        RETURN n
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    node_id=node_id,
                    workspace=self.workspace,
                    properties=properties
                )
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return False
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Success status
        """
        if properties is None:
            properties = {}
            
        query = f"""
        MATCH (a {{id: $source_id, workspace: $workspace}})
        MATCH (b {{id: $target_id, workspace: $workspace}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN r
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    workspace=self.workspace,
                    properties=properties
                )
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    async def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get neighbors of a node.
        
        Args:
            node_id: Node identifier
            relationship_type: Optional relationship type filter
            direction: Direction (in, out, both)
            
        Returns:
            List of neighbor nodes
        """
        rel_pattern = f":{relationship_type}" if relationship_type else ""
        
        if direction == "out":
            pattern = f"(n)-[r{rel_pattern}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n)<-[r{rel_pattern}]-(neighbor)"
        else:
            pattern = f"(n)-[r{rel_pattern}]-(neighbor)"
        
        query = f"""
        MATCH (n {{id: $node_id, workspace: $workspace}})
        MATCH {pattern}
        WHERE neighbor.workspace = $workspace
        RETURN neighbor, r
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    node_id=node_id,
                    workspace=self.workspace
                )
                
                neighbors = []
                async for record in result:
                    neighbors.append({
                        "node": dict(record["neighbor"]),
                        "relationship": dict(record["r"]) if record["r"] else None
                    })
                return neighbors
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []
    
    # Hybrid Operations (Graph + Vector)
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        graph_depth: int = 2,
        top_k: int = 10,
        collection: str = "Entity"
    ) -> List[Dict[str, Any]]:
        """Combined vector similarity + graph traversal.
        
        Args:
            query_embedding: Query vector
            graph_depth: Depth of graph traversal
            top_k: Number of results
            collection: Collection to search
            
        Returns:
            List of results with graph context
        """
        index_name = f"{collection.lower()}_embedding_index"
        
        # Build query with actual depth value embedded
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        WHERE node.workspace = $workspace
        WITH node, score
        OPTIONAL MATCH path = (node)-[*1..{graph_depth}]-(neighbor)
        WHERE neighbor.workspace = $workspace
        WITH node, score, 
             collect(distinct neighbor) as neighbors,
             collect(distinct path) as paths
        RETURN node, score, neighbors, 
               [p in paths | nodes(p)] as path_nodes
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    index_name=index_name,
                    top_k=top_k,
                    query_embedding=query_embedding,
                    workspace=self.workspace
                )
                
                results = []
                async for record in result:
                    results.append({
                        "node": dict(record["node"]),
                        "score": float(record["score"]),
                        "neighbors": [dict(n) for n in record["neighbors"]],
                        "paths": record["path_nodes"]
                    })
                return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    # Utility Methods
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        queries = {
            "total_nodes": "MATCH (n) WHERE n.workspace = $workspace RETURN count(n) as count",
            "entities_with_vectors": "MATCH (n:Entity) WHERE n.workspace = $workspace AND n.embedding IS NOT NULL RETURN count(n) as count",
            "chunks_with_vectors": "MATCH (n:Chunk) WHERE n.workspace = $workspace AND n.embedding IS NOT NULL RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        try:
            async with self.driver.session(database=self.database) as session:
                for key, query in queries.items():
                    result = await session.run(query, workspace=self.workspace)
                    record = await result.single()
                    stats[key] = record["count"] if record else 0
                    
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def clear_workspace(self) -> bool:
        """Clear all data in the current workspace.
        
        Returns:
            Success status
        """
        query = """
        MATCH (n {workspace: $workspace})
        DETACH DELETE n
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(query, workspace=self.workspace)
                logger.info(f"Cleared workspace: {self.workspace}")
                return True
        except Exception as e:
            logger.error(f"Failed to clear workspace: {e}")
            return False
    
    async def close(self):
        """Close the database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")