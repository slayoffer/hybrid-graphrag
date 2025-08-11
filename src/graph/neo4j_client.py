"""Neo4j client for direct graph operations."""

from typing import Dict, Any, List, Optional
from loguru import logger

try:
    from neo4j import AsyncGraphDatabase, GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning("Neo4j driver not installed")
    NEO4J_AVAILABLE = False

from ..config import Settings


class Neo4jClient:
    """Direct Neo4j client for graph operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Neo4j client.
        
        Args:
            settings: Application settings
        """
        if settings is None:
            from ..config import settings
            self.settings = settings
        else:
            self.settings = settings
        
        self.driver = None
        self.async_driver = None
        
        if NEO4J_AVAILABLE:
            self._initialize_drivers()
    
    def _initialize_drivers(self):
        """Initialize Neo4j drivers."""
        try:
            # Sync driver
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
                max_connection_pool_size=self.settings.neo4j_max_connection_pool_size
            )
            
            # Async driver
            self.async_driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
                max_connection_pool_size=self.settings.neo4j_max_connection_pool_size
            )
            
            logger.info("Neo4j drivers initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j drivers: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j connection health.
        
        Returns:
            Health status
        """
        if not NEO4J_AVAILABLE or not self.async_driver:
            return {"status": "unavailable", "error": "Neo4j driver not available"}
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("RETURN 1 as num")
                record = await result.single()
                
                if record and record["num"] == 1:
                    # Get version info
                    version_result = await session.run("CALL dbms.components()")
                    version_record = await version_result.single()
                    
                    return {
                        "status": "healthy",
                        "version": version_record["versions"][0] if version_record else "unknown"
                    }
                else:
                    return {"status": "unhealthy", "error": "Connection test failed"}
                    
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def create_indexes(self) -> List[str]:
        """Create necessary indexes for performance.
        
        Returns:
            List of created indexes
        """
        if not self.async_driver:
            return []
        
        created = []
        indexes = [
            ("entity_name_index", "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:testing) ON (e.entity_id)"),
            ("entity_type_index", "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:testing) ON (e.entity_type)"),
            ("relationship_weight_index", "CREATE INDEX rel_weight_index IF NOT EXISTS FOR ()-[r:DIRECTED]-() ON (r.weight)")
        ]
        
        async with self.async_driver.session() as session:
            for name, query in indexes:
                try:
                    await session.run(query)
                    created.append(name)
                    logger.debug(f"Created index: {name}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create index {name}: {e}")
        
        # Create fulltext index
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                await session.run("""
                    CALL db.index.fulltext.createNodeIndex(
                        'entityFulltext',
                        ['Entity'],
                        ['name', 'description']
                    )
                """)
                created.append("entityFulltext")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Failed to create fulltext index: {e}")
        
        return created
    
    async def create_vector_indexes(self) -> List[str]:
        """Create optimized vector indexes with HNSW configuration.
        
        Returns:
            List of created vector indexes
        """
        if not self.async_driver:
            return []
        
        created = []
        
        # Vector indexes with optimal HNSW parameters
        vector_indexes = [
            ("entity_embeddings", "Entity", "embedding"),
            ("chunk_embeddings", "Chunk", "embedding"),
            ("relationship_embeddings", "RELATED", "embedding")
        ]
        
        async with self.async_driver.session() as session:
            for index_name, label, property_name in vector_indexes:
                try:
                    if label == "RELATED":
                        # Relationship vector index
                        cypher = f"""
                            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                            FOR ()-[r:{label}]-() ON (r.{property_name})
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: {self.settings.neo4j_vector_dimensions},
                                    `vector.similarity_function`: '{self.settings.neo4j_vector_similarity}',
                                    `vector.quantization.enabled`: {str(self.settings.neo4j_vector_quantization).lower()},
                                    `vector.hnsw.m`: {self.settings.neo4j_hnsw_m},
                                    `vector.hnsw.ef_construction`: {self.settings.neo4j_hnsw_ef_construction}
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
                                    `vector.dimensions`: {self.settings.neo4j_vector_dimensions},
                                    `vector.similarity_function`: '{self.settings.neo4j_vector_similarity}',
                                    `vector.quantization.enabled`: {str(self.settings.neo4j_vector_quantization).lower()},
                                    `vector.hnsw.m`: {self.settings.neo4j_hnsw_m},
                                    `vector.hnsw.ef_construction`: {self.settings.neo4j_hnsw_ef_construction}
                                }}
                            }}
                        """
                    
                    await session.run(cypher)
                    created.append(index_name)
                    logger.info(f"Created vector index: {index_name} with HNSW optimization")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create vector index {index_name}: {e}")
        
        return created
    
    async def check_vector_index_status(self) -> Dict[str, Any]:
        """Check status of all vector indexes.
        
        Returns:
            Dictionary with vector index status information
        """
        if not self.async_driver:
            return {}
        
        status = {}
        
        try:
            async with self.async_driver.session() as session:
                # Get vector index information
                result = await session.run("SHOW VECTOR INDEXES")
                
                async for record in result:
                    index_name = record.get("name", "unknown")
                    status[index_name] = {
                        "state": record.get("state", "unknown"),
                        "type": record.get("type", "unknown"),
                        "entity_type": record.get("entityType", "unknown"),
                        "labelsOrTypes": record.get("labelsOrTypes", []),
                        "properties": record.get("properties", []),
                        "options": record.get("options", {})
                    }
                
                logger.debug(f"Retrieved status for {len(status)} vector indexes")
        except Exception as e:
            logger.error(f"Failed to check vector index status: {e}")
        
        return status
    
    async def get_vector_index_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for vector indexes.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.async_driver:
            return {}
        
        metrics = {}
        
        try:
            async with self.async_driver.session() as session:
                # Test vector search performance
                vector_indexes = ["entity_embeddings", "chunk_embeddings", "relationship_embeddings"]
                
                for index_name in vector_indexes:
                    try:
                        # Create a test vector (zeros for testing)
                        test_vector = [0.0] * self.settings.neo4j_vector_dimensions
                        
                        # Measure search performance
                        import time
                        start_time = time.time()
                        
                        result = await session.run(
                            f"""
                            CALL db.index.vector.queryNodes(
                                '{index_name}',
                                5,
                                $test_vector
                            ) YIELD node, score
                            RETURN count(*) as result_count
                            """,
                            test_vector=test_vector
                        )
                        
                        record = await result.single()
                        latency_ms = (time.time() - start_time) * 1000
                        
                        metrics[index_name] = {
                            "latency_ms": latency_ms,
                            "result_count": record["result_count"] if record else 0,
                            "performance_status": "good" if latency_ms < 100 else "needs_optimization"
                        }
                        
                    except Exception as e:
                        metrics[index_name] = {
                            "error": str(e),
                            "performance_status": "error"
                        }
                
                logger.debug(f"Collected performance metrics for {len(metrics)} vector indexes")
        except Exception as e:
            logger.error(f"Failed to get vector index performance metrics: {e}")
        
        return metrics
    
    async def optimize_vector_quantization(self) -> Dict[str, Any]:
        """Optimize vector quantization settings for memory efficiency.
        
        Returns:
            Optimization results
        """
        if not self.async_driver:
            return {}
        
        results = {}
        
        try:
            async with self.async_driver.session() as session:
                # Get current vector index configuration
                status = await self.check_vector_index_status()
                
                for index_name, info in status.items():
                    try:
                        options = info.get("options", {})
                        quantization_enabled = options.get("vector.quantization.enabled", False)
                        
                        if not quantization_enabled and self.settings.neo4j_vector_quantization:
                            logger.info(f"Vector quantization should be enabled for {index_name} "
                                       f"to reduce memory usage by 50-75%")
                        
                        # Calculate estimated memory usage
                        dimensions = options.get("vector.dimensions", self.settings.neo4j_vector_dimensions)
                        estimated_memory_per_vector = dimensions * 4  # 4 bytes per float32
                        
                        if quantization_enabled:
                            estimated_memory_per_vector = int(estimated_memory_per_vector * 0.25)  # 75% reduction
                        
                        results[index_name] = {
                            "quantization_enabled": quantization_enabled,
                            "dimensions": dimensions,
                            "estimated_memory_bytes_per_vector": estimated_memory_per_vector,
                            "memory_optimization": "75% reduction" if quantization_enabled else "not optimized"
                        }
                        
                    except Exception as e:
                        results[index_name] = {"error": str(e)}
                
                logger.info(f"Vector quantization analysis completed for {len(results)} indexes")
        except Exception as e:
            logger.error(f"Failed to optimize vector quantization: {e}")
        
        return results
    
    async def get_entity_count(self) -> int:
        """Get total number of entities.
        
        Returns:
            Entity count
        """
        if not self.async_driver:
            return 0
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("MATCH (n:testing) RETURN count(n) as count")
                record = await result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.error(f"Failed to get entity count: {e}")
            return 0
    
    async def get_relationship_count(self) -> int:
        """Get total number of relationships.
        
        Returns:
            Relationship count
        """
        if not self.async_driver:
            return 0
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0
    
    async def get_entity_types(self) -> Dict[str, int]:
        """Get distribution of entity types.
        
        Returns:
            Dictionary of entity types and counts
        """
        if not self.async_driver:
            return {}
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("""
                    MATCH (n:testing)
                    RETURN n.entity_type as type, count(*) as count
                    ORDER BY count DESC
                """)
                
                types = {}
                async for record in result:
                    types[record["type"]] = record["count"]
                
                return types
        except Exception as e:
            logger.error(f"Failed to get entity types: {e}")
            return {}
    
    async def get_top_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top entities by degree centrality.
        
        Args:
            limit: Number of entities to return
            
        Returns:
            List of top entities
        """
        if not self.async_driver:
            return []
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("""
                    MATCH (n:testing)
                    WITH n, 
                         COUNT{(n)-[]->()} as out_degree, 
                         COUNT{()-[]->(n)} as in_degree
                    RETURN n.entity_id as name, 
                           n.entity_type as type,
                           out_degree,
                           in_degree,
                           out_degree + in_degree as total_degree
                    ORDER BY total_degree DESC
                    LIMIT $limit
                """, limit=limit)
                
                entities = []
                async for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "out_degree": record["out_degree"],
                        "in_degree": record["in_degree"],
                        "total_degree": record["total_degree"]
                    })
                
                return entities
        except Exception as e:
            logger.error(f"Failed to get top entities: {e}")
            return []
    
    async def get_strongest_relationships(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get relationships with highest weights.
        
        Args:
            limit: Number of relationships to return
            
        Returns:
            List of strongest relationships
        """
        if not self.async_driver:
            return []
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run("""
                    MATCH (a:Entity)-[r:DIRECTED]->(b:Entity)
                    RETURN a.name as source,
                           r.weight as weight,
                           b.name as target,
                           r.description as description
                    ORDER BY r.weight DESC
                    LIMIT $limit
                """, limit=limit)
                
                relationships = []
                async for record in result:
                    relationships.append({
                        "source": record["source"],
                        "target": record["target"],
                        "weight": record["weight"],
                        "description": record["description"]
                    })
                
                return relationships
        except Exception as e:
            logger.error(f"Failed to get strongest relationships: {e}")
            return []
    
    async def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a custom Cypher query.
        
        Args:
            query: Cypher query
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self.async_driver:
            return []
        
        try:
            async with self.async_driver.session(database="chunk-entity-relation") as session:
                result = await session.run(query, parameters or {})
                
                records = []
                async for record in result:
                    records.append(dict(record))
                
                return records
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
    async def close(self):
        """Close Neo4j connections."""
        if self.driver:
            self.driver.close()
        if self.async_driver:
            await self.async_driver.close()
        logger.info("Neo4j connections closed")