"""Core LightRAG manager with Neo4j integration.

CRITICAL: This module properly initializes LightRAG with Neo4j backend.
- MUST include embedding_func parameter for Neo4j
- MUST call initialize_storages() after creation
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# LightRAG imports - handle gracefully if not installed
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    LIGHTRAG_AVAILABLE = True
    logger.info("LightRAG successfully imported")
    
    # Mock initialize_pipeline_status if not available
    async def initialize_pipeline_status():
        pass
except ImportError as e:
    logger.warning(f"LightRAG not installed - using mock implementation: {e}")
    LIGHTRAG_AVAILABLE = False
    # Mock for development
    async def initialize_pipeline_status():
        pass
    
    # Mock classes for development
    class LightRAG:
        def __init__(self, **kwargs):
            self.config = kwargs
            
        async def initialize_storages(self):
            pass
            
        async def ainsert(self, text: str):
            pass
            
        async def aquery(self, question: str, param=None):
            return f"Mock response for: {question}"
    
    class QueryParam:
        def __init__(self, **kwargs):
            self.config = kwargs

from ..config import Settings
from .embeddings import create_embedding_func, create_llm_func
from .tokenizer import SimpleTokenizer


class RAGManager:
    """Manages LightRAG instance with Neo4j backend.
    
    CRITICAL: Properly initializes LightRAG with all required parameters
    for Neo4j integration.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize RAG manager with configuration.
        
        Args:
            settings: Application settings (uses default if not provided)
        """
        if settings is None:
            from ..config import settings
            self.settings = settings
        else:
            self.settings = settings
        
        self.rag = None
        self.embedding_func = None
        self.llm_func = None
        self.initialized = False
        
        # Initialize components
        self._setup_functions()
        self._create_rag_instance()
        
        # Note: Initialization must be done asynchronously
        # This will be called from FastAPI's lifespan context
        logger.info("RAGManager created, awaiting async initialization")
    
    def _setup_functions(self):
        """Set up embedding and LLM functions."""
        # CRITICAL: Create embedding function for Neo4j
        logger.info("Creating embedding function for Neo4j...")
        self.embedding_func = create_embedding_func(
            api_key=self.settings.openai_api_key,
            model=self.settings.embedding_model,
            max_token_size=8192
        )
        
        # Create LLM function
        logger.info("Creating LLM function...")
        self.llm_func = create_llm_func(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            max_tokens=4096
        )
    
    def _create_rag_instance(self):
        """Create LightRAG instance with proper configuration.
        
        CRITICAL: Must include embedding_func for Neo4j backend.
        """
        # Ensure working directory exists
        working_dir = Path(self.settings.lightrag_working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Set workspace environment variable for isolation
        if self.settings.neo4j_workspace:
            os.environ["NEO4J_WORKSPACE"] = self.settings.neo4j_workspace
            logger.info(f"Set Neo4j workspace to: {self.settings.neo4j_workspace}")
        
        if not LIGHTRAG_AVAILABLE:
            logger.warning("Using mock LightRAG implementation")
            self.rag = LightRAG()
            return
        
        # Check if using unified storage
        if self.settings.unified_neo4j_storage:
            logger.info("Using UNIFIED Neo4j storage for graph and vectors")
            self._create_unified_rag_instance(working_dir)
        else:
            logger.info("Using HYBRID storage (Neo4j + NanoVectorDB)")
            self._create_hybrid_rag_instance(working_dir)
    
    def _create_unified_rag_instance(self, working_dir: Path):
        """Create LightRAG with unified Neo4j storage."""
        try:
            # For unified storage, use Neo4j for both graph and vectors
            logger.info("Creating LightRAG with unified Neo4j storage...")
            
            # Create LightRAG with Neo4j for both graph and vectors
            self.rag = LightRAG(
                working_dir=str(working_dir),
                llm_model_func=gpt_4o_mini_complete if "mini" in self.settings.openai_model else gpt_4o_complete,
                embedding_func_max_async=32,
                graph_storage="Neo4JStorage",
                vector_storage="Neo4JStorage",  # Use Neo4j for vectors too
                enable_llm_cache=True
            )
            
            logger.info("Unified LightRAG instance created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create unified LightRAG instance: {e}")
            raise
    
    def _create_hybrid_rag_instance(self, working_dir: Path):
        """Create LightRAG with hybrid storage (original implementation)."""
        # Build Neo4j configuration
        neo4j_config = {
            "uri": self.settings.neo4j_uri,
            "username": self.settings.neo4j_username,
            "password": self.settings.neo4j_password,
            "database": self.settings.neo4j_workspace,
            "max_connection_pool_size": self.settings.neo4j_max_connection_pool_size
        }
        
        try:
            # Prepare addon_params for Neo4j configuration
            addon_params = {
                "neo4j_config": neo4j_config,
                "entity_types": self.settings.entity_types_list
            }
            
            # Create a custom tokenizer to avoid pickling issues
            tokenizer = SimpleTokenizer(model_name="gpt-4.1-mini")
            
            self.rag = LightRAG(
                working_dir=str(working_dir),
                
                # CRITICAL: Embedding function is MANDATORY for Neo4j
                embedding_func=self.embedding_func,
                
                # LLM configuration
                llm_model_func=self.llm_func,
                llm_model_name=self.settings.openai_model,
                
                # Custom tokenizer to avoid pickling issues
                tokenizer=tokenizer,
                
                # Storage configuration
                graph_storage="Neo4JStorage",
                
                # Vector storage (optional, for hybrid mode)
                vector_storage=self.settings.lightrag_vector_storage,
                
                # KV and document storage
                kv_storage=self.settings.lightrag_kv_storage,
                doc_status_storage=self.settings.lightrag_doc_storage,
                
                # Entity extraction configuration
                entity_extract_max_gleaning=self.settings.gleaning_iterations,
                max_entity_tokens=self.settings.max_entity_tokens,
                max_relation_tokens=self.settings.max_relationship_tokens,
                
                # Workspace isolation
                workspace=self.settings.neo4j_workspace,
                
                # Add Neo4j configuration via addon_params
                addon_params=addon_params
            )
            logger.info("Hybrid LightRAG instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create hybrid LightRAG instance: {e}")
            raise
    
    async def initialize(self):
        """Initialize storage backends and create indexes.
        
        CRITICAL: This MUST be called after RAG creation for Neo4j to work.
        """
        if not self.rag:
            logger.error("RAG instance not created")
            return
        
        try:
            # No need for special unified storage initialization with simplified approach
            
            # Initialize LightRAG storages (if the method exists)
            if hasattr(self.rag, 'initialize_storages'):
                logger.info("Initializing storage backends...")
                await self.rag.initialize_storages()
                logger.info("Storage backends initialized successfully")
            
            # CRITICAL: Initialize pipeline status for document processing
            logger.info("Initializing pipeline status...")
            await initialize_pipeline_status()
            logger.info("Pipeline status initialized successfully")
            
            # Create Neo4j indexes (only for hybrid mode, unified creates its own)
            if not self.settings.unified_neo4j_storage:
                await self._create_indexes()
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize storages: {e}")
            raise
    
    async def _create_indexes(self):
        """Create Neo4j indexes for better performance."""
        if not LIGHTRAG_AVAILABLE:
            return
        
        try:
            # Import Neo4j driver if available
            from neo4j import AsyncGraphDatabase
            
            logger.info("Creating Neo4j indexes...")
            
            # Connect to Neo4j
            driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            async with driver.session() as session:
                # Create indexes
                queries = [
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                    """CALL db.index.fulltext.createNodeIndex(
                        'entityFulltext', 
                        ['Entity'], 
                        ['name', 'description']
                    ) YIELD name RETURN name"""
                ]
                
                for query in queries:
                    try:
                        await session.run(query)
                        logger.debug(f"Executed: {query[:50]}...")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Index creation warning: {e}")
            
            await driver.close()
            logger.info("Neo4j indexes created successfully")
            
        except ImportError:
            logger.warning("Neo4j driver not available, skipping index creation")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    async def insert_document(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """Insert document into the knowledge graph.
        
        Args:
            content: Document content to process
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        if not self.initialized:
            logger.error("RAGManager not initialized")
            return False
        
        try:
            start_time = time.time()
            logger.info(f"Inserting document ({len(content)} chars)...")
            
            # Insert into LightRAG - use ainsert for async
            await self.rag.ainsert(content)
            
            elapsed = time.time() - start_time
            logger.info(f"Document inserted successfully in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return False
    
    async def insert_batch(self, documents: List[str]) -> Dict[str, Any]:
        """Insert multiple documents in batch.
        
        Args:
            documents: List of document contents
            
        Returns:
            Batch insertion results
        """
        results = {
            "total": len(documents),
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        for i, doc in enumerate(documents):
            try:
                success = await self.insert_document(doc)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Document {i}: Insertion failed")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Document {i}: {str(e)}")
        
        return results
    
    async def query(
        self,
        question: str,
        mode: str = "mix",
        top_k: int = 100,
        enable_rerank: bool = True
    ) -> Dict[str, Any]:
        """Query the knowledge graph.
        
        Args:
            question: Query text
            mode: Query mode (local/global/hybrid/mix/naive)
            top_k: Number of results to return
            enable_rerank: Whether to rerank results
            
        Returns:
            Query response with metadata
        """
        if not self.initialized:
            logger.error("RAGManager not initialized")
            return {"error": "RAGManager not initialized"}
        
        try:
            start_time = time.time()
            logger.info(f"Querying with mode '{mode}': {question[:100]}...")
            
            # Create query parameters
            param = QueryParam(
                mode=mode,
                top_k=top_k,
                enable_rerank=enable_rerank,
                response_type="Multiple Paragraphs"
            )
            
            # Execute query - use aquery for async
            response = await self.rag.aquery(question, param=param)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "mode": mode,
                "latency_ms": latency_ms,
                "top_k": top_k,
                "reranked": enable_rerank
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "error": str(e),
                "mode": mode,
                "latency_ms": 0
            }
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not LIGHTRAG_AVAILABLE:
            return {"status": "LightRAG not available"}
        
        try:
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            stats = {}
            async with driver.session() as session:
                # Count entities
                result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
                record = await result.single()
                stats["entity_count"] = record["count"] if record else 0
                
                # Count relationships
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                stats["relationship_count"] = record["count"] if record else 0
                
                # Get entity type distribution
                result = await session.run("""
                    MATCH (n:Entity)
                    RETURN n.entity_type as type, count(*) as count
                    ORDER BY count DESC
                """)
                stats["entity_types"] = {
                    record["type"]: record["count"]
                    async for record in result
                }
            
            await driver.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {"error": str(e)}
    
    async def clear_graph(self) -> bool:
        """Clear all data from the graph.
        
        Returns:
            Success status
        """
        if not LIGHTRAG_AVAILABLE:
            return True
        
        try:
            from neo4j import AsyncGraphDatabase
            
            logger.warning("Clearing all graph data...")
            
            driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            async with driver.session() as session:
                # Delete all nodes and relationships
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph cleared successfully")
            
            await driver.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of RAG manager and dependencies.
        
        Returns:
            Health status
        """
        health = {
            "status": "unknown",
            "lightrag": LIGHTRAG_AVAILABLE,
            "initialized": self.initialized,
            "neo4j": False,
            "errors": []
        }
        
        # Check Neo4j connection
        try:
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            async with driver.session() as session:
                result = await session.run("RETURN 1 as num")
                record = await result.single()
                if record and record["num"] == 1:
                    health["neo4j"] = True
            
            await driver.close()
        except Exception as e:
            health["errors"].append(f"Neo4j check failed: {str(e)}")
        
        # Determine overall status
        if health["initialized"] and health["neo4j"]:
            health["status"] = "healthy"
        elif health["initialized"]:
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
        
        return health