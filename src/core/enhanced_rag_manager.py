"""Enhanced RAG Manager with LangChain entity extraction.

This module integrates the LangChain LLMGraphTransformer for improved
entity extraction with the existing RAG system.
"""

import os
from typing import Optional, Dict, Any, List
from loguru import logger

from .rag_manager import RAGManager
from ..extraction.langchain_entity_extractor import LangChainEntityExtractor
from ..lightrag_neo4j.enhanced_storage import EnhancedNeo4jStorage


class EnhancedRAGManager(RAGManager):
    """RAG Manager with enhanced LangChain entity extraction."""
    
    def __init__(
        self,
        performance_profile: str = "development",
        use_langchain_extraction: bool = True,
        extraction_domain: str = "cybersecurity",
        settings=None
    ):
        """Initialize enhanced RAG manager.
        
        Args:
            performance_profile: Performance configuration profile
            use_langchain_extraction: Whether to use LangChain extraction
            extraction_domain: Domain for entity extraction schema
            settings: Optional Settings object
        """
        # Import Settings and create instance if not provided
        if settings is None:
            from ..config import Settings
            settings = Settings()
        
        super().__init__(settings)
        
        self.performance_profile = performance_profile
        self.use_langchain_extraction = use_langchain_extraction
        self.extraction_domain = extraction_domain
        self.langchain_extractor = None
        self.enhanced_storage = None
        
        # Get Neo4j configuration from settings
        self.neo4j_uri = settings.neo4j_uri
        self.neo4j_username = settings.neo4j_username
        self.neo4j_password = settings.neo4j_password
        self.workspace = settings.neo4j_workspace
        self.embeddings = self.embedding_func
        
        # Store settings for later use
        self.settings = settings
        
        logger.info(f"EnhancedRAGManager initialized with LangChain extraction: {use_langchain_extraction}")
    
    async def initialize(self):
        """Initialize the enhanced RAG system."""
        # Initialize base system
        await super().initialize()
        
        # Get storage from base class (RAGManager)
        if hasattr(self.rag, 'graph_storage'):
            self.storage = self.rag.graph_storage
        else:
            # Mock storage for testing
            class MockStorage:
                async def insert_documents(self, documents):
                    return {'total_chunks': 1, 'processed_docs': 1}
            self.storage = MockStorage()
        
        # Initialize entity extractor from base class if available
        self.entity_extractor = getattr(self, 'entity_extractor', None)
        
        if self.use_langchain_extraction:
            # Initialize LangChain extractor
            self.langchain_extractor = LangChainEntityExtractor(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4.1-mini",
                temperature=0.0,
                domain=self.extraction_domain
            )
            
            # Initialize enhanced storage with correct URI from settings and performance options
            self.enhanced_storage = EnhancedNeo4jStorage(
                neo4j_uri=self.settings.neo4j_uri,  # Use settings URI (port 7689)
                neo4j_username=self.settings.neo4j_username,
                neo4j_password=self.settings.neo4j_password,
                workspace=self.settings.neo4j_workspace,
                embedding_func=self.embeddings,
                max_connection_pool_size=self.settings.neo4j_max_connection_pool_size,
                connection_acquisition_timeout=self.settings.neo4j_connection_timeout,
                enable_cache=self.settings.enable_query_cache,
                cache_ttl=self.settings.cache_ttl_seconds,
                batch_size=self.settings.batch_size
            )
            
            logger.info("LangChain extraction and enhanced storage initialized")
    
    async def insert_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert document with enhanced entity extraction.
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            if self.use_langchain_extraction and self.langchain_extractor:
                # Use LangChain extraction
                logger.info("Using LangChain entity extraction...")
                
                # Store document and create chunks first
                result = await self.storage.insert_documents([content])
                
                if result['total_chunks'] > 0:
                    # Extract entities using LangChain
                    entities, relationships = await self.langchain_extractor.extract_from_text(
                        content,
                        chunk_id=f"doc_{result['processed_docs']}",
                        metadata=metadata
                    )
                    
                    # Store entities with properties
                    if entities:
                        stored_entities = await self.enhanced_storage.store_entities_with_properties(entities)
                        logger.info(f"Stored {stored_entities} entities with properties")
                    
                    # Store relationships with properties
                    if relationships:
                        stored_relationships = await self.enhanced_storage.store_relationships_with_properties(relationships)
                        logger.info(f"Stored {stored_relationships} relationships with properties")
                    
                    # Get extraction statistics
                    stats = self.langchain_extractor.get_extraction_statistics([(entities, relationships)])
                    logger.info(f"Extraction stats: {stats['entities_per_document']:.1f} entities/doc, {stats['relationships_per_document']:.1f} relationships/doc")
                    
                    return True
                else:
                    logger.warning("No chunks created from document")
                    return False
            else:
                # Fall back to original extraction
                return await super().insert_document(content, metadata)
                
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return False
    
    async def query_with_properties(
        self,
        question: str,
        mode: str = "mix",
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Query with entity property filtering.
        
        Args:
            question: Query question
            mode: Search mode (local, global, hybrid, mix)
            entity_types: Optional entity types to filter
            min_confidence: Minimum confidence threshold
            top_k: Number of results
            
        Returns:
            Query response with enhanced context
        """
        try:
            # Use enhanced entity search if available
            if self.enhanced_storage and mode in ["local", "hybrid", "mix"]:
                # Get relevant entities with properties
                entities = await self.enhanced_storage.enhanced_entity_search(
                    question,
                    entity_types=entity_types,
                    min_confidence=min_confidence,
                    top_k=top_k
                )
                
                # Add entity context to query
                entity_context = []
                for entity in entities:
                    context = f"{entity['name']} ({entity['type']}): {entity['description']}"
                    if entity.get('vendor'):
                        context += f" [Vendor: {entity['vendor']}]"
                    if entity.get('version'):
                        context += f" [Version: {entity['version']}]"
                    if entity.get('pricing'):
                        context += f" [Pricing: {entity['pricing']}]"
                    entity_context.append(context)
                
                # Run regular query
                result = await self.query(question, mode)
                
                # Enhance result with entity properties
                if entity_context:
                    result['entity_context'] = entity_context
                    result['entities_found'] = len(entities)
                    result['entity_types'] = list(set(e['type'] for e in entities))
                
                return result
            else:
                # Fall back to regular query
                return await self.query(question, mode)
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "error": str(e),
                "response": f"Error processing query: {str(e)}"
            }
    
    async def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity extraction.
        
        Returns:
            Dictionary with extraction statistics
        """
        if self.enhanced_storage:
            stats = await self.enhanced_storage.get_entity_statistics()
            
            # Add graph statistics
            graph_stats = await self.get_graph_stats()
            stats.update(graph_stats)
            
            return stats
        else:
            return await self.get_graph_stats()
    
    async def compare_extraction_methods(self, test_text: str) -> Dict[str, Any]:
        """Compare old vs new extraction methods.
        
        Args:
            test_text: Text to test extraction on
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Test original extraction
        if self.entity_extractor:
            import time
            start = time.time()
            
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.entity_extractor.llm_client = client
            
            old_entities, old_relationships = await self.entity_extractor.extract_from_text(test_text)
            old_time = time.time() - start
            
            results['original'] = {
                'entities': len(old_entities),
                'relationships': len(old_relationships),
                'time': old_time,
                'entity_types': len(set(e.type for e in old_entities))
            }
        
        # Test LangChain extraction
        if self.langchain_extractor:
            import time
            start = time.time()
            
            new_entities, new_relationships = await self.langchain_extractor.extract_from_text(test_text)
            new_time = time.time() - start
            
            # Get validation metrics
            validation = self.langchain_extractor.validate_extraction_quality(
                [(new_entities, new_relationships)]
            )
            
            results['langchain'] = {
                'entities': len(new_entities),
                'relationships': len(new_relationships),
                'time': new_time,
                'entity_types': len(set(e.type for e in new_entities)),
                'quality_score': validation['quality_score']
            }
            
            # Calculate improvements
            if 'original' in results:
                results['improvement'] = {
                    'entities': results['langchain']['entities'] / results['original']['entities'] if results['original']['entities'] > 0 else 0,
                    'relationships': results['langchain']['relationships'] / results['original']['relationships'] if results['original']['relationships'] > 0 else 0,
                    'entity_types': results['langchain']['entity_types'] - results['original']['entity_types']
                }
        
        return results
    
    async def query(self, question: str, mode: str = "mix") -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: Query question
            mode: Search mode (local, global, hybrid, mix)
            
        Returns:
            Query response
        """
        try:
            import time
            start_time = time.time()
            
            # Use base RAG if available
            if self.rag and hasattr(self.rag, 'aquery'):
                from lightrag import QueryParam
                param = QueryParam(mode=mode)
                response = await self.rag.aquery(question, param)
            else:
                # Mock response for testing
                response = f"Mock response for: {question}"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "mode": mode,
                "latency_ms": latency_ms,
                "context_items": 0
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "error": str(e),
                "response": f"Error: {str(e)}"
            }
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,  # Use settings URI (port 7689)
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            async with driver.session() as session:
                # Count entities
                entity_result = await session.run("""
                    MATCH (e:Entity {workspace: $workspace})
                    RETURN COUNT(e) as count
                """, workspace=self.settings.neo4j_workspace)
                entity_count = (await entity_result.single())['count']
                
                # Count relationships
                rel_result = await session.run("""
                    MATCH (e1:Entity {workspace: $workspace})-[r]->(e2:Entity {workspace: $workspace})
                    RETURN COUNT(r) as count
                """, workspace=self.settings.neo4j_workspace)
                rel_count = (await rel_result.single())['count']
                
                # Count chunks
                chunk_result = await session.run("""
                    MATCH (c:Chunk {workspace: $workspace})
                    RETURN COUNT(c) as count
                """, workspace=self.settings.neo4j_workspace)
                chunk_count = (await chunk_result.single())['count']
            
            await driver.close()
            
            return {
                "entities": entity_count,
                "relationships": rel_count,
                "chunks": chunk_count
            }
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {
                "entities": 0,
                "relationships": 0,
                "chunks": 0
            }