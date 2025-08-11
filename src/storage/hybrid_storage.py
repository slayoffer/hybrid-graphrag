"""Hybrid Storage Adapter combining NanoVectorDB and Neo4j.

This module provides a hybrid storage solution that uses:
- NanoVectorDB for initial vector creation and temporary storage
- Neo4j for permanent unified storage of both graph and vectors
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger
from neo4j import AsyncGraphDatabase
from nano_vectordb import NanoVectorDB


class HybridStorage:
    """Hybrid storage adapter for NanoVectorDB + Neo4j integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid storage with configuration.
        
        Args:
            config: Configuration containing:
                - neo4j_uri: Neo4j connection URI
                - neo4j_username: Neo4j username
                - neo4j_password: Neo4j password
                - working_dir: Directory for NanoVectorDB files
                - workspace: Workspace for isolation
                - vector_dimensions: Vector dimensions (default: 1536)
                - embedding_func: Embedding function
        """
        self.config = config
        self.working_dir = Path(config.get("working_dir", "./rag_storage"))
        self.workspace = config.get("workspace", "default")
        self.vector_dimensions = config.get("vector_dimensions", 1536)
        self.embedding_func = config.get("embedding_func")
        
        # Neo4j configuration
        self.neo4j_uri = config["neo4j_uri"]
        self.neo4j_username = config["neo4j_username"]
        self.neo4j_password = config["neo4j_password"]
        
        # Initialize Neo4j driver
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize NanoVectorDB instances for each collection
        self.nano_dbs = {}
        self._init_nano_dbs()
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"HybridStorage initialized for workspace: {self.workspace}")
    
    def _init_nano_dbs(self):
        """Initialize NanoVectorDB instances for different collections."""
        collections = ["entities", "chunks", "relationships"]
        
        # Create workspace directory if needed
        workspace_dir = self.working_dir / self.workspace
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        for collection in collections:
            db_file = workspace_dir / f"vdb_{collection}.json"
            # NanoVectorDB expects dimension as first argument
            self.nano_dbs[collection] = NanoVectorDB(
                self.vector_dimensions,
                storage_file=str(db_file)
            )
            logger.debug(f"Initialized NanoVectorDB for {collection}: {db_file}")
    
    async def upsert_vector(
        self,
        collection: str,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert or update a vector in hybrid storage.
        
        Args:
            collection: Collection name (entities, chunks, relationships)
            doc_id: Document/entity ID
            content: Text content to embed
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if metadata is None:
            metadata = {}
        
        try:
            # Generate embedding if embedding function provided
            if self.embedding_func:
                embedding = await self.embedding_func([content])
                embedding = embedding[0] if isinstance(embedding, list) else embedding
            else:
                # Use random embedding for testing
                embedding = np.random.random(self.vector_dimensions).tolist()
            
            async with self._lock:
                # Store in NanoVectorDB first
                nano_db = self.nano_dbs.get(collection)
                if nano_db:
                    nano_data = {
                        "__id__": doc_id,
                        "__vector__": embedding,
                        "content": content,
                        **metadata
                    }
                    nano_db.upsert([nano_data])
                    nano_db.save()  # Persist to disk
                    logger.debug(f"Stored vector in NanoVectorDB: {collection}/{doc_id}")
                
                # Then sync to Neo4j
                await self._sync_to_neo4j(collection, doc_id, embedding, content, metadata)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            return False
    
    async def _sync_to_neo4j(
        self,
        collection: str,
        doc_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ):
        """Sync vector to Neo4j for unified storage.
        
        Args:
            collection: Collection name
            doc_id: Document ID
            embedding: Vector embedding
            content: Text content
            metadata: Additional metadata
        """
        # Map collection names to Neo4j labels
        label_map = {
            "entities": "Entity",
            "chunks": "Chunk",
            "relationships": "Relationship"
        }
        label = label_map.get(collection, "Document")
        
        query = f"""
        MERGE (n:{label} {{id: $doc_id, workspace: $workspace}})
        SET n.embedding = $embedding
        SET n.content = $content
        SET n += $metadata
        RETURN n
        """
        
        async with self.neo4j_driver.session() as session:
            await session.run(
                query,
                doc_id=doc_id,
                workspace=self.workspace,
                embedding=embedding,
                content=content,
                metadata=metadata
            )
            logger.debug(f"Synced vector to Neo4j: {label}/{doc_id}")
    
    async def search_similar(
        self,
        collection: str,
        query_text: str,
        top_k: int = 10,
        use_neo4j: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using hybrid approach.
        
        Args:
            collection: Collection to search
            query_text: Query text
            top_k: Number of results
            use_neo4j: Whether to use Neo4j (True) or NanoVectorDB (False)
            
        Returns:
            List of similar documents with scores
        """
        # Generate query embedding
        if self.embedding_func:
            query_embedding = await self.embedding_func([query_text])
            query_embedding = query_embedding[0] if isinstance(query_embedding, list) else query_embedding
        else:
            query_embedding = np.random.random(self.vector_dimensions).tolist()
        
        if use_neo4j:
            return await self._search_neo4j(collection, query_embedding, top_k)
        else:
            return await self._search_nano(collection, query_embedding, top_k)
    
    async def _search_nano(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search in NanoVectorDB."""
        nano_db = self.nano_dbs.get(collection)
        if not nano_db:
            return []
        
        results = nano_db.query(
            query=query_embedding,
            top_k=top_k,
            better_than_threshold=0.5  # Cosine similarity threshold
        )
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("__id__"),
                "content": result.get("content", ""),
                "score": float(result.get("__metrics__", 0)),
                "metadata": {k: v for k, v in result.items() 
                           if not k.startswith("__")}
            })
        
        return formatted_results
    
    async def _search_neo4j(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search in Neo4j using vector index."""
        label_map = {
            "entities": "Entity",
            "chunks": "Chunk",
            "relationships": "Relationship"
        }
        label = label_map.get(collection, "Document")
        
        # Use Cypher query for vector similarity
        query = f"""
        MATCH (n:{label})
        WHERE n.workspace = $workspace AND n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
        WHERE similarity > 0.5
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    query,
                    workspace=self.workspace,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
                
                results = []
                async for record in result:
                    node_data = dict(record["n"])
                    # Remove embedding from response
                    node_data.pop("embedding", None)
                    
                    results.append({
                        "id": node_data.get("id"),
                        "content": node_data.get("content", ""),
                        "score": float(record["similarity"]),
                        "metadata": {k: v for k, v in node_data.items()
                                   if k not in ["id", "content", "workspace"]}
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Neo4j search failed: {e}")
            # Fallback to NanoVectorDB
            return await self._search_nano(collection, query_embedding, top_k)
    
    async def batch_upsert(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """Batch insert multiple documents.
        
        Args:
            collection: Collection name
            documents: List of documents with 'id', 'content', and optional 'metadata'
            
        Returns:
            Number of documents inserted
        """
        success_count = 0
        
        for doc in documents:
            success = await self.upsert_vector(
                collection=collection,
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {})
            )
            if success:
                success_count += 1
        
        logger.info(f"Batch inserted {success_count}/{len(documents)} documents")
        return success_count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics from both NanoVectorDB and Neo4j.
        
        Returns:
            Combined statistics
        """
        stats = {
            "nano_vectordb": {},
            "neo4j": {},
            "workspace": self.workspace
        }
        
        # NanoVectorDB stats
        for collection, nano_db in self.nano_dbs.items():
            # NanoVectorDB doesn't expose data directly, count from file
            db_file = self.working_dir / self.workspace / f"vdb_{collection}.json"
            if db_file.exists():
                try:
                    with open(db_file, 'r') as f:
                        data = json.load(f)
                        stats["nano_vectordb"][collection] = len(data.get("data", []))
                except:
                    stats["nano_vectordb"][collection] = 0
            else:
                stats["nano_vectordb"][collection] = 0
        
        # Neo4j stats
        async with self.neo4j_driver.session() as session:
            # Count entities
            result = await session.run("""
                MATCH (n:Entity)
                WHERE n.workspace = $workspace
                RETURN count(n) as total,
                       count(n.embedding) as with_vectors
            """, workspace=self.workspace)
            record = await result.single()
            if record:
                stats["neo4j"]["entities"] = {
                    "total": record["total"],
                    "with_vectors": record["with_vectors"]
                }
            
            # Count chunks
            result = await session.run("""
                MATCH (n:Chunk)
                WHERE n.workspace = $workspace
                RETURN count(n) as total,
                       count(n.embedding) as with_vectors
            """, workspace=self.workspace)
            record = await result.single()
            if record:
                stats["neo4j"]["chunks"] = {
                    "total": record["total"],
                    "with_vectors": record["with_vectors"]
                }
            
            # Count relationships
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as count
            """)
            record = await result.single()
            if record:
                stats["neo4j"]["relationships"] = record["count"]
        
        return stats
    
    async def sync_all_to_neo4j(self) -> Dict[str, int]:
        """Sync all NanoVectorDB data to Neo4j.
        
        Returns:
            Sync statistics
        """
        from .vector_migrator import VectorMigrator
        
        migrator = VectorMigrator(
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            workspace=self.workspace,
            vector_dimensions=self.vector_dimensions
        )
        
        try:
            # Create vector indexes
            await migrator.create_vector_indexes()
            
            # Sync all collections
            workspace_dir = self.working_dir / self.workspace
            stats = await migrator.sync_all_collections(str(workspace_dir))
            
            return stats
            
        finally:
            await migrator.close()
    
    async def close(self):
        """Close connections and save data."""
        # Save NanoVectorDB data
        for nano_db in self.nano_dbs.values():
            nano_db.save()
        
        # Close Neo4j driver
        await self.neo4j_driver.close()
        
        logger.info("HybridStorage closed")


async def test_hybrid_storage():
    """Test the hybrid storage implementation."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create test configuration
    config = {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7689"),
        "neo4j_username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
        "working_dir": "./rag_storage",
        "workspace": "testing",
        "vector_dimensions": 1536
    }
    
    storage = HybridStorage(config)
    
    try:
        # Test upsert
        await storage.upsert_vector(
            collection="entities",
            doc_id="test_entity_1",
            content="LightRAG is a graph-enhanced retrieval system",
            metadata={"type": "system", "category": "RAG"}
        )
        
        # Test search
        results = await storage.search_similar(
            collection="entities",
            query_text="What is LightRAG?",
            top_k=5,
            use_neo4j=True
        )
        
        logger.info(f"Search results: {results}")
        
        # Get statistics
        stats = await storage.get_statistics()
        logger.info(f"Storage statistics: {stats}")
        
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(test_hybrid_storage())