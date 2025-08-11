"""Vector Migration Service for syncing NanoVectorDB to Neo4j.

This module handles the migration of vectors from NanoVectorDB (temporary storage)
to Neo4j (permanent unified storage) for hybrid graph-vector queries.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from loguru import logger
from neo4j import AsyncGraphDatabase


class VectorMigrator:
    """Migrates vectors from NanoVectorDB to Neo4j for unified storage."""
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        workspace: str = "default",
        vector_dimensions: int = 1536
    ):
        """Initialize the vector migrator.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            workspace: Workspace for data isolation
            vector_dimensions: Dimension of vectors (1536 for text-embedding-3-small)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.workspace = workspace
        self.vector_dimensions = vector_dimensions
        
        # Neo4j driver
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        logger.info(f"VectorMigrator initialized for workspace: {self.workspace}")
    
    async def load_nano_vectors(self, nano_db_path: str) -> Dict[str, Any]:
        """Load vectors from NanoVectorDB JSON file.
        
        Args:
            nano_db_path: Path to NanoVectorDB JSON file
            
        Returns:
            Dictionary of vectors and metadata
        """
        try:
            with open(nano_db_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data.get('data', []))} vectors from {nano_db_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load NanoVectorDB file: {e}")
            return {}
    
    async def create_vector_indexes(self):
        """Create Neo4j vector indexes for efficient similarity search.
        
        Creates HNSW indexes for entities, chunks, and relationships.
        """
        indexes = [
            {
                "name": "entity_embedding_index",
                "label": "Entity",
                "property": "embedding",
                "description": "Entity embeddings for semantic search"
            },
            {
                "name": "chunk_embedding_index", 
                "label": "Chunk",
                "property": "embedding",
                "description": "Document chunk embeddings"
            },
            {
                "name": "relationship_embedding_index",
                "label": "Relationship",
                "property": "embedding", 
                "description": "Relationship embeddings"
            }
        ]
        
        async with self.driver.session() as session:
            for index in indexes:
                try:
                    # Create vector index with HNSW configuration
                    query = f"""
                    CALL db.index.vector.createNodeIndex(
                        '{index['name']}',
                        '{index['label']}',
                        '{index['property']}',
                        {self.vector_dimensions},
                        'cosine'
                    )
                    """
                    await session.run(query)
                    logger.info(f"Created vector index: {index['name']}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.debug(f"Vector index {index['name']} already exists")
                    else:
                        logger.error(f"Failed to create index {index['name']}: {e}")
    
    async def migrate_vectors_to_neo4j(
        self,
        vectors_data: Dict[str, Any],
        collection_type: str = "Entity"
    ) -> int:
        """Migrate vectors from NanoVectorDB format to Neo4j.
        
        Args:
            vectors_data: Vector data from NanoVectorDB
            collection_type: Type of collection (Entity, Chunk, Relationship)
            
        Returns:
            Number of vectors migrated
        """
        if not vectors_data or 'data' not in vectors_data:
            logger.warning("No vector data to migrate")
            return 0
        
        migrated_count = 0
        batch_size = 100
        vectors = vectors_data['data']
        
        # Process in batches for efficiency
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            async with self.driver.session() as session:
                for item in batch:
                    try:
                        # Extract vector and metadata
                        vector_id = item.get('__id__', '')
                        embedding = item.get('__vector__', [])
                        metadata = {k: v for k, v in item.items() 
                                  if not k.startswith('__')}
                        
                        # Store in Neo4j
                        query = f"""
                        MERGE (n:{collection_type} {{id: $id, workspace: $workspace}})
                        SET n.embedding = $embedding
                        SET n += $metadata
                        RETURN n
                        """
                        
                        await session.run(
                            query,
                            id=vector_id,
                            workspace=self.workspace,
                            embedding=embedding,
                            metadata=metadata
                        )
                        migrated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate vector {vector_id}: {e}")
            
            logger.debug(f"Migrated batch {i//batch_size + 1}, total: {migrated_count}")
        
        logger.info(f"Successfully migrated {migrated_count} vectors to Neo4j")
        return migrated_count
    
    async def sync_all_collections(self, working_dir: str) -> Dict[str, int]:
        """Sync all vector collections from NanoVectorDB to Neo4j.
        
        Args:
            working_dir: LightRAG working directory containing NanoVectorDB files
            
        Returns:
            Dictionary with migration statistics
        """
        stats = {}
        collections = [
            ("vdb_entities.json", "Entity"),
            ("vdb_chunks.json", "Chunk"),
            ("vdb_relationships.json", "Relationship")
        ]
        
        for filename, collection_type in collections:
            file_path = Path(working_dir) / filename
            
            if file_path.exists():
                logger.info(f"Syncing {collection_type} vectors from {filename}")
                vectors_data = await self.load_nano_vectors(str(file_path))
                count = await self.migrate_vectors_to_neo4j(vectors_data, collection_type)
                stats[collection_type] = count
            else:
                logger.warning(f"File not found: {file_path}")
                stats[collection_type] = 0
        
        return stats
    
    async def verify_migration(self) -> Dict[str, Any]:
        """Verify that vectors have been successfully migrated to Neo4j.
        
        Returns:
            Verification statistics
        """
        stats = {}
        
        async with self.driver.session() as session:
            # Count entities with embeddings
            result = await session.run("""
                MATCH (n:Entity)
                WHERE n.workspace = $workspace AND n.embedding IS NOT NULL
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats['entities_with_vectors'] = record['count'] if record else 0
            
            # Count chunks with embeddings
            result = await session.run("""
                MATCH (n:Chunk)
                WHERE n.workspace = $workspace AND n.embedding IS NOT NULL
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats['chunks_with_vectors'] = record['count'] if record else 0
            
            # Count relationships with embeddings
            result = await session.run("""
                MATCH (n:Relationship)
                WHERE n.workspace = $workspace AND n.embedding IS NOT NULL
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats['relationships_with_vectors'] = record['count'] if record else 0
            
            # Check vector indexes
            result = await session.run("SHOW INDEXES")
            indexes = []
            async for record in result:
                if 'vector' in record.get('type', '').lower():
                    indexes.append(record['name'])
            stats['vector_indexes'] = indexes
        
        return stats
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        collection: str = "Entity",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and graph relationships.
        
        Args:
            query_embedding: Query vector
            collection: Collection to search
            top_k: Number of results
            
        Returns:
            Search results with scores
        """
        async with self.driver.session() as session:
            # Vector similarity search
            query = f"""
            CALL db.index.vector.queryNodes(
                '{collection.lower()}_embedding_index',
                $top_k,
                $query_embedding
            ) YIELD node, score
            WHERE node.workspace = $workspace
            RETURN node, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            result = await session.run(
                query,
                top_k=top_k,
                query_embedding=query_embedding,
                workspace=self.workspace
            )
            
            results = []
            async for record in result:
                node_data = dict(record['node'])
                # Remove embedding from response to reduce size
                node_data.pop('embedding', None)
                results.append({
                    'node': node_data,
                    'score': float(record['score']),
                    'id': node_data.get('id'),
                    'content': node_data.get('content', '')
                })
            
            return results
    
    async def close(self):
        """Close Neo4j driver connection."""
        await self.driver.close()


async def main():
    """Example usage of VectorMigrator."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize migrator
    migrator = VectorMigrator(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7689"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        workspace="testing"
    )
    
    try:
        # Create vector indexes
        await migrator.create_vector_indexes()
        
        # Sync all collections
        stats = await migrator.sync_all_collections("./rag_storage")
        logger.info(f"Migration statistics: {stats}")
        
        # Verify migration
        verification = await migrator.verify_migration()
        logger.info(f"Verification: {verification}")
        
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())