"""Migration script to move vectors from NanoVectorDB JSON files to Neo4j.

This script handles the migration of vector embeddings from the current
NanoVectorDB JSON storage to Neo4j 5.13+ vector properties.
"""

import json
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from neo4j import AsyncGraphDatabase
import numpy as np
import base64


class VectorMigration:
    """Migrate vectors from NanoVectorDB to Neo4j."""
    
    def __init__(self, neo4j_config: Dict[str, Any], storage_path: str = "rag_storage/testing"):
        """Initialize migration with configuration.
        
        Args:
            neo4j_config: Neo4j connection configuration
            storage_path: Path to storage directory with JSON files
        """
        self.neo4j_config = neo4j_config
        self.storage_path = Path(storage_path)
        self.driver = None
        self.database = neo4j_config.get("database", "neo4j")
        self.workspace = neo4j_config.get("workspace", "testing")
        
        # Statistics
        self.stats = {
            "entities_migrated": 0,
            "chunks_migrated": 0,
            "relationships_migrated": 0,
            "errors": []
        }
        
    async def connect(self):
        """Establish connection to Neo4j."""
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_config["uri"],
            auth=(self.neo4j_config["username"], self.neo4j_config["password"])
        )
        logger.info(f"Connected to Neo4j at {self.neo4j_config['uri']}")
        
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def backup_data(self):
        """Backup existing data before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.storage_path.parent / f"backup_{timestamp}"
        
        if self.storage_path.exists():
            shutil.copytree(self.storage_path, backup_dir)
            logger.info(f"✅ Backup created at: {backup_dir}")
            return backup_dir
        else:
            logger.warning(f"Storage path not found: {self.storage_path}")
            return None
    
    def _decode_matrix(self, matrix_str: str, shape: tuple) -> np.ndarray:
        """Decode base64 matrix string to numpy array.
        
        Args:
            matrix_str: Base64 encoded matrix
            shape: Expected shape of the matrix
            
        Returns:
            Numpy array
        """
        try:
            # Decode base64 to bytes
            matrix_bytes = base64.b64decode(matrix_str)
            # Convert to numpy array
            matrix = np.frombuffer(matrix_bytes, dtype=np.float32)
            # Reshape to expected dimensions
            matrix = matrix.reshape(shape)
            return matrix
        except Exception as e:
            logger.error(f"Failed to decode matrix: {e}")
            return None
    
    async def migrate_entities(self):
        """Migrate entity vectors from vdb_entities.json."""
        entities_file = self.storage_path / "vdb_entities.json"
        
        if not entities_file.exists():
            logger.warning(f"Entities file not found: {entities_file}")
            return
        
        logger.info(f"Loading entities from {entities_file}...")
        
        with open(entities_file, 'r') as f:
            vdb_data = json.load(f)
        
        # NanoVectorDB structure
        embedding_dim = vdb_data.get("embedding_dim", 1536)
        entities_data = vdb_data.get("data", [])
        matrix_str = vdb_data.get("matrix", "")
        
        if not entities_data:
            logger.warning("No entities found in vdb_entities.json")
            return
        
        logger.info(f"Found {len(entities_data)} entities with {embedding_dim}-dimensional embeddings")
        
        # Decode the embedding matrix if it exists
        embeddings_matrix = None
        if matrix_str:
            embeddings_matrix = self._decode_matrix(
                matrix_str, 
                (len(entities_data), embedding_dim)
            )
        
        # Process entities in batches
        batch_size = 50
        for i in range(0, len(entities_data), batch_size):
            batch = entities_data[i:i+batch_size]
            batch_embeddings = embeddings_matrix[i:i+batch_size] if embeddings_matrix is not None else None
            
            await self._migrate_entity_batch(batch, batch_embeddings, i)
            logger.info(f"Migrated {min(i+batch_size, len(entities_data))}/{len(entities_data)} entities")
        
        self.stats["entities_migrated"] = len(entities_data)
    
    async def _migrate_entity_batch(self, entities: List[Any], embeddings: Optional[np.ndarray], start_idx: int):
        """Migrate a batch of entities with their embeddings.
        
        Args:
            entities: List of entity data
            embeddings: Numpy array of embeddings
            start_idx: Starting index in the original list
        """
        if not self.driver:
            await self.connect()
        
        # Prepare batch data
        batch_data = []
        for idx, entity in enumerate(entities):
            # Handle different possible structures
            if isinstance(entity, dict):
                entity_name = entity.get("entity_name", entity.get("name", f"entity_{start_idx + idx}"))
                entity_type = entity.get("entity_type", "Entity")
                content = entity.get("content", "")
            elif isinstance(entity, list) and len(entity) >= 2:
                # Format: [entity_name, entity_type, ...]
                entity_name = entity[0]
                entity_type = entity[1] if len(entity) > 1 else "Entity"
                content = entity[2] if len(entity) > 2 else ""
            else:
                logger.warning(f"Unknown entity format: {entity}")
                continue
            
            # Get embedding
            embedding = None
            if embeddings is not None and idx < len(embeddings):
                embedding = embeddings[idx].tolist()
            
            batch_data.append({
                "name": entity_name,
                "entity_type": entity_type,
                "content": content,
                "embedding": embedding,
                "workspace": self.workspace
            })
        
        # Execute batch update
        query = """
        UNWIND $batch_data as entity
        MERGE (n:Entity {name: entity.name, workspace: entity.workspace})
        SET n.entity_type = entity.entity_type,
            n.content = entity.content,
            n.embedding = entity.embedding,
            n.migrated_at = datetime()
        RETURN count(n) as updated
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, batch_data=batch_data)
                record = await result.single()
                updated = record["updated"] if record else 0
                logger.debug(f"Updated {updated} entities in batch")
        except Exception as e:
            logger.error(f"Failed to migrate entity batch: {e}")
            self.stats["errors"].append(f"Entity batch migration: {str(e)}")
    
    async def migrate_chunks(self):
        """Migrate chunk vectors from vdb_chunks.json."""
        chunks_file = self.storage_path / "vdb_chunks.json"
        
        if not chunks_file.exists():
            logger.warning(f"Chunks file not found: {chunks_file}")
            return
        
        logger.info(f"Loading chunks from {chunks_file}...")
        
        with open(chunks_file, 'r') as f:
            vdb_data = json.load(f)
        
        embedding_dim = vdb_data.get("embedding_dim", 1536)
        chunks_data = vdb_data.get("data", [])
        matrix_str = vdb_data.get("matrix", "")
        
        if not chunks_data:
            logger.warning("No chunks found in vdb_chunks.json")
            return
        
        logger.info(f"Found {len(chunks_data)} chunks with {embedding_dim}-dimensional embeddings")
        
        # Decode embeddings matrix
        embeddings_matrix = None
        if matrix_str:
            embeddings_matrix = self._decode_matrix(
                matrix_str,
                (len(chunks_data), embedding_dim)
            )
        
        # Process chunks in batches
        batch_size = 50
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i+batch_size]
            batch_embeddings = embeddings_matrix[i:i+batch_size] if embeddings_matrix is not None else None
            
            await self._migrate_chunk_batch(batch, batch_embeddings, i)
            logger.info(f"Migrated {min(i+batch_size, len(chunks_data))}/{len(chunks_data)} chunks")
        
        self.stats["chunks_migrated"] = len(chunks_data)
    
    async def _migrate_chunk_batch(self, chunks: List[Any], embeddings: Optional[np.ndarray], start_idx: int):
        """Migrate a batch of chunks with their embeddings."""
        if not self.driver:
            await self.connect()
        
        batch_data = []
        for idx, chunk in enumerate(chunks):
            # Handle different chunk formats
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id", f"chunk_{start_idx + idx}")
                content = chunk.get("content", chunk.get("text", ""))
                tokens = chunk.get("tokens", 0)
            elif isinstance(chunk, (list, tuple)) and len(chunk) >= 1:
                content = chunk[0] if chunk else ""
                chunk_id = f"chunk_{start_idx + idx}"
                tokens = chunk[1] if len(chunk) > 1 else 0
            else:
                content = str(chunk)
                chunk_id = f"chunk_{start_idx + idx}"
                tokens = 0
            
            # Get embedding
            embedding = None
            if embeddings is not None and idx < len(embeddings):
                embedding = embeddings[idx].tolist()
            
            batch_data.append({
                "chunk_id": chunk_id,
                "content": content,
                "tokens": tokens,
                "embedding": embedding,
                "workspace": self.workspace
            })
        
        query = """
        UNWIND $batch_data as chunk
        MERGE (n:Chunk {chunk_id: chunk.chunk_id, workspace: chunk.workspace})
        SET n.content = chunk.content,
            n.tokens = chunk.tokens,
            n.embedding = chunk.embedding,
            n.migrated_at = datetime()
        RETURN count(n) as updated
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, batch_data=batch_data)
                record = await result.single()
                updated = record["updated"] if record else 0
                logger.debug(f"Updated {updated} chunks in batch")
        except Exception as e:
            logger.error(f"Failed to migrate chunk batch: {e}")
            self.stats["errors"].append(f"Chunk batch migration: {str(e)}")
    
    async def create_vector_indexes(self):
        """Create vector indexes in Neo4j."""
        if not self.driver:
            await self.connect()
        
        logger.info("Creating vector indexes...")
        
        indexes = [
            """
            CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
            FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            """
            CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (n:Chunk)
            ON (n.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """
        ]
        
        async with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                    logger.info("✅ Vector index created/verified")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.error(f"Failed to create index: {e}")
                        self.stats["errors"].append(f"Index creation: {str(e)}")
    
    async def verify_migration(self):
        """Verify that migration was successful."""
        if not self.driver:
            await self.connect()
        
        logger.info("Verifying migration...")
        
        async with self.driver.session(database=self.database) as session:
            # Count entities with embeddings
            result = await session.run("""
                MATCH (n:Entity {workspace: $workspace})
                WHERE n.embedding IS NOT NULL
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            entity_count = record["count"] if record else 0
            
            # Count chunks with embeddings
            result = await session.run("""
                MATCH (n:Chunk {workspace: $workspace})
                WHERE n.embedding IS NOT NULL
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            chunk_count = record["count"] if record else 0
            
            # Test vector search on entities
            search_works = False
            if entity_count > 0:
                result = await session.run("""
                    MATCH (n:Entity {workspace: $workspace})
                    WHERE n.embedding IS NOT NULL
                    WITH n LIMIT 1
                    CALL db.index.vector.queryNodes(
                        'entity_embedding_index',
                        5,
                        n.embedding
                    ) YIELD node, score
                    RETURN count(node) as matches
                """, workspace=self.workspace)
                record = await result.single()
                search_works = (record["matches"] > 0) if record else False
            
            logger.info(f"✅ Verification Results:")
            logger.info(f"  - Entities with embeddings: {entity_count}")
            logger.info(f"  - Chunks with embeddings: {chunk_count}")
            logger.info(f"  - Vector search functional: {search_works}")
            
            if not search_works and entity_count > 0:
                logger.warning("⚠️ Vector search not working properly!")
                self.stats["errors"].append("Vector search verification failed")
            
            return {
                "entities": entity_count,
                "chunks": chunk_count,
                "search_functional": search_works
            }
    
    async def migrate_all(self):
        """Execute complete migration process."""
        logger.info("=" * 60)
        logger.info("Starting Vector Migration to Neo4j")
        logger.info("=" * 60)
        
        try:
            # Step 1: Backup
            logger.info("\n📦 Step 1: Creating backup...")
            backup_path = await self.backup_data()
            
            # Step 2: Connect to Neo4j
            logger.info("\n🔌 Step 2: Connecting to Neo4j...")
            await self.connect()
            
            # Step 3: Migrate entities
            logger.info("\n📊 Step 3: Migrating entity vectors...")
            await self.migrate_entities()
            
            # Step 4: Migrate chunks
            logger.info("\n📄 Step 4: Migrating chunk vectors...")
            await self.migrate_chunks()
            
            # Step 5: Create indexes
            logger.info("\n🔍 Step 5: Creating vector indexes...")
            await self.create_vector_indexes()
            
            # Step 6: Verify
            logger.info("\n✔️ Step 6: Verifying migration...")
            verification = await self.verify_migration()
            
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("Migration Summary")
            logger.info("=" * 60)
            logger.info(f"✅ Entities migrated: {self.stats['entities_migrated']}")
            logger.info(f"✅ Chunks migrated: {self.stats['chunks_migrated']}")
            logger.info(f"✅ Relationships migrated: {self.stats['relationships_migrated']}")
            
            if self.stats["errors"]:
                logger.warning(f"⚠️ Errors encountered: {len(self.stats['errors'])}")
                for error in self.stats["errors"]:
                    logger.warning(f"  - {error}")
            else:
                logger.info("✅ No errors encountered")
            
            logger.info("\n🎉 Migration completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.stats["errors"].append(f"Migration failed: {str(e)}")
            return False
        finally:
            await self.close()


async def main():
    """Main migration entry point."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Configuration for Neo4j connection
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7688"),  # Using unified port
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "lightrag123"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "workspace": os.getenv("NEO4J_WORKSPACE", "testing")
    }
    
    # Create migration instance
    migration = VectorMigration(neo4j_config)
    
    # Run migration
    success = await migration.migrate_all()
    
    return success


if __name__ == "__main__":
    # Run the migration
    result = asyncio.run(main())
    exit(0 if result else 1)