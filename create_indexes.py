#!/usr/bin/env python3
"""Create necessary vector indexes for Neo4j."""

import asyncio
import sys
from loguru import logger
from neo4j import AsyncGraphDatabase
from src.config import Settings

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


async def create_vector_indexes():
    """Create all necessary vector indexes."""
    
    logger.info("=" * 60)
    logger.info("Creating Vector Indexes")
    logger.info("=" * 60)
    
    # Load settings
    settings = Settings()
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    workspace = "test_10_questions_enhanced"
    
    try:
        async with driver.session() as session:
            # Create chunk vector index
            logger.info("Creating chunk vector index...")
            try:
                await session.run("""
                    CREATE VECTOR INDEX chunk_embedding_index_v2 IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                logger.info("  ✅ Chunk vector index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Chunk index: {e}")
            
            # Create entity vector index
            logger.info("Creating entity vector index...")
            try:
                await session.run("""
                    CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON e.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                logger.info("  ✅ Entity vector index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Entity index: {e}")
            
            # Create relationship vector index
            logger.info("Creating relationship vector index...")
            try:
                await session.run("""
                    CREATE VECTOR INDEX relationship_embedding_index IF NOT EXISTS
                    FOR (r:Relationship)
                    ON r.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                logger.info("  ✅ Relationship vector index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Relationship index: {e}")
            
            # Create text indexes for better search
            logger.info("Creating text indexes...")
            
            # Entity name index
            try:
                await session.run("""
                    CREATE TEXT INDEX entity_name_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON e.name
                """)
                logger.info("  ✅ Entity name text index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Entity name index: {e}")
            
            # Entity description index
            try:
                await session.run("""
                    CREATE TEXT INDEX entity_description_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON e.description
                """)
                logger.info("  ✅ Entity description text index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Entity description index: {e}")
            
            # Chunk content index
            try:
                await session.run("""
                    CREATE TEXT INDEX chunk_content_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.content
                """)
                logger.info("  ✅ Chunk content text index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Chunk content index: {e}")
            
            # Create additional performance indexes
            logger.info("Creating performance indexes...")
            
            # Workspace index
            try:
                await session.run("""
                    CREATE INDEX entity_workspace_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON (e.workspace)
                """)
                logger.info("  ✅ Entity workspace index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Entity workspace index: {e}")
            
            try:
                await session.run("""
                    CREATE INDEX chunk_workspace_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON (c.workspace)
                """)
                logger.info("  ✅ Chunk workspace index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Chunk workspace index: {e}")
            
            # Entity type index
            try:
                await session.run("""
                    CREATE INDEX entity_type_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON (e.type)
                """)
                logger.info("  ✅ Entity type index created/exists")
            except Exception as e:
                logger.warning(f"  ⚠️ Entity type index: {e}")
            
            # List all indexes
            logger.info("\n📋 All indexes:")
            result = await session.run("SHOW INDEXES")
            index_count = 0
            async for record in result:
                index_count += 1
                name = record.get('name', 'unnamed')
                state = record.get('state', 'unknown')
                if state == 'ONLINE':
                    logger.info(f"  ✅ {name}: {state}")
                else:
                    logger.warning(f"  ⚠️ {name}: {state}")
            
            logger.info(f"\nTotal indexes: {index_count}")
            
            # Check for data with embeddings
            logger.info("\n📊 Checking for data with embeddings:")
            
            # Count chunks with embeddings
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as count
            """, workspace=workspace)
            record = await result.single()
            chunks_with_embeddings = record["count"] if record else 0
            logger.info(f"  Chunks with embeddings: {chunks_with_embeddings}")
            
            # Count entities with embeddings
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                WHERE e.embedding IS NOT NULL
                RETURN count(e) as count
            """, workspace=workspace)
            record = await result.single()
            entities_with_embeddings = record["count"] if record else 0
            logger.info(f"  Entities with embeddings: {entities_with_embeddings}")
            
            if chunks_with_embeddings == 0 or entities_with_embeddings == 0:
                logger.warning("\n⚠️ Warning: No embeddings found! You may need to re-ingest with embeddings enabled.")
            
    finally:
        await driver.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("Index creation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(create_vector_indexes())