#!/usr/bin/env python3
"""Clean up Neo4j database for fresh start."""

import asyncio
import sys
from loguru import logger
from neo4j import AsyncGraphDatabase
from src.config import Settings

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


async def cleanup_workspace(workspace: str = "test_10_questions_enhanced"):
    """Clean up all data for a specific workspace."""
    
    logger.info("=" * 60)
    logger.info(f"Cleaning up workspace: {workspace}")
    logger.info("=" * 60)
    
    # Load settings
    settings = Settings()
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Get initial counts
            logger.info("📊 Current database state:")
            
            # Count nodes
            for label in ["Chunk", "Document", "Entity"]:
                result = await session.run(f"""
                    MATCH (n:{label} {{workspace: $workspace}})
                    RETURN count(n) as count
                """, workspace=workspace)
                record = await result.single()
                count = record["count"] if record else 0
                logger.info(f"  {label}: {count} nodes")
            
            # Count relationships
            result = await session.run("""
                MATCH (e1:Entity {workspace: $workspace})-[r]-(e2:Entity {workspace: $workspace})
                RETURN count(DISTINCT r) as count
            """, workspace=workspace)
            record = await result.single()
            rel_count = record["count"] if record else 0
            logger.info(f"  Relationships: {rel_count}")
            
            # Delete all relationships first
            logger.info("\n🗑️ Deleting relationships...")
            result = await session.run("""
                MATCH (n {workspace: $workspace})-[r]-()
                DELETE r
                RETURN count(r) as deleted
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Deleted {record['deleted'] if record else 0} relationships")
            
            # Delete all nodes
            logger.info("\n🗑️ Deleting nodes...")
            
            # Delete Chunks
            result = await session.run("""
                MATCH (n:Chunk {workspace: $workspace})
                DELETE n
                RETURN count(n) as deleted
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Deleted {record['deleted'] if record else 0} chunks")
            
            # Delete Documents
            result = await session.run("""
                MATCH (n:Document {workspace: $workspace})
                DELETE n
                RETURN count(n) as deleted
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Deleted {record['deleted'] if record else 0} documents")
            
            # Delete Entities
            result = await session.run("""
                MATCH (n:Entity {workspace: $workspace})
                DELETE n
                RETURN count(n) as deleted
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Deleted {record['deleted'] if record else 0} entities")
            
            # Verify cleanup
            logger.info("\n✅ Verifying cleanup...")
            
            result = await session.run("""
                MATCH (n {workspace: $workspace})
                RETURN count(n) as remaining
            """, workspace=workspace)
            record = await result.single()
            remaining = record["remaining"] if record else 0
            
            if remaining == 0:
                logger.info(f"  ✅ Workspace '{workspace}' is completely clean!")
            else:
                logger.warning(f"  ⚠️ {remaining} nodes still remain in workspace")
            
            # List all indexes (for information)
            logger.info("\n📋 Database indexes (keeping intact):")
            result = await session.run("SHOW INDEXES")
            index_count = 0
            async for record in result:
                index_count += 1
                logger.debug(f"  - {record.get('name', 'unnamed')}: {record.get('state', 'unknown')}")
            logger.info(f"  Total indexes: {index_count}")
            
    finally:
        await driver.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup complete!")
    logger.info("=" * 60)


async def cleanup_all_workspaces():
    """Clean up all workspaces (use with caution!)."""
    
    logger.warning("⚠️ This will delete ALL data from ALL workspaces!")
    response = input("Are you sure? Type 'yes' to continue: ")
    
    if response.lower() != 'yes':
        logger.info("Cancelled.")
        return
    
    settings = Settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Get all workspaces
            result = await session.run("""
                MATCH (n)
                WHERE n.workspace IS NOT NULL
                RETURN DISTINCT n.workspace as workspace
            """)
            
            workspaces = []
            async for record in result:
                workspaces.append(record["workspace"])
            
            logger.info(f"Found {len(workspaces)} workspaces: {workspaces}")
            
            # Clean each workspace
            for ws in workspaces:
                await cleanup_workspace(ws)
    finally:
        await driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Neo4j database")
    parser.add_argument(
        "--workspace",
        default="test_10_questions_enhanced",
        help="Workspace to clean (default: test_10_questions_enhanced)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean ALL workspaces (use with caution!)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(cleanup_all_workspaces())
    else:
        asyncio.run(cleanup_workspace(args.workspace))