#!/usr/bin/env python3
"""Document ingestion script for Anton Evseev profile."""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from neo4j import AsyncGraphDatabase
from src.config import Settings
from src.extraction.langchain_entity_extractor import LangChainEntityExtractor
from src.core.knn_graph import KNNGraph
import hashlib
from typing import List, Dict, Any
import openai

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def create_chunks(content: str, chunk_size: int = 250, overlap: int = 150) -> List[Dict[str, Any]]:
    """Create overlapping chunks from document content.
    
    Using smaller chunks (250 chars) with high overlap (150 chars) for better 
    graph connectivity and more precise retrieval.
    """
    chunks = []
    
    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i:i+chunk_size]
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:8]
        
        chunks.append({
            "id": f"chunk_{i//(chunk_size-overlap)}_{chunk_id}",
            "content": chunk,
            "index": i // (chunk_size - overlap),
            "start_char": i,
            "end_char": min(i + chunk_size, len(content))
        })
    
    return chunks


async def main():
    """Ingest Anton Evseev profile document."""
    
    logger.info("=" * 80)
    logger.info("Anton Evseev Profile Document Ingestion")
    logger.info("=" * 80)
    
    # Load settings
    settings = Settings()
    workspace = "anton_profile"
    
    # Load the document
    doc_path = Path("data/mixed/About Me - Anton Evseev - Confluence.md")
    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    logger.info(f"📄 Loaded document: {doc_path.name}")
    logger.info(f"  Size: {len(content):,} characters")
    
    # Create chunks with smaller size for graph traversal
    logger.info("\n📦 Creating document chunks...")
    chunks = create_chunks(content, chunk_size=250, overlap=150)
    logger.info(f"  Created {len(chunks)} chunks (250 chars each with 150 char overlap)")
    
    # Generate embeddings for chunks
    logger.info("\n🧮 Generating embeddings...")
    openai.api_key = settings.openai_api_key
    
    for chunk in chunks:
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk["content"]
            )
            chunk["embedding"] = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {chunk['id']}: {e}")
            chunk["embedding"] = [0.0] * 1536  # Default embedding
    
    logger.info(f"  Generated embeddings for {len(chunks)} chunks")
    
    # Connect to Neo4j and store data
    logger.info("\n💾 Storing in Neo4j...")
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Clear existing data for this workspace
            await session.run("""
                MATCH (n {workspace: $workspace})
                DETACH DELETE n
            """, workspace=workspace)
            logger.info("  Cleared existing data")
            
            # Store the document
            await session.run("""
                CREATE (d:Document {
                    workspace: $workspace,
                    id: $doc_id,
                    title: $title,
                    source: $source,
                    content: $content,
                    char_count: $char_count
                })
            """, 
                workspace=workspace,
                doc_id="anton_profile_doc",
                title="About Me - Anton Evseev",
                source=str(doc_path),
                content=content[:5000],
                char_count=len(content)
            )
            logger.info("  ✅ Document stored")
            
            # Store chunks with embeddings
            chunk_count = 0
            for chunk in chunks:
                await session.run("""
                    CREATE (c:Chunk {
                        workspace: $workspace,
                        id: $chunk_id,
                        content: $content,
                        chunk_index: $index,
                        doc_id: $doc_id,
                        start_char: $start_char,
                        end_char: $end_char,
                        embedding: $embedding
                    })
                """,
                    workspace=workspace,
                    chunk_id=chunk["id"],
                    content=chunk["content"],
                    index=chunk["index"],
                    doc_id="anton_profile_doc",
                    start_char=chunk["start_char"],
                    end_char=chunk["end_char"],
                    embedding=chunk["embedding"]
                )
                chunk_count += 1
            
            logger.info(f"  ✅ {chunk_count} chunks stored with embeddings")
            
            # Build KNN graph relationships
            logger.info("\n🔗 Building KNN graph relationships...")
            knn_graph = KNNGraph(
                min_similarity=0.85,  # Higher threshold for quality relationships
                max_neighbors=5,      # Fewer but higher quality neighbors
                batch_size=50
            )
            
            num_relationships = await knn_graph.build_knn_relationships(
                session,
                workspace=workspace,
                rebuild=True
            )
            logger.info(f"  ✅ Created {num_relationships} SIMILAR relationships")
            
            # Get graph stats
            stats = await knn_graph.get_graph_stats(session, workspace)
            logger.info(f"  📊 Graph stats:")
            logger.info(f"     - Avg similarity score: {stats['avg_similarity_score']:.3f}")
            logger.info(f"     - Avg neighbors per chunk: {stats['avg_neighbors_per_chunk']:.1f}")
            
            # Verify storage
            logger.info("\n✅ Verification:")
            
            # Count chunks
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                RETURN count(c) as count
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Total chunks: {record['count']}")
            
            # Count documents
            result = await session.run("""
                MATCH (d:Document {workspace: $workspace})
                RETURN count(d) as count
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Total documents: {record['count']}")
            
    finally:
        await driver.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Anton Evseev profile ingestion complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())