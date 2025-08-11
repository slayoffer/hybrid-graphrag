#!/usr/bin/env python3
"""Debug similarity scores to understand retrieval issues."""

import asyncio
import numpy as np
from neo4j import AsyncGraphDatabase
from src.config import Settings
from src.core.embeddings import OpenAIEmbeddings

async def analyze_similarity_scores():
    settings = Settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    embedding_gen = OpenAIEmbeddings(api_key=settings.openai_api_key)
    
    # Test query
    query = 'What languages can Anton speak?'
    query_embedding = (await embedding_gen.async_embed_batch([query]))[0]
    
    async with driver.session() as session:
        # Get all chunks with their embeddings
        result = await session.run('''
            MATCH (c:Chunk {workspace: $workspace})
            RETURN c.id as id, c.chunk_index as idx, c.content as content, c.embedding as embedding
            ORDER BY c.chunk_index
        ''', workspace='anton_profile')
        
        chunks = []
        async for record in result:
            chunks.append({
                'id': record['id'],
                'idx': record['idx'],
                'content': record['content'],
                'embedding': record['embedding']
            })
        
        print(f"Query: '{query}'")
        print("="*80)
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            # Cosine similarity
            dot_product = np.dot(query_embedding, chunk['embedding'])
            norm_a = np.linalg.norm(query_embedding)
            norm_b = np.linalg.norm(chunk['embedding'])
            similarity = dot_product / (norm_a * norm_b)
            similarities.append((chunk['id'], chunk['idx'], similarity, chunk['content']))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        print("\nTop 10 chunks by similarity:")
        for i, (chunk_id, idx, sim, content) in enumerate(similarities[:10]):
            print(f"\n{i+1}. Chunk {idx} ({chunk_id}): similarity = {sim:.4f}")
            print(f"   Content preview: {content[:150]}...")
            if 'language' in content.lower() or 'russian' in content.lower() or 'korean' in content.lower():
                print("   ⭐ CONTAINS TARGET INFO!")
        
        # Find our target chunk
        target_chunk = 'chunk_26_df73fd4e'
        for i, (chunk_id, idx, sim, content) in enumerate(similarities):
            if chunk_id == target_chunk:
                print(f"\n🎯 Target chunk {target_chunk} is at position {i+1} with similarity {sim:.4f}")
                print(f"   Content: {content}")
                break
        
        print("\n" + "="*80)
        print("Analysis:")
        print(f"- Total chunks: {len(chunks)}")
        print(f"- Top similarity score: {similarities[0][2]:.4f}")
        print(f"- Target chunk position: {[i for i, (cid, _, _, _) in enumerate(similarities) if cid == target_chunk][0] + 1}")
    
    await driver.close()

if __name__ == "__main__":
    asyncio.run(analyze_similarity_scores())