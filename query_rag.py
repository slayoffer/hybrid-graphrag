#!/usr/bin/env python3
"""Simple script to query the RAG system."""

import asyncio
import sys
from loguru import logger
from src.rag_system import RAGSystem

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


async def main():
    """Interactive RAG query interface."""
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = RAGSystem()
    
    # Get and display statistics
    stats = await rag.get_statistics()
    logger.info(f"Database loaded: {stats['entities']} entities, {stats['chunks']} chunks, {stats['relationships']} relationships")
    
    if stats['chunks'] == 0:
        logger.error("No data found! Please run 'python ingest_documents.py' first.")
        await rag.close()
        return
    
    logger.info("\nRAG System Ready! Type 'quit' to exit.\n")
    
    # Interactive query loop
    while True:
        try:
            # Get user question
            question = input("\n📝 Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            # Query the system
            logger.info("Searching...")
            answer = await rag.query(question, search_mode="vector_chunks")
            
            # Display answer
            print("\n🤖 Answer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Clean up
    logger.info("\nClosing connection...")
    await rag.close()
    logger.info("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())