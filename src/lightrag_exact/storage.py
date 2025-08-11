"""Exact LightRAG storage implementation with Neo4j backend."""

import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from neo4j import AsyncGraphDatabase
from loguru import logger
import numpy as np

from .chunking import compute_mdhash_id, Tokenizer, chunking_by_token_size, clean_text


class LightRAGExactStorage:
    """Exact LightRAG storage implementation using Neo4j."""
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        workspace: str = "default",
        working_dir: str = "./rag_storage",
        embedding_func: Optional[callable] = None,
        chunk_token_size: int = 1024,
        chunk_overlap_token_size: int = 128,
        tokenizer_model: str = "gpt-4.1-mini"
    ):
        """Initialize storage with Neo4j backend."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.workspace = workspace
        self.working_dir = Path(working_dir) / workspace
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(tokenizer_model)
        
        # Chunking parameters
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        
        # Embedding function
        self.embedding_func = embedding_func
        
        # Neo4j driver
        self.driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Local KV store for text chunks (like LightRAG)
        self.text_chunks_file = self.working_dir / "kv_store_text_chunks.json"
        self.text_chunks_data = self._load_text_chunks()
        
        # Local full docs store
        self.full_docs_file = self.working_dir / "kv_store_full_docs.json"
        self.full_docs_data = self._load_full_docs()
        
        logger.info(f"LightRAGExactStorage initialized for workspace: {workspace}")
    
    def _load_text_chunks(self) -> Dict[str, Any]:
        """Load text chunks from local KV store."""
        if self.text_chunks_file.exists():
            with open(self.text_chunks_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_text_chunks(self):
        """Save text chunks to local KV store."""
        with open(self.text_chunks_file, 'w') as f:
            json.dump(self.text_chunks_data, f, indent=2)
    
    def _load_full_docs(self) -> Dict[str, Any]:
        """Load full documents from local KV store."""
        if self.full_docs_file.exists():
            with open(self.full_docs_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_full_docs(self):
        """Save full documents to local KV store."""
        with open(self.full_docs_file, 'w') as f:
            json.dump(self.full_docs_data, f, indent=2)
    
    async def insert_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False
    ) -> Dict[str, Any]:
        """
        Insert documents exactly like LightRAG does.
        
        Args:
            documents: List of document strings
            doc_ids: Optional document IDs (will generate MD5 if not provided)
            file_paths: Optional file paths for documents
            split_by_character: Optional character to split by
            split_by_character_only: If True, only split by character
            
        Returns:
            Processing results
        """
        if isinstance(documents, str):
            documents = [documents]
        
        if file_paths is None:
            file_paths = ["unknown_source"] * len(documents)
        elif isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Clean documents
        cleaned_docs = [clean_text(doc) for doc in documents]
        
        # Generate document IDs if not provided
        if doc_ids is None:
            doc_ids = [
                compute_mdhash_id(doc, prefix="doc-")
                for doc in cleaned_docs
            ]
        
        results = {
            "processed_docs": 0,
            "total_chunks": 0,
            "chunks_by_doc": {}
        }
        
        async with self.driver.session() as session:
            # Clear existing data for clean test
            await session.run("MATCH (n) WHERE n.workspace = $workspace DETACH DELETE n", 
                            workspace=self.workspace)
        
        for doc_id, content, file_path in zip(doc_ids, cleaned_docs, file_paths):
            # Store full document
            self.full_docs_data[doc_id] = {
                "content": content,
                "file_path": file_path,
                "created_at": int(time.time())
            }
            
            # Generate chunks exactly like LightRAG
            chunks_raw = chunking_by_token_size(
                self.tokenizer,
                content,
                split_by_character,
                split_by_character_only,
                self.chunk_overlap_token_size,
                self.chunk_token_size
            )
            
            # Process chunks
            chunks_data = {}
            for chunk_info in chunks_raw:
                chunk_id = compute_mdhash_id(chunk_info["content"], prefix="chunk-")
                
                # Prepare chunk data with all LightRAG fields
                chunk_data = {
                    "tokens": chunk_info["tokens"],
                    "content": chunk_info["content"],
                    "chunk_order_index": chunk_info["chunk_order_index"],
                    "full_doc_id": doc_id,
                    "file_path": file_path,
                    "llm_cache_list": [],  # Empty initially
                    "create_time": int(time.time()),
                    "update_time": int(time.time()),
                    "_id": chunk_id
                }
                
                chunks_data[chunk_id] = chunk_data
                
                # Store in text chunks KV store
                self.text_chunks_data[chunk_id] = chunk_data
            
            # Process chunks for vector storage
            await self._store_chunks_in_neo4j(chunks_data)
            
            results["processed_docs"] += 1
            results["total_chunks"] += len(chunks_data)
            results["chunks_by_doc"][doc_id] = len(chunks_data)
        
        # Save local KV stores
        self._save_text_chunks()
        self._save_full_docs()
        
        return results
    
    async def _store_chunks_in_neo4j(self, chunks: Dict[str, Any]):
        """Store chunks in Neo4j with embeddings."""
        async with self.driver.session() as session:
            for chunk_id, chunk_data in chunks.items():
                # Generate embedding if function provided
                if self.embedding_func:
                    # Call the func attribute of EmbeddingFunc object
                    if hasattr(self.embedding_func, 'func'):
                        embedding = await self.embedding_func.func([chunk_data["content"]])
                    else:
                        embedding = await self.embedding_func([chunk_data["content"]])
                    # Handle numpy array or list
                    if hasattr(embedding, 'tolist'):  # numpy array
                        embedding_vector = embedding[0].tolist() if len(embedding.shape) > 1 else embedding.tolist()
                    elif isinstance(embedding, list) and len(embedding) > 0:
                        embedding_vector = embedding[0] if isinstance(embedding[0], list) else embedding
                    else:
                        embedding_vector = embedding
                else:
                    # Random embedding for testing
                    embedding_vector = np.random.random(1536).tolist()
                
                # Store in Neo4j with minimal fields (like LightRAG vector DB)
                await session.run("""
                    CREATE (n:Chunk {
                        id: $chunk_id,
                        content: $content,
                        full_doc_id: $full_doc_id,
                        file_path: $file_path,
                        created_at: $created_at,
                        embedding: $embedding,
                        workspace: $workspace
                    })
                """,
                chunk_id=chunk_id,
                content=chunk_data["content"],
                full_doc_id=chunk_data["full_doc_id"],
                file_path=chunk_data["file_path"],
                created_at=chunk_data["create_time"],
                embedding=embedding_vector,
                workspace=self.workspace)
                
                logger.debug(f"Stored chunk in Neo4j: {chunk_id}")
    
    async def extract_entities_and_relationships(
        self,
        llm_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from chunks.
        
        Note: This is a simplified version. Real LightRAG uses complex LLM prompts.
        """
        results = {
            "entities": [],
            "relationships": []
        }
        
        # For demo purposes, create some example entities
        # Real implementation would use LLM to extract these
        example_entities = [
            {
                "id": compute_mdhash_id("Aikido", prefix="ent-"),
                "name": "Aikido",
                "description": "A cloud-native security platform",
                "source_chunks": []
            },
            {
                "id": compute_mdhash_id("GitLab CI/CD", prefix="ent-"),
                "name": "GitLab CI/CD",
                "description": "Continuous integration and deployment platform",
                "source_chunks": []
            }
        ]
        
        example_relationships = [
            {
                "id": compute_mdhash_id("GitLab CI/CD-integrates_with-HashiCorp Vault", prefix="rel-"),
                "source": "GitLab CI/CD",
                "target": "HashiCorp Vault",
                "relationship": "integrates_with",
                "description": "GitLab CI/CD integrates with HashiCorp Vault for secure credential management"
            }
        ]
        
        # Store entities in Neo4j
        async with self.driver.session() as session:
            for entity in example_entities:
                if self.embedding_func:
                    # Call the func attribute of EmbeddingFunc object
                    if hasattr(self.embedding_func, 'func'):
                        embedding = await self.embedding_func.func([entity["description"]])
                    else:
                        embedding = await self.embedding_func([entity["description"]])
                    # Handle numpy array or list
                    if hasattr(embedding, 'tolist'):
                        embedding_vector = embedding[0].tolist() if len(embedding.shape) > 1 else embedding.tolist()
                    else:
                        embedding_vector = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
                else:
                    embedding_vector = np.random.random(1536).tolist()
                
                await session.run("""
                    CREATE (n:Entity {
                        id: $entity_id,
                        entity_name: $name,
                        content: $description,
                        embedding: $embedding,
                        created_at: $created_at,
                        file_path: 'extracted_entities',
                        workspace: $workspace
                    })
                """,
                entity_id=entity["id"],
                name=entity["name"],
                description=entity["description"],
                embedding=embedding_vector,
                created_at=int(time.time()),
                workspace=self.workspace)
                
                results["entities"].append(entity)
            
            # Store relationships
            for rel in example_relationships:
                if self.embedding_func:
                    rel_content = f"{rel['source']}\t{rel['target']}\n{rel['relationship']}\n{rel['description']}"
                    # Call the func attribute of EmbeddingFunc object
                    if hasattr(self.embedding_func, 'func'):
                        embedding = await self.embedding_func.func([rel_content])
                    else:
                        embedding = await self.embedding_func([rel_content])
                    # Handle numpy array or list
                    if hasattr(embedding, 'tolist'):
                        embedding_vector = embedding[0].tolist() if len(embedding.shape) > 1 else embedding.tolist()
                    else:
                        embedding_vector = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
                else:
                    embedding_vector = np.random.random(1536).tolist()
                
                await session.run("""
                    CREATE (n:Relationship {
                        id: $rel_id,
                        src_id: $source,
                        tgt_id: $target,
                        content: $description,
                        relationship_type: $rel_type,
                        embedding: $embedding,
                        created_at: $created_at,
                        file_path: 'extracted_relationships',
                        workspace: $workspace
                    })
                """,
                rel_id=rel["id"],
                source=rel["source"],
                target=rel["target"],
                description=rel["description"],
                rel_type=rel["relationship"],
                embedding=embedding_vector,
                created_at=int(time.time()),
                workspace=self.workspace)
                
                results["relationships"].append(rel)
        
        return results
    
    async def create_vector_indexes(self):
        """Create Neo4j vector indexes."""
        async with self.driver.session() as session:
            indexes = [
                ("chunk_embedding_index", "Chunk", "embedding"),
                ("entity_embedding_index", "Entity", "embedding"),
                ("relationship_embedding_index", "Relationship", "embedding")
            ]
            
            for index_name, label, property in indexes:
                try:
                    await session.run(f"""
                        CALL db.index.vector.createNodeIndex(
                            '{index_name}',
                            '{label}',
                            '{property}',
                            1536,
                            'cosine'
                        )
                    """)
                    logger.info(f"Created vector index: {index_name}")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"Vector index already exists: {index_name}")
                    else:
                        logger.error(f"Failed to create index {index_name}: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "workspace": self.workspace,
            "text_chunks": len(self.text_chunks_data),
            "full_docs": len(self.full_docs_data),
            "neo4j": {}
        }
        
        async with self.driver.session() as session:
            # Count chunks
            result = await session.run("""
                MATCH (n:Chunk)
                WHERE n.workspace = $workspace
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["neo4j"]["chunks"] = record["count"] if record else 0
            
            # Count entities
            result = await session.run("""
                MATCH (n:Entity)
                WHERE n.workspace = $workspace
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["neo4j"]["entities"] = record["count"] if record else 0
            
            # Count relationships
            result = await session.run("""
                MATCH (n:Relationship)
                WHERE n.workspace = $workspace
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["neo4j"]["relationships"] = record["count"] if record else 0
        
        return stats
    
    async def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search on chunks.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        if self.embedding_func:
            if hasattr(self.embedding_func, 'func'):
                query_embedding = await self.embedding_func.func([query])
            else:
                query_embedding = await self.embedding_func([query])
            
            # Handle numpy array or list
            if hasattr(query_embedding, 'tolist'):
                query_vector = query_embedding[0].tolist() if len(query_embedding.shape) > 1 else query_embedding.tolist()
            elif isinstance(query_embedding, list) and len(query_embedding) > 0:
                query_vector = query_embedding[0] if isinstance(query_embedding[0], list) else query_embedding
            else:
                query_vector = query_embedding
        else:
            # Random embedding for testing
            query_vector = np.random.random(1536).tolist()
        
        # Perform vector search in Neo4j
        async with self.driver.session() as session:
            result = await session.run("""
                CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_vector)
                YIELD node, score
                WHERE node.workspace = $workspace
                RETURN node.id as id, node.content as content, node.file_path as file_path, score
                ORDER BY score DESC
                LIMIT $top_k
            """,
            query_vector=query_vector,
            top_k=top_k,
            workspace=self.workspace)
            
            results = []
            async for record in result:
                results.append({
                    "id": record["id"],
                    "content": record["content"],
                    "file_path": record["file_path"],
                    "score": record["score"]
                })
            
            return results
    
    async def close(self):
        """Close connections and save data."""
        self._save_text_chunks()
        self._save_full_docs()
        await self.driver.close()
        logger.info("LightRAGExactStorage closed")