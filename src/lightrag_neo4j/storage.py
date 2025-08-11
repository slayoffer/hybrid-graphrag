"""
Neo4j-only storage implementation for LightRAG.
Stores all metadata directly in Neo4j, eliminating the need for JSON KV store.
Production-ready with ACID compliance and concurrent access support.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from functools import lru_cache
import asyncio
import numpy as np
from neo4j import AsyncGraphDatabase
from loguru import logger

import tiktoken
from src.lightrag_exact.chunking import chunking_by_token_size, clean_text

class Tokenizer:
    """Simple tokenizer wrapper."""
    def __init__(self, model: str = "cl100k_base"):
        # Use get_encoding directly for encoding names
        self.encoding = tiktoken.get_encoding(model)
    
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)


class QueryCache:
    """Simple LRU cache for query results to improve performance."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize cache with max size and TTL."""
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query: str, params: Dict[str, Any]) -> str:
        """Create cache key from query and parameters."""
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        return hashlib.md5(f"{query}{sorted_params}".encode()).hexdigest()
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(query, params)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return result
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, params: Dict[str, Any], result: Any):
        """Cache query result."""
        key = self._make_key(query, params)
        
        # Add to cache
        self.cache[key] = (result, time.time())
        self.cache.move_to_end(key)
        
        # Evict oldest if over size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }


class Neo4jOnlyStorage:
    """
    Production-ready storage using only Neo4j for all data.
    No JSON files, everything stored as node properties.
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        workspace: str = "default",
        tokenizer_model: str = "cl100k_base",
        chunk_token_size: int = 1024,
        chunk_overlap_token_size: int = 128,
        embedding_func: Optional[Any] = None,
        max_connection_pool_size: int = 100,
        connection_acquisition_timeout: int = 30,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        batch_size: int = 100
    ):
        """Initialize Neo4j-only storage with performance optimizations."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.workspace = workspace
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(tokenizer_model)
        
        # Chunking parameters
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        
        # Embedding function
        self.embedding_func = embedding_func
        
        # Performance settings
        self.batch_size = batch_size
        
        # Initialize query cache
        self.cache = QueryCache(ttl_seconds=cache_ttl) if enable_cache else None
        
        # Neo4j driver with connection pooling
        self.driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout,
            max_transaction_retry_time=30
        )
        
        logger.info(f"Neo4jOnlyStorage initialized for workspace: {workspace}")
        logger.info(f"Performance: pool_size={max_connection_pool_size}, cache={enable_cache}, batch_size={batch_size}")
    
    def _generate_id(self, content: str) -> str:
        """Generate MD5 hash-based ID for content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def insert_documents(
        self,
        documents: Union[List[str], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Insert documents and create chunks with all metadata in Neo4j.
        
        Args:
            documents: List of document strings or dicts with content
            
        Returns:
            Dictionary with insertion statistics
        """
        results = {
            "processed_docs": 0,
            "total_chunks": 0,
            "chunks_by_doc": {}
        }
        
        # Clean documents
        if isinstance(documents[0], dict):
            cleaned_docs = [clean_text(doc.get("content", doc)) for doc in documents]
        else:
            cleaned_docs = [clean_text(doc) for doc in documents]
        
        for doc_index, doc_content in enumerate(cleaned_docs):
            # Generate document ID
            doc_id = f"doc-{self._generate_id(doc_content)}"
            
            # Create chunks with exact LightRAG structure
            chunks = chunking_by_token_size(
                self.tokenizer,
                doc_content,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size
            )
            
            # Store full document in Neo4j
            await self._store_document(doc_id, doc_content, doc_index)
            
            # Process each chunk
            for chunk_index, chunk_data in enumerate(chunks):
                chunk_id = f"chunk-{self._generate_id(chunk_data['content'])}"
                
                # Prepare all metadata
                chunk_metadata = {
                    "id": chunk_id,
                    "content": chunk_data["content"],
                    "tokens": chunk_data["tokens"],
                    "chunk_order_index": chunk_index,
                    "full_doc_id": doc_id,
                    "file_path": f"doc_{doc_index}.md",
                    "llm_cache_list": [],  # Empty list for compatibility
                    "create_time": int(time.time()),
                    "update_time": int(time.time()),
                    "workspace": self.workspace
                }
                
                # Generate embedding if available
                if self.embedding_func:
                    embedding = await self._generate_embedding(chunk_data["content"])
                else:
                    embedding = np.random.random(1536).tolist()
                
                chunk_metadata["embedding"] = embedding
                
                # Store chunk in Neo4j with ALL metadata
                await self._store_chunk(chunk_metadata)
            
            results["processed_docs"] += 1
            results["total_chunks"] += len(chunks)
            results["chunks_by_doc"][doc_id] = len(chunks)
        
        return results
    
    async def batch_insert_documents(
        self,
        documents: Union[List[str], List[Dict[str, Any]]],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Batch insert documents for improved performance.
        
        Args:
            documents: List of document strings or dicts with content
            batch_size: Optional batch size override
            
        Returns:
            Dictionary with insertion statistics
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        results = {
            "processed_docs": 0,
            "total_chunks": 0,
            "chunks_by_doc": {},
            "batch_timings": []
        }
        
        # Clean documents
        if isinstance(documents[0], dict):
            cleaned_docs = [clean_text(doc.get("content", doc)) for doc in documents]
        else:
            cleaned_docs = [clean_text(doc) for doc in documents]
        
        # Process in batches for better performance
        for batch_start in range(0, len(cleaned_docs), batch_size):
            batch_end = min(batch_start + batch_size, len(cleaned_docs))
            batch_docs = cleaned_docs[batch_start:batch_end]
            
            start_time = time.time()
            
            # Prepare all chunks for this batch
            batch_chunks = []
            batch_doc_nodes = []
            
            for doc_index, doc_content in enumerate(batch_docs, start=batch_start):
                # Generate document ID
                doc_id = f"doc-{self._generate_id(doc_content)}"
                
                # Create chunks with exact LightRAG structure
                chunks = chunking_by_token_size(
                    self.tokenizer,
                    doc_content,
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size
                )
                
                # Prepare document node
                batch_doc_nodes.append({
                    "doc_id": doc_id,
                    "content": doc_content,
                    "index": doc_index,
                    "workspace": self.workspace
                })
                
                # Prepare chunks for batch insert
                for chunk_index, chunk_data in enumerate(chunks):
                    chunk_id = f"chunk-{self._generate_id(chunk_data['content'])}"
                    
                    # Generate embedding if available
                    if self.embedding_func:
                        embedding = await self._generate_embedding(chunk_data["content"])
                    else:
                        embedding = np.random.random(1536).tolist()
                    
                    batch_chunks.append({
                        "id": chunk_id,
                        "content": chunk_data["content"],
                        "tokens": chunk_data["tokens"],
                        "chunk_order_index": chunk_index,
                        "full_doc_id": doc_id,
                        "file_path": f"doc_{doc_index}.md",
                        "llm_cache_list": [],
                        "create_time": int(time.time()),
                        "update_time": int(time.time()),
                        "workspace": self.workspace,
                        "embedding": embedding
                    })
                
                results["chunks_by_doc"][doc_id] = len(chunks)
                results["total_chunks"] += len(chunks)
            
            # Batch insert documents
            await self._batch_store_documents(batch_doc_nodes)
            
            # Batch insert chunks
            await self._batch_store_chunks(batch_chunks)
            
            results["processed_docs"] += len(batch_docs)
            
            batch_time = time.time() - start_time
            results["batch_timings"].append(batch_time)
            logger.info(f"Batch {batch_start//batch_size + 1}: Processed {len(batch_docs)} docs with {len(batch_chunks)} chunks in {batch_time:.2f}s")
        
        # Calculate average batch time
        if results["batch_timings"]:
            avg_batch_time = sum(results["batch_timings"]) / len(results["batch_timings"])
            results["avg_batch_time"] = avg_batch_time
            logger.info(f"Batch processing complete: {results['processed_docs']} docs, {results['total_chunks']} chunks, avg batch time: {avg_batch_time:.2f}s")
        
        return results
    
    async def _batch_store_documents(self, doc_nodes: List[Dict[str, Any]]):
        """Batch store multiple documents in Neo4j."""
        async with self.driver.session() as session:
            await session.run("""
                UNWIND $docs as doc
                MERGE (d:Document {id: doc.doc_id})
                SET d.content = doc.content,
                    d.doc_index = doc.index,
                    d.workspace = doc.workspace,
                    d.created_at = timestamp(),
                    d.updated_at = timestamp()
            """, docs=doc_nodes)
    
    async def _batch_store_chunks(self, chunks: List[Dict[str, Any]]):
        """Batch store multiple chunks in Neo4j with all metadata."""
        async with self.driver.session() as session:
            # Batch insert chunks
            await session.run("""
                UNWIND $chunks as chunk
                CREATE (c:Chunk {
                    id: chunk.id,
                    content: chunk.content,
                    tokens: chunk.tokens,
                    chunk_order_index: chunk.chunk_order_index,
                    full_doc_id: chunk.full_doc_id,
                    file_path: chunk.file_path,
                    llm_cache_list: chunk.llm_cache_list,
                    create_time: chunk.create_time,
                    update_time: chunk.update_time,
                    workspace: chunk.workspace,
                    embedding: chunk.embedding
                })
            """, chunks=chunks)
            
            # Batch create relationships to documents
            await session.run("""
                UNWIND $chunks as chunk
                MATCH (c:Chunk {id: chunk.id})
                MATCH (d:Document {id: chunk.full_doc_id})
                CREATE (c)-[:BELONGS_TO]->(d)
            """, chunks=chunks)
            
            logger.debug(f"Batch stored {len(chunks)} chunks in Neo4j")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if hasattr(self.embedding_func, 'func'):
            embedding = await self.embedding_func.func([text])
        else:
            embedding = await self.embedding_func([text])
        
        # Handle numpy array or list
        if hasattr(embedding, 'tolist'):
            return embedding[0].tolist() if len(embedding.shape) > 1 else embedding.tolist()
        elif isinstance(embedding, list) and len(embedding) > 0:
            return embedding[0] if isinstance(embedding[0], list) else embedding
        else:
            return embedding
    
    async def _store_document(self, doc_id: str, content: str, index: int):
        """Store full document in Neo4j."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.content = $content,
                    d.doc_index = $index,
                    d.workspace = $workspace,
                    d.created_at = timestamp(),
                    d.updated_at = timestamp()
            """,
            doc_id=doc_id,
            content=content,
            index=index,
            workspace=self.workspace)
    
    async def _store_chunk(self, chunk_data: Dict[str, Any]):
        """Store chunk with all metadata in Neo4j."""
        async with self.driver.session() as session:
            # Store ALL fields as node properties
            await session.run("""
                CREATE (c:Chunk {
                    id: $id,
                    content: $content,
                    tokens: $tokens,
                    chunk_order_index: $chunk_order_index,
                    full_doc_id: $full_doc_id,
                    file_path: $file_path,
                    llm_cache_list: $llm_cache_list,
                    create_time: $create_time,
                    update_time: $update_time,
                    workspace: $workspace,
                    embedding: $embedding
                })
                WITH c
                MATCH (d:Document {id: $full_doc_id})
                CREATE (c)-[:BELONGS_TO]->(d)
            """,
            **chunk_data)
            
            logger.debug(f"Stored chunk in Neo4j with all metadata: {chunk_data['id']}")
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on chunks using workspace-specific index.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            return_metadata: If True, return all metadata fields
            
        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        if self.embedding_func:
            query_vector = await self._generate_embedding(query)
        else:
            query_vector = np.random.random(1536).tolist()
        
        # Build return clause based on metadata flag
        if return_metadata:
            return_clause = """
                       node.id as id,
                       node.content as content,
                       node.tokens as tokens,
                       node.chunk_order_index as chunk_order_index,
                       node.full_doc_id as full_doc_id,
                       node.file_path as file_path,
                       node.llm_cache_list as llm_cache_list,
                       node.create_time as create_time,
                       node.update_time as update_time,
            """
        else:
            return_clause = """
                       node.id as id,
                       node.content as content,
                       node.file_path as file_path,
            """
        
        async with self.driver.session() as session:
            try:
                # Optimized query: Get more results and filter by workspace
                # This ensures we always get results from the correct workspace
                result = await session.run(f"""
                    CALL db.index.vector.queryNodes('chunk_embedding_index_v2', $query_limit, $query_vector)
                    YIELD node, score
                    WHERE node.workspace = $workspace
                    RETURN {return_clause}
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """,
                query_vector=query_vector,
                query_limit=min(top_k * 10, 100),  # Get 10x results or max 100 to ensure workspace matches
                top_k=top_k,
                workspace=self.workspace)
                
                results = []
                async for record in result:
                    results.append(dict(record))
                
                # If we got results, return them
                if results:
                    return results
                    
            except Exception as e:
                logger.debug(f"Vector index not available, using fallback: {e}")
            
            # Fallback: Direct similarity calculation if workspace index doesn't exist
            # This is simpler and only used during initial setup
            logger.debug("Using fallback similarity calculation")
            result = await session.run(f"""
                MATCH (c:Chunk)
                WHERE c.workspace = $workspace
                WITH c, c.embedding as embedding
                LIMIT 100
                RETURN c.id as id,
                       c.content as content,
                       c.file_path as file_path,
                       {"c.tokens as tokens, c.chunk_order_index as chunk_order_index, c.full_doc_id as full_doc_id, c.llm_cache_list as llm_cache_list, c.create_time as create_time, c.update_time as update_time," if return_metadata else ""}
                       embedding
            """,
            workspace=self.workspace)
            
            all_chunks = []
            async for record in result:
                chunk_dict = dict(record)
                if chunk_dict.get('embedding'):
                    # Calculate cosine similarity
                    embedding = np.array(chunk_dict['embedding'])
                    query_vec = np.array(query_vector)
                    score = np.dot(embedding, query_vec) / (np.linalg.norm(embedding) * np.linalg.norm(query_vec))
                    chunk_dict['score'] = float(score)
                    del chunk_dict['embedding']
                    all_chunks.append(chunk_dict)
            
            # Sort by score and return top k
            all_chunks.sort(key=lambda x: x['score'], reverse=True)
            return all_chunks[:top_k]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by ID with all metadata.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Complete chunk data or None if not found
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (c:Chunk {id: $chunk_id, workspace: $workspace})
                RETURN c {.*, embedding: null} as chunk
            """,
            chunk_id=chunk_id,
            workspace=self.workspace)
            
            record = await result.single()
            return dict(record["chunk"]) if record else None
    
    async def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document, ordered by chunk_order_index.
        
        Args:
            doc_id: The document ID
            
        Returns:
            List of chunks in order
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (c:Chunk {full_doc_id: $doc_id, workspace: $workspace})
                RETURN c {.*, embedding: null} as chunk
                ORDER BY c.chunk_order_index
            """,
            doc_id=doc_id,
            workspace=self.workspace)
            
            chunks = []
            async for record in result:
                chunks.append(dict(record["chunk"]))
            
            return chunks
    
    async def create_vector_indexes(self):
        """Create optimized vector index for similarity search."""
        async with self.driver.session() as session:
            # Use a single global index but optimize queries
            index_name = "chunk_embedding_index_v2"
            
            # Check if index exists
            result = await session.run("""
                SHOW INDEXES
                YIELD name
                WHERE name = $index_name
                RETURN count(*) as count
            """, index_name=index_name)
            record = await result.single()
            
            if record and record["count"] == 0:
                # Create global index for all chunks
                try:
                    await session.run("""
                        CREATE VECTOR INDEX chunk_embedding_index_v2
                        FOR (n:Chunk)
                        ON n.embedding
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    logger.info(f"Created optimized vector index: {index_name}")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"Vector index {index_name} already exists")
                    else:
                        logger.warning(f"Could not create vector index: {e}")
            else:
                logger.info(f"Vector index {index_name} already exists")
        
        # Try to tune HNSW parameters for better performance
        await self.tune_hnsw_parameters()
    
    async def tune_hnsw_parameters(self):
        """Tune HNSW algorithm parameters for optimal performance (Enterprise Edition only)."""
        try:
            async with self.driver.session() as session:
                # HNSW tuning parameters for better performance
                tuning_params = {
                    'm': 16,  # Number of bi-directional links (higher = better quality, more memory)
                    'efConstruction': 200,  # Size of dynamic candidate list (higher = better quality, slower indexing)
                    'ef': 50  # Size of dynamic list for search (higher = better quality, slower search)
                }
                
                # Try to tune both chunk and entity indexes
                indexes = ['chunk_embedding_index_v2', 'entity_embedding_index']
                
                for index_name in indexes:
                    try:
                        # Check if index exists first
                        check_result = await session.run("""
                            SHOW INDEXES
                            YIELD name
                            WHERE name = $index_name
                            RETURN count(*) as count
                        """, index_name=index_name)
                        check_record = await check_result.single()
                        
                        if check_record and check_record["count"] > 0:
                            # Try to set HNSW parameters (Enterprise only)
                            await session.run("""
                                CALL db.index.vector.setConfig($index_name, {
                                    m: $m,
                                    efConstruction: $ef_construction,
                                    ef: $ef
                                })
                            """, 
                            index_name=index_name,
                            m=tuning_params['m'],
                            ef_construction=tuning_params['efConstruction'],
                            ef=tuning_params['ef'])
                            
                            logger.info(f"HNSW parameters tuned for {index_name}: m={tuning_params['m']}, efConstruction={tuning_params['efConstruction']}, ef={tuning_params['ef']}")
                    except Exception as e:
                        # This is expected for Community Edition
                        logger.debug(f"Could not tune HNSW for {index_name} (likely Community Edition): {e}")
        except Exception as e:
            logger.debug(f"HNSW tuning not available (Community Edition or missing index): {e}")
    
    async def verify_index_population(self) -> bool:
        """Verify that vector index is populated with chunks."""
        async with self.driver.session() as session:
            # Check if chunks have embeddings
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as indexed_count
            """, workspace=self.workspace)
            record = await result.single()
            indexed_count = record["indexed_count"] if record else 0
            
            # Check total chunks
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                RETURN count(c) as total_count
            """, workspace=self.workspace)
            record = await result.single()
            total_count = record["total_count"] if record else 0
            
            if total_count > 0:
                logger.info(f"Index verification: {indexed_count}/{total_count} chunks have embeddings")
                return indexed_count == total_count
            return True
    
    async def refresh_vector_index(self):
        """Force refresh of vector index (drop and recreate)."""
        async with self.driver.session() as session:
            try:
                # Drop existing index
                await session.run("DROP INDEX chunk_embedding_index IF EXISTS")
                logger.info("Dropped existing vector index")
                
                # Wait a moment
                import asyncio
                await asyncio.sleep(1)
                
                # Recreate index
                await session.run("""
                    CREATE VECTOR INDEX chunk_embedding_index
                    FOR (n:Chunk)
                    ON n.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Recreated vector index")
                
                # Wait for index to be ready
                await asyncio.sleep(2)
                return True
                
            except Exception as e:
                logger.error(f"Failed to refresh vector index: {e}")
                return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from Neo4j."""
        stats = {
            "workspace": self.workspace,
            "chunks": 0,
            "documents": 0,
            "entities": 0,
            "relationships": 0
        }
        
        async with self.driver.session() as session:
            # Count chunks
            result = await session.run("""
                MATCH (n:Chunk {workspace: $workspace})
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["chunks"] = record["count"] if record else 0
            
            # Count documents
            result = await session.run("""
                MATCH (n:Document {workspace: $workspace})
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["documents"] = record["count"] if record else 0
            
            # Count entities
            result = await session.run("""
                MATCH (n:Entity {workspace: $workspace})
                RETURN count(n) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["entities"] = record["count"] if record else 0
            
            # Count relationships (edges between entities)
            result = await session.run("""
                MATCH (e1:Entity {workspace: $workspace})-[r:RELATED]-(e2:Entity {workspace: $workspace})
                RETURN count(DISTINCT r) as count
            """, workspace=self.workspace)
            record = await result.single()
            stats["relationships"] = record["count"] if record else 0
        
        return stats
    
    async def drop_workspace_index(self):
        """Drop the workspace-specific vector index."""
        index_name = f"chunk_embedding_{self.workspace.replace('-', '_')}"
        async with self.driver.session() as session:
            try:
                await session.run(f"DROP INDEX {index_name} IF EXISTS")
                logger.info(f"Dropped workspace index: {index_name}")
            except Exception as e:
                logger.debug(f"Could not drop index {index_name}: {e}")
    
    async def clear_workspace(self):
        """Clear all data for this workspace and drop its index."""
        # First drop the workspace-specific index
        await self.drop_workspace_index()
        
        # Then delete all data for this workspace
        async with self.driver.session() as session:
            await session.run("""
                MATCH (n {workspace: $workspace})
                DETACH DELETE n
            """, workspace=self.workspace)
            logger.info(f"Cleared all data for workspace: {self.workspace}")
    
    # ==================== Graph Operations ====================
    
    async def store_entities(self, entities: List[Any]) -> int:
        """Store entities in Neo4j with embeddings.
        
        Args:
            entities: List of Entity objects
            
        Returns:
            Number of entities stored
        """
        # Use batch method for better performance
        if len(entities) > 10:
            return await self.batch_store_entities(entities)
        
        stored_count = 0
        
        async with self.driver.session() as session:
            for entity in entities:
                # Generate embedding for entity
                entity_content = f"{entity.name}\n{entity.description}"
                if self.embedding_func:
                    embedding = await self._generate_embedding(entity_content)
                else:
                    embedding = np.random.random(1536).tolist()
                
                # Store entity
                await session.run("""
                    MERGE (e:Entity {id: $id, workspace: $workspace})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.embedding = $embedding,
                        e.source_chunk_ids = $source_chunk_ids,
                        e.created_at = $created_at,
                        e.updated_at = $updated_at
                """,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                embedding=embedding,
                source_chunk_ids=entity.source_chunk_ids,
                created_at=int(time.time()),
                updated_at=int(time.time()),
                workspace=self.workspace)
                
                stored_count += 1
                
                # Link entity to source chunks
                for chunk_id in entity.source_chunk_ids:
                    await session.run("""
                        MATCH (e:Entity {id: $entity_id, workspace: $workspace})
                        MATCH (c:Chunk {id: $chunk_id, workspace: $workspace})
                        MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    entity_id=entity.id,
                    chunk_id=chunk_id,
                    workspace=self.workspace)
        
        logger.info(f"Stored {stored_count} entities in Neo4j")
        return stored_count
    
    async def batch_store_entities(self, entities: List[Any], batch_size: Optional[int] = None) -> int:
        """Batch store entities for improved performance.
        
        Args:
            entities: List of Entity objects
            batch_size: Optional batch size override
            
        Returns:
            Number of entities stored
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        stored_count = 0
        
        for batch_start in range(0, len(entities), batch_size):
            batch_end = min(batch_start + batch_size, len(entities))
            batch_entities = entities[batch_start:batch_end]
            
            # Prepare batch data
            batch_data = []
            chunk_links = []
            
            for entity in batch_entities:
                # Generate embedding
                entity_content = f"{entity.name}\n{entity.description}"
                if self.embedding_func:
                    embedding = await self._generate_embedding(entity_content)
                else:
                    embedding = np.random.random(1536).tolist()
                
                batch_data.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "embedding": embedding,
                    "source_chunk_ids": entity.source_chunk_ids,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                    "workspace": self.workspace
                })
                
                # Prepare chunk links
                for chunk_id in entity.source_chunk_ids:
                    chunk_links.append({
                        "entity_id": entity.id,
                        "chunk_id": chunk_id,
                        "workspace": self.workspace
                    })
            
            async with self.driver.session() as session:
                # Batch insert entities
                await session.run("""
                    UNWIND $entities as entity
                    MERGE (e:Entity {id: entity.id, workspace: entity.workspace})
                    SET e += entity
                """, entities=batch_data)
                
                # Batch create chunk relationships
                if chunk_links:
                    await session.run("""
                        UNWIND $links as link
                        MATCH (e:Entity {id: link.entity_id, workspace: link.workspace})
                        MATCH (c:Chunk {id: link.chunk_id, workspace: link.workspace})
                        MERGE (e)-[:MENTIONED_IN]->(c)
                    """, links=chunk_links)
            
            stored_count += len(batch_entities)
            logger.debug(f"Batch stored {len(batch_entities)} entities")
        
        logger.info(f"Batch stored total {stored_count} entities in Neo4j")
        return stored_count
    
    async def store_relationships(self, relationships: List[Any]) -> int:
        """Store relationships in Neo4j with embeddings.
        
        Args:
            relationships: List of Relationship objects
            
        Returns:
            Number of relationships stored
        """
        stored_count = 0
        
        async with self.driver.session() as session:
            for rel in relationships:
                # Generate embedding for relationship
                rel_content = f"{rel.source} {rel.target}\n{rel.description}\n{' '.join(rel.keywords)}"
                if self.embedding_func:
                    embedding = await self._generate_embedding(rel_content)
                else:
                    embedding = np.random.random(1536).tolist()
                
                # Store relationship as a node (for vector search)
                await session.run("""
                    MERGE (r:Relationship {id: $id, workspace: $workspace})
                    SET r.source = $source,
                        r.target = $target,
                        r.description = $description,
                        r.keywords = $keywords,
                        r.strength = $strength,
                        r.embedding = $embedding,
                        r.source_chunk_ids = $source_chunk_ids,
                        r.created_at = $created_at,
                        r.updated_at = $updated_at
                """,
                id=rel.id,
                source=rel.source,
                target=rel.target,
                description=rel.description,
                keywords=rel.keywords,
                strength=rel.strength,
                embedding=embedding,
                source_chunk_ids=rel.source_chunk_ids,
                created_at=int(time.time()),
                updated_at=int(time.time()),
                workspace=self.workspace)
                
                # Create actual Neo4j relationship between entities
                await session.run("""
                    MATCH (e1:Entity {name: $source, workspace: $workspace})
                    MATCH (e2:Entity {name: $target, workspace: $workspace})
                    MERGE (e1)-[r:RELATED_TO {
                        relationship_id: $rel_id,
                        description: $description,
                        strength: $strength,
                        keywords: $keywords
                    }]->(e2)
                """,
                source=rel.source,
                target=rel.target,
                rel_id=rel.id,
                description=rel.description,
                strength=rel.strength,
                keywords=rel.keywords,
                workspace=self.workspace)
                
                stored_count += 1
        
        logger.info(f"Stored {stored_count} relationships in Neo4j")
        return stored_count
    
    async def extract_and_store_graph(
        self,
        chunk_ids: List[str],
        extractor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract entities and relationships from chunks and store them.
        
        Args:
            chunk_ids: List of chunk IDs to process
            extractor: EntityExtractor instance
            
        Returns:
            Extraction statistics
        """
        if not extractor:
            logger.warning("No extractor provided, skipping graph extraction")
            return {"entities": 0, "relationships": 0}
        
        stats = {
            "chunks_processed": 0,
            "entities": 0,
            "relationships": 0
        }
        
        # Get chunks from Neo4j
        async with self.driver.session() as session:
            for chunk_id in chunk_ids:
                result = await session.run("""
                    MATCH (c:Chunk {id: $chunk_id, workspace: $workspace})
                    RETURN c.content as content
                """,
                chunk_id=chunk_id,
                workspace=self.workspace)
                
                record = await result.single()
                if not record:
                    continue
                
                content = record["content"]
                
                # Extract entities and relationships
                entities, relationships = await extractor.extract_from_text(
                    content, 
                    chunk_id=chunk_id
                )
                
                # Store extracted items
                if entities:
                    stored = await self.store_entities(entities)
                    stats["entities"] += stored
                
                if relationships:
                    stored = await self.store_relationships(relationships)
                    stats["relationships"] += stored
                
                stats["chunks_processed"] += 1
        
        logger.info(f"Graph extraction complete: {stats}")
        return stats
    
    async def entity_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for entities using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of matching entities
        """
        # Generate query embedding
        if self.embedding_func:
            query_vector = await self._generate_embedding(query)
        else:
            query_vector = np.random.random(1536).tolist()
        
        async with self.driver.session() as session:
            try:
                # Try using vector index
                result = await session.run("""
                    CALL db.index.vector.queryNodes('entity_embedding_index', $top_k, $query_vector)
                    YIELD node, score
                    WHERE node.workspace = $workspace
                    RETURN node.id as id,
                           node.name as name,
                           node.type as type,
                           node.description as description,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """,
                query_vector=query_vector,
                top_k=top_k * 5,  # Get more to ensure workspace matches
                workspace=self.workspace)
                
            except:
                # Fallback to cosine similarity
                result = await session.run("""
                    MATCH (e:Entity {workspace: $workspace})
                    WITH e, vector.similarity.cosine(e.embedding, $query_vector) as score
                    WHERE score > 0.3
                    RETURN e.id as id,
                           e.name as name,
                           e.type as type,
                           e.description as description,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """,
                query_vector=query_vector,
                top_k=top_k,
                workspace=self.workspace)
            
            results = []
            async for record in result:
                results.append(dict(record))
            
            return results
    
    async def graph_search(
        self,
        entity_names: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Search graph starting from specific entities.
        
        Args:
            entity_names: Starting entity names
            max_depth: Maximum traversal depth
            
        Returns:
            Graph subgraph with entities and relationships
        """
        async with self.driver.session() as session:
            # Get entities and their relationships (standard Cypher, no APOC)
            # Use pattern matching up to max_depth
            if max_depth == 1:
                pattern = "(e:Entity)-[r:RELATED_TO]-(neighbor:Entity)"
            elif max_depth == 2:
                pattern = "(e:Entity)-[r:RELATED_TO*1..2]-(neighbor:Entity)"
            else:
                pattern = f"(e:Entity)-[r:RELATED_TO*1..{max_depth}]-(neighbor:Entity)"
            
            result = await session.run(f"""
                MATCH {pattern}
                WHERE e.workspace = $workspace
                AND e.name IN $entity_names
                AND neighbor.workspace = $workspace
                RETURN DISTINCT e, neighbor, r
            """,
            entity_names=entity_names,
            workspace=self.workspace)
            
            entities = {}
            relationships = []
            
            async for record in result:
                # Add source entity
                e = record["e"]
                if e["id"] not in entities:
                    entities[e["id"]] = {
                        "id": e.get("id"),
                        "name": e.get("name"),
                        "type": e.get("type"),
                        "description": e.get("description")
                    }
                
                # Add neighbor entity
                n = record["neighbor"]
                if n["id"] not in entities:
                    entities[n["id"]] = {
                        "id": n.get("id"),
                        "name": n.get("name"),
                        "type": n.get("type"),
                        "description": n.get("description")
                    }
                
                # Add relationships (r might be a list if depth > 1)
                rel_list = record["r"] if isinstance(record["r"], list) else [record["r"]]
                for rel in rel_list:
                    if rel:  # Check if relationship exists
                        relationships.append({
                            "source": e.get("name"),
                            "target": n.get("name"),
                            "description": rel.get("description", ""),
                            "strength": rel.get("strength", 0.5)
                        })
            
            return {
                "entities": list(entities.values()),
                "relationships": relationships
            }
    
    async def local_search(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Local search mode - entity-focused search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Search results with entities and related chunks
        """
        # First find relevant entities
        entities = await self.entity_search(query, top_k=5)
        
        if not entities:
            # Fallback to vector search
            chunks = await self.vector_search(query, top_k=top_k)
            return {"entities": [], "chunks": chunks}
        
        # Get entity names
        entity_names = [e["name"] for e in entities]
        
        # Get graph context
        graph_context = await self.graph_search(entity_names, max_depth=1)
        
        # Get chunks mentioning these entities
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})-[:MENTIONED_IN]->(c:Chunk)
                WHERE e.name IN $entity_names
                RETURN DISTINCT c.id as id,
                       c.content as content,
                       c.file_path as file_path
                LIMIT $limit
            """,
            entity_names=entity_names,
            workspace=self.workspace,
            limit=top_k)
            
            chunks = []
            async for record in result:
                chunks.append(dict(record))
        
        return {
            "entities": entities,
            "relationships": graph_context["relationships"],
            "chunks": chunks
        }
    
    async def global_search(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Global search mode - high-level thematic search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Search results with high-level entities
        """
        # Get high-degree entities (most connected)
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})-[r:RELATED_TO]-()
                WITH e, count(r) as degree
                ORDER BY degree DESC
                LIMIT 20
                RETURN e.id as id,
                       e.name as name,
                       e.type as type,
                       e.description as description,
                       degree
            """,
            workspace=self.workspace)
            
            high_degree_entities = []
            async for record in result:
                high_degree_entities.append(dict(record))
        
        # Score entities by relevance to query
        if self.embedding_func:
            query_vector = await self._generate_embedding(query)
            
            # Re-rank by similarity
            for entity in high_degree_entities:
                entity_content = f"{entity['name']}\n{entity['description']}"
                entity_vector = await self._generate_embedding(entity_content)
                
                # Compute cosine similarity
                similarity = np.dot(query_vector, entity_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(entity_vector)
                )
                entity["relevance_score"] = similarity * entity["degree"]
            
            # Sort by combined score
            high_degree_entities.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Get top entities
        top_entities = high_degree_entities[:top_k]
        
        return {
            "entities": top_entities,
            "theme": "High-level overview based on most connected entities"
        }
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Hybrid search - combines local and global approaches.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Combined search results
        """
        # Get both local and global results
        local_results = await self.local_search(query, top_k=top_k//2)
        global_results = await self.global_search(query, top_k=top_k//2)
        
        # Combine results
        return {
            "local_entities": local_results.get("entities", []),
            "global_entities": global_results.get("entities", []),
            "relationships": local_results.get("relationships", []),
            "chunks": local_results.get("chunks", []),
            "theme": global_results.get("theme", "")
        }
    
    async def mix_search(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Mix mode - best performance combining graph and vector.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Mixed search results
        """
        # Get entity search results
        entities = await self.entity_search(query, top_k=5)
        
        # Get vector search results
        chunks = await self.vector_search(query, top_k=top_k)
        
        # Get relationship context if entities found
        relationships = []
        if entities:
            entity_names = [e["name"] for e in entities]
            graph_context = await self.graph_search(entity_names, max_depth=2)
            relationships = graph_context["relationships"]
        
        # Combine and score
        combined_results = {
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks,
            "mode": "mix"
        }
        
        return combined_results
    
    async def batch_vector_search(
        self,
        queries: List[str],
        top_k: int = 5,
        return_metadata: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch vector similarity search for multiple queries.
        
        Args:
            queries: List of query texts to search for
            top_k: Number of results per query
            return_metadata: If True, return all metadata fields
            
        Returns:
            List of result lists, one per query
        """
        # Generate embeddings for all queries
        query_embeddings = []
        for query in queries:
            if self.embedding_func:
                embedding = await self._generate_embedding(query)
            else:
                embedding = np.random.random(1536).tolist()
            query_embeddings.append(embedding)
        
        # Execute searches in parallel
        tasks = []
        for query, embedding in zip(queries, query_embeddings):
            # Use cached result if available
            if self.cache:
                cache_key_params = {"query": query, "top_k": top_k, "workspace": self.workspace}
                cached = self.cache.get("vector_search", cache_key_params)
                if cached:
                    tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task for cached result
                    continue
            
            # Create search task
            task = self._vector_search_with_embedding(embedding, top_k, return_metadata)
            tasks.append(asyncio.create_task(task))
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks)
        
        # Cache results
        if self.cache:
            for query, result in zip(queries, results):
                if result:  # Only cache successful results
                    cache_key_params = {"query": query, "top_k": top_k, "workspace": self.workspace}
                    self.cache.set("vector_search", cache_key_params, result)
        
        logger.info(f"Batch vector search complete: {len(queries)} queries, {top_k} results each")
        return results
    
    async def _vector_search_with_embedding(
        self,
        query_vector: List[float],
        top_k: int = 5,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Internal vector search using pre-computed embedding."""
        # Build return clause based on metadata flag
        if return_metadata:
            return_clause = """
                       node.id as id,
                       node.content as content,
                       node.tokens as tokens,
                       node.chunk_order_index as chunk_order_index,
                       node.full_doc_id as full_doc_id,
                       node.file_path as file_path,
                       node.llm_cache_list as llm_cache_list,
                       node.create_time as create_time,
                       node.update_time as update_time,
            """
        else:
            return_clause = """
                       node.id as id,
                       node.content as content,
                       node.file_path as file_path,
            """
        
        async with self.driver.session() as session:
            try:
                # Optimized query: Get more results and filter by workspace
                result = await session.run(f"""
                    CALL db.index.vector.queryNodes('chunk_embedding_index_v2', $query_limit, $query_vector)
                    YIELD node, score
                    WHERE node.workspace = $workspace
                    RETURN {return_clause}
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """,
                query_vector=query_vector,
                query_limit=min(top_k * 10, 100),
                top_k=top_k,
                workspace=self.workspace)
                
                results = []
                async for record in result:
                    results.append(dict(record))
                
                if results:
                    return results
                    
            except Exception as e:
                logger.debug(f"Vector index not available, using fallback: {e}")
            
            # Fallback for Community Edition
            return await self._fallback_vector_search(query_vector, top_k, return_metadata)
    
    async def _fallback_vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Fallback vector search using manual cosine similarity."""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (c:Chunk)
                WHERE c.workspace = $workspace
                WITH c, c.embedding as embedding
                LIMIT 100
                RETURN c.id as id,
                       c.content as content,
                       c.file_path as file_path,
                       {("c.tokens as tokens, c.chunk_order_index as chunk_order_index, c.full_doc_id as full_doc_id, c.llm_cache_list as llm_cache_list, c.create_time as create_time, c.update_time as update_time," if return_metadata else "")}
                       embedding
            """,
            workspace=self.workspace)
            
            all_chunks = []
            async for record in result:
                chunk_dict = dict(record)
                if chunk_dict.get('embedding'):
                    # Calculate cosine similarity
                    embedding = np.array(chunk_dict['embedding'])
                    query_vec = np.array(query_vector)
                    score = np.dot(embedding, query_vec) / (np.linalg.norm(embedding) * np.linalg.norm(query_vec))
                    chunk_dict['score'] = float(score)
                    del chunk_dict['embedding']
                    all_chunks.append(chunk_dict)
            
            # Sort by score and return top k
            all_chunks.sort(key=lambda x: x['score'], reverse=True)
            return all_chunks[:top_k]
    
    async def create_performance_indexes(self):
        """Create composite indexes for improved query performance."""
        async with self.driver.session() as session:
            # Composite indexes for common query patterns
            indexes = [
                # Workspace + type indexes for filtered searches
                "CREATE INDEX chunk_workspace_order IF NOT EXISTS FOR (c:Chunk) ON (c.workspace, c.chunk_order_index)",
                "CREATE INDEX entity_workspace_type IF NOT EXISTS FOR (e:Entity) ON (e.workspace, e.type)",
                "CREATE INDEX entity_workspace_name IF NOT EXISTS FOR (e:Entity) ON (e.workspace, e.name)",
                
                # Temporal indexes for time-based queries
                "CREATE INDEX chunk_workspace_time IF NOT EXISTS FOR (c:Chunk) ON (c.workspace, c.create_time)",
                "CREATE INDEX entity_workspace_time IF NOT EXISTS FOR (e:Entity) ON (e.workspace, e.created_at)",
                
                # Document indexes
                "CREATE INDEX doc_workspace IF NOT EXISTS FOR (d:Document) ON (d.workspace)",
                "CREATE INDEX doc_workspace_index IF NOT EXISTS FOR (d:Document) ON (d.workspace, d.doc_index)",
                
                # Relationship indexes for faster traversal
                "CREATE INDEX rel_workspace IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.workspace)",
                "CREATE INDEX rel_strength IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.strength)"
            ]
            
            for index_query in indexes:
                try:
                    await session.run(index_query)
                    index_name = index_query.split("INDEX ")[1].split(" ")[0]
                    logger.debug(f"Created/verified composite index: {index_name}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Could not create index: {e}")
            
            logger.info("Performance indexes created/verified")
    
    async def create_entity_vector_index(self):
        """Create vector index for entities."""
        async with self.driver.session() as session:
            try:
                await session.run("""
                    CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                    FOR (n:Entity)
                    ON n.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Created entity vector index")
            except Exception as e:
                logger.debug(f"Entity index may already exist: {e}")
        
        # Also create performance indexes
        await self.create_performance_indexes()
    
    async def create_relationship_vector_index(self):
        """Create vector index for relationships."""
        async with self.driver.session() as session:
            try:
                await session.run("""
                    CREATE VECTOR INDEX relationship_embedding_index IF NOT EXISTS
                    FOR (n:Relationship)
                    ON n.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Created relationship vector index")
            except Exception as e:
                logger.debug(f"Relationship index may already exist: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        stats = {
            "storage": {},
            "indexes": {},
            "cache": {},
            "performance": {}
        }
        
        async with self.driver.session() as session:
            # Storage statistics
            storage_queries = {
                "total_chunks": "MATCH (c:Chunk {workspace: $workspace}) RETURN count(c) as count",
                "total_entities": "MATCH (e:Entity {workspace: $workspace}) RETURN count(e) as count",
                "total_relationships": "MATCH (e1:Entity {workspace: $workspace})-[r]-(e2:Entity {workspace: $workspace}) RETURN count(r) as count",
                "total_documents": "MATCH (d:Document {workspace: $workspace}) RETURN count(d) as count"
            }
            
            for key, query in storage_queries.items():
                result = await session.run(query, workspace=self.workspace)
                record = await result.single()
                stats["storage"][key] = record["count"] if record else 0
            
            # Index statistics
            try:
                # Check which indexes exist
                index_result = await session.run("""
                    SHOW INDEXES
                    YIELD name, type, state, populationPercent
                    RETURN name, type, state, populationPercent
                """)
                
                indexes = []
                async for record in index_result:
                    indexes.append({
                        "name": record["name"],
                        "type": record["type"],
                        "state": record["state"],
                        "population": record.get("populationPercent", 100)
                    })
                stats["indexes"]["list"] = indexes
                stats["indexes"]["total"] = len(indexes)
                stats["indexes"]["vector_indexes"] = sum(1 for i in indexes if i["type"] == "VECTOR")
            except Exception as e:
                logger.debug(f"Could not get index stats: {e}")
                stats["indexes"]["error"] = str(e)
            
            # Memory and performance metrics
            try:
                # Get database size estimate
                size_result = await session.run("""
                    MATCH (n {workspace: $workspace})
                    WITH count(n) as node_count,
                         sum(size(keys(n))) as total_properties
                    RETURN node_count, total_properties
                """, workspace=self.workspace)
                size_record = await size_result.single()
                
                if size_record:
                    stats["performance"]["node_count"] = size_record["node_count"]
                    stats["performance"]["avg_properties"] = (
                        size_record["total_properties"] / size_record["node_count"] 
                        if size_record["node_count"] > 0 else 0
                    )
                    # Rough estimate: assume 1KB per node average
                    stats["performance"]["estimated_size_mb"] = size_record["node_count"] * 0.001
            except Exception as e:
                logger.debug(f"Could not get performance stats: {e}")
        
        # Cache statistics
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        else:
            stats["cache"]["enabled"] = False
        
        # Connection pool stats
        stats["performance"]["connection_pool"] = {
            "max_size": self.driver._max_connection_pool_size if hasattr(self.driver, '_max_connection_pool_size') else "unknown",
            "in_use": "unknown"  # Would need driver internals
        }
        
        # Batch processing capabilities
        stats["performance"]["batch_size"] = self.batch_size
        stats["performance"]["supports_hnsw"] = "unknown"  # Detected at runtime
        
        return stats
    
    async def optimize_database(self):
        """Run optimization procedures on the database.
        
        This includes:
        - Refreshing statistics
        - Optimizing indexes
        - Cleaning up orphaned nodes
        """
        optimization_results = {
            "indexes_optimized": 0,
            "orphans_removed": 0,
            "statistics_refreshed": False
        }
        
        async with self.driver.session() as session:
            try:
                # Refresh index statistics (Enterprise Edition)
                await session.run("CALL db.stats.clear()")
                await session.run("CALL db.stats.collect()")
                optimization_results["statistics_refreshed"] = True
            except Exception as e:
                logger.debug(f"Statistics refresh not available: {e}")
            
            try:
                # Clean up orphaned chunks (chunks without documents)
                orphan_result = await session.run("""
                    MATCH (c:Chunk {workspace: $workspace})
                    WHERE NOT EXISTS ((c)-[:BELONGS_TO]->(:Document))
                    DELETE c
                    RETURN count(c) as deleted
                """, workspace=self.workspace)
                orphan_record = await orphan_result.single()
                optimization_results["orphans_removed"] = orphan_record["deleted"] if orphan_record else 0
            except Exception as e:
                logger.error(f"Failed to clean orphans: {e}")
            
            # Force index population if needed
            try:
                await session.run("CALL db.awaitIndexes()")
                optimization_results["indexes_optimized"] = True
            except Exception as e:
                logger.debug(f"Index await not available: {e}")
        
        logger.info(f"Database optimization complete: {optimization_results}")
        return optimization_results
    
    async def close(self):
        """Close Neo4j connection."""
        # Log final statistics if cache is enabled
        if self.cache:
            cache_stats = self.cache.get_stats()
            logger.info(f"Cache statistics at close: {cache_stats}")
        
        await self.driver.close()
        logger.info("Neo4jOnlyStorage closed")