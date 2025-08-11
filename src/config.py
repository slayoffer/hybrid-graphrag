"""Configuration management for LightRAG GraphRAG Testing Framework."""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic v2."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for LLM and embeddings"
    )
    openai_model: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model for text generation"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for embeddings"
    )
    
    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7689",
        description="Neo4j connection URI"
    )
    neo4j_username: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="lightrag123",
        description="Neo4j password"
    )
    neo4j_workspace: str = Field(
        default="testing",
        description="Neo4j workspace for data isolation"
    )
    neo4j_max_connection_pool_size: int = Field(
        default=100,
        description="Maximum Neo4j connection pool size (increased for vector operations)"
    )
    neo4j_connection_timeout: int = Field(
        default=30,
        description="Connection acquisition timeout in seconds"
    )
    
    # LightRAG Configuration
    lightrag_working_dir: str = Field(
        default="/app/rag_storage",
        description="LightRAG working directory for storage"
    )
    lightrag_graph_storage: str = Field(
        default="Neo4JStorage",
        description="Graph storage backend type"
    )
    lightrag_vector_storage: str = Field(
        default="Neo4JStorage",
        description="Vector storage backend type (unified with graph storage)"
    )
    lightrag_kv_storage: str = Field(
        default="JsonKVStorage",
        description="Key-value storage backend type"
    )
    lightrag_doc_storage: str = Field(
        default="JsonDocStatusStorage",
        description="Document storage backend type"
    )
    max_entity_tokens: int = Field(
        default=15000,
        description="Maximum tokens for entity extraction"
    )
    max_relationship_tokens: int = Field(
        default=10000,
        description="Maximum tokens for relationship extraction"
    )
    gleaning_iterations: int = Field(
        default=3,
        description="Number of gleaning iterations for extraction"
    )
    llm_context_size: int = Field(
        default=32768,
        description="LLM context window size"
    )
    
    # Query Configuration
    default_query_mode: str = Field(
        default="mix",
        description="Default query mode (local/global/hybrid/mix/naive)"
    )
    default_top_k: int = Field(
        default=100,
        description="Default number of results to return"
    )
    enable_rerank: bool = Field(
        default=True,
        description="Enable result reranking"
    )
    
    # Performance Configuration
    batch_size: int = Field(
        default=100,
        description="Batch size for bulk operations"
    )
    enable_query_cache: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Query cache TTL in seconds"
    )
    cache_max_size: int = Field(
        default=100,
        description="Maximum cache size (number of queries)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Entity Extraction Configuration
    entity_types: str = Field(
        default="organization,person,geo,event,category,technology,concept",
        description="Comma-separated list of entity types to extract"
    )
    
    # API Configuration
    api_port: int = Field(
        default=9621,
        description="API server port"
    )
    api_auth_token: Optional[str] = Field(
        default=None,
        description="Optional API authentication token"
    )
    
    # Unified Neo4j Vector Storage Configuration
    unified_neo4j_storage: bool = Field(
        default=True,
        description="Use unified Neo4j storage for graph and vectors"
    )
    neo4j_vector_dimensions: int = Field(
        default=1536,
        description="Vector embedding dimensions for OpenAI text-embedding-3-small"
    )
    neo4j_vector_similarity: str = Field(
        default="cosine",
        description="Vector similarity function (cosine, euclidean, dot)"
    )
    neo4j_vector_quantization: bool = Field(
        default=True,
        description="Enable vector quantization for 50-75% memory reduction"
    )
    neo4j_hnsw_m: int = Field(
        default=16,
        description="HNSW m parameter - optimal connections per node"
    )
    neo4j_hnsw_ef_construction: int = Field(
        default=200,
        description="HNSW efConstruction parameter - quality vs speed balance"
    )
    neo4j_hnsw_ef: int = Field(
        default=50,
        description="HNSW ef parameter - query time accuracy/speed tradeoff"
    )
    
    # Memory Optimization Settings
    neo4j_memory_heap_initial: str = Field(
        default="8G",
        description="Neo4j initial heap size"
    )
    neo4j_memory_heap_max: str = Field(
        default="8G",
        description="Neo4j maximum heap size"
    )
    neo4j_memory_pagecache: str = Field(
        default="16G",
        description="Neo4j page cache size"
    )
    
    @property
    def entity_types_list(self) -> list[str]:
        """Return entity types as a list."""
        return [t.strip() for t in self.entity_types.split(",")]


# Create global settings instance
settings = Settings()