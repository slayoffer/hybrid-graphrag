"""Pydantic schemas for API endpoints."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


class QueryMode(str, Enum):
    """Query modes for retrieval."""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"
    NAIVE = "naive"


class DocumentInsertRequest(BaseModel):
    """Request model for document insertion."""
    
    model_config = ConfigDict(extra="forbid")
    
    content: str = Field(
        ...,
        min_length=1,
        description="Document content to insert"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional document metadata"
    )


class DocumentInsertResponse(BaseModel):
    """Response model for document insertion."""
    
    status: str = Field(..., description="Status of the operation")
    message: Optional[str] = Field(default=None, description="Optional message")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QueryRequest(BaseModel):
    """Request model for querying."""
    
    model_config = ConfigDict(extra="forbid")
    
    query: str = Field(
        ...,
        min_length=1,
        description="Query text"
    )
    mode: QueryMode = Field(
        default=QueryMode.MIX,
        description="Query mode to use"
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of results to return"
    )
    enable_rerank: bool = Field(
        default=True,
        description="Whether to rerank results"
    )


class QueryResponse(BaseModel):
    """Response model for queries."""
    
    query: str = Field(..., description="Original query")
    mode: str = Field(..., description="Query mode used")
    response: str = Field(..., description="Query response")
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    lightrag: bool = Field(..., description="LightRAG availability")
    neo4j: bool = Field(..., description="Neo4j connectivity")
    initialized: bool = Field(..., description="System initialization status")
    errors: List[str] = Field(default_factory=list, description="Any errors")


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    
    entity_count: int = Field(..., description="Total number of entities")
    relationship_count: int = Field(..., description="Total number of relationships")
    entity_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of entity types"
    )
    top_entities: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top entities by degree"
    )


class BatchInsertRequest(BaseModel):
    """Request model for batch document insertion."""
    
    model_config = ConfigDict(extra="forbid")
    
    documents: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of documents to insert"
    )


class BatchInsertResponse(BaseModel):
    """Response model for batch insertion."""
    
    total: int = Field(..., description="Total documents processed")
    success: int = Field(..., description="Successfully inserted")
    failed: int = Field(..., description="Failed insertions")
    errors: List[str] = Field(default_factory=list, description="Error messages")


class EvaluationMetrics(BaseModel):
    """Model for evaluation metrics."""
    
    accuracy: float = Field(
        ...,
        ge=0,
        le=1,
        description="Accuracy score"
    )
    completeness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Completeness score"
    )
    semantic_fidelity: float = Field(
        ...,
        ge=0,
        le=1,
        description="Semantic fidelity score"
    )
    query_latency_ms: float = Field(
        ...,
        ge=0,
        description="Average query latency"
    )
    extraction_time_s: float = Field(
        ...,
        ge=0,
        description="Entity extraction time"
    )


class CompareModeRequest(BaseModel):
    """Request model for comparing query modes."""
    
    model_config = ConfigDict(extra="forbid")
    
    query: str = Field(
        ...,
        min_length=1,
        description="Query to test"
    )
    modes: Optional[List[QueryMode]] = Field(
        default=None,
        description="Modes to compare (default: all)"
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of results"
    )


class CompareModeResponse(BaseModel):
    """Response model for mode comparison."""
    
    query: str = Field(..., description="Original query")
    results: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Results for each mode"
    )
    
    
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


# Vector Operations Schemas

class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""
    
    model_config = ConfigDict(extra="forbid")
    
    query_text: str = Field(
        ...,
        min_length=1,
        description="Text to search for using embeddings"
    )
    collection: str = Field(
        default="Entity",
        description="Collection to search (Entity, Chunk, etc.)"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply"
    )


class VectorSearchResponse(BaseModel):
    """Response model for vector similarity search."""
    
    query_text: str = Field(..., description="Original query text")
    collection: str = Field(..., description="Collection searched")
    results: List[Dict[str, Any]] = Field(..., description="Search results with scores")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    total_found: int = Field(..., description="Total number of results found")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid graph + vector search."""
    
    model_config = ConfigDict(extra="forbid")
    
    query_text: str = Field(
        ...,
        min_length=1,
        description="Text query for hybrid search"
    )
    graph_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum graph traversal depth"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    collection: str = Field(
        default="Entity",
        description="Vector collection to search"
    )


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    
    query_text: str = Field(..., description="Original query text")
    results: List[Dict[str, Any]] = Field(..., description="Hybrid search results")
    graph_depth: int = Field(..., description="Graph traversal depth used")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    vector_matches: int = Field(..., description="Number of vector matches found")


class VectorIndexStatusResponse(BaseModel):
    """Response model for vector index status."""
    
    indexes: Dict[str, Dict[str, Any]] = Field(..., description="Vector index status information")
    total_indexes: int = Field(..., description="Total number of vector indexes")
    online_indexes: int = Field(..., description="Number of online indexes")
    optimization_enabled: int = Field(..., description="Number of indexes with quantization")


class VectorIndexCreateRequest(BaseModel):
    """Request model for creating vector indexes."""
    
    model_config = ConfigDict(extra="forbid")
    
    force_recreate: bool = Field(
        default=False,
        description="Force recreation of existing indexes"
    )


class VectorIndexCreateResponse(BaseModel):
    """Response model for vector index creation."""
    
    created_indexes: List[str] = Field(..., description="List of created/verified indexes")
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Result message")


class VectorPerformanceMetricsResponse(BaseModel):
    """Response model for vector performance metrics."""
    
    metrics: Dict[str, Dict[str, Any]] = Field(..., description="Performance metrics by index")
    overall_performance: Dict[str, Any] = Field(..., description="Overall performance summary")
    recommendations: List[str] = Field(default_factory=list, description="Performance recommendations")


class VectorQuantizationResponse(BaseModel):
    """Response model for vector quantization optimization."""
    
    optimization_results: Dict[str, Dict[str, Any]] = Field(..., description="Quantization analysis results")
    memory_savings: Dict[str, str] = Field(..., description="Estimated memory savings by index")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")


class UnifiedStorageStatsResponse(BaseModel):
    """Response model for unified storage statistics."""
    
    total_nodes: int = Field(..., description="Total nodes in storage")
    entities_with_vectors: int = Field(..., description="Entities with vector embeddings")
    chunks_with_vectors: int = Field(..., description="Chunks with vector embeddings")
    total_relationships: int = Field(..., description="Total relationships")
    workspace: str = Field(..., description="Current workspace")
    vector_dimensions: int = Field(..., description="Vector embedding dimensions")
    quantization_enabled: bool = Field(..., description="Whether vector quantization is enabled")