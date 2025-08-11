"""API routes for LightRAG GraphRAG Testing Framework."""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional

from loguru import logger

from .schemas import (
    DocumentInsertRequest, DocumentInsertResponse,
    QueryRequest, QueryResponse,
    HealthResponse, GraphStatsResponse,
    BatchInsertRequest, BatchInsertResponse,
    CompareModeRequest, CompareModeResponse,
    ErrorResponse,
    # Vector Operations Schemas
    VectorSearchRequest, VectorSearchResponse,
    HybridSearchRequest, HybridSearchResponse,
    VectorIndexStatusResponse,
    VectorIndexCreateRequest, VectorIndexCreateResponse,
    VectorPerformanceMetricsResponse,
    VectorQuantizationResponse,
    UnifiedStorageStatsResponse
)
from ..queries.query_executor import QueryExecutor
from ..graph.neo4j_client import Neo4jClient


router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check system health status."""
    try:
        # Get RAG manager from app state
        rag_manager = request.app.state.rag_manager
        health = await rag_manager.health_check()
        
        return HealthResponse(
            status=health.get("status", "unknown"),
            lightrag=health.get("lightrag", False),
            neo4j=health.get("neo4j", False),
            initialized=health.get("initialized", False),
            errors=health.get("errors", [])
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            lightrag=False,
            neo4j=False,
            initialized=False,
            errors=[str(e)]
        )


@router.post("/insert", response_model=DocumentInsertResponse)
async def insert_document(
    request: Request,
    doc_request: DocumentInsertRequest
):
    """Insert a document into the knowledge graph."""
    try:
        rag_manager = request.app.state.rag_manager
        
        success = await rag_manager.insert_document(
            content=doc_request.content,
            metadata=doc_request.metadata
        )
        
        if success:
            return DocumentInsertResponse(
                status="success",
                message="Document inserted successfully"
            )
        else:
            return DocumentInsertResponse(
                status="failed",
                error="Failed to insert document"
            )
            
    except Exception as e:
        logger.error(f"Document insertion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insert_batch", response_model=BatchInsertResponse)
async def insert_batch(
    request: Request,
    batch_request: BatchInsertRequest
):
    """Insert multiple documents in batch."""
    try:
        rag_manager = request.app.state.rag_manager
        
        results = await rag_manager.insert_batch(batch_request.documents)
        
        return BatchInsertResponse(
            total=results["total"],
            success=results["success"],
            failed=results["failed"],
            errors=results["errors"]
        )
        
    except Exception as e:
        logger.error(f"Batch insertion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    query_request: QueryRequest
):
    """Query the knowledge graph."""
    try:
        query_executor = request.app.state.query_executor
        
        response = await query_executor.execute(
            query=query_request.query,
            mode=query_request.mode,
            top_k=query_request.top_k,
            enable_rerank=query_request.enable_rerank
        )
        
        return QueryResponse(
            query=response.query,
            mode=response.mode,
            response=response.result,
            latency_ms=response.metadata.get("latency_ms", 0),
            metadata=response.metadata
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare_modes", response_model=CompareModeResponse)
async def compare_modes(
    request: Request,
    compare_request: CompareModeRequest
):
    """Compare query results across different modes."""
    try:
        query_executor = request.app.state.query_executor
        
        results = await query_executor.compare_modes(
            query=compare_request.query,
            modes=compare_request.modes,
            top_k=compare_request.top_k
        )
        
        # Convert results to serializable format
        response_results = {}
        for mode, response in results.items():
            response_results[mode] = {
                "response": response.result,
                "latency_ms": response.metadata.get("latency_ms", 0),
                "metadata": response.metadata
            }
        
        return CompareModeResponse(
            query=compare_request.query,
            results=response_results
        )
        
    except Exception as e:
        logger.error(f"Mode comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(request: Request):
    """Get statistics about the knowledge graph."""
    try:
        neo4j_client = request.app.state.neo4j_client
        
        entity_count = await neo4j_client.get_entity_count()
        relationship_count = await neo4j_client.get_relationship_count()
        entity_types = await neo4j_client.get_entity_types()
        top_entities = await neo4j_client.get_top_entities(limit=10)
        
        return GraphStatsResponse(
            entity_count=entity_count,
            relationship_count=relationship_count,
            entity_types=entity_types,
            top_entities=top_entities
        )
        
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/graph/clear")
async def clear_graph(request: Request):
    """Clear all data from the knowledge graph."""
    try:
        rag_manager = request.app.state.rag_manager
        
        success = await rag_manager.clear_graph()
        
        if success:
            return {"status": "success", "message": "Graph cleared successfully"}
        else:
            return {"status": "failed", "error": "Failed to clear graph"}
            
    except Exception as e:
        logger.error(f"Failed to clear graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector Operations Endpoints

@router.post("/vector/search", response_model=VectorSearchResponse)
async def vector_search(
    request: Request,
    search_request: VectorSearchRequest
):
    """Perform vector similarity search using embeddings."""
    try:
        from ..core.embeddings import create_embedding_func
        from ..config import settings
        
        # Check if unified storage is available
        rag_manager = request.app.state.rag_manager
        if not settings.unified_neo4j_storage:
            raise HTTPException(
                status_code=400, 
                detail="Vector search requires unified Neo4j storage to be enabled"
            )
        
        if not hasattr(rag_manager, 'storage_backend'):
            raise HTTPException(
                status_code=500,
                detail="Unified storage backend not available"
            )
        
        # Create embedding for the query text
        embedding_func = create_embedding_func(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            max_token_size=8192
        )
        
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await embedding_func(search_request.query_text)
        
        # Perform vector search
        results = await rag_manager.storage_backend.search_similar(
            collection=search_request.collection,
            query_embedding=query_embedding,
            top_k=search_request.top_k,
            filter_dict=search_request.filters
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return VectorSearchResponse(
            query_text=search_request.query_text,
            collection=search_request.collection,
            results=results,
            latency_ms=latency_ms,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: Request,
    search_request: HybridSearchRequest
):
    """Perform hybrid graph + vector search."""
    try:
        from ..core.embeddings import create_embedding_func
        from ..config import settings
        
        # Check if unified storage is available
        rag_manager = request.app.state.rag_manager
        if not settings.unified_neo4j_storage:
            raise HTTPException(
                status_code=400,
                detail="Hybrid search requires unified Neo4j storage to be enabled"
            )
        
        if not hasattr(rag_manager, 'storage_backend'):
            raise HTTPException(
                status_code=500,
                detail="Unified storage backend not available"
            )
        
        # Create embedding for the query text
        embedding_func = create_embedding_func(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            max_token_size=8192
        )
        
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await embedding_func(search_request.query_text)
        
        # Perform hybrid search
        results = await rag_manager.storage_backend.hybrid_search(
            query_embedding=query_embedding,
            graph_depth=search_request.graph_depth,
            top_k=search_request.top_k,
            collection=search_request.collection
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return HybridSearchResponse(
            query_text=search_request.query_text,
            results=results,
            graph_depth=search_request.graph_depth,
            latency_ms=latency_ms,
            vector_matches=len([r for r in results if r.get("score", 0) > 0])
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/indexes/status", response_model=VectorIndexStatusResponse)
async def get_vector_index_status(request: Request):
    """Get status of all vector indexes."""
    try:
        neo4j_client = request.app.state.neo4j_client
        
        # Get index status from Neo4j client
        status = await neo4j_client.check_vector_index_status()
        
        # Count different types of indexes
        total_indexes = len(status)
        online_indexes = len([idx for idx in status.values() if idx.get("state", "").lower() == "online"])
        optimization_enabled = len([
            idx for idx in status.values() 
            if idx.get("options", {}).get("vector.quantization.enabled", False)
        ])
        
        return VectorIndexStatusResponse(
            indexes=status,
            total_indexes=total_indexes,
            online_indexes=online_indexes,
            optimization_enabled=optimization_enabled
        )
        
    except Exception as e:
        logger.error(f"Failed to get vector index status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector/indexes/create", response_model=VectorIndexCreateResponse)
async def create_vector_indexes(
    request: Request,
    create_request: VectorIndexCreateRequest
):
    """Create or verify vector indexes with optimal HNSW configuration."""
    try:
        neo4j_client = request.app.state.neo4j_client
        
        # Create vector indexes
        created_indexes = await neo4j_client.create_vector_indexes()
        
        return VectorIndexCreateResponse(
            created_indexes=created_indexes,
            status="success",
            message=f"Successfully created/verified {len(created_indexes)} vector indexes"
        )
        
    except Exception as e:
        logger.error(f"Failed to create vector indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/performance/metrics", response_model=VectorPerformanceMetricsResponse)
async def get_vector_performance_metrics(request: Request):
    """Get performance metrics for vector indexes."""
    try:
        neo4j_client = request.app.state.neo4j_client
        
        # Get performance metrics
        metrics = await neo4j_client.get_vector_index_performance_metrics()
        
        # Calculate overall performance summary
        all_latencies = [m.get("latency_ms", 0) for m in metrics.values() if "latency_ms" in m]
        
        overall_performance = {}
        if all_latencies:
            overall_performance = {
                "average_latency_ms": sum(all_latencies) / len(all_latencies),
                "max_latency_ms": max(all_latencies),
                "min_latency_ms": min(all_latencies),
                "total_indexes_tested": len(all_latencies)
            }
        
        # Generate recommendations
        recommendations = []
        for index_name, metric in metrics.items():
            if metric.get("latency_ms", 0) > 500:
                recommendations.append(f"Index {index_name} has high latency ({metric['latency_ms']:.1f}ms) - consider optimization")
            if metric.get("performance_status") == "needs_optimization":
                recommendations.append(f"Index {index_name} performance needs optimization")
        
        return VectorPerformanceMetricsResponse(
            metrics=metrics,
            overall_performance=overall_performance,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get vector performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/optimization/quantization", response_model=VectorQuantizationResponse)
async def get_vector_quantization_optimization(request: Request):
    """Get vector quantization optimization analysis."""
    try:
        neo4j_client = request.app.state.neo4j_client
        
        # Get quantization optimization results
        optimization_results = await neo4j_client.optimize_vector_quantization()
        
        # Extract memory savings information
        memory_savings = {}
        recommendations = []
        
        for index_name, result in optimization_results.items():
            if "error" not in result:
                memory_opt = result.get("memory_optimization", "not optimized")
                memory_savings[index_name] = memory_opt
                
                if memory_opt == "not optimized":
                    recommendations.append(f"Enable quantization for {index_name} to reduce memory usage by 50-75%")
                elif memory_opt == "75% reduction":
                    recommendations.append(f"Index {index_name} is optimally configured with quantization")
        
        return VectorQuantizationResponse(
            optimization_results=optimization_results,
            memory_savings=memory_savings,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get vector quantization optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/unified/stats", response_model=UnifiedStorageStatsResponse)
async def get_unified_storage_stats(request: Request):
    """Get statistics for unified Neo4j storage."""
    try:
        from ..config import settings
        
        # Check if unified storage is enabled
        if not settings.unified_neo4j_storage:
            raise HTTPException(
                status_code=400,
                detail="Unified storage is not enabled"
            )
        
        rag_manager = request.app.state.rag_manager
        if not hasattr(rag_manager, 'storage_backend'):
            raise HTTPException(
                status_code=500,
                detail="Unified storage backend not available"
            )
        
        # Get storage statistics
        stats = await rag_manager.storage_backend.get_statistics()
        
        return UnifiedStorageStatsResponse(
            total_nodes=stats.get("total_nodes", 0),
            entities_with_vectors=stats.get("entities_with_vectors", 0),
            chunks_with_vectors=stats.get("chunks_with_vectors", 0),
            total_relationships=stats.get("total_relationships", 0),
            workspace=settings.neo4j_workspace,
            vector_dimensions=settings.neo4j_vector_dimensions,
            quantization_enabled=settings.neo4j_vector_quantization
        )
        
    except Exception as e:
        logger.error(f"Failed to get unified storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_info(request: Request):
    """Get system information."""
    try:
        from ..config import settings
        
        return {
            "name": "LightRAG GraphRAG Testing Framework",
            "version": "1.0.0",
            "configuration": {
                "openai_model": settings.openai_model,
                "embedding_model": settings.embedding_model,
                "neo4j_workspace": settings.neo4j_workspace,
                "default_query_mode": settings.default_query_mode,
                "entity_types": settings.entity_types_list,
                "unified_neo4j_storage": settings.unified_neo4j_storage,
                "vector_dimensions": settings.neo4j_vector_dimensions,
                "vector_quantization": settings.neo4j_vector_quantization,
                "hnsw_parameters": {
                    "m": settings.neo4j_hnsw_m,
                    "ef_construction": settings.neo4j_hnsw_ef_construction
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        raise HTTPException(status_code=500, detail=str(e))