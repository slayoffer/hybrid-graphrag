"""Query executor for multi-mode retrieval."""

import time
from typing import Dict, Any, Optional
from enum import Enum

from loguru import logger

from ..core.rag_manager import RAGManager
from ..config import Settings


class QueryMode(str, Enum):
    """Query modes supported by LightRAG."""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"
    NAIVE = "naive"


class QueryResponse:
    """Response from query execution."""
    
    def __init__(
        self,
        query: str,
        mode: QueryMode,
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize query response.
        
        Args:
            query: Original query text
            mode: Query mode used
            result: Query result
            metadata: Additional metadata
        """
        self.query = query
        self.mode = mode
        self.result = result
        self.metadata = metadata or {}
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "query": self.query,
            "mode": self.mode,
            "result": self.result,
            "metadata": self.metadata
        }


class QueryExecutor:
    """Execute queries across different retrieval modes."""
    
    def __init__(self, rag_manager: RAGManager):
        """Initialize query executor.
        
        Args:
            rag_manager: RAG manager instance
        """
        self.rag_manager = rag_manager
        
        # Mode handlers - all use the same underlying query method
        # but with different mode parameters
        self.mode_handlers = {
            QueryMode.LOCAL: self._query_local,
            QueryMode.GLOBAL: self._query_global,
            QueryMode.HYBRID: self._query_hybrid,
            QueryMode.MIX: self._query_mix,
            QueryMode.NAIVE: self._query_naive
        }
        
        logger.info("QueryExecutor initialized with all 5 modes")
    
    async def execute(
        self,
        query: str,
        mode: QueryMode = QueryMode.MIX,
        top_k: int = 100,
        enable_rerank: bool = True,
        **kwargs
    ) -> QueryResponse:
        """Execute query with specified mode.
        
        Args:
            query: Query text
            mode: Query mode to use
            top_k: Number of results
            enable_rerank: Whether to rerank results
            **kwargs: Additional parameters
            
        Returns:
            Query response
        """
        # Validate mode
        if mode not in self.mode_handlers:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(self.mode_handlers.keys())}")
        
        logger.info(f"Executing query with mode '{mode}': {query[:100]}...")
        
        # Get handler for mode
        handler = self.mode_handlers[mode]
        
        # Execute query
        result = await handler(query, top_k=top_k, enable_rerank=enable_rerank, **kwargs)
        
        # Create response
        response = QueryResponse(
            query=query,
            mode=mode,
            result=result.get("response", ""),
            metadata=self._extract_metadata(result)
        )
        
        return response
    
    async def _query_local(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute local mode query.
        
        Local mode focuses on specific entities and their immediate relationships.
        Best for detailed, entity-specific questions.
        
        Args:
            query: Query text
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        logger.debug("Executing LOCAL mode query")
        return await self.rag_manager.query(
            question=query,
            mode="local",
            **kwargs
        )
    
    async def _query_global(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute global mode query.
        
        Global mode uses high-level themes and concepts.
        Best for summary and thematic questions.
        
        Args:
            query: Query text
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        logger.debug("Executing GLOBAL mode query")
        return await self.rag_manager.query(
            question=query,
            mode="global",
            **kwargs
        )
    
    async def _query_hybrid(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute hybrid mode query.
        
        Hybrid mode combines local and global approaches.
        Best for complex questions requiring both detail and context.
        
        Args:
            query: Query text
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        logger.debug("Executing HYBRID mode query")
        return await self.rag_manager.query(
            question=query,
            mode="hybrid",
            **kwargs
        )
    
    async def _query_mix(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute mix mode query.
        
        Mix mode integrates knowledge graph with vector retrieval.
        Often provides the best overall performance.
        
        Args:
            query: Query text
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        logger.debug("Executing MIX mode query")
        return await self.rag_manager.query(
            question=query,
            mode="mix",
            **kwargs
        )
    
    async def _query_naive(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute naive mode query.
        
        Naive mode uses basic retrieval without advanced techniques.
        Useful as a baseline for comparison.
        
        Args:
            query: Query text
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        logger.debug("Executing NAIVE mode query")
        return await self.rag_manager.query(
            question=query,
            mode="naive",
            **kwargs
        )
    
    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from query result.
        
        Args:
            result: Query result
            
        Returns:
            Extracted metadata
        """
        metadata = {
            "latency_ms": result.get("latency_ms", 0),
            "top_k": result.get("top_k", 100),
            "reranked": result.get("reranked", False)
        }
        
        # Add any error information
        if "error" in result:
            metadata["error"] = result["error"]
        
        return metadata
    
    async def compare_modes(
        self,
        query: str,
        modes: Optional[list] = None,
        **kwargs
    ) -> Dict[str, QueryResponse]:
        """Compare query results across multiple modes.
        
        Args:
            query: Query text
            modes: List of modes to compare (default: all)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping modes to responses
        """
        if modes is None:
            modes = list(QueryMode)
        
        results = {}
        
        for mode in modes:
            try:
                response = await self.execute(query, mode=mode, **kwargs)
                results[mode] = response
                logger.info(f"Mode {mode}: {response.metadata.get('latency_ms', 0):.2f}ms")
            except Exception as e:
                logger.error(f"Failed to execute {mode} mode: {e}")
                results[mode] = QueryResponse(
                    query=query,
                    mode=mode,
                    result="",
                    metadata={"error": str(e)}
                )
        
        return results
    
    async def benchmark_modes(
        self,
        queries: list,
        modes: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark performance across modes.
        
        Args:
            queries: List of queries to test
            modes: List of modes to benchmark
            **kwargs: Additional parameters
            
        Returns:
            Benchmark results
        """
        if modes is None:
            modes = list(QueryMode)
        
        benchmark = {
            "total_queries": len(queries),
            "modes_tested": modes,
            "results": {}
        }
        
        for mode in modes:
            mode_stats = {
                "total_time_ms": 0,
                "avg_time_ms": 0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0,
                "successful": 0,
                "failed": 0
            }
            
            for query in queries:
                try:
                    start = time.time()
                    response = await self.execute(query, mode=mode, **kwargs)
                    elapsed_ms = (time.time() - start) * 1000
                    
                    mode_stats["total_time_ms"] += elapsed_ms
                    mode_stats["min_time_ms"] = min(mode_stats["min_time_ms"], elapsed_ms)
                    mode_stats["max_time_ms"] = max(mode_stats["max_time_ms"], elapsed_ms)
                    mode_stats["successful"] += 1
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {mode} on query: {e}")
                    mode_stats["failed"] += 1
            
            if mode_stats["successful"] > 0:
                mode_stats["avg_time_ms"] = mode_stats["total_time_ms"] / mode_stats["successful"]
            
            benchmark["results"][mode] = mode_stats
        
        return benchmark