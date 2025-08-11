"""Main application entry point for LightRAG GraphRAG Testing Framework."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from .config import settings
from .core.rag_manager import RAGManager
from .queries.query_executor import QueryExecutor
from .graph.neo4j_client import Neo4jClient
from .api.routes import router as api_router


# Configure logging
logger.add(
    "logs/lightrag.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting LightRAG GraphRAG Testing Framework...")
    
    try:
        # Initialize RAG manager
        logger.info("Creating RAG Manager...")
        app.state.rag_manager = RAGManager(settings)
        
        # CRITICAL: Initialize RAG storages asynchronously
        logger.info("Initializing RAG Manager storages...")
        await app.state.rag_manager.initialize()
        
        # Initialize query executor
        logger.info("Initializing Query Executor...")
        app.state.query_executor = QueryExecutor(app.state.rag_manager)
        
        # Initialize Neo4j client
        logger.info("Initializing Neo4j Client...")
        app.state.neo4j_client = Neo4jClient(settings)
        
        # Create Neo4j indexes
        logger.info("Creating Neo4j indexes...")
        indexes = await app.state.neo4j_client.create_indexes()
        logger.info(f"Created indexes: {indexes}")
        
        # Check health
        health = await app.state.rag_manager.health_check()
        logger.info(f"System health: {health}")
        
        logger.info("✅ LightRAG GraphRAG Testing Framework started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LightRAG GraphRAG Testing Framework...")
    
    try:
        # Close Neo4j connections
        if hasattr(app.state, "neo4j_client"):
            await app.state.neo4j_client.close()
        
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="LightRAG GraphRAG Testing Framework",
    description="Testing framework for measuring LLM performance with graph-based knowledge representation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.log_level == "DEBUG" else None
        }
    )


# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LightRAG GraphRAG Testing Framework",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


def run_server(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    reload: bool = False
):
    """Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to (uses settings if not provided)
        reload: Enable auto-reload for development
    """
    if port is None:
        port = settings.api_port
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    # Run the server
    run_server(reload=False)