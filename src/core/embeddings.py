"""OpenAI embedding function wrapper for LightRAG with Neo4j compatibility."""

import asyncio
from typing import List, Optional, Union
import numpy as np
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

# Import LightRAG embedding utilities
try:
    from lightrag.utils import EmbeddingFunc
except ImportError:
    logger.warning("LightRAG not installed, using mock EmbeddingFunc")
    # Mock for development
    class EmbeddingFunc:
        def __init__(self, embedding_dim: int, max_token_size: int, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func


class OpenAIEmbeddings:
    """OpenAI embeddings wrapper with async support and retry logic."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        timeout: int = 60
    ):
        """Initialize OpenAI embeddings wrapper.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize both sync and async clients
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        
        # Set embedding dimensions based on model
        self.embedding_dim = self._get_embedding_dim(model)
        
        logger.info(f"Initialized OpenAI embeddings with model: {model}")
    
    def _get_embedding_dim(self, model: str) -> int:
        """Get embedding dimensions for the model.
        
        Args:
            model: Model name
            
        Returns:
            Embedding dimensions
        """
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dim_map.get(model, 1536)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _async_embed_single(self, text: str) -> List[float]:
        """Embed a single text asynchronously with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.async_client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    async def async_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts asynchronously.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Handle batch size limits (OpenAI max is typically 2048 inputs)
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i//batch_size + 1} with {len(batch)} texts")
            
            try:
                response = await self.async_client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Fall back to single embedding
                for text in batch:
                    embedding = await self._async_embed_single(text)
                    all_embeddings.append(embedding)
        
        return all_embeddings
    
    def embed_sync(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Synchronous embedding function.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if single else embeddings
        except Exception as e:
            logger.error(f"Error in sync embedding: {e}")
            raise
    
    async def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Make the class callable for LightRAG compatibility.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = await self.async_embed_batch(texts)
        return np.array(embeddings, dtype=np.float32)


def create_embedding_func(
    api_key: str,
    model: str = "text-embedding-3-large",
    max_token_size: int = 8192
) -> EmbeddingFunc:
    """Create LightRAG-compatible embedding function for Neo4j.
    
    CRITICAL: This function is REQUIRED for Neo4j backend in LightRAG.
    Without this, LightRAG will fail when using Neo4j storage.
    
    Args:
        api_key: OpenAI API key
        model: Embedding model name
        max_token_size: Maximum token size for the model
        
    Returns:
        EmbeddingFunc instance configured for OpenAI
    """
    # Create embeddings wrapper
    embeddings = OpenAIEmbeddings(api_key=api_key, model=model)
    
    # Create async wrapper function for LightRAG
    async def embedding_func(texts: List[str]) -> np.ndarray:
        """Async embedding function for LightRAG.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Use the embeddings wrapper
        return await embeddings(texts)
    
    # Create sync wrapper for compatibility
    def embedding_func_sync(texts: List[str]) -> np.ndarray:
        """Sync wrapper for embedding function.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(embedding_func(texts))
        finally:
            loop.close()
    
    # Return LightRAG-compatible EmbeddingFunc
    return EmbeddingFunc(
        embedding_dim=embeddings.embedding_dim,
        max_token_size=max_token_size,
        func=embedding_func  # Use async version
    )


def create_llm_func(
    api_key: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 4096,
    temperature: float = 0.7
):
    """Create LLM function for LightRAG.
    
    Args:
        api_key: OpenAI API key
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Temperature for sampling
        
    Returns:
        Async LLM function for LightRAG
    """
    client = AsyncOpenAI(api_key=api_key)
    
    async def llm_func(prompt: str, **kwargs) -> str:
        """Async LLM function for LightRAG.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters including system_prompt
            
        Returns:
            Model response
        """
        try:
            # Use the system_prompt from kwargs if provided, otherwise use default
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", max_tokens),
                temperature=kwargs.get("temperature", temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            raise
    
    return llm_func