"""Exact LightRAG chunking implementation."""

from typing import Any, List, Dict, Optional
import tiktoken
import hashlib


class Tokenizer:
    """Tokenizer wrapper for tiktoken."""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        """Initialize tokenizer with model."""
        self.encoding = tiktoken.encoding_for_model(model)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        return self.encoding.decode(tokens)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Generate MD5 hash ID like LightRAG does."""
    hash_obj = hashlib.md5(content.encode())
    return f"{prefix}{hash_obj.hexdigest()}"


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: Optional[str] = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Exact implementation of LightRAG's chunking_by_token_size function.
    
    Args:
        tokenizer: Tokenizer instance
        content: Content to chunk
        split_by_character: Optional character to split by (e.g., "\n\n")
        split_by_character_only: If True, only split by character without further chunking
        overlap_token_size: Number of overlapping tokens between chunks
        max_token_size: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries with tokens, content, and chunk_order_index
    """
    tokens = tokenizer.encode(content)
    results: List[Dict[str, Any]] = []
    
    if split_by_character:
        # Split by character first
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        
        if split_by_character_only:
            # Only split by character, don't further chunk
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            # Split by character and further chunk if needed
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > max_token_size:
                    # Chunk is too large, split it further
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        
        # Create result chunks
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        # No character splitting, chunk by token size with overlap
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + max_token_size])
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    
    return results


def clean_text(text: str) -> str:
    """Clean text like LightRAG does."""
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text.strip()