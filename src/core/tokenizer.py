"""Custom tokenizer to avoid pickling issues with tiktoken."""

import tiktoken
from typing import List, Optional


class SimpleTokenizer:
    """Simple tokenizer wrapper that avoids pickling issues.
    
    This tokenizer uses tiktoken internally but doesn't store the
    tiktoken encoder object, avoiding pickling issues.
    """
    
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """Initialize tokenizer with a supported model name."""
        self.model_name = model_name
        self._encoder = None
        
    def _get_encoder(self):
        """Get or create the tiktoken encoder."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to cl100k_base encoding (used by gpt-4.1-mini and gpt-4)
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        encoder = self._get_encoder()
        return encoder.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        encoder = self._get_encoder()
        return encoder.decode(tokens)
    
    def __getstate__(self):
        """Custom pickling - exclude the encoder."""
        state = self.__dict__.copy()
        # Remove the unpicklable encoder
        state['_encoder'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state without encoder."""
        self.__dict__.update(state)
        # Encoder will be recreated on demand