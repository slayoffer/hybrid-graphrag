"""Query classification for adaptive retrieval strategies."""

import re
from enum import Enum
from typing import Dict, Tuple

class QueryType(Enum):
    """Types of queries for adaptive retrieval."""
    FACTUAL = "factual"  # Simple fact extraction (who, what, when, where)
    ANALYTICAL = "analytical"  # Comparison, analysis, reasoning
    COMPLEX = "complex"  # Multi-hop, synthesis, deep understanding
    
class QueryClassifier:
    """Classifies queries to determine optimal retrieval strategy."""
    
    def __init__(self):
        """Initialize query classifier with patterns."""
        # Factual query patterns
        self.factual_patterns = [
            r"^what is\b",
            r"^who is\b",
            r"^when did\b",
            r"^where is\b",
            r"^how many\b",
            r"^what (was|were)\b",
            r"position at",
            r"current role",
            r"speak$",
            r"personality type",
            r"^is\b",
            r"^are\b",
        ]
        
        # Complex query patterns
        self.complex_patterns = [
            r"compare",
            r"contrast",
            r"analyze",
            r"explain",
            r"how does.*work",
            r"relationship between",
            r"compliance.*standards",
            r"ensure.*meet",
            r"unusual.*and.*how many",
            r"courses.*complete",
            r"would.*like to",
            r"wants to",
        ]
        
    def classify(self, query: str) -> Tuple[QueryType, Dict[str, float]]:
        """
        Classify a query and return type with confidence scores.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (QueryType, weights dict for retrieval)
        """
        query_lower = query.lower().strip()
        
        # Check for factual patterns
        is_factual = any(
            re.search(pattern, query_lower) 
            for pattern in self.factual_patterns
        )
        
        # Check for complex patterns
        is_complex = any(
            re.search(pattern, query_lower) 
            for pattern in self.complex_patterns
        )
        
        # Determine query type and weights
        if is_factual and not is_complex:
            query_type = QueryType.FACTUAL
            weights = {
                "vector": 0.8,
                "graph": 0.2
            }
        elif is_complex:
            query_type = QueryType.COMPLEX
            weights = {
                "vector": 0.3,
                "graph": 0.7
            }
        else:
            # Default to analytical for everything else
            query_type = QueryType.ANALYTICAL
            weights = {
                "vector": 0.5,
                "graph": 0.5
            }
        
        # Adjust weights based on query length (longer queries benefit from graph)
        word_count = len(query.split())
        if word_count > 10:
            weights["graph"] = min(weights["graph"] + 0.1, 0.8)
            weights["vector"] = 1.0 - weights["graph"]
        
        return query_type, weights
    
    def get_retrieval_params(self, query_type: QueryType) -> Dict:
        """
        Get optimal retrieval parameters for a query type.
        
        Args:
            query_type: The classified query type
            
        Returns:
            Dictionary of retrieval parameters
        """
        if query_type == QueryType.FACTUAL:
            return {
                "initial_chunks": 3,
                "traversal_depth": 1,
                "similarity_threshold": 0.8,
                "max_graph_chunks": 5
            }
        elif query_type == QueryType.COMPLEX:
            return {
                "initial_chunks": 5,
                "traversal_depth": 2,
                "similarity_threshold": 0.7,
                "max_graph_chunks": 15
            }
        else:  # ANALYTICAL
            return {
                "initial_chunks": 4,
                "traversal_depth": 2,
                "similarity_threshold": 0.75,
                "max_graph_chunks": 10
            }