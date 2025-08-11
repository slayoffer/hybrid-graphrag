"""LightRAG prompts module."""

from .extraction_prompts import (
    get_extraction_prompt,
    get_summary_prompt,
    get_query_analysis_prompt,
    get_answer_prompt
)

__all__ = [
    'get_extraction_prompt',
    'get_summary_prompt', 
    'get_query_analysis_prompt',
    'get_answer_prompt'
]