"""Document processing and chunking for LightRAG."""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from loguru import logger
import markdown
from bs4 import BeautifulSoup


class DocumentProcessor:
    """Process and prepare documents for LightRAG ingestion."""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def process_markdown(self, content: str) -> str:
        """Process markdown content to plain text.
        
        Args:
            content: Markdown content
            
        Returns:
            Cleaned plain text
        """
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=['extra', 'codehilite'])
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"\(\)\[\]]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_size = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        overlap_sentences.insert(0, sent)
                        if overlap_size >= self.chunk_overlap:
                            break
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_file(self, file_path: Path) -> List[str]:
        """Process a file and return chunks.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of text chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Process based on file type
        if file_path.suffix.lower() in ['.md', '.markdown']:
            content = self.process_markdown(content)
        
        # Clean text
        content = self.clean_text(content)
        
        # Chunk text
        chunks = self.chunk_text(content)
        
        logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        return chunks
    
    def process_directory(self, dir_path: Path, pattern: str = "*.md") -> Dict[str, List[str]]:
        """Process all matching files in a directory.
        
        Args:
            dir_path: Directory path
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping file paths to chunks
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return {}
        
        results = {}
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                chunks = self.process_file(file_path)
                if chunks:
                    results[str(file_path)] = chunks
        
        logger.info(f"Processed {len(results)} files from {dir_path}")
        return results
    
    def create_document_metadata(self, content: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Create metadata for a document.
        
        Args:
            content: Document content
            source: Optional source identifier
            
        Returns:
            Document metadata
        """
        # Generate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Extract basic statistics
        word_count = len(content.split())
        char_count = len(content)
        
        # Estimate reading time (200 words per minute)
        reading_time_minutes = max(1, word_count // 200)
        
        metadata = {
            "content_hash": content_hash,
            "word_count": word_count,
            "char_count": char_count,
            "reading_time_minutes": reading_time_minutes,
            "source": source or "unknown"
        }
        
        return metadata
    
    def prepare_for_extraction(self, text: str) -> str:
        """Prepare text specifically for entity extraction.
        
        Args:
            text: Raw text
            
        Returns:
            Text optimized for entity extraction
        """
        # Keep more punctuation and structure for better extraction
        # Don't over-clean as it might remove important context
        
        # Just normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure sentences end with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text