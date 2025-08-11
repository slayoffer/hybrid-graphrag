"""Entity extraction using LangChain LLMGraphTransformer.

This module implements cybersecurity-specific entity extraction from documents
using LangChain's experimental graph transformers with OpenAI models.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import asyncio
from loguru import logger

try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    from langchain_community.graphs.graph_document import GraphDocument
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain dependencies not installed: {e}")
    LANGCHAIN_AVAILABLE = False
    # Mock classes for development
    class LLMGraphTransformer:
        pass
    class ChatOpenAI:
        pass
    class Document:
        pass
    class GraphDocument:
        pass

from .entity_schemas import (
    CYBERSECURITY_ENTITY_TYPES,
    CYBERSECURITY_RELATIONSHIP_TYPES,
    ENTITY_PROPERTIES,
    RELATIONSHIP_PROPERTIES,
    get_extraction_config
)

# Import existing Entity and Relationship dataclasses
from .entity_extractor import Entity, Relationship


class LangChainEntityExtractor:
    """Entity extraction pipeline using LangChain LLMGraphTransformer."""
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        domain: str = "cybersecurity"
    ):
        """Initialize the LangChain entity extractor.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use for extraction
            temperature: Model temperature for consistency
            domain: Domain configuration to use (cybersecurity, generic)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain dependencies not installed. Run: pip install langchain-experimental langchain-openai langchain-core langchain-community")
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.domain = domain
        
        # Get domain-specific configuration
        config = get_extraction_config(domain)
        self.entity_types = config["entity_types"]
        self.relationship_types = config["relationship_types"]
        self.entity_properties = config["entity_properties"]
        self.relationship_properties = config["relationship_properties"]
        self.strict_mode = config["strict_mode"]
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=temperature
        )
        
        # Initialize LLMGraphTransformer with cybersecurity schema
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.entity_types,
            allowed_relationships=self.relationship_types,
            node_properties=self.entity_properties,
            relationship_properties=self.relationship_properties,
            strict_mode=self.strict_mode
        )
        
        logger.info(f"Initialized LangChainEntityExtractor with model: {model}, domain: {domain}")
        logger.info(f"Entity types: {len(self.entity_types)}, Relationship types: {len(self.relationship_types)}")
    
    async def extract_from_text(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text using LangChain.
        
        Args:
            text: Text to extract from
            chunk_id: Optional chunk ID for tracking
            metadata: Optional metadata about the source
            
        Returns:
            Tuple of (entities, relationships) as Entity and Relationship objects
        """
        try:
            # Create document object
            document = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # Extract entities and relationships using LangChain
            logger.debug("Calling LangChain LLMGraphTransformer...")
            graph_documents = await self.transformer.aconvert_to_graph_documents([document])
            
            if not graph_documents:
                logger.warning("No graph documents generated from text")
                return [], []
            
            graph_doc = graph_documents[0]
            
            # Convert LangChain nodes to our Entity objects
            entities = []
            entity_map = {}  # Map node IDs to entity names for relationship conversion
            
            for node in graph_doc.nodes:
                # Extract entity properties
                props = node.properties if hasattr(node, 'properties') else {}
                
                # Create Entity object
                entity = Entity(
                    name=node.id,  # LangChain uses 'id' for entity name
                    type=node.type if hasattr(node, 'type') else "Unknown",
                    description=props.get('description', f"{node.type} entity: {node.id}"),
                    source_chunk_ids=[chunk_id] if chunk_id else []
                )
                
                # Store additional properties as metadata (for future use)
                entity.metadata = {
                    'confidence': props.get('confidence', 0.8),
                    'category': props.get('category'),
                    'pricing': props.get('pricing'),
                    'rating': props.get('rating'),
                    'version': props.get('version'),
                    'vendor': props.get('vendor'),
                    'compliance': props.get('compliance'),
                    'security_rating': props.get('security_rating')
                }
                
                entities.append(entity)
                entity_map[node.id] = entity.name
            
            # Convert LangChain relationships to our Relationship objects
            relationships = []
            
            for rel in graph_doc.relationships:
                # Extract relationship properties
                props = rel.properties if hasattr(rel, 'properties') else {}
                
                # Get source and target entity names
                source_name = rel.source.id if hasattr(rel.source, 'id') else str(rel.source)
                target_name = rel.target.id if hasattr(rel.target, 'id') else str(rel.target)
                
                # Create Relationship object
                relationship = Relationship(
                    source=source_name,
                    target=target_name,
                    description=props.get('description', rel.type if hasattr(rel, 'type') else 'RELATED'),
                    keywords=self._extract_keywords(props.get('context', '')),
                    strength=float(props.get('strength', 0.6)),
                    source_chunk_ids=[chunk_id] if chunk_id else []
                )
                
                # Store additional properties as metadata
                relationship.metadata = {
                    'type': rel.type if hasattr(rel, 'type') else 'RELATED',
                    'confidence': props.get('confidence', 0.7),
                    'context': props.get('context'),
                    'evidence': props.get('evidence')
                }
                
                relationships.append(relationship)
            
            logger.info(f"LangChain extraction complete: {len(entities)} entities, {len(relationships)} relationships")
            
            # Log entity type distribution
            entity_types_count = {}
            for entity in entities:
                entity_types_count[entity.type] = entity_types_count.get(entity.type, 0) + 1
            logger.debug(f"Entity type distribution: {entity_types_count}")
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"LangChain entity extraction failed: {e}")
            return [], []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Simple keyword extraction - split by common delimiters
        keywords = []
        for delimiter in [',', ';', '|', ' and ', ' or ']:
            if delimiter in text:
                keywords.extend([k.strip() for k in text.split(delimiter)])
                break
        
        # If no delimiters found, take first few words
        if not keywords:
            words = text.split()[:5]
            keywords = [w for w in words if len(w) > 3]
        
        return keywords[:10]  # Limit to 10 keywords
    
    async def extract_from_documents(self, documents: List[Document]) -> List[Tuple[List[Entity], List[Relationship]]]:
        """Extract entities from multiple documents.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            List of tuples (entities, relationships) for each document
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                chunk_id = f"doc_{i:03d}"
                entities, relationships = await self.extract_from_text(
                    doc.page_content,
                    chunk_id=chunk_id,
                    metadata=doc.metadata
                )
                results.append((entities, relationships))
                
            except Exception as e:
                logger.error(f"Failed to extract from document {i}: {e}")
                results.append(([], []))
        
        # Log summary statistics
        total_entities = sum(len(e) for e, _ in results)
        total_relationships = sum(len(r) for _, r in results)
        logger.info(f"Processed {len(documents)} documents: {total_entities} entities, {total_relationships} relationships")
        
        return results
    
    def get_extraction_statistics(self, extraction_results: List[Tuple[List[Entity], List[Relationship]]]) -> Dict[str, Any]:
        """Calculate extraction statistics.
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            'total_documents': len(extraction_results),
            'total_entities': 0,
            'total_relationships': 0,
            'entity_types': {},
            'relationship_types': {},
            'avg_confidence': 0.0,
            'entities_per_document': 0.0,
            'relationships_per_document': 0.0
        }
        
        total_confidence = 0.0
        confidence_count = 0
        
        for entities, relationships in extraction_results:
            stats['total_entities'] += len(entities)
            stats['total_relationships'] += len(relationships)
            
            # Count entity types
            for entity in entities:
                stats['entity_types'][entity.type] = stats['entity_types'].get(entity.type, 0) + 1
                
                # Aggregate confidence scores
                if hasattr(entity, 'metadata') and entity.metadata.get('confidence'):
                    total_confidence += float(entity.metadata['confidence'])
                    confidence_count += 1
            
            # Count relationship types
            for rel in relationships:
                rel_type = rel.metadata.get('type', 'UNKNOWN') if hasattr(rel, 'metadata') else 'UNKNOWN'
                stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
                
                # Aggregate confidence scores
                if hasattr(rel, 'metadata') and rel.metadata.get('confidence'):
                    total_confidence += float(rel.metadata['confidence'])
                    confidence_count += 1
        
        # Calculate averages
        if stats['total_documents'] > 0:
            stats['entities_per_document'] = stats['total_entities'] / stats['total_documents']
            stats['relationships_per_document'] = stats['total_relationships'] / stats['total_documents']
        
        if confidence_count > 0:
            stats['avg_confidence'] = total_confidence / confidence_count
        
        return stats
    
    def validate_extraction_quality(
        self,
        extraction_results: List[Tuple[List[Entity], List[Relationship]]],
        min_entities_per_doc: int = 5
    ) -> Dict[str, Any]:
        """Validate the quality of entity extraction.
        
        Args:
            extraction_results: List of extraction results to validate
            min_entities_per_doc: Minimum expected entities per document
            
        Returns:
            Validation results with quality metrics
        """
        stats = self.get_extraction_statistics(extraction_results)
        
        validation = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'quality_score': 0.0,
            'statistics': stats
        }
        
        # Check entity extraction rate
        if stats['entities_per_document'] < min_entities_per_doc:
            validation['warnings'].append(
                f"Low entity extraction rate: {stats['entities_per_document']:.1f} per document (expected: {min_entities_per_doc}+)"
            )
        
        # Check relationship extraction
        if stats['total_relationships'] == 0:
            validation['errors'].append("No relationships extracted")
            validation['passed'] = False
        
        # Check confidence levels
        if stats['avg_confidence'] < 0.6:
            validation['warnings'].append(
                f"Low extraction confidence: {stats['avg_confidence']:.2f} (expected: 0.6+)"
            )
        
        # Check entity type diversity
        if len(stats['entity_types']) < 3:
            validation['warnings'].append(
                f"Low entity type diversity: {len(stats['entity_types'])} types (expected: 3+)"
            )
        
        # Calculate quality score (0.0 - 1.0)
        quality_factors = [
            min(stats['entities_per_document'] / min_entities_per_doc, 1.0) * 0.3,  # Entity rate (30%)
            min(stats['total_relationships'] / stats['total_entities'], 1.0) * 0.2 if stats['total_entities'] > 0 else 0,  # Relationship ratio (20%)
            stats['avg_confidence'] * 0.3,  # Confidence (30%)
            min(len(stats['entity_types']) / 5, 1.0) * 0.2  # Type diversity (20%)
        ]
        
        validation['quality_score'] = sum(quality_factors)
        
        return validation