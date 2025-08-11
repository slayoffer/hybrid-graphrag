"""Entity extraction module following LightRAG best practices."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from loguru import logger
import asyncio
import openai


@dataclass
class Entity:
    """Entity data class."""
    name: str
    type: str
    description: str
    source_chunk_ids: List[str] = None
    
    def __post_init__(self):
        if self.source_chunk_ids is None:
            self.source_chunk_ids = []
    
    @property
    def id(self) -> str:
        """Generate unique ID for entity."""
        return f"ent-{hashlib.md5(f'{self.name}:{self.type}'.encode()).hexdigest()[:16]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_chunk_ids": self.source_chunk_ids
        }


@dataclass 
class Relationship:
    """Relationship data class."""
    source: str
    target: str
    description: str
    keywords: List[str]
    strength: float  # 0.0 to 1.0
    source_chunk_ids: List[str] = None
    
    def __post_init__(self):
        if self.source_chunk_ids is None:
            self.source_chunk_ids = []
        # Ensure strength is between 0 and 1
        self.strength = max(0.0, min(1.0, self.strength))
    
    @property
    def id(self) -> str:
        """Generate unique ID for relationship."""
        rel_str = f"{self.source}-{self.target}-{self.description[:50]}"
        return f"rel-{hashlib.md5(rel_str.encode()).hexdigest()[:16]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "keywords": self.keywords,
            "strength": self.strength,
            "source_chunk_ids": self.source_chunk_ids
        }


class EntityExtractor:
    """Extract entities and relationships from text using LLM."""
    
    # Default entity types following LightRAG
    DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        entity_types: Optional[List[str]] = None,
        max_gleaning: int = 1,  # Number of extraction passes
        language: str = "English"
    ):
        """Initialize entity extractor.
        
        Args:
            llm_client: OpenAI client or compatible
            entity_types: List of entity types to extract
            max_gleaning: Maximum number of extraction passes
            language: Output language
        """
        self.llm_client = llm_client
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.max_gleaning = max_gleaning
        self.language = language
        
        logger.info(f"EntityExtractor initialized with types: {self.entity_types}")
        logger.info(f"Max gleaning passes: {self.max_gleaning}")
    
    def build_extraction_prompt(self, text: str, gleaning_round: int = 1) -> str:
        """Build entity extraction prompt following LightRAG format.
        
        Args:
            text: Text to extract from
            gleaning_round: Current gleaning iteration
            
        Returns:
            Formatted prompt
        """
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""Goal: Given a text document, identify all entities and their relationships.

Steps:
1. Identify all entities. For each, extract:
   - entity_name: Name as it appears in the text (capitalize if {self.language})
   - entity_type: One of [{entity_types_str}]
   - entity_description: Comprehensive description based on the text
   
2. Identify all relationships between entities. For each, extract:
   - source_entity: Name of the source entity
   - target_entity: Name of the target entity  
   - relationship_description: Nature of the relationship
   - relationship_keywords: Key terms describing the relationship (list)
   - relationship_strength: Numeric score 0.0-1.0 indicating strength/importance

Output format:
- Use "<|>" to separate fields within a record
- Use "##" to separate different records
- Entity format: ("entity"<|><name><|><type><|><description>)##
- Relationship format: ("relationship"<|><source><|><target><|><description><|><keywords><|><strength>)##
- End output with: <|COMPLETE|>

Text to analyze:
{text}

{"Note: This is gleaning round " + str(gleaning_round) + ". Focus on finding entities and relationships not found in previous rounds." if gleaning_round > 1 else ""}

Begin extraction:"""
        
        return prompt
    
    async def extract_from_text(
        self,
        text: str,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text.
        
        Args:
            text: Text to extract from
            chunk_id: Optional chunk ID for tracking
            
        Returns:
            Tuple of (entities, relationships)
        """
        all_entities = []
        all_relationships = []
        seen_entities = set()
        seen_relationships = set()
        
        for gleaning_round in range(1, self.max_gleaning + 1):
            logger.debug(f"Starting gleaning round {gleaning_round}/{self.max_gleaning}")
            
            # Build prompt
            prompt = self.build_extraction_prompt(text, gleaning_round)
            
            try:
                # Call LLM
                response = await self._call_llm(prompt)
                
                # Parse response
                entities, relationships = self._parse_extraction_response(response)
                
                # Deduplicate and add new items
                for entity in entities:
                    entity_key = (entity.name.lower(), entity.type)
                    if entity_key not in seen_entities:
                        if chunk_id:
                            entity.source_chunk_ids.append(chunk_id)
                        all_entities.append(entity)
                        seen_entities.add(entity_key)
                
                for rel in relationships:
                    rel_key = (rel.source.lower(), rel.target.lower(), rel.description[:50])
                    if rel_key not in seen_relationships:
                        if chunk_id:
                            rel.source_chunk_ids.append(chunk_id)
                        all_relationships.append(rel)
                        seen_relationships.add(rel_key)
                
                logger.debug(f"Round {gleaning_round}: Found {len(entities)} entities, {len(relationships)} relationships")
                if gleaning_round == 1:
                    logger.info(f"LLM response preview (first 1000 chars): {response[:1000]}")
                    if len(relationships) == 0:
                        logger.warning("No relationships found in LLM response")
                
                # If nothing new found, stop gleaning
                if gleaning_round > 1 and len(entities) == 0 and len(relationships) == 0:
                    logger.debug("No new entities/relationships found, stopping gleaning")
                    break
                    
            except Exception as e:
                logger.error(f"Extraction failed in round {gleaning_round}: {e}")
                if gleaning_round == 1:
                    # If first round fails, re-raise
                    raise
                # Otherwise continue with what we have
                break
        
        logger.info(f"Extraction complete: {len(all_entities)} entities, {len(all_relationships)} relationships")
        return all_entities, all_relationships
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for extraction.
        
        Args:
            prompt: Extraction prompt
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured")
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 for better extraction
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from text. Follow the output format exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_extraction_response(self, response: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM extraction response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        
        # Clean response
        response = response.strip()
        if not response:
            return entities, relationships
        
        # Remove completion marker
        if "<|COMPLETE|>" in response:
            response = response.split("<|COMPLETE|>")[0]
        
        # Split by record delimiter
        records = response.split("##")
        
        for record in records:
            record = record.strip()
            if not record:
                continue
            
            # Remove parentheses and quotes
            record = record.strip('()"')
            
            # Split by field delimiter
            fields = record.split("<|>")
            
            if len(fields) < 2:
                logger.debug(f"Skipping record with insufficient fields: {record}")
                continue
            
            record_type = fields[0].strip().lower()
            
            # Handle records that start with parentheses
            if record_type.startswith("("):
                record_type = record_type[1:]
            if record_type.startswith('"'):
                record_type = record_type[1:]
            record_type = record_type.strip('"')
            
            if record_type == "entity" and len(fields) >= 4:
                try:
                    entity = Entity(
                        name=fields[1].strip(),
                        type=fields[2].strip().lower(),
                        description=fields[3].strip()
                    )
                    # Validate entity type
                    if entity.type in [t.lower() for t in self.entity_types]:
                        entities.append(entity)
                    else:
                        logger.warning(f"Unknown entity type: {entity.type}")
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {e}")
            
            elif record_type == "relationship" and len(fields) >= 6:
                try:
                    # Parse keywords (might be a comma-separated string)
                    keywords_str = fields[4].strip()
                    # Remove any quotes or brackets
                    keywords_str = keywords_str.strip('"\'[]')
                    
                    # Split by comma
                    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                    
                    # Parse strength
                    try:
                        strength = float(fields[5].strip())
                    except:
                        strength = 0.5  # Default strength
                    
                    relationship = Relationship(
                        source=fields[1].strip(),
                        target=fields[2].strip(),
                        description=fields[3].strip(),
                        keywords=keywords,
                        strength=strength
                    )
                    relationships.append(relationship)
                except Exception as e:
                    logger.warning(f"Failed to parse relationship: {e}")
        
        return entities, relationships
    
    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Merged entities
        """
        entity_map = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            
            if key in entity_map:
                # Merge - combine descriptions and source chunks
                existing = entity_map[key]
                if entity.description not in existing.description:
                    existing.description += f" {entity.description}"
                existing.source_chunk_ids.extend(entity.source_chunk_ids)
                existing.source_chunk_ids = list(set(existing.source_chunk_ids))
            else:
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def merge_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Merge duplicate relationships.
        
        Args:
            relationships: List of relationships to merge
            
        Returns:
            Merged relationships
        """
        rel_map = {}
        
        for rel in relationships:
            # Use source, target, and first 50 chars of description as key
            key = (rel.source.lower(), rel.target.lower(), rel.description[:50])
            
            if key in rel_map:
                # Merge - average strength, combine keywords and chunks
                existing = rel_map[key]
                existing.strength = (existing.strength + rel.strength) / 2
                existing.keywords = list(set(existing.keywords + rel.keywords))
                existing.source_chunk_ids.extend(rel.source_chunk_ids)
                existing.source_chunk_ids = list(set(existing.source_chunk_ids))
            else:
                rel_map[key] = rel
        
        return list(rel_map.values())