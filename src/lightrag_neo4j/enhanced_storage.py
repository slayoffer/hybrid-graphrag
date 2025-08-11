"""Enhanced Neo4j storage with support for entity/relationship properties.

This module extends the base Neo4j storage to handle the rich properties
extracted by LangChain LLMGraphTransformer.
"""

from typing import Any, Dict, List, Optional
import time
from loguru import logger
from neo4j import AsyncGraphDatabase
import numpy as np

from .storage import Neo4jOnlyStorage


class EnhancedNeo4jStorage(Neo4jOnlyStorage):
    """Enhanced storage with support for entity and relationship properties."""
    
    async def store_entities_with_properties(self, entities: List[Any]) -> int:
        """Store entities with all their properties in Neo4j.
        
        Args:
            entities: List of Entity objects with metadata properties
            
        Returns:
            Number of entities stored
        """
        stored_count = 0
        
        async with self.driver.session() as session:
            for entity in entities:
                # Generate embedding for entity
                entity_content = f"{entity.name}\n{entity.description}"
                if self.embedding_func:
                    embedding = await self._generate_embedding(entity_content)
                else:
                    embedding = np.random.random(1536).tolist()
                
                # Extract metadata properties if available
                metadata = getattr(entity, 'metadata', {})
                
                # Store entity with all properties
                await session.run("""
                    MERGE (e:Entity {id: $id, workspace: $workspace})
                    SET e.name = $name,
                        e.type = $type,
                        e.entity_type = $type,
                        e.description = $description,
                        e.embedding = $embedding,
                        e.source_chunk_ids = $source_chunk_ids,
                        e.created_at = timestamp(),
                        e.updated_at = timestamp(),
                        e.confidence = $confidence,
                        e.category = $category,
                        e.pricing = $pricing,
                        e.rating = $rating,
                        e.version = $version,
                        e.vendor = $vendor,
                        e.compliance = $compliance,
                        e.security_rating = $security_rating
                """,
                id=entity.id,
                workspace=self.workspace,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                embedding=embedding,
                source_chunk_ids=entity.source_chunk_ids,
                confidence=metadata.get('confidence', 0.8),
                category=metadata.get('category'),
                pricing=metadata.get('pricing'),
                rating=metadata.get('rating'),
                version=metadata.get('version'),
                vendor=metadata.get('vendor'),
                compliance=metadata.get('compliance'),
                security_rating=metadata.get('security_rating'))
                
                stored_count += 1
                
                # Link entity to source chunks
                for chunk_id in entity.source_chunk_ids:
                    await session.run("""
                        MATCH (c:Chunk {id: $chunk_id, workspace: $workspace})
                        MATCH (e:Entity {id: $entity_id, workspace: $workspace})
                        MERGE (c)-[:HAS_ENTITY]->(e)
                    """,
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    workspace=self.workspace)
        
        logger.info(f"Stored {stored_count} entities with properties")
        return stored_count
    
    async def store_relationships_with_properties(self, relationships: List[Any]) -> int:
        """Store relationships with all their properties in Neo4j.
        
        Args:
            relationships: List of Relationship objects with metadata properties
            
        Returns:
            Number of relationships stored
        """
        stored_count = 0
        
        async with self.driver.session() as session:
            for rel in relationships:
                # Extract metadata properties if available
                metadata = getattr(rel, 'metadata', {})
                rel_type = metadata.get('type', 'RELATED')
                
                # Store relationship with all properties
                result = await session.run("""
                    MATCH (e1:Entity {name: $source, workspace: $workspace})
                    MATCH (e2:Entity {name: $target, workspace: $workspace})
                    MERGE (e1)-[r:RELATED {id: $id}]->(e2)
                    SET r.type = $type,
                        r.description = $description,
                        r.keywords = $keywords,
                        r.strength = $strength,
                        r.confidence = $confidence,
                        r.context = $context,
                        r.evidence = $evidence,
                        r.source_chunk_ids = $source_chunk_ids,
                        r.created_at = timestamp(),
                        r.updated_at = timestamp()
                    RETURN COUNT(r) as count
                """,
                source=rel.source,
                target=rel.target,
                workspace=self.workspace,
                id=rel.id,
                type=rel_type,
                description=rel.description,
                keywords=rel.keywords,
                strength=rel.strength,
                confidence=metadata.get('confidence', 0.7),
                context=metadata.get('context'),
                evidence=metadata.get('evidence'),
                source_chunk_ids=rel.source_chunk_ids)
                
                # Check if relationship was created
                record = await result.single()
                if record and record['count'] > 0:
                    stored_count += 1
                    
                    # Also create typed relationship for better querying
                    if rel_type and rel_type != 'RELATED':
                        await session.run(f"""
                            MATCH (e1:Entity {{name: $source, workspace: $workspace}})
                            MATCH (e2:Entity {{name: $target, workspace: $workspace}})
                            MERGE (e1)-[r:{rel_type}]->(e2)
                            SET r.description = $description,
                                r.strength = $strength,
                                r.confidence = $confidence
                        """,
                        source=rel.source,
                        target=rel.target,
                        workspace=self.workspace,
                        description=rel.description,
                        strength=rel.strength,
                        confidence=metadata.get('confidence', 0.7))
        
        logger.info(f"Stored {stored_count} relationships with properties")
        return stored_count
    
    async def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entity dictionaries
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                WHERE e.type = $entity_type OR e.entity_type = $entity_type
                RETURN e
                ORDER BY e.confidence DESC
                LIMIT 100
            """,
            workspace=self.workspace,
            entity_type=entity_type)
            
            entities = []
            async for record in result:
                entity = record['e']
                entities.append({
                    'id': entity.get('id'),
                    'name': entity.get('name'),
                    'type': entity.get('type'),
                    'description': entity.get('description'),
                    'confidence': entity.get('confidence'),
                    'category': entity.get('category'),
                    'vendor': entity.get('vendor'),
                    'version': entity.get('version'),
                    'pricing': entity.get('pricing'),
                    'compliance': entity.get('compliance'),
                    'security_rating': entity.get('security_rating')
                })
            
            return entities
    
    async def get_relationships_by_type(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get all relationships of a specific type.
        
        Args:
            rel_type: Type of relationships to retrieve
            
        Returns:
            List of relationship dictionaries
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e1:Entity {workspace: $workspace})-[r:RELATED]->(e2:Entity {workspace: $workspace})
                WHERE r.type = $rel_type
                RETURN e1.name as source, e2.name as target, r
                ORDER BY r.confidence DESC, r.strength DESC
                LIMIT 100
            """,
            workspace=self.workspace,
            rel_type=rel_type)
            
            relationships = []
            async for record in result:
                relationships.append({
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['r'].get('type'),
                    'description': record['r'].get('description'),
                    'strength': record['r'].get('strength'),
                    'confidence': record['r'].get('confidence'),
                    'context': record['r'].get('context'),
                    'evidence': record['r'].get('evidence')
                })
            
            return relationships
    
    async def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about entities and their properties.
        
        Returns:
            Dictionary with entity statistics
        """
        async with self.driver.session() as session:
            # Get entity type distribution
            type_result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                RETURN e.type as type, COUNT(e) as count
                ORDER BY count DESC
            """, workspace=self.workspace)
            
            type_distribution = {}
            async for record in type_result:
                type_distribution[record['type']] = record['count']
            
            # Get average confidence
            confidence_result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                WHERE e.confidence IS NOT NULL
                RETURN AVG(e.confidence) as avg_confidence
            """, workspace=self.workspace)
            
            confidence_record = await confidence_result.single()
            avg_confidence = confidence_record['avg_confidence'] if confidence_record else 0
            
            # Get property coverage
            property_result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                RETURN 
                    COUNT(CASE WHEN e.category IS NOT NULL THEN 1 END) as with_category,
                    COUNT(CASE WHEN e.vendor IS NOT NULL THEN 1 END) as with_vendor,
                    COUNT(CASE WHEN e.version IS NOT NULL THEN 1 END) as with_version,
                    COUNT(CASE WHEN e.pricing IS NOT NULL THEN 1 END) as with_pricing,
                    COUNT(CASE WHEN e.compliance IS NOT NULL THEN 1 END) as with_compliance,
                    COUNT(e) as total
            """, workspace=self.workspace)
            
            property_record = await property_result.single()
            
            return {
                'type_distribution': type_distribution,
                'avg_confidence': avg_confidence,
                'property_coverage': {
                    'category': property_record['with_category'] / property_record['total'] if property_record['total'] > 0 else 0,
                    'vendor': property_record['with_vendor'] / property_record['total'] if property_record['total'] > 0 else 0,
                    'version': property_record['with_version'] / property_record['total'] if property_record['total'] > 0 else 0,
                    'pricing': property_record['with_pricing'] / property_record['total'] if property_record['total'] > 0 else 0,
                    'compliance': property_record['with_compliance'] / property_record['total'] if property_record['total'] > 0 else 0
                },
                'total_entities': property_record['total'] if property_record else 0
            }
    
    async def enhanced_entity_search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Enhanced entity search with property filtering.
        
        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            min_confidence: Minimum confidence threshold
            top_k: Number of results to return
            
        Returns:
            List of matching entities with properties
        """
        # Generate query embedding
        if self.embedding_func:
            query_embedding = await self._generate_embedding(query)
        else:
            query_embedding = np.random.random(1536).tolist()
        
        async with self.driver.session() as session:
            # Build type filter
            type_filter = ""
            if entity_types:
                type_list = ", ".join([f"'{t}'" for t in entity_types])
                type_filter = f"AND (node.type IN [{type_list}] OR node.entity_type IN [{type_list}])"
            
            # For Community Edition, just do a simple match without vector similarity
            # (Neo4j Community doesn't support vector indexes)
            type_filter_e = ""
            if entity_types:
                type_list = ", ".join([f"'{t}'" for t in entity_types])
                type_filter_e = f"AND (e.type IN [{type_list}] OR e.entity_type IN [{type_list}])"
            
            result = await session.run(f"""
                MATCH (e:Entity {{workspace: $workspace}})
                WHERE e.embedding IS NOT NULL
                AND e.confidence >= $min_confidence
                {type_filter_e}
                RETURN e, 1.0 as similarity
                LIMIT $top_k
            """,
            workspace=self.workspace,
            min_confidence=min_confidence,
            top_k=top_k)
            
            entities = []
            async for record in result:
                entity = record['e']
                entities.append({
                    'id': entity.get('id'),
                    'name': entity.get('name'),
                    'type': entity.get('type'),
                    'description': entity.get('description'),
                    'similarity': record['similarity'],
                    'confidence': entity.get('confidence'),
                    'category': entity.get('category'),
                    'vendor': entity.get('vendor'),
                    'version': entity.get('version'),
                    'pricing': entity.get('pricing'),
                    'compliance': entity.get('compliance'),
                    'security_rating': entity.get('security_rating')
                })
            
            return entities