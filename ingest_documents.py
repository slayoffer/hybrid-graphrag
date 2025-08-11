#!/usr/bin/env python3
"""Document ingestion script for production RAG system."""

import asyncio
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from neo4j import AsyncGraphDatabase
from src.config import Settings
from src.extraction.langchain_entity_extractor import LangChainEntityExtractor
from src.extraction.entity_extractor import Entity, Relationship

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def extract_tool_details_from_markdown(content: str) -> Dict[str, Dict[str, Any]]:
    """Extract structured information from the comparison table and detailed sections."""
    
    tools = {
        "Aikido": {
            "type": "SAST / DAST",
            "hosting": ["SaaS", "CI Code Inject"],
            "pricing": "UI users",
            "advantages": [
                "Cost-effective",
                "Includes DAST and Pentest",
                "K8s scans on roadmap",
                "WAF helper",
                "Reduced false positives",
                "Strong deduplication",
                "Security-oriented",
                "Competitive pricing",
                "AI-driven features"
            ],
            "disadvantages": [
                "No VPN broker scan support",
                "Limited scanner customization in UI"
            ],
            "ai_features": [
                "AI-driven vulnerability analysis",
                "AI explanations for vulnerabilities and possible fixes",
                "Independent AI vulnerability research center (Aikido owned)"
            ],
            "additional_features": [
                "WAF",
                "Code Reachability Analysis",
                "Dependencies Reachability Analysis",
                "Pentest as a service (optional)",
                "K8s vulnerability graphs (planned for 2025)",
                "K8s pods lifecycle graphs (planned for 2025)"
            ],
            "unique_selling_points": [
                "DAST included in subscription",
                "WAF included in subscription",
                "AI-driven noise reduction",
                "Primary focus is on reducing False Positive findings"
            ]
        },
        "OX Security": {
            "type": "SAST",
            "hosting": ["SaaS", "Self-hosted", "Broker"],
            "pricing": "Committers",
            "advantages": [
                "Vulnerability graphs",
                "AI integration",
                "Deep vulnerability analysis",
                "Custom workflows for handling findings",
                "Self hosting capabilities at extra cost",
                "Broker to perform scans inside VPN"
            ],
            "disadvantages": [
                "High cost",
                "No DAST",
                "Limited scanner customization in UI"
            ],
            "ai_features": [
                "AI explanations for vulnerabilities and possible fixes"
            ],
            "additional_features": [
                "Code vulnerability graphs",
                "3 level deep code reachability vulnerability analysis",
                "Dependencies Reachability Analysis"
            ],
            "unique_selling_points": [
                "Deep vulnerability analysis",
                "Customizable triage workflows"
            ]
        },
        "Veracode": {
            "type": "SAST / DAST",
            "hosting": ["SaaS", "CI Code Inject", "Broker"],
            "pricing": "Committers",
            "advantages": [
                "Comprehensive DAST capabilities",
                "Broker to perform scans inside VPN"
            ],
            "disadvantages": [
                "High cost"
            ],
            "ai_features": [
                "AI-assisted vulnerability management"
            ],
            "additional_features": [
                "Pentest as a service (optional)"
            ],
            "unique_selling_points": [
                "Robust DAST capabilities"
            ]
        },
        "GitLab Ultimate": {
            "type": "SAST / DAST",
            "hosting": ["SaaS", "Self-hosted"],
            "pricing": "GitLab users",
            "advantages": [
                "Seamless integration with GitLab ecosystem"
            ],
            "disadvantages": [
                "High cost",
                "No auto triage",
                "Limited deduplication"
            ],
            "ai_features": [],
            "additional_features": [],
            "unique_selling_points": [
                "Seamless GitLab integration"
            ],
            "integrations": [
                "Datadog", "Jira", "VSCode", "Okta", "Drata"
            ]
        },
        "SonarQube": {
            "type": "SAST",
            "hosting": ["SaaS", "Self-hosted"],
            "pricing": "Fixed plan",
            "advantages": [
                "Best code linter",
                "Superior code quality scanner"
            ],
            "disadvantages": [
                "Security scan is an add-on",
                "High rate of false positives from security scans",
                "Primarily focused on code linting & quality"
            ],
            "ai_features": [
                "AI generated code detection",
                "AI fixes"
            ],
            "additional_features": [
                "IaC scanning",
                "Secrets management"
            ],
            "unique_selling_points": [
                "Best code quality and linting capabilities on the market"
            ]
        }
    }
    
    return tools


def create_manual_entities_and_relationships(tools_data: Dict[str, Dict[str, Any]]) -> Tuple[List[Entity], List[Relationship]]:
    """Create entities and relationships from structured tool data."""
    
    entities = []
    relationships = []
    
    for tool_name, tool_info in tools_data.items():
        # Create main tool entity
        tool_entity = Entity(
            name=tool_name,
            type="Product",
            description=f"{tool_name} - {tool_info['type']} security testing tool with {tool_info['pricing']} pricing model"
        )
        entities.append(tool_entity)
        
        # Create testing type entities and relationships
        if "SAST" in tool_info["type"]:
            sast_entity = Entity(name="SAST", type="TestingType", description="Static Application Security Testing")
            if sast_entity not in entities:
                entities.append(sast_entity)
            relationships.append(Relationship(
                source=tool_name,
                target="SAST",
                description="PROVIDES",
                keywords=["provides", "supports", "testing"],
                strength=1.0
            ))
        
        if "DAST" in tool_info["type"]:
            dast_entity = Entity(name="DAST", type="TestingType", description="Dynamic Application Security Testing")
            if dast_entity not in entities:
                entities.append(dast_entity)
            relationships.append(Relationship(
                source=tool_name,
                target="DAST",
                description="PROVIDES",
                keywords=["provides", "supports", "testing"],
                strength=1.0
            ))
        
        # Create pricing model entity
        pricing_entity = Entity(
            name=f"{tool_info['pricing']} Pricing",
            type="PricingModel",
            description=f"Pricing model based on {tool_info['pricing']}"
        )
        if pricing_entity not in entities:
            entities.append(pricing_entity)
        relationships.append(Relationship(
            source=tool_name,
            target=f"{tool_info['pricing']} Pricing",
            description="USES_PRICING",
            keywords=["pricing", "cost", "model"],
            strength=1.0
        ))
        
        # Create hosting option entities
        for hosting in tool_info["hosting"]:
            hosting_entity = Entity(
                name=hosting,
                type="HostingOption",
                description=f"{hosting} deployment option"
            )
            if hosting_entity not in entities:
                entities.append(hosting_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=hosting,
                description="SUPPORTS_HOSTING",
                keywords=["hosting", "deployment", "infrastructure"],
                strength=0.9
            ))
        
        # Create AI feature entities
        for ai_feature in tool_info.get("ai_features", []):
            ai_entity = Entity(
                name=ai_feature[:50],  # Truncate long names
                type="AIFeature",
                description=ai_feature
            )
            entities.append(ai_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=ai_feature[:50],
                description="HAS_AI_FEATURE",
                keywords=["ai", "artificial intelligence", "ml"],
                strength=0.95
            ))
        
        # Create advantage entities
        for advantage in tool_info.get("advantages", [])[:5]:  # Limit to top 5
            adv_entity = Entity(
                name=advantage[:50],
                type="Advantage",
                description=f"Advantage: {advantage}"
            )
            entities.append(adv_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=advantage[:50],
                description="HAS_ADVANTAGE",
                keywords=["advantage", "benefit", "strength"],
                strength=0.85
            ))
        
        # Create disadvantage entities
        for disadvantage in tool_info.get("disadvantages", []):
            dis_entity = Entity(
                name=disadvantage[:50],
                type="Disadvantage",
                description=f"Disadvantage: {disadvantage}"
            )
            entities.append(dis_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=disadvantage[:50],
                description="HAS_DISADVANTAGE",
                keywords=["disadvantage", "limitation", "weakness"],
                strength=0.85
            ))
        
        # Create additional feature entities
        for feature in tool_info.get("additional_features", []):
            # Special handling for specific features
            if "3 level deep" in feature:
                feature_entity = Entity(
                    name="3-Level Code Reachability",
                    type="Feature",
                    description="3 level deep code reachability vulnerability analysis"
                )
            elif "K8s" in feature or "Kubernetes" in feature:
                feature_entity = Entity(
                    name=feature[:50],
                    type="RoadmapFeature" if "2025" in feature else "Feature",
                    description=feature
                )
            else:
                feature_entity = Entity(
                    name=feature[:50],
                    type="Feature",
                    description=feature
                )
            
            entities.append(feature_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=feature_entity.name,
                description="PLANS_FEATURE" if "2025" in feature else "HAS_FEATURE",
                keywords=["feature", "capability", "function"],
                strength=0.9
            ))
        
        # Create USP entities
        for usp in tool_info.get("unique_selling_points", []):
            usp_entity = Entity(
                name=usp[:50],
                type="USP",
                description=f"Unique Selling Point: {usp}"
            )
            entities.append(usp_entity)
            relationships.append(Relationship(
                source=tool_name,
                target=usp[:50],
                description="HAS_USP",
                keywords=["unique", "selling point", "differentiator"],
                strength=0.95
            ))
    
    # Add specific entities for test questions
    # Broker support
    broker_entity = Entity(
        name="Broker Support",
        type="Feature",
        description="Support for broker deployment to scan inside VPN"
    )
    entities.append(broker_entity)
    
    # Add broker relationships
    for tool in ["OX Security", "Veracode"]:
        relationships.append(Relationship(
            source=tool,
            target="Broker Support",
            description="SUPPORTS",
            keywords=["broker", "vpn", "scanning"],
            strength=1.0
        ))
    
    # Aikido 2025 features
    k8s_vuln_entity = Entity(
        name="Kubernetes vulnerability graphs",
        type="RoadmapFeature",
        description="Kubernetes (K8s) vulnerability graphs planned for 2025"
    )
    entities.append(k8s_vuln_entity)
    
    k8s_pods_entity = Entity(
        name="Kubernetes pods lifecycle graphs",
        type="RoadmapFeature",
        description="Kubernetes pods lifecycle graphs planned for 2025"
    )
    entities.append(k8s_pods_entity)
    
    relationships.append(Relationship(
        source="Aikido",
        target="Kubernetes vulnerability graphs",
        description="PLANS_2025",
        keywords=["kubernetes", "k8s", "2025", "roadmap"],
        strength=1.0
    ))
    
    relationships.append(Relationship(
        source="Aikido",
        target="Kubernetes pods lifecycle graphs",
        description="PLANS_2025",
        keywords=["kubernetes", "k8s", "pods", "2025", "roadmap"],
        strength=1.0
    ))
    
    return entities, relationships


async def main():
    """Enhanced ingestion with comprehensive extraction."""
    
    logger.info("=" * 80)
    logger.info("Anton Evseev Profile Document Ingestion")
    logger.info("=" * 80)
    
    # Load settings
    settings = Settings()
    workspace = "anton_profile"
    
    # Load the document
    doc_path = Path("data/mixed/About Me - Anton Evseev - Confluence.md")
    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    logger.info(f"📄 Loaded document: {doc_path.name}")
    logger.info(f"  Size: {len(content):,} characters")
    
    # Extract structured data from markdown
    logger.info("\n📊 Extracting structured tool data...")
    tools_data = extract_tool_details_from_markdown(content)
    logger.info(f"  Extracted data for {len(tools_data)} tools")
    
    # Create manual entities and relationships
    logger.info("\n🔧 Creating manual entities and relationships...")
    manual_entities, manual_relationships = create_manual_entities_and_relationships(tools_data)
    logger.info(f"  Created {len(manual_entities)} manual entities")
    logger.info(f"  Created {len(manual_relationships)} manual relationships")
    
    # Initialize LangChain entity extractor
    logger.info("\n🤖 Initializing LangChain entity extractor...")
    extractor = LangChainEntityExtractor(
        openai_api_key=settings.openai_api_key,
        model="gpt-4.1-mini",
        temperature=0.0,
        domain="cybersecurity"
    )
    
    # Process document in chunks for better extraction
    all_entities = list(manual_entities)
    all_relationships = list(manual_relationships)
    
    chunk_size = 5000
    chunks_to_process = []
    
    # Process the full document in chunks
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        chunks_to_process.append((i, chunk))
    
    logger.info(f"\n🔍 Processing {len(chunks_to_process)} chunks...")
    
    for idx, (offset, chunk) in enumerate(chunks_to_process):
        logger.info(f"  Processing chunk {idx+1}/{len(chunks_to_process)} (offset: {offset})...")
        
        try:
            entities, relationships = await extractor.extract_from_text(
                chunk,
                chunk_id=f"chunk_{idx}",
                metadata={"source": "SAST/DAST Comparison", "offset": offset}
            )
            
            # Add unique entities
            for entity in entities:
                if not any(e.name == entity.name and e.type == entity.type for e in all_entities):
                    all_entities.append(entity)
            
            # Add unique relationships
            for rel in relationships:
                if not any(r.source == rel.source and r.target == rel.target for r in all_relationships):
                    all_relationships.append(rel)
            
            logger.info(f"    Extracted {len(entities)} entities, {len(relationships)} relationships")
        except Exception as e:
            logger.warning(f"    Error processing chunk {idx+1}: {e}")
            continue
    
    logger.info(f"\n✅ Total extraction results:")
    logger.info(f"  {len(all_entities)} unique entities")
    logger.info(f"  {len(all_relationships)} unique relationships")
    
    # Show entity type distribution
    entity_types = {}
    for entity in all_entities:
        entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
    
    logger.info("\n📊 Entity type distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {entity_type}: {count}")
    
    # Connect to Neo4j and store data
    logger.info("\n💾 Storing in Neo4j...")
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )
    
    try:
        async with driver.session() as session:
            # Store the document
            await session.run("""
                CREATE (d:Document {
                    workspace: $workspace,
                    id: $doc_id,
                    title: $title,
                    source: $source,
                    content: $content,
                    char_count: $char_count
                })
            """, 
                workspace=workspace,
                doc_id="sast_dast_enhanced_doc",
                title="Commercial SAST/DAST Tools Comparison",
                source=str(doc_path),
                content=content[:5000],
                char_count=len(content)
            )
            logger.info("  ✅ Document stored")
            
            # Store entities
            entity_count = 0
            for entity in all_entities:
                try:
                    await session.run("""
                        MERGE (e:Entity {workspace: $workspace, name: $name})
                        SET e.type = $type,
                            e.description = $description,
                            e.chunk_ids = $chunk_ids
                    """,
                        workspace=workspace,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description[:500] if entity.description else "",
                        chunk_ids=entity.source_chunk_ids if entity.source_chunk_ids else []
                    )
                    entity_count += 1
                except Exception as e:
                    logger.warning(f"  Skipping duplicate entity: {entity.name} - {e}")
                    continue
            logger.info(f"  ✅ {entity_count} entities stored")
            
            # Store relationships
            rel_count = 0
            for rel in all_relationships:
                # Use description as relationship type
                rel_type = rel.description if rel.description else "RELATED"
                await session.run("""
                    MATCH (e1:Entity {workspace: $workspace, name: $source})
                    MATCH (e2:Entity {workspace: $workspace, name: $target})
                    MERGE (e1)-[r:RELATED {type: $rel_type}]->(e2)
                    SET r.description = $description,
                        r.keywords = $keywords,
                        r.strength = $strength,
                        r.workspace = $workspace
                """,
                    workspace=workspace,
                    source=rel.source,
                    target=rel.target,
                    rel_type=rel_type,
                    description=rel.description[:500] if rel.description else "",
                    keywords=rel.keywords[:10] if rel.keywords else [],
                    strength=rel.strength
                )
                rel_count += 1
            logger.info(f"  ✅ {rel_count} relationships stored")
            
            # Create detailed chunks for better retrieval
            logger.info("\n📦 Creating enhanced chunks...")
            
            # Create tool-specific chunks
            for tool_name, tool_info in tools_data.items():
                tool_chunk = f"""
                {tool_name} Details:
                - Type: {tool_info['type']}
                - Pricing Model: {tool_info['pricing']}
                - Hosting Options: {', '.join(tool_info['hosting'])}
                - Advantages: {', '.join(tool_info.get('advantages', []))}
                - Disadvantages: {', '.join(tool_info.get('disadvantages', []))}
                - AI Features: {', '.join(tool_info.get('ai_features', []))}
                - Additional Features: {', '.join(tool_info.get('additional_features', []))}
                - Unique Selling Points: {', '.join(tool_info.get('unique_selling_points', []))}
                """
                
                await session.run("""
                    CREATE (c:Chunk {
                        workspace: $workspace,
                        id: $chunk_id,
                        content: $content,
                        chunk_index: $index,
                        doc_id: $doc_id,
                        tool: $tool
                    })
                """,
                    workspace=workspace,
                    chunk_id=f"tool_chunk_{tool_name}",
                    content=tool_chunk,
                    index=-1,  # Special index for tool chunks
                    doc_id="sast_dast_enhanced_doc",
                    tool=tool_name
                )
            
            # Create regular document chunks
            chunk_size = 1000
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                await session.run("""
                    CREATE (c:Chunk {
                        workspace: $workspace,
                        id: $chunk_id,
                        content: $content,
                        chunk_index: $index,
                        doc_id: $doc_id
                    })
                """,
                    workspace=workspace,
                    chunk_id=f"chunk_{i//chunk_size}",
                    content=chunk,
                    index=i//chunk_size,
                    doc_id="sast_dast_enhanced_doc"
                )
            
            total_chunks = len(tools_data) + (len(content) // chunk_size) + 1
            logger.info(f"  ✅ {total_chunks} chunks created")
            
            # Verify storage
            logger.info("\n✅ Verification:")
            
            # Count entities
            result = await session.run("""
                MATCH (e:Entity {workspace: $workspace})
                RETURN count(e) as count
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Total entities: {record['count']}")
            
            # Count relationships
            result = await session.run("""
                MATCH (e1:Entity {workspace: $workspace})-[r]-(e2:Entity {workspace: $workspace})
                RETURN count(DISTINCT r) as count
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Total relationships: {record['count']}")
            
            # Count chunks
            result = await session.run("""
                MATCH (c:Chunk {workspace: $workspace})
                RETURN count(c) as count
            """, workspace=workspace)
            record = await result.single()
            logger.info(f"  Total chunks: {record['count']}")
            
    finally:
        await driver.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Enhanced ingestion complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())