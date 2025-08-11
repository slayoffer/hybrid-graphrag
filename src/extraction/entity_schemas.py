"""Entity and relationship schemas for cybersecurity domain.

This module defines the entity types, relationship types, and properties
used by the LangChain LLMGraphTransformer for entity extraction.
"""

from typing import List, Tuple

# Cybersecurity-specific entity types based on domain requirements
CYBERSECURITY_ENTITY_TYPES = [
    "Organization",      # Companies, vendors, teams
    "Product",          # Software products, tools, services
    "Platform",         # Operating systems, cloud platforms, CI/CD systems
    "Feature",          # Product features, capabilities
    "Technology",       # Programming languages, frameworks, protocols
    "Standard",         # Compliance standards, security frameworks, methodologies
    "Vulnerability",    # Security vulnerabilities, CVEs, weaknesses
    "Integration",      # API integrations, connectors, plugins
    "Methodology",      # Security practices, development frameworks
    "Decision",         # Business decisions, architectural choices, recommendations
    "Requirement",      # Technical or business requirements, specifications
    "Comparison"        # Comparative analysis, evaluations, trade-offs
]

# Cybersecurity-specific relationship types with allowed source/target combinations
CYBERSECURITY_RELATIONSHIP_TYPES = [
    ("Product", "PROVIDES", "Feature"),           # Product provides specific features
    ("Organization", "OFFERS", "Product"),        # Company offers products/services
    ("Product", "INTEGRATES_WITH", "Product"),   # Integration between products
    ("Product", "COMPETES_WITH", "Product"),     # Competitive relationships
    ("Product", "SUPPORTS", "Technology"),       # Product supports languages/frameworks
    ("Product", "REQUIRES", "Standard"),         # Compliance requirements
    ("Feature", "COSTS", "Comparison"),          # Pricing information
    ("Platform", "INCLUDES", "Feature"),         # Platform built-in features
    ("Organization", "IMPLEMENTS", "Standard"),  # Company compliance
    ("Vulnerability", "AFFECTS", "Product"),     # Security impact
    ("Product", "REPLACES", "Product"),          # Migration/replacement
    ("Technology", "ENABLES", "Feature"),        # Tech enables capabilities
    ("Standard", "MANDATES", "Requirement"),     # Compliance requirements
    ("Decision", "CONSIDERS", "Comparison"),     # Decision factors
    ("Integration", "CONNECTS", "Product"),      # Integration connections
    ("Methodology", "GUIDES", "Decision")        # Framework guidance
]

# Entity properties to extract for each entity type
ENTITY_PROPERTIES = [
    "description",      # Detailed description of the entity
    "category",         # Category or sub-classification
    "confidence",       # Extraction confidence score (0.0-1.0)
    "pricing",          # Cost information if applicable
    "rating",           # Quality or performance rating
    "version",          # Version information
    "vendor",           # Vendor or provider name
    "compliance",       # Compliance or certification information
    "security_rating"   # Security assessment score
]

# Relationship properties to extract
RELATIONSHIP_PROPERTIES = [
    "confidence",       # Extraction confidence score (0.0-1.0)
    "description",      # Relationship description
    "strength",         # Relationship strength (0.0-1.0)
    "context",          # Context where relationship was found
    "evidence"          # Supporting evidence for the relationship
]

# Generic entity types for fallback (original LightRAG types)
GENERIC_ENTITY_TYPES = [
    "organization",
    "person",
    "geo",
    "event", 
    "category"
]

# Configuration presets for different domains
EXTRACTION_CONFIGS = {
    "cybersecurity": {
        "entity_types": CYBERSECURITY_ENTITY_TYPES,
        "relationship_types": CYBERSECURITY_RELATIONSHIP_TYPES,
        "entity_properties": ENTITY_PROPERTIES,
        "relationship_properties": RELATIONSHIP_PROPERTIES,
        "strict_mode": False  # Allow flexibility for edge cases
    },
    "generic": {
        "entity_types": GENERIC_ENTITY_TYPES,
        "relationship_types": [],  # No predefined relationships
        "entity_properties": ["description"],
        "relationship_properties": ["description", "strength"],
        "strict_mode": False
    }
}

def get_extraction_config(domain: str = "cybersecurity") -> dict:
    """Get extraction configuration for a specific domain.
    
    Args:
        domain: Domain name (cybersecurity, generic)
        
    Returns:
        Configuration dictionary with entity types, relationships, and properties
    """
    return EXTRACTION_CONFIGS.get(domain, EXTRACTION_CONFIGS["generic"])

def validate_relationship(source_type: str, rel_type: str, target_type: str) -> bool:
    """Validate if a relationship is allowed by the schema.
    
    Args:
        source_type: Type of source entity
        rel_type: Relationship type
        target_type: Type of target entity
        
    Returns:
        True if relationship is valid according to schema
    """
    for allowed_source, allowed_rel, allowed_target in CYBERSECURITY_RELATIONSHIP_TYPES:
        if (source_type == allowed_source and 
            rel_type == allowed_rel and 
            target_type == allowed_target):
            return True
    return False

def get_entity_type_description(entity_type: str) -> str:
    """Get description for an entity type.
    
    Args:
        entity_type: Entity type name
        
    Returns:
        Description of what this entity type represents
    """
    descriptions = {
        "Organization": "Companies, vendors, teams, or any organizational entity",
        "Product": "Software products, tools, services, or applications",
        "Platform": "Operating systems, cloud platforms, CI/CD systems, or infrastructure",
        "Feature": "Specific capabilities, functionalities, or features of products",
        "Technology": "Programming languages, frameworks, protocols, or technical standards",
        "Standard": "Compliance standards, security frameworks, or methodologies",
        "Vulnerability": "Security vulnerabilities, CVEs, weaknesses, or threats",
        "Integration": "API integrations, connectors, plugins, or interoperability features",
        "Methodology": "Security practices, development frameworks, or processes",
        "Decision": "Business decisions, architectural choices, or recommendations",
        "Requirement": "Technical or business requirements, specifications, or constraints",
        "Comparison": "Comparative analysis, evaluations, trade-offs, or benchmarks"
    }
    return descriptions.get(entity_type, "Unknown entity type")

def get_relationship_type_description(rel_type: str) -> str:
    """Get description for a relationship type.
    
    Args:
        rel_type: Relationship type name
        
    Returns:
        Description of what this relationship represents
    """
    descriptions = {
        "PROVIDES": "Offers or supplies a capability or feature",
        "OFFERS": "Makes available as a product or service",
        "INTEGRATES_WITH": "Has built-in integration or compatibility",
        "COMPETES_WITH": "Is in competition or serves as alternative",
        "SUPPORTS": "Has compatibility or works with",
        "REQUIRES": "Has as a dependency or prerequisite",
        "COSTS": "Has associated pricing or cost",
        "INCLUDES": "Contains or has built-in",
        "IMPLEMENTS": "Follows or adheres to",
        "AFFECTS": "Has impact on or compromises",
        "REPLACES": "Serves as replacement or migration path",
        "ENABLES": "Makes possible or facilitates",
        "MANDATES": "Requires or enforces",
        "CONSIDERS": "Takes into account or evaluates",
        "CONNECTS": "Links or bridges between",
        "GUIDES": "Provides direction or framework for"
    }
    return descriptions.get(rel_type, "Unknown relationship type")