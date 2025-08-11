"""LightRAG-style extraction prompts."""

# LightRAG entity extraction prompt template
ENTITY_EXTRACTION_PROMPT = """Goal: Given a text document that has been split into chunks, identify all entities and their relationships.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text, capitalize if English
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities based on the input text

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of the relationship between the two entities
- relationship_keywords: one or more high-level keywords that summarize the relationship
- relationship_strength: a numeric score between 0 and 1 indicating the strength/importance of the relationship

3. Return output in the following format:
- Use "<|>" to separate fields within a record
- Use "##" to separate different records
- Entity format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)##
- Relationship format: ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)##
- Finish the output with <|COMPLETE|>

######################
-Examples-
######################

Example 1:

Input text:
"The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called 'The Great Explainer'. The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961-1963."

Output:
("entity"<|>The Feynman Lectures on Physics<|>event<|>A physics textbook based on lectures by Richard Feynman, presented at Caltech during 1961-1963)##
("entity"<|>Richard Feynman<|>person<|>Nobel laureate physicist known as 'The Great Explainer' who gave lectures at Caltech)##
("entity"<|>California Institute of Technology<|>organization<|>University (Caltech) where Feynman presented his physics lectures to undergraduate students during 1961-1963)##
("entity"<|>Nobel Prize<|>event<|>Prestigious award that Richard Feynman received for his contributions to physics)##
("relationship"<|>Richard Feynman<|>The Feynman Lectures on Physics<|>Richard Feynman authored the lectures that became The Feynman Lectures on Physics textbook<|>authored, created, lectures<|>0.9)##
("relationship"<|>Richard Feynman<|>California Institute of Technology<|>Richard Feynman presented lectures at Caltech during 1961-1963<|>presented, taught, lectures<|>0.8)##
("relationship"<|>The Feynman Lectures on Physics<|>California Institute of Technology<|>The lectures were presented at Caltech to undergraduate students<|>presented at, location, education<|>0.7)##
("relationship"<|>Richard Feynman<|>Nobel Prize<|>Richard Feynman received the Nobel Prize for his work in physics<|>awarded, recipient, achievement<|>0.85)##
<|COMPLETE|>

Example 2:

Input text:
"Amazon Web Services (AWS) is a subsidiary of Amazon that provides on-demand cloud computing platforms. AWS was launched in 2006 and has data centers in multiple regions including US-East (Virginia), EU-West (Ireland), and Asia-Pacific (Singapore). Jeff Bezos, the founder of Amazon, announced AWS at the 2006 Web 2.0 conference."

Output:
("entity"<|>Amazon Web Services<|>organization<|>Subsidiary of Amazon providing on-demand cloud computing platforms, launched in 2006 with global data centers)##
("entity"<|>Amazon<|>organization<|>Parent company of AWS, founded by Jeff Bezos)##
("entity"<|>Jeff Bezos<|>person<|>Founder of Amazon who announced AWS at the 2006 Web 2.0 conference)##
("entity"<|>Web 2.0 conference<|>event<|>2006 conference where Jeff Bezos announced AWS)##
("entity"<|>US-East (Virginia)<|>geo<|>AWS data center region located in Virginia, United States)##
("entity"<|>EU-West (Ireland)<|>geo<|>AWS data center region located in Ireland, Europe)##
("entity"<|>Asia-Pacific (Singapore)<|>geo<|>AWS data center region located in Singapore, Asia-Pacific)##
("relationship"<|>Amazon Web Services<|>Amazon<|>AWS is a subsidiary of Amazon<|>subsidiary, owned by, part of<|>1.0)##
("relationship"<|>Jeff Bezos<|>Amazon<|>Jeff Bezos founded Amazon<|>founded, created, CEO<|>1.0)##
("relationship"<|>Jeff Bezos<|>Amazon Web Services<|>Jeff Bezos announced AWS at the 2006 conference<|>announced, launched, introduced<|>0.8)##
("relationship"<|>Amazon Web Services<|>US-East (Virginia)<|>AWS has a data center region in Virginia<|>data center, infrastructure, location<|>0.7)##
("relationship"<|>Amazon Web Services<|>EU-West (Ireland)<|>AWS has a data center region in Ireland<|>data center, infrastructure, location<|>0.7)##
("relationship"<|>Amazon Web Services<|>Asia-Pacific (Singapore)<|>AWS has a data center region in Singapore<|>data center, infrastructure, location<|>0.7)##
("relationship"<|>Jeff Bezos<|>Web 2.0 conference<|>Jeff Bezos announced AWS at this conference<|>announced at, presented, speaker<|>0.6)##
<|COMPLETE|>

######################
-Real Data-
######################

Text to analyze:
{input_text}

{gleaning_note}

Output:
"""

# Prompt for knowledge graph summarization (for global search)
GRAPH_SUMMARY_PROMPT = """Goal: Create a comprehensive summary of the knowledge graph focusing on key themes and high-level concepts.

Analyze the provided entities and relationships to identify:
1. Main themes and topics
2. Key organizations and their roles
3. Important people and their contributions
4. Significant events and their impacts
5. Geographic distributions and patterns
6. Overall narrative or story

Provide a summary that captures the essence of the knowledge graph at a high level, suitable for answering thematic and overview questions.

Entities:
{entities}

Relationships:
{relationships}

Summary:
"""

# Prompt for community detection and summarization
COMMUNITY_SUMMARY_PROMPT = """Goal: Analyze a community of closely related entities and summarize their collective significance.

Given a group of entities that form a community (highly interconnected), provide:
1. The main theme or purpose of this community
2. Key members and their roles
3. Internal dynamics and relationships
4. External connections to other communities
5. Overall importance in the broader context

Community entities:
{entities}

Community relationships:
{relationships}

Community summary:
"""

# Prompt for query understanding (to determine search mode)
QUERY_ANALYSIS_PROMPT = """Goal: Analyze the user query to determine the best search strategy.

Given a user query, determine:
1. Query type: Is it asking for specific details (LOCAL), overview/themes (GLOBAL), or both (HYBRID)?
2. Key entities mentioned (if any)
3. Relationship focus (if any)
4. Level of detail required

Query: {query}

Analysis:
- Query type: [LOCAL/GLOBAL/HYBRID]
- Key entities: [list of entity names or "none"]
- Relationship focus: [yes/no]
- Detail level: [high/medium/low]
- Recommended search mode: [local/global/hybrid/mix]
"""

# Prompt for answer generation from retrieved context
ANSWER_GENERATION_PROMPT = """Goal: Generate a comprehensive answer using the retrieved context.

Given the user query and retrieved information, provide a detailed answer that:
1. Directly addresses the user's question
2. Uses specific information from the context
3. Maintains accuracy without hallucination
4. Provides appropriate level of detail

Query: {query}

Retrieved entities:
{entities}

Retrieved relationships:
{relationships}

Retrieved text chunks:
{chunks}

Answer:
"""

def get_extraction_prompt(
    text: str,
    entity_types: list = None,
    gleaning_round: int = 1,
    language: str = "English"
) -> str:
    """Get formatted extraction prompt.
    
    Args:
        text: Text to extract from
        entity_types: List of entity types
        gleaning_round: Current gleaning iteration
        language: Output language
        
    Returns:
        Formatted prompt
    """
    if entity_types is None:
        entity_types = ["organization", "person", "geo", "event", "category"]
    
    entity_types_str = ", ".join(entity_types)
    
    gleaning_note = ""
    if gleaning_round > 1:
        gleaning_note = f"\nNote: This is gleaning round {gleaning_round}. Focus on finding entities and relationships not found in previous rounds.\n"
    
    return ENTITY_EXTRACTION_PROMPT.format(
        entity_types=entity_types_str,
        input_text=text,
        gleaning_note=gleaning_note
    )

def get_summary_prompt(entities: list, relationships: list) -> str:
    """Get graph summary prompt.
    
    Args:
        entities: List of entities
        relationships: List of relationships
        
    Returns:
        Formatted prompt
    """
    entities_str = "\n".join([f"- {e.name} ({e.type}): {e.description}" for e in entities])
    relationships_str = "\n".join([f"- {r.source} -> {r.target}: {r.description} (strength: {r.strength})" for r in relationships])
    
    return GRAPH_SUMMARY_PROMPT.format(
        entities=entities_str,
        relationships=relationships_str
    )

def get_query_analysis_prompt(query: str) -> str:
    """Get query analysis prompt.
    
    Args:
        query: User query
        
    Returns:
        Formatted prompt
    """
    return QUERY_ANALYSIS_PROMPT.format(query=query)

def get_answer_prompt(
    query: str,
    entities: list = None,
    relationships: list = None,
    chunks: list = None
) -> str:
    """Get answer generation prompt.
    
    Args:
        query: User query
        entities: Retrieved entities
        relationships: Retrieved relationships
        chunks: Retrieved text chunks
        
    Returns:
        Formatted prompt
    """
    entities_str = ""
    if entities:
        entities_str = "\n".join([f"- {e.get('name', '')} ({e.get('type', '')}): {e.get('description', '')}" for e in entities])
    
    relationships_str = ""
    if relationships:
        relationships_str = "\n".join([f"- {r.get('source', '')} -> {r.get('target', '')}: {r.get('description', '')} (strength: {r.get('strength', 0.5)})" for r in relationships])
    
    chunks_str = ""
    if chunks:
        chunks_str = "\n---\n".join([c.get('content', '') for c in chunks[:5]])  # Limit to 5 chunks
    
    return ANSWER_GENERATION_PROMPT.format(
        query=query,
        entities=entities_str or "None",
        relationships=relationships_str or "None",
        chunks=chunks_str or "None"
    )