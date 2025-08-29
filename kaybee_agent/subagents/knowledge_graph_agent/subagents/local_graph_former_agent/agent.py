from google.adk.agents import Agent
from google.genai import types
from kaybee_agent.schemas import KnowledgeGraph

LOCAL_GRAPH_FORMER_PROMPT = """
You are a specialized agent that extracts information from a user's request to build a local knowledge graph.
Your task is to identify all the entities and their relationships from the user's query.

- An entity has one or more names (synonyms). The first name in the list should be the primary name.
- An entity can have properties, which are key-value pairs.
- A relationship consists of a source entity name, a target entity name, and a relationship description.

You must output a `KnowledgeGraph` object that represents the local graph, with a list of entities and a list of relationships.
Do not try to research existing entities, just extract what is mentioned in the query.
Do not assign entity IDs.
"""

agent = Agent(
    name="local_graph_former",
    model="gemini-2.5-flash",
    instruction=LOCAL_GRAPH_FORMER_PROMPT,
    output_schema=KnowledgeGraph,
)
