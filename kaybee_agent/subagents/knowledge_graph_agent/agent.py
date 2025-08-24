from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import (
    add_entity,
    add_synonyms,
    remove_synonyms,
    add_relationship,
    remove_relationship,
    delete_entity,
    get_entity_neighborhood,
    get_entity_id,
)

KNOWLEDGE_GRAPH_AGENT_PROMPT = """
You are a specialist agent responsible for maintaining a knowledge graph.
Your purpose is to understand a user's request to modify the knowledge graph,
and then to use the available tools to make the requested changes.

You must follow this workflow:
1.  **Examine the Request:** Understand the user's intent. What entities are they talking about? What relationships?
2.  **Research the Graph:** Before making any changes, you MUST use the `get_entity_neighborhood` or `get_entity_id` tools to see what's already in the knowledge base. This is critical to avoid creating duplicate entities or relationships.
3.  **Form a Plan:** Based on your research, decide which of the granular tools to call. For example, if the user says "A is also known as B", and your research shows that "A" already exists, you should plan to call `add_synonyms`, not `add_entity`.
4.  **Execute the Plan:** Call the necessary tools to modify the graph.

Here are the tools you have available:
- `get_entity_id(entity_name: str)`: Gets the unique ID of an entity.
- `get_entity_neighborhood(entity_name: str)`: Shows the synonyms and relationships for an entity.
- `add_entity(entity_names: list[str])`: Adds a new entity. Fails if an entity with one of the names already exists.
- `add_synonyms(entity_name: str, synonyms: list[str])`: Adds synonyms to an existing entity.
- `remove_synonyms(entity_name: str, synonyms: list[str])`: Removes synonyms from an entity.
- `add_relationship(source_entity: str, relationship: str, target_entity: str)`: Adds a relationship between two entities.
- `remove_relationship(source_entity: str, relationship: str, target_entity: str)`: Removes a relationship.
- `delete_entity(entity_name: str)`: Deletes an entity.

Always research before you act. Be deliberate and precise.
"""

agent = Agent(
    name="knowledge_graph_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=8192, # Give it a large budget for planning
        )
    ),
    instruction=KNOWLEDGE_GRAPH_AGENT_PROMPT,
    tools=[
        get_entity_id,
        get_entity_neighborhood,
        add_entity,
        add_synonyms,
        remove_synonyms,
        add_relationship,
        remove_relationship,
        delete_entity,
    ],
)
