from google.adk.agents import Agent
from google.genai import types
from kaybee_agent.schemas import KnowledgeGraph, StoreResult

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

local_graph_former = Agent(
    name="local_graph_former",
    model="gemini-2.5-flash",
    instruction=LOCAL_GRAPH_FORMER_PROMPT,
    output_schema=KnowledgeGraph,
)

from .tools import get_relevant_neighborhoods, store_graph

NEIGHBORHOOD_RESEARCHER_PROMPT = """
You are a specialized agent that researches the main knowledge graph.
Your input is a `KnowledgeGraph` object from the `local_graph_former` agent.
Your task is to extract all the entity names from the entities in the input graph and use the `get_relevant_neighborhoods` tool to find the relevant neighborhood in the main knowledge graph.

You must call the `get_relevant_neighborhoods` tool with a list of all entity names.
The output of this agent will be the output of the `get_relevant_neighborhoods` tool.
"""

neighborhood_researcher = Agent(
    name="neighborhood_researcher",
    model="gemini-2.5-flash",
    instruction=NEIGHBORHOOD_RESEARCHER_PROMPT,
    tools=[get_relevant_neighborhoods],
)

GRAPH_MERGER_PROMPT = """
You are a specialized agent that merges two knowledge graphs.
You will receive two inputs:
1.  A "local graph" from the `local_graph_former` agent, containing the new information from the user's request.
2.  A "neighborhood graph" from the `neighborhood_researcher` agent, containing the existing information from the main knowledge graph.

Your task is to intelligently merge the "local graph" into the "neighborhood graph".
You must follow these rules:
-   **Local graph wins:** If there is any conflict between the local graph and the neighborhood graph, the local graph's information takes precedence.
-   **Add new entities:** If an entity from the local graph does not exist in the neighborhood graph, add it.
-   **Update existing entities:** If an entity from the local graph already exists in the neighborhood graph (based on name matching), update its properties and relationships. The new properties and relationships from the local graph should be added. If a property or relationship from the local graph already exists on the entity, its value should be updated.
-   **Preserve existing information:** Preserve all the information in the neighborhood graph that is not affected by the local graph.
-   **Handle deletions:** If the user's request was to delete an entity or relationship, you will need to identify it and remove it from the graph. The user's intent should be present in the context.

You must output the final, merged graph as a `KnowledgeGraph` object.
"""

graph_merger = Agent(
    name="graph_merger",
    model="gemini-2.5-flash",
    instruction=GRAPH_MERGER_PROMPT,
    output_schema=KnowledgeGraph,
)

GRAPH_STORER_PROMPT = """
You are a specialized agent that stores a knowledge graph.
Your input is a `KnowledgeGraph` object from the `graph_merger` agent.
Your task is to call the `store_graph` tool with the received graph.
The graph needs to be converted to a dictionary before calling the tool.
Your final output should be the result of the `store_graph` tool call.
"""

graph_storer = Agent(
    name="graph_storer",
    model="gemini-2.5-flash",
    instruction=GRAPH_STORER_PROMPT,
    tools=[store_graph],
)
