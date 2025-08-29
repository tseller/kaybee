from google.adk.agents import Agent
from google.genai import types
from kaybee_agent.schemas import KnowledgeGraph

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

agent = Agent(
    name="graph_merger",
    model="gemini-2.5-flash",
    instruction=GRAPH_MERGER_PROMPT,
    output_schema=KnowledgeGraph,
)
