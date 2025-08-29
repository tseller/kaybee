from google.adk.agents import SequentialAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .subagents.local_graph_former_agent import local_graph_former_agent
from .subagents.neighborhood_researcher_agent import neighborhood_researcher_agent
from .subagents.graph_merger_agent import graph_merger_agent
from .subagents.graph_storer_agent import graph_storer_agent

KNOWLEDGE_GRAPH_AGENT_PROMPT = """
You are a master agent responsible for maintaining a knowledge graph.
Your purpose is to understand a user's request and orchestrate a sequence of sub-agents to add, remove, or change information in the knowledge graph.

You will proceed in four steps:
1.  **Form Local Graph:** First, a sub-agent will analyze the user's request to form a "local graph" of the entities and relationships mentioned.
2.  **Research Neighborhood:** Next, a second sub-agent will research the existing knowledge graph to find the neighborhood of entities relevant to the user's request.
3.  **Merge Graphs:** A third sub-agent will merge the "local graph" with the "neighborhood graph" to produce a final, updated version of the graph.
4.  **Store Graph:** Finally, a fourth sub-agent will store the merged graph.
"""

agent = SequentialAgent(
    name="knowledge_graph_agent",
    sub_agents=[
        local_graph_former_agent,
        neighborhood_researcher_agent,
        graph_merger_agent,
        graph_storer_agent,
    ],
)
