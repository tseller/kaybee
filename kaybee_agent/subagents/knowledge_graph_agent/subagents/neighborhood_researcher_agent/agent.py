from google.adk.agents import Agent
from google.genai import types
from kaybee_agent.subagents.knowledge_graph_agent.tools import get_relevant_neighborhoods

NEIGHBORHOOD_RESEARCHER_PROMPT = """
You are a specialized agent that researches the main knowledge graph.
Your input is a `KnowledgeGraph` object from the `local_graph_former` agent.
Your task is to extract all the entity names from the entities in the input graph and use the `get_relevant_neighborhoods` tool to find the relevant neighborhood in the main knowledge graph.

You must call the `get_relevant_neighborhoods` tool with a list of all entity names.
The output of this agent will be the output of the `get_relevant_neighborhoods` tool.
"""

agent = Agent(
    name="neighborhood_researcher",
    model="gemini-2.5-flash",
    instruction=NEIGHBORHOOD_RESEARCHER_PROMPT,
    tools=[get_relevant_neighborhoods],
)
