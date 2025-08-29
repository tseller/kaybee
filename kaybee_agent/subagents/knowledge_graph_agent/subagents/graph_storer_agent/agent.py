from google.adk.agents import Agent
from google.genai import types
from kaybee_agent.subagents.knowledge_graph_agent.tools import store_graph

GRAPH_STORER_PROMPT = """
You are a specialized agent that stores a knowledge graph.
Your input is a `KnowledgeGraph` object from the `graph_merger` agent.
Your task is to call the `store_graph` tool with the received graph.
The graph needs to be converted to a dictionary before calling the tool.
Your final output should be the result of the `store_graph` tool call.
"""

agent = Agent(
    name="graph_storer",
    model="gemini-2.5-flash",
    instruction=GRAPH_STORER_PROMPT,
    tools=[store_graph],
)
