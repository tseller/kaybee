from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .schemas import KnowledgeGraph
from .tools import store_graph

PROMPT = """
You are a specialized agent that updates knowledge graphs.

Given a knowledge graph and a list of updates, your task is to produce an updated version of the knowledge graph.

Here's the knowledge graph that should be updated:

    {existing_knowledge}

Here are updates that need to be applied to the existing knowledge:

    {knowledge_updates}

The resulting knowledge graph must:
-   **Include all the new knowledge** from the updates.
-   **Preserve existing knowledge** to the extent it is not updated by new knowledge.

You must output the final, merged graph as a `KnowledgeGraph` object.
"""

def check_for_updates(callback_context: CallbackContext) -> Optional[types.Content]:
    if not callback_context.state['knowledge_updates']['knowledge']:
        # Return Content to skip the agent's run
        return types.Content(
            parts=[types.Part(text=f"Agent {callback_context.agent_name} skipped by before_agent_callback due to state.")],
            role="model" # Assign model role to the overriding response
        )
    else:
        # Return None to allow the LlmAgent's normal execution
        return None

agent = Agent(
    name="merge_knowledge_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=1024,
        )
    ),
    instruction=PROMPT,
    output_schema=KnowledgeGraph,
    before_agent_callback=check_for_updates,
    after_model_callback=store_graph
)
