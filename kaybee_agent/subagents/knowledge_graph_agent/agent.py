from google.adk.agents import SequentialAgent, ParallelAgent

from .subagents.new_knowledge_agent import agent as new_knowledge_agent
from .subagents.existing_knowledge_agent import agent as existing_knowledge_agent
from .subagents.merge_knowledge_agent import agent as merge_knowledge_agent

new_and_existing_knowledge_agent = ParallelAgent(
    name='new_and_existing_knowledge_agent',
    sub_agents=[
        new_knowledge_agent,
        existing_knowledge_agent,
    ]
)

root_agent = SequentialAgent(
    name="knowledge_graph_agent",
    description='This agent updates the knowledge graph when potentially new information is encountered.',
    sub_agents=[
        new_and_existing_knowledge_agent,
        merge_knowledge_agent,
    ],
)
