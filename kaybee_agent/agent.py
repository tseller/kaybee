import json
import os
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.planners import BuiltInPlanner
from google.genai import types
from google.cloud import logging as google_cloud_logging
from google.cloud import storage
from typing import Optional

from .knowledge_graph_tool import expand_query
from .knowledge_graph_agent import knowledge_graph_agent
from .prompt import get_prompt

# Load environment variables from .env file in root directory
root_dir = Path(__file__).parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Use default project from credentials if not in .env
_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

logging_client = google_cloud_logging.Client()
logger = logging_client.logger("kaybee-agent")
storage_client = storage.Client()

KNOWLEDGE_GRAPH_BUCKET = storage_client.get_bucket(
        os.environ["KNOWLEDGE_GRAPH_BUCKET"])

def process_user_input(
        callback_context: CallbackContext) -> Optional[types.Content]:
    if text := callback_context.user_content.parts[-1].text:
        if kb_context := expand_query(text):
            callback_context.user_content.parts.append(kb_context)

root_agent = Agent(
    name="knowledge_base_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    instruction=get_prompt(),
    tools=[
        knowledge_graph_agent,
    ],
    before_agent_callback=process_user_input,
)
