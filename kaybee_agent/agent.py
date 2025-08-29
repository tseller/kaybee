import json
import os
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.planners import BuiltInPlanner
from google.genai import types
from typing import Optional

from .subagents.knowledge_graph_agent import agent as knowledge_graph_agent
from .prompt import get_prompt

def setup_environment():
    # Load environment variables from .env file in root directory
    root_dir = Path(__file__).parent.parent
    dotenv_path = root_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    # Use default project from credentials if not in .env
    try:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    except google.auth.exceptions.DefaultCredentialsError:
        # This will happen in the test environment.
        # The tests will set the required environment variables.
        pass

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
    sub_agents=[
        knowledge_graph_agent
    ],
)
