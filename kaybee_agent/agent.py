import json
import os
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from google.cloud import logging as google_cloud_logging
from typing import Optional

from .subagents.knowledge_graph_agent.tools import expand_query
from .subagents.knowledge_graph_agent import agent as knowledge_graph_agent
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

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams, StdioServerParameters
def process_user_input(
        callback_context: CallbackContext) -> Optional[types.Content]:
    if text := callback_context.user_content.parts[-1].text:
        if kb_context := expand_query(text):
            callback_context.user_content.parts.append(kb_context)

root_agent = LlmAgent(
    model="gemini-1.5-flash",
    name='filesystem_assistant_agent',
    instruction='Help the user manage their files. You can list files, read files, etc.',
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params = StdioServerParameters(
                    command='npx',
                    args=[
                        "-y",  # Argument for npx to auto-confirm install
                        "@modelcontextprotocol/server-filesystem",
                        # IMPORTANT: This MUST be an ABSOLUTE path to a folder the
                        # npx process can access.
                        # Use a safe, dedicated folder instead of root.
                        str((Path(__file__).parent / "mcp_files").resolve()),
                    ],
                ),
                timeout=30,
            ),
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['list_directory', 'read_file']
        )
    ],
)
