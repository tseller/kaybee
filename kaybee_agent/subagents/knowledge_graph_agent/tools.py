import json
import os
import networkx as nx
from dotenv import load_dotenv
from floggit import flog
from typing import Optional

from google.adk.tools import ToolContext
from google.genai import types
from google.cloud import storage
from thefuzz import fuzz

from kaybee_agent.schemas import Entity, RelationshipIdentifier

load_dotenv()

storage_client = storage.Client()

def _get_bucket():
    bucket_name = os.environ.get("KNOWLEDGE_GRAPH_BUCKET")
    if not bucket_name:
        raise ValueError("KNOWLEDGE_GRAPH_BUCKET environment variable not set.")
    return storage_client.get_bucket(bucket_name)


def _fetch_knowledge_graph(graph_id: str) -> dict:
    """Fetches the knowledge graph from the Google Cloud Storage bucket."""
    bucket = _get_bucket()
    blob = bucket.blob(f"{graph_id}.json")
    if not blob.exists():
        return {"entities": {}, "relationships": []}
    else:
        content = blob.download_as_text()
        return json.loads(content)


def _store_knowledge_graph(knowledge_graph: dict, graph_id: str) -> None:
    """Stores the knowledge graph in the Google Cloud Storage bucket."""
    bucket = _get_bucket()
    blob = bucket.blob(f"{graph_id}.json")
    blob.upload_from_string(
        json.dumps(knowledge_graph, indent=2), content_type="application/json"
    )


def _knowledge_graph_to_nx(g: dict) -> "nx.MultiDiGraph":
    """Converts the knowledge graph dictionary to a NetworkX MultiDiGraph."""
    mdg = nx.MultiDiGraph()
    mdg.add_nodes_from((k, v) for k, v in g["entities"].items())
    mdg.add_edges_from(
        (
            rel["source_entity_id"],
            rel["target_entity_id"],
            {"relationship": rel["relationship"]},
        )
        for rel in g.get("relationships", [])
    )
    return mdg


def _find_entity_ids_by_name(
    entity_name: str, g: dict, threshold: int = 80
) -> list[str]:
    """Finds an entity by its name or one of its synonyms using fuzzy string matching."""
    return [
        entity_id
        for entity_id, entity_data in g["entities"].items()
        for name in entity_data["entity_names"]
        if fuzz.ratio(entity_name.lower(), name.lower()) > threshold
    ]


@flog
def get_relevant_neighborhoods(entity_names: list[str], tool_context: ToolContext) -> str:
    """
    Retrieves the neighborhoods of all potentially relevant existing entities and relationships.
    """
    graph_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=graph_id)

    relevant_entity_ids = set().union(*[
        _find_entity_ids_by_name(entity_name, g)
        for entity_name in entity_names
    ])
    mdg = _knowledge_graph_to_nx(g)

    nbrs1 = {
            nbr for entity_id in relevant_entity_ids
            for nbr in mdg.to_undirected().neighbors(entity_id)
    }
    nbrs2 = {
            nbr for entity_id in nbrs1
            for nbr in mdg.to_undirected().neighbors(entity_id)
    }

    subgraph = mdg.subgraph(relevant_entity_ids|nbrs1|nbrs2)
    j = nx.node_link_data(subgraph, edges="links")

    return {
        'entities': j['nodes'],
        'relationships': j['links']
    }

@flog
def store_graph(graph: dict, tool_context: ToolContext) -> str:
    """
    Stores the provided graph in the knowledge graph store.
    This will overwrite the existing graph.
    """
    graph_id = tool_context._invocation_context.user_id
    _store_knowledge_graph(knowledge_graph=graph, graph_id=graph_id)
    return "Graph stored successfully."
