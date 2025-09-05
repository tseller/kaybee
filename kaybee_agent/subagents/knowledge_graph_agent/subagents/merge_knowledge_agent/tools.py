import json
import os
from dotenv import load_dotenv
from typing import Optional
import uuid

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.cloud import storage

load_dotenv()

def _get_bucket():
    storage_client = storage.Client()
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


def _reformat_graph(g: dict) -> dict:
    '''
    Args:
        g (dict): A knowledge graph as a dict, with g['entities'] as a list.

    Returns:
        dict: g, but with new entity IDs, and with g['entities'] now as a dict.'''

    id_mapping = {
            entity_id: str(uuid.uuid4())
            for entity_id in [
                entity['entity_id']
                for entity in g['entities']]
    }

    g['entities'] = {
            id_mapping[entity['entity_id']]: entity | {'entity_id': id_mapping[entity['entity_id']]}
            for entity in g['entities']
    }

    g['relationships'] = [
            rel | {
                'source_entity_id': id_mapping.get(rel['source_entity_id'], rel['source_entity_id']),
                'target_entity_id': id_mapping.get(rel['target_entity_id'], rel['target_entity_id'])
            }
            for rel in g['relationships']
    ]

    return g

def store_graph(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """
    Stores the provided graph in the knowledge graph store.
    This will overwrite the existing graph.
    """
    if llm_response.partial:
        return

    existing_knowledge_subgraph = callback_context.state['existing_knowledge']
    updated_knowledge_subgraph = json.loads(llm_response.content.parts[-1].text)
    updated_knowledge_subgraph = _reformat_graph(updated_knowledge_subgraph)

    graph_id = callback_context._invocation_context.user_id
    full_knowledge_graph = _fetch_knowledge_graph(graph_id)

    # Excise existing_knowledge_graph
    full_knowledge_graph['entities'] = {
            k: v
            for k, v in full_knowledge_graph['entities'].items()
            if k not in existing_knowledge_subgraph['entities']
    }
    full_knowledge_graph['relationships'] = [
            rel for rel in full_knowledge_graph['relationships']
            if (rel['source_entity_id'], rel['target_entity_id']) not in [
                (r['source_entity_id'], r['target_entity_id'])
                for r in existing_knowledge_subgraph['relationships']
            ]
    ]

    # Insert updated_knowledge_graph
    full_knowledge_graph['entities'].update(updated_knowledge_subgraph['entities'])
    full_knowledge_graph['relationships'].extend(updated_knowledge_subgraph['relationships'])

    _store_knowledge_graph(knowledge_graph=full_knowledge_graph, graph_id=graph_id)
