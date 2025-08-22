import json
import os
import uuid
import networkx as nx
from dotenv import load_dotenv
from floggit import flog
from pydantic import BaseModel, Field
from typing import Optional

from google.genai import types
from google.cloud import storage

from .schemas import KnowledgeGraphEntity

# Load environment variables from .env file in root directory
#root_dir = Path(__file__).parent.parent
#dotenv_path = root_dir / ".env"
#load_dotenv(dotenv_path=dotenv_path)
load_dotenv()

storage_client = storage.Client()

KNOWLEDGE_GRAPH_BUCKET = storage_client.get_bucket(
        os.environ["KNOWLEDGE_GRAPH_BUCKET"])

@flog
def fetch_knowledge_graph() -> dict:
    """Fetches the knowledge graph from the Google Cloud Storage bucket.

    Returns:
        dict: The knowledge graph as a dictionary.
    """
    blob = KNOWLEDGE_GRAPH_BUCKET.blob("knowledge_graph.json")
    if not blob.exists():
        return {'entities': {}}
    else:
        content = blob.download_as_text()
        return json.loads(content)


@flog
def store_knowledge_graph(knowledge_graph: dict) -> None:
    """Stores the knowledge graph in the Google Cloud Storage bucket.

    Args:
        knowledge_graph (dict): The knowledge graph to be stored.
    """
    blob = KNOWLEDGE_GRAPH_BUCKET.blob("knowledge_graph.json")
    blob.upload_from_string(json.dumps(knowledge_graph), content_type='application/json')


@flog
def expand_query(query: str) -> Optional[types.Part]:
    """Expands a query by fetching related entities from the knowledge graph and appending them to the query."""

    g = fetch_knowledge_graph()

    # Find entities related to the query
    relevant_entities = {}
    for k, v in g['entities'].items():
        if any(name.lower() in query.lower() for name in v['entity_names']):
            relevant_entities[k] = v['entity_names']
    relevant_entities_str = ', '.join(
        f"{names[0]} is also known as: {', '.join(names[1:])}"
        for names in relevant_entities.values() if len(names) > 1
    )

    relationships = []
    mdg = knowledge_graph_to_nx(g)
    for e1, e2, edge_labels in mdg.out_edges(relevant_entities.keys(), data=True):
        relationships.append(
            (
                mdg.nodes[e1]['entity_names'][0],
                edge_labels['relationship'],
                mdg.nodes[e2]['entity_names'][0]
            )
        )
    relationships_str = '\n'.join(
        f"{e1} {relationship} {e2}"
        for e1, relationship, e2 in relationships
    )

    if relevant_entities_str or relationships_str:
        return types.Part(
            text=f"(FYI, according to the knowledge base: {relevant_entities_str}\n{relationships_str}.)"
        )


def knowledge_graph_to_nx(g: dict) -> 'nx.MultiDiGraph':
    """Converts the knowledge graph dictionary to a NetworkX MultiDiGraph."""
    mdg = nx.MultiDiGraph()
    mdg.add_nodes_from((k, v) for k, v in g['entities'].items())
    mdg.add_edges_from(
        (rel['source_entity_id'], rel['target_entity_id'], {'relationship': rel['relationship']})
        for rel in g.get('relationships', [])
    )

    return mdg

@flog
def delete_knowledge(entity_names: list[str]) -> None:
    """Deletes a set of entities from the knowledge graph.

    Args:
        entity_names: The name of the entities to be deleted.
    Returns:
        None
    """
    g = fetch_knowledge_graph()

    # Find the entity_id for the given entity name
    entity_ids = []
    for k, v in g['entities'].items():
        # Check if any of the entity names intersect with existing entity names
        # This allows for synonyms or alternate names to match
        if set(en.lower() for en in entity_names).intersection(
                en.lower() for en in v['entity_names']):
            entity_ids.append(v['entity_id'])

    g['entities'] = {
            k: v for k, v in g['entities'].items() if k not in entity_ids}
    g['relationships'] = [
        relationship for relationship in g.get('relationships', [])
        if relationship['source_entity_id'] not in entity_ids and
           relationship['target_entity_id'] not in entity_ids
    ]

    store_knowledge_graph(g)


@flog
def update_knowledge(entity: KnowledgeGraphEntity):
    """Adds/updates an entity, its synonyms, and its relationships to other entities.'''

    Args:
        entity: An entity, its synonyms, and its relationships to other entities.

    Returns:
        None
    """
    @flog
    def _get_entity_ids(entity_names) -> str:
        # Get existing entity_id, if it exists
        return [
            v['entity_id'] for k, v in g['entities'].items()
            if set(en.lower() for en in entity_names).intersection(
                en.lower() for en in v['entity_names'])
        ]

    g = fetch_knowledge_graph()
    entity_names = [entity['name']] + entity.get('synonyms', [])
    entity_ids = _get_entity_ids(entity_names)

    # Remove entity (if it exists)
    if entity_ids:
        g['entities'] = {
            k: v for k, v in g['entities'].items() if k not in entity_ids
        }

    # Remove existing relationships from this entity
    g['relationships'] = [
        relationship for relationship in g.get('relationships', [])
        if relationship['source_entity_id'] not in entity_ids
    ]

    # Add updated entity to the knowledge graph
    new_entity_id = str(uuid.uuid4())
    g['entities'][new_entity_id] = {
        'entity_id': new_entity_id,
        'entity_names': entity_names
    }

    # Add updated relationships _from_ this entity
    for relationship in entity.get('relationships', []):
        target_entity_ids = _get_entity_ids([relationship['target_entity']])
        if not target_entity_ids:
            new_target_entity_id = str(uuid.uuid4())
            target_entity_ids = [new_target_entity_id]
            g['entities'][new_target_entity_id] = {
                'entity_id': new_target_entity_id,
                'entity_names': [relationship['target_entity']]
            }
        for target_id in target_entity_ids:
            g['relationships'].append({
                'source_entity_id': new_entity_id,
                'target_entity_id': target_id,
                'relationship': relationship['relationship']
            })

    # Repair relationships _to_ this entity
    for relationship in g['relationships']:
        if relationship['target_entity_id'] in entity_ids:
            relationship['target_entity_id'] = new_entity_id

    store_knowledge_graph(g)
