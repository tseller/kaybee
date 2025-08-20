import json
import os
import uuid
import networkx as nx
from dotenv import load_dotenv
from floggit import flog

from google.genai import types
from google.cloud import storage

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
def expand_query(query: str) -> types.Content:
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

    if relevant_entities:

        return types.Content(
            parts=[types.Part(
                text=f"For context, {relevant_entities_str}\n{relationships_str}"
            )],
            role='user'
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

from pydantic import BaseModel, Field
class EntityRelationship(BaseModel):
    target_entity: str = Field(..., description="The name of the related entity")
    relationship: str = Field(..., description="The type of relationship (e.g., 'is a', 'part of', etc.)")
class KnowledgeGraphEntity(BaseModel):
    name: str = Field(description="The name of the entity")
    synonyms: list[str] = Field(
            description="A list of synonyms or alternate spellings for the entity",
            default=[],)
    relationships: list[EntityRelationship] = Field(
            description="A list of relationships from this entity to others",
            default=[])


@flog
def update_knowledge(entity: KnowledgeGraphEntity):
    """Updates the knowledge base with new information.

    Args:
        entity: A new or updated entity, with its synonyms and relationships to other entities.

    Returns:
        None
    """
    @flog
    def _get_entity_id(entity_names: list[str]) -> str:
        # Get existing entity_id, if it exists
        entity_id = None
        for k,v in g['entities'].items():
            # Check if any of the entity names intersect with existing entity names
            # This allows for synonyms or alternate names to match
            if entity_names.intersection(v['entity_names']):
                entity_id = v['entity_id']
                break
        if entity_id is None:
            entity_id = str(uuid.uuid4())

        return entity_id

    @flog
    def _upsert_entity(entity):
        # Create or update the entity in the knowledge graph
        entity_names = set([entity['name']] + entity['synonyms'])
        entity_id = _get_entity_id(entity_names)
        g['entities'][entity_id] = {
            'entity_id': entity_id,
            'entity_names': list(entity_names)
        }

        return entity_id

    g = fetch_knowledge_graph()
    entity_id = _upsert_entity(entity)

    # Delete existing relationships for this entity
    g['relationships'] = [
            relationship for relationship in g.get('relationships',[])
            if relationship['source_entity_id'] != entity_id
    ]
    # Add updated relationships
    for relationship in entity.get('relationships', []):
        target_entity_id = _upsert_entity(
                KnowledgeGraphEntity(name=relationship['target_entity']))
        g['relationships'].append({
            'source_entity_id': entity_id,
            'target_entity_id': target_entity_id,
            'relationship': relationship['relationship']
        })

    store_knowledge_graph(g)
