import json
import os
import uuid
import networkx as nx
from dotenv import load_dotenv
from floggit import flog
from pydantic import BaseModel, Field
from typing import Optional
from thefuzz import fuzz

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

def fetch_knowledge_graph() -> dict:
    """Fetches the knowledge graph from the Google Cloud Storage bucket.

    Returns:
        dict: The knowledge graph as a dictionary.
    """
    blob = KNOWLEDGE_GRAPH_BUCKET.blob("knowledge_graph.json")
    if not blob.exists():
        return {'entities': {}, 'relationships': []}
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
    blob.upload_from_string(json.dumps(knowledge_graph, indent=2), content_type='application/json')


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
    for e1, e2, edge_labels in mdg.out_edges(list(relevant_entities.keys()), data=True):
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

def _find_entity_id_by_name(entity_name: str, g: dict, threshold: int = 80) -> Optional[str]:
    """Finds an entity by its name or one of its synonyms using fuzzy string matching."""
    best_match_score = 0
    best_match_id = None

    for entity_id, entity_data in g['entities'].items():
        for name in entity_data['entity_names']:
            score = fuzz.ratio(entity_name.lower(), name.lower())
            if score > best_match_score:
                best_match_score = score
                best_match_id = entity_id

    if best_match_score >= threshold:
        return best_match_id
    else:
        return None

@flog
def add_entity(entity_names: list[str]) -> str:
    """Adds a new entity to the knowledge graph.

    Args:
        entity_names: A list of names for the entity, where the first name is the primary name.

    Returns:
        A message indicating success or failure.
    """
    if not entity_names:
        return "Error: At least one name must be provided for the entity."

    g = fetch_knowledge_graph()

    for name in entity_names:
        if _find_entity_id_by_name(name, g):
            return f"Error: An entity with the name '{name}' already exists."

    new_entity_id = str(uuid.uuid4())
    g['entities'][new_entity_id] = {
        'entity_id': new_entity_id,
        'entity_names': entity_names
    }
    store_knowledge_graph(g)
    return f"Entity '{entity_names[0]}' added successfully."


@flog
def add_synonyms(entity_id: str, synonyms: list[str]) -> str:
    """Adds synonyms to an existing entity.

    Args:
        entity_id: The ID of the entity to add synonyms to.
        synonyms: A list of synonyms to add.

    Returns:
        A message indicating success or failure.
    """
    g = fetch_knowledge_graph()
    if entity_id not in g['entities']:
        return f"Error: Entity with ID '{entity_id}' not found."

    for s in synonyms:
        if _find_entity_id_by_name(s, g):
              return f"Error: An entity with the name '{s}' already exists, so it cannot be a synonym."

    entity = g['entities'][entity_id]
    existing_synonyms = set(name.lower() for name in entity['entity_names'])
    for synonym in synonyms:
        if synonym.lower() not in existing_synonyms:
            entity['entity_names'].append(synonym)

    store_knowledge_graph(g)
    primary_name = g['entities'][entity_id]['entity_names'][0]
    return f"Synonyms added successfully to entity '{primary_name}'."


@flog
def remove_synonyms(entity_id: str, synonyms: list[str]) -> str:
    """Removes synonyms from an entity.

    Args:
        entity_id: The ID of the entity to remove synonyms from.
        synonyms: The list of synonyms to remove.

    Returns:
        A message indicating success or failure.
    """
    g = fetch_knowledge_graph()
    if entity_id not in g['entities']:
        return f"Error: Entity with ID '{entity_id}' not found."

    entity = g['entities'][entity_id]
    original_names = entity['entity_names']
    primary_name = original_names[0]

    # Create a lowercase version of synonyms for case-insensitive comparison
    lower_synonyms_to_remove = {s.lower() for s in synonyms}

    new_names = [name for name in original_names if name.lower() not in lower_synonyms_to_remove]

    if not new_names:
        return f"Error: Cannot remove all names from entity '{primary_name}'. An entity must have at least one name."

    g['entities'][entity_id]['entity_names'] = new_names
    store_knowledge_graph(g)
    return f"Synonyms removed successfully from entity '{primary_name}'."


@flog
def add_relationship(source_entity_id: str, relationship: str, target_entity_id: str) -> str:
    """Adds a relationship between two entities.

    Args:
        source_entity_id: The ID of the source entity.
        relationship: The description of the relationship.
        target_entity_id: The ID of the target entity.

    Returns:
        A message indicating success or failure.
    """
    g = fetch_knowledge_graph()
    if source_entity_id not in g['entities']:
        return f"Error: Source entity with ID '{source_entity_id}' not found."
    if target_entity_id not in g['entities']:
        return f"Error: Target entity with ID '{target_entity_id}' not found."

    source_primary_name = g['entities'][source_entity_id]['entity_names'][0]
    target_primary_name = g['entities'][target_entity_id]['entity_names'][0]

    # Check if the relationship already exists
    for rel in g.get('relationships', []):
        if (rel['source_entity_id'] == source_entity_id and
            rel['target_entity_id'] == target_entity_id and
            rel['relationship'].lower() == relationship.lower()):
            return f"Relationship '{source_primary_name} -> {relationship} -> {target_primary_name}' already exists."


    g.setdefault('relationships', []).append({
        'source_entity_id': source_entity_id,
        'target_entity_id': target_entity_id,
        'relationship': relationship
    })
    store_knowledge_graph(g)
    return f"Relationship '{source_primary_name} -> {relationship} -> {target_primary_name}' added successfully."

@flog
def remove_relationship(source_entity_id: str, relationship: str, target_entity_id: str) -> str:
    """Removes a relationship between two entities.

    Args:
        source_entity_id: The ID of the source entity.
        relationship: The description of the relationship.
        target_entity_id: The ID of the target entity.

    Returns:
        A message indicating success or failure.
    """
    g = fetch_knowledge_graph()
    if source_entity_id not in g['entities']:
        return f"Error: Source entity with ID '{source_entity_id}' not found."
    if target_entity_id not in g['entities']:
        return f"Error: Target entity with ID '{target_entity_id}' not found."

    source_primary_name = g['entities'][source_entity_id]['entity_names'][0]
    target_primary_name = g['entities'][target_entity_id]['entity_names'][0]

    initial_rel_count = len(g.get('relationships', []))
    g['relationships'] = [
        rel for rel in g.get('relationships', [])
        if not (rel['source_entity_id'] == source_entity_id and
                rel['target_entity_id'] == target_entity_id and
                rel['relationship'].lower() == relationship.lower())
    ]

    if len(g.get('relationships', [])) == initial_rel_count:
        return f"Error: Relationship '{source_primary_name} -> {relationship} -> {target_primary_name}' not found."

    store_knowledge_graph(g)
    return f"Relationship '{source_primary_name} -> {relationship} -> {target_primary_name}' removed successfully."


@flog
def delete_entity(entity_id: str) -> str:
    """Deletes an entity and all its relationships from the knowledge graph.

    Args:
        entity_id: The ID of the entity to delete.

    Returns:
        A message indicating success or failure.
    """
    g = fetch_knowledge_graph()
    if entity_id not in g['entities']:
        return f"Error: Entity with ID '{entity_id}' not found."

    primary_name = g['entities'][entity_id]['entity_names'][0]

    # Delete the entity
    del g['entities'][entity_id]

    # Delete all relationships to or from this entity
    if 'relationships' in g:
        g['relationships'] = [
            rel for rel in g['relationships']
            if rel['source_entity_id'] != entity_id and rel['target_entity_id'] != entity_id
        ]

    store_knowledge_graph(g)
    return f"Entity '{primary_name}' and its relationships deleted successfully."


@flog
def get_entity_neighborhood(entity_names: list[str]) -> str:
    """
    Retrieves the neighborhood of given entities as a JSON subgraph.

    Args:
        entity_names: A list of names of the entities to look up.

    Returns:
        A JSON string representing the combined neighborhood of the entities as a subgraph.
        If an entity is not found, it is simply omitted from the subgraph.
    """
    g = fetch_knowledge_graph()

    subgraph = {
        'entities': {},
        'relationships': []
    }

    entity_ids_to_process = set()

    for entity_name in entity_names:
        entity_id = _find_entity_id_by_name(entity_name, g)
        if entity_id:
            entity_ids_to_process.add(entity_id)
            subgraph['entities'][entity_id] = g['entities'][entity_id]

    if not entity_ids_to_process:
        return json.dumps(subgraph, indent=2)

    if 'relationships' in g:
        for rel in g['relationships']:
            if rel['source_entity_id'] in entity_ids_to_process or rel['target_entity_id'] in entity_ids_to_process:
                subgraph['relationships'].append(rel)

                # Add neighbor entities to the subgraph
                source_id = rel['source_entity_id']
                target_id = rel['target_entity_id']
                if source_id in g['entities'] and source_id not in subgraph['entities']:
                    subgraph['entities'][source_id] = g['entities'][source_id]
                if target_id in g['entities'] and target_id not in subgraph['entities']:
                    subgraph['entities'][target_id] = g['entities'][target_id]

    return json.dumps(subgraph, indent=2)

