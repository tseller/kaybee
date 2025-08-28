import json
import os
import uuid
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
KNOWLEDGE_GRAPH_BUCKET = storage_client.get_bucket(
    os.environ["KNOWLEDGE_GRAPH_BUCKET"]
)


def _fetch_knowledge_graph(graph_id: str) -> dict:
    """Fetches the knowledge graph from the Google Cloud Storage bucket."""
    blob = KNOWLEDGE_GRAPH_BUCKET.blob(f"{graph_id}.json")
    if not blob.exists():
        return {"entities": {}, "relationships": []}
    else:
        content = blob.download_as_text()
        return json.loads(content)


def _store_knowledge_graph(knowledge_graph: dict, graph_id: str) -> None:
    """Stores the knowledge graph in the Google Cloud Storage bucket."""
    blob = KNOWLEDGE_GRAPH_BUCKET.blob(f"{graph_id}.json")
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


def _find_entity_id_by_name(
    entity_name: str, g: dict, threshold: int = 80
) -> Optional[str]:
    """Finds an entity by its name or one of its synonyms using fuzzy string matching."""
    best_match_score = 0
    best_match_id = None

    for entity_id, entity_data in g["entities"].items():
        for name in entity_data["entity_names"]:
            score = fuzz.ratio(entity_name.lower(), name.lower())
            if score > best_match_score:
                best_match_score = score
                best_match_id = entity_id

    if best_match_score >= threshold:
        return best_match_id
    else:
        return None

def _find_entity_id_by_name_exact(entity_name: str, g: dict) -> Optional[str]:
    """Finds an entity by its name or one of its synonyms using exact string matching."""
    for entity_id, entity_data in g["entities"].items():
        if any(name.lower() == entity_name.lower() for name in entity_data["entity_names"]):
            return entity_id
    return None


@flog
def upsert_entities(entities: list[Entity], tool_context: ToolContext) -> str:
    """
    Upserts (updates or inserts) entities into the knowledge graph.
    - If an entity has an ID, it will be updated.
    - If an entity does not have an ID, it will be created.
    """
    graph_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=graph_id)

    # A map to resolve entity names to IDs, including newly created ones.
    entity_name_to_id_map = {
        name.lower(): entity_id
        for entity_id, entity_data in g["entities"].items()
        for name in entity_data["entity_names"]
    }

    for entity_data in entities:
        if hasattr(entity_data, "entity_id") and entity_data.entity_id:
            # This is an update to an existing entity
            entity_id = entity_data.entity_id
            if entity_id not in g['entities']:
                # This should not happen if the agent is working correctly
                continue

            # Update names
            existing_names = set(n.lower() for n in g["entities"][entity_id]["entity_names"])
            for name in entity_data.entity_names:
                if name.lower() not in existing_names:
                    g["entities"][entity_id]["entity_names"].append(name)
                    entity_name_to_id_map[name.lower()] = entity_id

            # Update properties
            if "properties" not in g["entities"][entity_id]:
                g["entities"][entity_id]["properties"] = {}
            g["entities"][entity_id]["properties"].update(entity_data.properties)

        else:
            # This is a new entity to be created
            # Check for exact name conflict first.
            primary_name = entity_data.entity_names[0]
            if _find_entity_id_by_name_exact(primary_name, g):
                # An entity with this name already exists.
                # The agent should have known this. We will skip this one.
                continue

            entity_id = str(uuid.uuid4())
            g["entities"][entity_id] = {
                "entity_id": entity_id,
                "entity_names": entity_data.entity_names,
                "properties": entity_data.properties or {},
            }
            for name in entity_data.entity_names:
                entity_name_to_id_map[name.lower()] = entity_id

        # Add relationships
        for rel in entity_data.relationships:
            target_entity_id = entity_name_to_id_map.get(rel.target_entity_name.lower())
            if not target_entity_id:
                target_entity_id = _find_entity_id_by_name_exact(rel.target_entity_name, g)

            if not target_entity_id:
                continue

            rel_exists = any(
                r["source_entity_id"] == entity_id
                and r["target_entity_id"] == target_entity_id
                and r["relationship"].lower() == rel.relationship.lower()
                for r in g.get("relationships", [])
            )
            if not rel_exists:
                g.setdefault("relationships", []).append(
                    {
                        "source_entity_id": entity_id,
                        "target_entity_id": target_entity_id,
                        "relationship": rel.relationship,
                    }
                )

    _store_knowledge_graph(knowledge_graph=g, graph_id=graph_id)
    return "Entities upserted successfully."


@flog
def remove_entities(entity_names: list[str], tool_context: ToolContext) -> str:
    """Removes entities and all their relationships from the knowledge graph."""
    graph_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=graph_id)
    ids_to_delete = {
        _find_entity_id_by_name_exact(name, g) for name in entity_names if _find_entity_id_by_name_exact(name, g)
    }

    if not ids_to_delete:
        return "No entities found with the given names."

    g["entities"] = {k: v for k, v in g["entities"].items() if k not in ids_to_delete}
    if "relationships" in g:
        g["relationships"] = [
            rel
            for rel in g["relationships"]
            if rel["source_entity_id"] not in ids_to_delete
            and rel["target_entity_id"] not in ids_to_delete
        ]

    _store_knowledge_graph(knowledge_graph=g, graph_id=graph_id)
    return "Entities and their relationships removed successfully."


@flog
def remove_relationships(
    relationships: list[RelationshipIdentifier], tool_context: ToolContext
) -> str:
    """Removes specified relationships from the knowledge graph."""
    graph_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=graph_id)

    rels_to_remove_set = set()
    for rel in relationships:
        source_id = _find_entity_id_by_name_exact(rel.source_entity_name, g)
        target_id = _find_entity_id_by_name_exact(rel.target_entity_name, g)
        if source_id and target_id:
            rels_to_remove_set.add((source_id, target_id, rel.relationship.lower()))

    if not rels_to_remove_set:
        return "No valid relationships to remove were identified."

    initial_rel_count = len(g.get("relationships", []))
    g["relationships"] = [
        r
        for r in g.get("relationships", [])
        if (
            r["source_entity_id"],
            r["target_entity_id"],
            r["relationship"].lower(),
        )
        not in rels_to_remove_set
    ]

    if len(g.get("relationships", [])) == initial_rel_count:
        return "No matching relationships were found to remove."

    _store_knowledge_graph(knowledge_graph=g, graph_id=graph_id)
    return "Relationships removed successfully."


@flog
def get_entity_neighborhood(entity_names: list[str], tool_context: ToolContext) -> str:
    """
    Retrieves the neighborhood of given entities as a JSON subgraph.
    """
    graph_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=graph_id)
    subgraph = {"entities": {}, "relationships": []}
    entity_ids_to_process = set()

    for entity_name in entity_names:
        entity_id = _find_entity_id_by_name(entity_name, g)
        if entity_id:
            entity_ids_to_process.add(entity_id)
            subgraph["entities"][entity_id] = g["entities"][entity_id]

    if not entity_ids_to_process:
        return json.dumps(subgraph, indent=2)

    if "relationships" in g:
        for rel in g["relationships"]:
            if (
                rel["source_entity_id"] in entity_ids_to_process
                or rel["target_entity_id"] in entity_ids_to_process
            ):
                subgraph["relationships"].append(rel)
                source_id = rel["source_entity_id"]
                target_id = rel["target_entity_id"]
                if source_id in g["entities"] and source_id not in subgraph["entities"]:
                    subgraph["entities"][source_id] = g["entities"][source_id]
                if target_id in g["entities"] and target_id not in subgraph["entities"]:
                    subgraph["entities"][target_id] = g["entities"][target_id]

    return json.dumps(subgraph, indent=2)


@flog
def expand_query(query: str, graph_id: str) -> Optional[types.Part]:
    """Expands a query by fetching related entities from the knowledge graph and appending them to the query."""
    g = _fetch_knowledge_graph(graph_id=graph_id)
    relevant_entities = {}
    for k, v in g["entities"].items():
        if any(name.lower() in query.lower() for name in v["entity_names"]):
            relevant_entities[k] = v["entity_names"]

    relevant_entities_str = ", ".join(
        f"{names[0]} is also known as: {', '.join(names[1:])}"
        for names in relevant_entities.values()
        if len(names) > 1
    )

    relationships = []
    mdg = _knowledge_graph_to_nx(g)
    for e1, e2, edge_labels in mdg.out_edges(list(relevant_entities.keys()), data=True):
        relationships.append(
            (
                mdg.nodes[e1]["entity_names"][0],
                edge_labels["relationship"],
                mdg.nodes[e2]["entity_names"][0],
            )
        )
    relationships_str = "\n".join(
        f"{e1} {relationship} {e2}" for e1, relationship, e2 in relationships
    )

    if relevant_entities_str or relationships_str:
        return types.Part(
            text=f"(FYI, according to the knowledge base: {relevant_entities_str}\n{relationships_str}.)"
        )
