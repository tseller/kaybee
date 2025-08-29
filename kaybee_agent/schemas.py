from pydantic import BaseModel, Field
from typing import Optional, Any


class Relationship(BaseModel):
    """A relationship between two entities."""
    source_entity_name: str = Field(
        ...,
        description="The name of the source entity."
    )
    target_entity_name: str = Field(
        ...,
        description="The name of the target entity."
    )
    relationship: str = Field(
        ...,
        description="The description of the relationship."
    )


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""
    entity_id: Optional[str] = Field(
        default=None,
        description="The ID of the entity to update. If not provided, a new entity will be created."
    )
    entity_names: list[str] = Field(
        ...,
        description="A list of names for the entity, with the first being the primary name."
    )
    properties: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of properties for the entity."
    )
    relationships: Optional[list[Relationship]] = Field(
        default_factory=list,
        description="A list of relationships for the entity."
    )

class RelationshipIdentifier(BaseModel):
    """Identifies a relationship to be removed."""
    source_entity_name: str = Field(..., description="The name of the source entity.")
    target_entity_name: str = Field(..., description="The name of the target entity.")
    relationship: str = Field(..., description="The description of the relationship.")

class KnowledgeGraph(BaseModel):
    """Represents a knowledge graph with entities and relationships."""
    entities: list[Entity]
    relationships: list[Relationship]

class StoreResult(BaseModel):
    """The result of storing a graph."""
    message: str
