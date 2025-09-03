from pydantic import BaseModel, Field
from typing import Optional, Any


class Relationship(BaseModel):
    """A relationship between two entities."""
    source_entity_id: str = Field(
        ...,
        description="The name of the source entity."
    )
    target_entity_id: str = Field(
        ...,
        description="The name of the target entity."
    )
    relationship: str = Field(
        ...,
        description="The description of the relationship."
    )


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""
    entity_id: str = Field(
        ...,
        description="The ID of the entity to update."
    )
    entity_names: list[str] = Field(
        ...,
        description="A list of names for the entity, with the first being the primary name."
    )
    properties: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of properties for the entity."
    )

class KnowledgeGraph(BaseModel):
    """Represents a knowledge graph with entities and relationships."""
    entities: list[Entity]
    relationships: list[Relationship]

class StoreResult(BaseModel):
    """The result of storing a graph."""
    message: str
