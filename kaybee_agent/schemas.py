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
