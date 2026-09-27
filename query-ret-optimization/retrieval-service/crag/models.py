from enum import Enum

from pydantic import BaseModel, Field


class RetrievalDecision(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class Chunk(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    decision: RetrievalDecision
    reason: str