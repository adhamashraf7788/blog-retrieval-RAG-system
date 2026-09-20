"""
Public data contract for the query service.

This module is the ONLY thing other teams/services should ever depend on.
Internals (strategies, router, LLM client) can change freely as long as
these shapes stay stable.
"""

from enum import Enum
from pydantic import BaseModel, Field


class QueryStrategy(str, Enum):
    PASSTHROUGH = "passthrough"
    REWRITE = "rewrite"
    EXPAND = "expand"
    DECOMPOSE = "decompose"


class QueryRequest(BaseModel):
    query: str
    # Optional escape hatch for testing/eval — force a specific strategy
    # instead of letting the router decide.
    force_strategy: QueryStrategy | None = None


class ProcessedQuery(BaseModel):
    """
    The service's entire deliverable. Retrieval only needs `.queries`.
    Everything else is optional context — never required for correctness.
    """

    original_query: str
    strategy_used: QueryStrategy
    queries: list[str] = Field(
        description="One or more retrieval-ready queries, in priority order."
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Optional hints for downstream consumers, e.g. "
            "{'weights': [0.7, 0.3], 'reasoning': '...'}. "
            "Retrieval must work correctly even if this is ignored entirely."
        ),
    )
