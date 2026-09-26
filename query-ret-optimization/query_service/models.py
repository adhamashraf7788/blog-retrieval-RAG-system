"""
Public data contract for the query service.

This module is the ONLY thing other teams/services should ever depend on.
Internals (strategies, router, LLM client) can change freely as long as
these shapes stay stable.
"""

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class QueryStrategy(str, Enum):
    PASSTHROUGH = "passthrough"
    REWRITE = "rewrite"
    EXPAND = "expand"
    DECOMPOSE = "decompose"


class QueryRequest(BaseModel):
    query: str
    # Explicit chain to run, in order — e.g. ["decompose", "rewrite"].
    # If set, the router is skipped entirely. Use for testing/eval.
    force_strategies: list[QueryStrategy] | None = None
    # Deprecated single-strategy escape hatch — kept for backward compat.
    # If both are set, force_strategies wins.
    force_strategy: QueryStrategy | None = None


class ProcessedQuery(BaseModel):
    """
    The service's entire deliverable. Retrieval only needs `.queries`.
    Everything else is optional context — never required for correctness.
    """

    original_query: str
    strategies_used: list[QueryStrategy] = Field(
        description="Ordered chain that was applied, e.g. ['decompose', 'rewrite']."
    )
    # Deprecated alias for the last strategy in the chain — kept so old
    # retrieval/eval code reading `.strategy_used` keeps working.
    strategy_used: QueryStrategy | None = Field(
        default=None,
        description="Deprecated: equals strategies_used[-1]. Prefer strategies_used.",
    )
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

    @model_validator(mode="after")
    def _fill_strategy_used(self):
        # Keep the deprecated alias in sync so old readers keep working.
        if self.strategies_used and self.strategy_used is None:
            self.strategy_used = self.strategies_used[-1]
        return self
