import asyncio

from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import QueryStrategy
from query_service.strategies.base import QueryStrategyBase

EXPAND_PROMPT = """Generate {n} alternative phrasings of the query below, \
covering likely synonyms and related terminology, to improve recall in a \
retrieval system.

Query: {query}

Return only the alternative phrasings."""


class _ExpandOutput(BaseModel):
    expanded_queries: list[str]


class ExpandStrategy(QueryStrategyBase):
    name = QueryStrategy.EXPAND

    def __init__(self, llm_client: LLMClient, n_expansions: int = 3):
        self.llm_client = llm_client
        self.n_expansions = n_expansions

    async def _expand_one(self, query: str) -> list[str]:
        result = await self.llm_client.generate_structured(
            prompt=EXPAND_PROMPT.format(query=query, n=self.n_expansions),
            schema=_ExpandOutput,
        )
        # Original query stays in the set — expansions supplement, not replace.
        return [query] + result.expanded_queries

    async def run_many(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        expanded = await asyncio.gather(*(self._expand_one(q) for q in queries))
        # Flatten, preserving per-query order.
        return [q for group in expanded for q in group]
