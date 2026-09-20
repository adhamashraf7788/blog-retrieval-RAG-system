from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import ProcessedQuery, QueryStrategy
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

    async def run(self, query: str) -> ProcessedQuery:
        result = await self.llm_client.generate_structured(
            prompt=EXPAND_PROMPT.format(query=query, n=self.n_expansions),
            schema=_ExpandOutput,
        )
        # Original query stays in the set — expansions supplement, not replace.
        all_queries = [query] + result.expanded_queries
        return ProcessedQuery(
            original_query=query,
            strategy_used=self.name,
            queries=all_queries,
        )
