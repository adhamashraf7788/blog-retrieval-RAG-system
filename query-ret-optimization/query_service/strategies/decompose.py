from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import ProcessedQuery, QueryStrategy
from query_service.strategies.base import QueryStrategyBase

DECOMPOSE_PROMPT = """Break the query below into independent, self-contained \
sub-questions if it contains multiple distinct information needs. If it is \
already a single atomic question, return it unchanged as the only sub-query.

Query: {query}

Rules:
- Each sub-query must be retrievable on its own, with no shared context.
- Replace every pronoun or implicit reference (e.g. "them", "it", "this") \
with the actual noun it refers to, pulled from elsewhere in the original \
query. A sub-query must make sense to someone who has never seen the \
others.

Example:
Query: How does attention differ from RNNs, and why did transformers replace them?
Sub-queries: ["How does attention differ from RNNs?", "Why did transformers replace RNNs?"]
(note: "them" was resolved to "RNNs", not left as a pronoun)

Return the sub-queries."""


class _DecomposeOutput(BaseModel):
    sub_queries: list[str]


class DecomposeStrategy(QueryStrategyBase):
    """
    Placeholder for today's LLM-based decomposition. This slot is also
    where a future reasoning-model-based decomposition strategy gets
    swapped in later — same interface, no changes needed upstream.
    """

    name = QueryStrategy.DECOMPOSE

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def run(self, query: str) -> ProcessedQuery:
        result = await self.llm_client.generate_structured(
            prompt=DECOMPOSE_PROMPT.format(query=query),
            schema=_DecomposeOutput,
        )
        return ProcessedQuery(
            original_query=query,
            strategy_used=self.name,
            queries=result.sub_queries,
        )