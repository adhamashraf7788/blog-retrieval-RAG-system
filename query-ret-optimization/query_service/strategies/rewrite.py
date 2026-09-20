from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import ProcessedQuery, QueryStrategy
from query_service.strategies.base import QueryStrategyBase

REWRITE_PROMPT = """You rewrite a user's search query to be clearer and more \
specific for a retrieval system, without changing its meaning.

Query: {query}

Return only the rewritten query, nothing else."""


class _RewriteOutput(BaseModel):
    rewritten_query: str


class RewriteStrategy(QueryStrategyBase):
    name = QueryStrategy.REWRITE

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def run(self, query: str) -> ProcessedQuery:
        result = await self.llm_client.generate_structured(
            prompt=REWRITE_PROMPT.format(query=query),
            schema=_RewriteOutput,
        )
        return ProcessedQuery(
            original_query=query,
            strategy_used=self.name,
            queries=[result.rewritten_query],
        )
