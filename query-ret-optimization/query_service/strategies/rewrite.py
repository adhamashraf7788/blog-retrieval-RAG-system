import asyncio

from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import QueryStrategy
from query_service.strategies.base import QueryStrategyBase

REWRITE_PROMPT = """You rewrite a user's search query to be clearer and more \
specific for a retrieval system, without changing its meaning.

Query: {query}

Return only the rewritten query, nothing else."""


class _RewriteOutput(BaseModel):
    rewritten_query: str


class RewriteStrategy(QueryStrategyBase):
    name = QueryStrategy.REWRITE

    def __init__(self, llm_client: LLMClient, max_concurrency: int = 1):
        self.llm_client = llm_client
        # Sequential by default (1): spreads TPM usage over time instead of
        # spiking the Groq bucket with N simultaneous calls. Each call is
        # independent (same prompt/model, temp=0), so quality is identical —
        # only timing changes. Raise to 2+ if p99 latency matters more.
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _rewrite_one(self, query: str) -> str:
        async with self._sem:
            result = await self.llm_client.generate_structured(
                prompt=REWRITE_PROMPT.format(query=query),
                schema=_RewriteOutput,
            )
        return result.rewritten_query

    async def run_many(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        # One LLM call per query; semaphore spaces them to avoid TPM spikes.
        return list(await asyncio.gather(*(self._rewrite_one(q) for q in queries)))
