from abc import ABC, abstractmethod

from pydantic import BaseModel

from query_service.llm_client import LLMClient
from query_service.models import QueryStrategy

# Heuristic knobs — tune/eval these against a labeled query set.
# Only genuinely trivial queries (e.g. "positional encoding") should skip
# the LLM classifier via word count alone — short queries can still be
# vague ("kinda", "best practices for X"), so keep this low.
SIMPLE_WORD_THRESHOLD = 3
MULTI_HOP_KEYWORDS = ("compare", " vs ", "difference between", " and why", " and how")

CLASSIFY_PROMPT = """Classify the query below into exactly one category:

- "passthrough": simple, single, unambiguous lookup — no transformation needed
- "rewrite": ambiguous or oddly phrased, but still a single intent
- "expand": clear single intent, but retrieval would benefit from synonym/
  related-term coverage
- "decompose": contains multiple distinct sub-questions, joined by "and",
  multiple question marks, or requiring separate pieces of evidence to
  answer fully — even if it reads as one grammatical sentence

Examples:

Query: What is backpropagation?
Category: passthrough

Query: How does attention differ from RNNs, and why did transformers replace them?
Category: decompose
(reason: two distinct asks — a comparison, and a separate causal question)

Query: What's the deal with that transformer thing everyone talks about?
Category: rewrite
(reason: single intent, but vague phrasing needs clarifying)

Query: Best practices for model evaluation
Category: expand
(reason: single intent, but broad — synonyms/related terms help recall)

Query: {query}

Return your answer as JSON in exactly this form: {{"strategy": "<category>"}}
Do not return anything else — no explanation, no bare word, only that JSON object."""


class _ClassificationOutput(BaseModel):
    strategy: QueryStrategy


class RouterBase(ABC):
    @abstractmethod
    async def decide(self, query: str) -> QueryStrategy:
        ...


class HeuristicRouter(RouterBase):
    """Free, no LLM call. Good first pass / fallback."""

    async def decide(self, query: str) -> QueryStrategy:
        lowered = query.lower()
        if any(kw in lowered for kw in MULTI_HOP_KEYWORDS):
            return QueryStrategy.DECOMPOSE
        if len(query.split()) <= SIMPLE_WORD_THRESHOLD:
            return QueryStrategy.PASSTHROUGH
        return QueryStrategy.REWRITE


class LLMRouter(RouterBase):
    """One small/fast LLM call to classify intent."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def decide(self, query: str) -> QueryStrategy:
        result = await self.llm_client.generate_structured(
            prompt=CLASSIFY_PROMPT.format(query=query),
            schema=_ClassificationOutput,
        )
        return result.strategy


class HybridRouter(RouterBase):
    """Heuristic pre-filter for obvious cases, LLM router for the rest.
    Recommended default — cuts LLM router calls for cheap, easy queries."""

    def __init__(self, heuristic: HeuristicRouter, llm_router: LLMRouter):
        self.heuristic = heuristic
        self.llm_router = llm_router

    async def decide(self, query: str) -> QueryStrategy:
        lowered = query.lower()
        # Free, zero-latency catch for obvious multi-hop signals —
        # skip the LLM call entirely when we already know the answer.
        if any(kw in lowered for kw in MULTI_HOP_KEYWORDS):
            return QueryStrategy.DECOMPOSE
        if len(query.split()) <= SIMPLE_WORD_THRESHOLD:
            return QueryStrategy.PASSTHROUGH
        return await self.llm_router.decide(query)