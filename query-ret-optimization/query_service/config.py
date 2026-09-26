from query_service.llm_client import LLMClient
from query_service.models import QueryStrategy
from query_service.pipeline import QueryPipeline
from query_service.router import HeuristicRouter, LLMRouter, HybridRouter
from query_service.strategies.decompose import DecomposeStrategy
from query_service.strategies.expand import ExpandStrategy
from query_service.strategies.passthrough import PassthroughStrategy
from query_service.strategies.rewrite import RewriteStrategy


# Max simultaneous LLM calls inside RewriteStrategy.run_many.
# 1 = sequential: spreads TPM over time, minimal 429 risk, same quality
# (calls are independent). Raise to 2 if p99 latency matters more.
REWRITE_CONCURRENCY = 1


# Default chains per router entry point. Decompose gets a rewrite polish
# pass (pronoun-resolved AND clarified). Rewrite/expand stay single-step
# by default to bound LLM cost — force an explicit chain like
# ["rewrite", "expand"] or ["decompose", "rewrite", "expand"] to test more.
DEFAULT_CHAINS: dict[QueryStrategy, list[QueryStrategy]] = {
    QueryStrategy.PASSTHROUGH: [QueryStrategy.PASSTHROUGH],
    QueryStrategy.REWRITE: [QueryStrategy.REWRITE],
    QueryStrategy.EXPAND: [QueryStrategy.EXPAND],
    QueryStrategy.DECOMPOSE: [QueryStrategy.DECOMPOSE, QueryStrategy.REWRITE],
}


def build_default_pipeline() -> QueryPipeline:
    """Single place that wires concrete implementations together.
    Swap routers/strategies here without touching pipeline.py or api.py."""

    # cheap/fast model — router classification, rewrite, expand
    fast_client = LLMClient(model="openai/gpt-oss-20b")
    # stronger model — decompose needs more reasoning to split well.
    # Using the same model for now since it's confirmed working on this
    # account; swap to "openai/gpt-oss-120b" here if it shows up in
    # `curl https://api.groq.com/openai/v1/models`.
    strong_client = LLMClient(model="openai/gpt-oss-20b")

    strategies = {
        QueryStrategy.PASSTHROUGH: PassthroughStrategy(),
        QueryStrategy.REWRITE: RewriteStrategy(fast_client, max_concurrency=REWRITE_CONCURRENCY),
        QueryStrategy.EXPAND: ExpandStrategy(fast_client),
        QueryStrategy.DECOMPOSE: DecomposeStrategy(strong_client),
    }

    router = HybridRouter(
        heuristic=HeuristicRouter(),
        llm_router=LLMRouter(fast_client),
    )

    return QueryPipeline(router=router, strategies=strategies, chain_map=DEFAULT_CHAINS)
