from query_service.llm_client import LLMClient
from query_service.models import QueryStrategy
from query_service.pipeline import QueryPipeline
from query_service.router import HeuristicRouter, LLMRouter, HybridRouter
from query_service.strategies.decompose import DecomposeStrategy
from query_service.strategies.expand import ExpandStrategy
from query_service.strategies.passthrough import PassthroughStrategy
from query_service.strategies.rewrite import RewriteStrategy


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
        QueryStrategy.REWRITE: RewriteStrategy(fast_client),
        QueryStrategy.EXPAND: ExpandStrategy(fast_client),
        QueryStrategy.DECOMPOSE: DecomposeStrategy(strong_client),
    }

    router = HybridRouter(
        heuristic=HeuristicRouter(),
        llm_router=LLMRouter(fast_client),
    )

    return QueryPipeline(router=router, strategies=strategies)