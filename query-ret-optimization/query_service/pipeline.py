from query_service.models import ProcessedQuery, QueryRequest, QueryStrategy
from query_service.router import RouterBase
from query_service.strategies.base import QueryStrategyBase


class QueryPipeline:
    """
    The one object other code should import and call — either directly
    as a library (same process) or wrapped behind FastAPI (api.py) for
    out-of-process callers. No retrieval, no merging, no I/O beyond the
    LLM calls made inside strategies.

    Runs an ordered chain of strategies: the output list of one feeds
    the next. The router picks the entry point; `chain_map` expands it
    to a full chain (e.g. decompose -> [decompose, rewrite]).
    """

    def __init__(
        self,
        router: RouterBase,
        strategies: dict[QueryStrategy, QueryStrategyBase],
        chain_map: dict[QueryStrategy, list[QueryStrategy]] | None = None,
        max_final_queries: int = 8,
    ):
        self.router = router
        self.strategies = strategies
        # Default chains: decompose gets a rewrite polish pass so its
        # sub-questions are self-contained AND clear. Everything else
        # runs alone unless explicitly forced — keeps LLM cost bounded.
        self.chain_map = chain_map or {
            QueryStrategy.PASSTHROUGH: [QueryStrategy.PASSTHROUGH],
            QueryStrategy.REWRITE: [QueryStrategy.REWRITE],
            QueryStrategy.EXPAND: [QueryStrategy.EXPAND],
            QueryStrategy.DECOMPOSE: [QueryStrategy.DECOMPOSE, QueryStrategy.REWRITE],
        }
        self.max_final_queries = max_final_queries

    def resolve_chain(self, request: QueryRequest, entry: QueryStrategy | None = None) -> list[QueryStrategy]:
        # Explicit chain wins (testing/eval). Deprecated single force
        # is respected as a one-element chain (router skipped).
        if request.force_strategies:
            return list(request.force_strategies)
        if request.force_strategy:
            return [request.force_strategy]
        assert entry is not None
        return list(self.chain_map.get(entry, [entry]))

    async def process(self, request: QueryRequest) -> ProcessedQuery:
        if request.force_strategies or request.force_strategy:
            chain = self.resolve_chain(request)
        else:
            entry = await self.router.decide(request.query)
            chain = self.resolve_chain(request, entry)

        queries: list[str] = [request.query]
        for step in chain:
            strategy = self.strategies[step]
            queries = await strategy.run_many(queries)
            if len(queries) > self.max_final_queries:
                queries = queries[: self.max_final_queries]
                break

        return ProcessedQuery(
            original_query=request.query,
            strategies_used=chain,
            strategy_used=chain[-1],
            queries=queries,
            metadata={"chain": [s.value for s in chain]},
        )
