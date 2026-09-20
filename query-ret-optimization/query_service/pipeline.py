from query_service.models import ProcessedQuery, QueryRequest, QueryStrategy
from query_service.router import RouterBase
from query_service.strategies.base import QueryStrategyBase


class QueryPipeline:
    """
    The one object other code should import and call — either directly
    as a library (same process) or wrapped behind FastAPI (api.py) for
    out-of-process callers. No retrieval, no merging, no I/O beyond the
    LLM calls made inside strategies.
    """

    def __init__(
        self,
        router: RouterBase,
        strategies: dict[QueryStrategy, QueryStrategyBase],
    ):
        self.router = router
        self.strategies = strategies

    async def process(self, request: QueryRequest) -> ProcessedQuery:
        strategy_name = request.force_strategy or await self.router.decide(
            request.query
        )
        strategy = self.strategies[strategy_name]
        return await strategy.run(request.query)
