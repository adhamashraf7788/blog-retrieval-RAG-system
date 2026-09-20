from query_service.models import ProcessedQuery, QueryStrategy
from query_service.strategies.base import QueryStrategyBase


class PassthroughStrategy(QueryStrategyBase):
    """No transformation — used when the router decides the query is
    already retrieval-ready as-is."""

    name = QueryStrategy.PASSTHROUGH

    async def run(self, query: str) -> ProcessedQuery:
        return ProcessedQuery(
            original_query=query,
            strategy_used=self.name,
            queries=[query],
        )
