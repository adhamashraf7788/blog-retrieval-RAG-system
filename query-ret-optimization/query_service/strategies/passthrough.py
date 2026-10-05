from query_service.models import QueryStrategy
from query_service.strategies.base import QueryStrategyBase


class PassthroughStrategy(QueryStrategyBase):
    """No transformation — used when the router decides the query is
    already retrieval-ready as-is."""

    name = QueryStrategy.PASSTHROUGH

    async def run_many(self, queries: list[str]) -> list[str]:
        return list(queries)
