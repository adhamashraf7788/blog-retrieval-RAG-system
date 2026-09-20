from abc import ABC, abstractmethod

from query_service.models import ProcessedQuery, QueryStrategy


class QueryStrategyBase(ABC):
    """
    Every strategy (rewrite, expand, decompose, future reasoning-based
    decomposition, ...) implements this. The router and pipeline only
    ever depend on this interface, never on a concrete strategy class.
    """

    name: QueryStrategy

    @abstractmethod
    async def run(self, query: str) -> ProcessedQuery:
        ...
