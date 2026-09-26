from abc import ABC, abstractmethod

from query_service.models import QueryStrategy


class QueryStrategyBase(ABC):
    """
    Every strategy (rewrite, expand, decompose, future reasoning-based
    decomposition, ...) implements this. The router and pipeline only
    ever depend on this interface, never on a concrete strategy class.
    """

    name: QueryStrategy

    @abstractmethod
    async def run_many(self, queries: list[str]) -> list[str]:
        """Transform a list of queries, preserving order where possible.
        This is what the pipeline chains — output of one strategy feeds
        the next. Implementations must handle empty lists (return [])."""
        ...
