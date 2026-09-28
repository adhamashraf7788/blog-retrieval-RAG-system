from crag.models import Chunk

class FakeRetriever:
    """
    Temporary retriever used while the real retrieval system
    is still being developed.

    First retrieval returns insufficient chunks.
    Second retrieval returns sufficient chunks.
    """

    def __init__(self):
        self.calls = 0

    async def retrieve(self, query: str) -> list[Chunk]:
        self.calls += 1
        if self.calls == 1:
            return self._insufficient_chunks()
        return self._sufficient_chunks()

    def _insufficient_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                id="bad-1",
                content=(
                    "Company A announced a new product in 2025."
                ),
            ),
            Chunk(
                id="bad-2",
                content=(
                    "Company A opened a new office in London."
                ),
            ),
            Chunk(
                id="bad-3",
                content=(
                    "Company A's revenue increased by 15%."
                ),
            ),
        ]

    def _sufficient_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                id="good-1",
                content=(
                    "John Smith is the CEO of Company A."
                ),
            ),
            Chunk(
                id="good-2",
                content=(
                    "Before joining Company A, John Smith "
                    "worked at Company B as Chief Operating Officer."
                ),
            ),
        ]