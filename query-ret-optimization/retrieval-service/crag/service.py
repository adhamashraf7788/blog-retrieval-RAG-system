from .evaluator import CRAGEvaluator
from .models import (
    Chunk,
    EvaluationResult,
    RetrievalDecision,
)


class CRAGService:
    def __init__(
        self,
        evaluator: CRAGEvaluator,
        retriever,
        max_attempts: int = 2,
    ):
        self.evaluator = evaluator
        self.retriever = retriever
        self.max_attempts = max_attempts

    async def process(self, query: str):
        attempts = []

        for attempt in range(1, self.max_attempts + 1):

            print(f"\n--- Retrieval attempt {attempt} ---")

            chunks = await self.retriever.retrieve(query)

            print(f"Retrieved {len(chunks)} chunks")

            evaluation = await self.evaluator.evaluate(
                query=query,
                chunks=chunks,
            )

            attempts.append(
                {
                    "attempt": attempt,
                    "chunks": chunks,
                    "evaluation": evaluation,
                }
            )

            print(f"Decision: {evaluation.decision}")
            print(f"Reason: {evaluation.reason}")

            if evaluation.decision == RetrievalDecision.SUFFICIENT:
                return {
                    "query": query,
                    "status": "sufficient",
                    "chunks": chunks,
                    "evaluation": evaluation,
                    "attempts": attempts,
                }

            print("Retrieval was insufficient. Trying again...")

        return {
            "query": query,
            "status": "insufficient",
            "chunks": attempts[-1]["chunks"],
            "evaluation": attempts[-1]["evaluation"],
            "attempts": attempts,
        }