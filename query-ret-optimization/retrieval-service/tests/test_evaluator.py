import asyncio
from crag.evaluator import CRAGEvaluator
from crag.models import Chunk

async def main():
    evaluator = CRAGEvaluator()
    query = "Where was the CEO of Company A working before?"
    chunks = [
        Chunk(
            id="1",
            content=(
                "Company A announced a new product in 2025."
            ),
        ),
        Chunk(
            id="2",
            content=(
                "Company A opened a new office in London."
            ),
        ),
        Chunk(
            id="3",
            content=(
                "Company A's revenue increased by 15%."
            ),
        ),
    ]
    result = await evaluator.evaluate(
        query=query,
        chunks=chunks,
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())