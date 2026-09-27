import asyncio

from crag.evaluator import CRAGEvaluator
from crag.service import CRAGService
from fake_retriever.fake_retriever import FakeRetriever


async def main():

    query = "Where was the CEO of Company A working before?"

    evaluator = CRAGEvaluator()
    retriever = FakeRetriever()

    crag = CRAGService(
        evaluator=evaluator,
        retriever=retriever,
        max_attempts=2,
    )

    result = await crag.process(query)

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print("Status:", result["status"])
    print("Decision:", result["evaluation"].decision)
    print("Reason:", result["evaluation"].reason)

    print("\nFinal chunks:")

    for chunk in result["chunks"]:
        print(f"\n[{chunk.id}]")
        print(chunk.content)


if __name__ == "__main__":
    asyncio.run(main())