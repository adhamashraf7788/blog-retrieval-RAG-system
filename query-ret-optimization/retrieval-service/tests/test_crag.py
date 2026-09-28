import asyncio

from crag.evaluator import CRAGEvaluator
from crag.service import CRAGService
from fake_retriever.fake_retriever import FakeRetriever


async def main():

    query = "Where was the CEO of Company A working before?"

    evaluator = CRAGEvaluator() # query + chunks --> LLM --> judgment
    retriever = FakeRetriever()

    crag = CRAGService(         
        evaluator=evaluator,
        retriever=retriever,
        max_attempts=2,
    )
    # CRAGService:
    # 1. Retrieve
    # 2. Evaluate
    # 3. If sufficient → stop
    # 4. If insufficient → retrieve again
    # 5. Evaluate again
    # 6. Stop after max attempts

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