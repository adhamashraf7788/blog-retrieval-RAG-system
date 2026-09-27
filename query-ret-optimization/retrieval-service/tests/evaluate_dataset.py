import asyncio
import json
from pathlib import Path

from crag.evaluator import CRAGEvaluator
from crag.models import Chunk


async def main():

    dataset_path = Path(__file__).parent / "test_cases.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    evaluator = CRAGEvaluator()

    correct = 0

    for case in cases:

        chunks = [
            Chunk(
                id=f"{case['id']}-{i}",
                content=content,
            )
            for i, content in enumerate(case["chunks"])
        ]

        result = await evaluator.evaluate(
            query=case["query"],
            chunks=chunks,
        )

        predicted = result.decision.value
        expected = case["expected"]

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(
            f"{case['id']} | "
            f"expected={expected} | "
            f"predicted={predicted} | "
            f"{'PASS' if is_correct else 'FAIL'}"
        )

        if not is_correct:
            print(f"  Reason: {result.reason}")

    accuracy = correct / len(cases)

    print("\n==============================")
    print("EVALUATION")
    print("==============================")

    print(f"Correct: {correct}/{len(cases)}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(main())