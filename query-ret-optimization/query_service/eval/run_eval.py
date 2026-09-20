"""
Run every query in eval/queries.json against the running service and
print what strategy was chosen, what queries came out, and whether the
chosen strategy matched what you expected. Also saves the full run to
eval/results/ as a timestamped JSON file for later comparison.

Usage:
    1. Start the service:  uvicorn query_service.api:app --reload
    2. In another terminal:  python eval/run_eval.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

SERVICE_URL = "http://localhost:8000/process_query"
QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    cases = json.loads(QUERIES_PATH.read_text())

    correct = 0
    run_results = []

    with httpx.Client(timeout=30.0) as client:
        for i, case in enumerate(cases, 1):
            query = case["query"]
            expected = case.get("expected_strategy")

            response = client.post(SERVICE_URL, json={"query": query})
            response.raise_for_status()
            result = response.json()

            actual = result["strategy_used"]
            is_match = actual == expected
            match = "✅" if is_match else "❌"
            if is_match:
                correct += 1

            print(f"[{i}] {match} expected={expected!r} actual={actual!r}")
            print(f"    query: {query}")
            for q in result["queries"]:
                print(f"      -> {q}")
            print()

            run_results.append({
                "query": query,
                "expected_strategy": expected,
                "actual_strategy": actual,
                "match": is_match,
                "queries_out": result["queries"],
                "metadata": result.get("metadata", {}),
            })

    score = f"{correct}/{len(cases)}"
    print(f"{score} matched expected strategy")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "score": score,
        "correct": correct,
        "total": len(cases),
        "results": run_results,
    }, indent=2))

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()