"""
Run every query in eval/queries.json against the running service and
print what chain was chosen, what queries came out, and whether the
chosen chain matched what you expected. Also saves the full run to
eval/results/ as a timestamped JSON file for later comparison.

Usage:
    1. Start the service:  uvicorn query_service.api:app --reload
    2. In another terminal:  python eval/run_eval.py [--force-chain decompose,rewrite] [--delay 9]

Options:
    --force-chain: comma-separated chain applied to EVERY query, e.g.
        --force-chain rewrite,expand  (tests a chain without router input)
    Per-case override: add "force_strategies": [...] to a case in
        queries.json to force a chain for just that query.
    Expected labels: each case uses "expected_strategies": [...] (list).
        The old single "expected_strategy": "..." is still accepted and
        treated as a one-element chain.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SERVICE_URL = "http://localhost:8000/process_query"
QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Free-tier LLM APIs (e.g. Groq) have tight tokens-per-minute limits.
# A pause between requests keeps this eval from burning through
# the limit and getting rate-limited or truncated mid-run.
# Chains make up to 2-5 LLM calls per query (router + stages, with
# rewrite fan-out per sub-query), so this is more important now than
# when each query cost 1-2 calls. Default 9s; override with --delay
# (higher when sharing quota, lower on a paid tier).
DEFAULT_DELAY_BETWEEN_REQUESTS_SECONDS = 9.0


def expected_chain(case: dict) -> list[str]:
    if "expected_strategies" in case:
        return list(case["expected_strategies"])
    if "expected_strategy" in case:
        return [case["expected_strategy"]]
    return []


def actual_chain(result: dict) -> list[str]:
    # New contract prefers strategies_used; fall back to strategy_used.
    if "strategies_used" in result and result["strategies_used"]:
        return list(result["strategies_used"])
    if "strategy_used" in result and result["strategy_used"]:
        return [result["strategy_used"]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval the query service chains.")
    parser.add_argument(
        "--force-chain",
        default=None,
        help='Comma-separated chain for ALL queries, e.g. "rewrite,expand".',
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_REQUESTS_SECONDS,
        help="Seconds to wait between requests (Groq TPM throttle).",
    )
    args = parser.parse_args()
    delay = args.delay
    global_force = (
        [s.strip() for s in args.force_chain.split(",") if s.strip()]
        if args.force_chain
        else None
    )

    cases = json.loads(QUERIES_PATH.read_text())

    correct = 0
    errored = 0
    rate_limited = 0
    run_results = []

    def post_once(payload: dict) -> httpx.Response:
        return client.post(SERVICE_URL, json=payload)

    with httpx.Client(timeout=90.0) as client:
        for i, case in enumerate(cases, 1):
            query = case["query"]
            expected = expected_chain(case)
            force = global_force or case.get("force_strategies")

            payload: dict = {"query": query}
            if force:
                payload["force_strategies"] = force

            try:
                response = post_once(payload)
                if response.status_code == 429:
                    # Truthful backpressure from the service (see api.py):
                    # honor Retry-After once, then resend a single time.
                    rate_limited += 1
                    try:
                        wait = float(response.json().get("retry_after_seconds", 5.0))
                    except Exception:
                        wait = 5.0
                    print(f"[{i}] 429 — waiting {wait:.1f}s then retrying once")
                    print(f"    query: {query}")
                    time.sleep(wait + 1.0)
                    response = post_once(payload)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPStatusError as e:
                # Don't let one failed call (rate limit, malformed
                # generation, etc.) kill the rest of the run — log it
                # and move on so you still get a full report.
                errored += 1
                print(f"[{i}] ERROR ({e.response.status_code}) — skipped")
                print(f"    query: {query}")
                print(f"    {e.response.text[:300]}")
                print()
                run_results.append({
                    "query": query,
                    "expected_strategies": expected,
                    "actual_strategies": None,
                    "match": False,
                    "error": e.response.text[:500],
                })
                time.sleep(delay)
                continue

            actual = actual_chain(result)
            is_match = actual == expected
            match = "PASS" if is_match else "FAIL"
            if is_match:
                correct += 1

            print(f"[{i}] {match} expected={expected} actual={actual}")
            print(f"    query: {query}")
            for q in result["queries"]:
                print(f"      -> {q}")
            print()

            run_results.append({
                "query": query,
                "expected_strategies": expected,
                "actual_strategies": actual,
                "match": is_match,
                "queries_out": result["queries"],
                "metadata": result.get("metadata", {}),
            })

            time.sleep(delay)

    score = f"{correct}/{len(cases)}"
    suffix = ""
    if errored:
        suffix += f" ({errored} errored"
        if rate_limited:
            suffix += f", {rate_limited} hit 429"
        suffix += ")"
    elif rate_limited:
        suffix += f" ({rate_limited} hit 429, recovered on retry)"
    print(f"{score} matched expected chain" + suffix)
    if global_force:
        print(f"(forced chain for all queries: {global_force})")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "score": score,
        "correct": correct,
        "errored": errored,
        "total": len(cases),
        "forced_chain": global_force,
        "rate_limited": rate_limited,
        "results": run_results,
    }, indent=2))

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
