# Query Service

A pluggable query-transformation service for RAG. Takes a raw user query,
returns one or more retrieval-ready queries — via rewriting, expansion,
decomposition, or passthrough, depending on what the query needs.

This service owns **query understanding only**. It does not call retrieval
and does not merge/rerank results — that stays entirely on the retrieval
side, by design (see "Design principles" below).

## What it does

```
raw query  →  [router decides strategy]  →  [strategy runs]  →  retrieval-ready queries
```

- **Passthrough** — simple queries pass through unchanged
- **Rewrite** — clarifies a single ambiguous query
- **Expand** — generates synonym/related-term variants to improve recall
- **Decompose** — splits a multi-hop query into independent sub-questions

The router (heuristic + LLM classifier) picks the strategy automatically,
or you can force one via `force_strategy` for testing.

## API

One endpoint. This is the entire public contract — anything downstream
(retrieval, eval scripts, teammates) should only ever talk to this:

```
POST /process_query
{
  "query": "How does attention differ from RNNs, and why did transformers replace them?",
  "force_strategy": null   // optional: "passthrough" | "rewrite" | "expand" | "decompose"
}
```

Response:

```json
{
  "original_query": "...",
  "strategy_used": "decompose",
  "queries": [
    "How does the attention mechanism work compared to RNNs?",
    "Why did transformers replace RNNs in NLP?"
  ],
  "metadata": {}
}
```

`queries` is the only field retrieval strictly needs. `metadata` is optional
context (e.g. per-sub-query weights) — retrieval must work even if it
ignores it entirely.

`GET /health` for a liveness check.

## Project structure

```
query_service/
├── models.py              # QueryRequest / ProcessedQuery — the public contract
├── llm_client.py          # thin LLM wrapper (generate_structured)
├── router.py                # HeuristicRouter, LLMRouter, HybridRouter
├── pipeline.py              # QueryPipeline — orchestrates router + strategy
├── config.py                 # build_default_pipeline() — wiring
├── api.py                    # FastAPI: POST /process_query, GET /health
└── strategies/
    ├── base.py                # QueryStrategyBase interface
    ├── passthrough.py
    ├── rewrite.py
    ├── expand.py
    └── decompose.py
```

## Design principles

- **One public contract.** `/process_query` is the only endpoint anyone
  outside this service should depend on. Internals can change freely.
- **No retrieval coupling.** This service never calls a vector store or
  merges results. It only produces `list[str]`. Keeps it swappable
  independently of whatever retrieval stack the team ends up using.
- **Strategy pattern.** Each transformation implements `QueryStrategyBase`.
  Adding a new one (e.g. a future reasoning-model-based decomposition) means
  dropping in a new file under `strategies/` — no changes needed elsewhere.
- **Router is swappable/evallable on its own.** `HeuristicRouter` (free,
  rule-based), `LLMRouter` (classification call), `HybridRouter` (default —
  heuristic pre-filter, LLM fallback).

## Setup

```bash
pip install fastapi uvicorn pydantic
# + your LLM SDK of choice (anthropic, openai, instructor, etc.)
```

Fill in `llm_client.py`'s `generate_structured()` with a real call to your
provider (see "Models to try" below).

Run:

```bash
uvicorn query_service.api:app --reload
```

## TODO

- [ ] Wire a real LLM SDK call into `llm_client.py`
- [ ] Pick/pin models per strategy in `config.py` (router + rewrite/expand
      can use a cheap model; decompose benefits from a stronger one)
- [ ] Build a small labeled eval set (~30-50 queries) to check router
      accuracy and decompose quality before trusting it in the pipeline
- [ ] Decide with the retrieval owner what (if anything) goes in `metadata`