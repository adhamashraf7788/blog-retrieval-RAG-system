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
- **Decompose** — splits a multi-hop query into independent, self-contained
  sub-questions (pronouns/implicit references resolved, e.g. "them" → "RNNs")

The router (heuristic pre-filter + LLM classifier) picks the strategy
automatically, or you can force one via `force_strategy` for testing.

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
    "How does attention differ from RNNs?",
    "Why did transformers replace RNNs?"
  ],
  "metadata": {}
}
```

`queries` is the only field retrieval strictly needs. `metadata` is reserved
for optional future context — currently always empty; retrieval must work
even if it's ignored entirely.

`GET /health` for a liveness check.

## Project structure

```
query_service/
├── models.py              # QueryRequest / ProcessedQuery — the public contract
├── llm_client.py          # Groq + instructor wrapper (generate_structured)
├── router.py                # HeuristicRouter, LLMRouter, HybridRouter
├── pipeline.py              # QueryPipeline — orchestrates router + strategy
├── config.py                 # build_default_pipeline() — wiring, model selection
├── api.py                    # FastAPI: POST /process_query, GET /health
├── requirements.txt
├── strategies/
│   ├── base.py                # QueryStrategyBase interface
│   ├── passthrough.py
│   ├── rewrite.py
│   ├── expand.py
│   └── decompose.py
└── eval/
    ├── queries.json            # labeled test queries (expected strategy per query)
    ├── run_eval.py              # runs queries.json against the live service
    └── results/                 # timestamped JSON output per eval run (gitignored)
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
  rule-based — multi-hop keywords + a low word-count passthrough shortcut),
  `LLMRouter` (classification call), `HybridRouter` (default — heuristic
  pre-filter, LLM fallback for anything not obviously simple/multi-hop).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the project root:

```bash
GROQ_API_KEY=gsk_your_key_here
```

Run:

```bash
uvicorn query_service.api:app --reload
```

Test it:

```bash
curl -X POST http://localhost:8000/process_query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does attention differ from RNNs, and why did transformers replace them?"}'
```

## Models

Currently on Groq's free tier, using `openai/gpt-oss-20b` for everything
(router classification, rewrite, expand, decompose). Model selection lives
in `config.py` — swap in a stronger model for decompose specifically if
quality needs it (e.g. `openai/gpt-oss-120b`, if available on your account —
check with `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`,
since Groq's available model list changes over time).

`llm_client.py` uses `instructor` in **JSON mode** (not tool-calling mode) —
gpt-oss models on Groq were unreliable with forced tool calls but work
consistently with plain JSON-mode structured output. `max_retries=3` is set
on the instructor call as a safety net for occasional malformed JSON replies.

## Eval

`eval/queries.json` has ~17 labeled queries spanning all four strategies,
including intentionally ambiguous edge cases. Run against the live service:

```bash
uvicorn query_service.api:app --reload   # terminal 1
python eval/run_eval.py                  # terminal 2
```

Prints per-query pass/fail plus a final score, and saves a full timestamped
JSON record to `eval/results/`.

**Note:** LLM classification is non-deterministic — the same query can be
routed differently across runs. Run the eval multiple times rather than
trusting a single score; consistency across runs matters as much as the
score itself.

## Known limitations / TODO

- [ ] **Classifier occasionally goes conversational instead of returning
      JSON** — `LLMRouter` sometimes gets a reply like "I don't see a
      query, could you provide one?" instead of a classification, causing
      a 500. Seems to correlate with being near the rate limit (degraded
      generation) and/or the length of `CLASSIFY_PROMPT`'s few-shot
      examples burying the actual query. Mitigated with `temperature=0`
      and an explicit `>>> QUERY TO CLASSIFY <<<` marker in the prompt,
      but not fully eliminated — worth watching, and a candidate for
      moving to a shorter prompt or a more reliable model if it persists.
- [x] **No rate-limit handling** — FIXED 2026-09-26: `llm_client.py` now
      retries twice on 429s honoring Groq's `try again in Xs` wait (+ jitter),
      then `api.py` returns truthful HTTP 429 + `Retry-After` instead of 500.
      `instructor max_retries=3` still covers malformed JSON separately.
      Chains cost 2-5 LLM calls per query now, so eval defaults to `--delay 9`.
- [ ] Router occasionally misclassifies vague/indirect-reference queries
      (e.g. "tell me about the thing that replaced X") as passthrough —
      `CLASSIFY_PROMPT`'s few-shot examples don't yet cover this pattern.
      Partly mitigated 2026-09-26 with a `HEDGE_WORDS` shortcut + hedge
      few-shot example (covers `kinda`/`sorta`-style phrasing) — the
      indirect-reference gap above remains.
- [ ] Consider rewrite-first fallback for garbled input — default chain stays
      `decompose -> rewrite -> expand` (split before polish: rewriting a
      multi-intent query first risks blending intents and erasing split
      signals). If decompose ever fails on severely malformed queries, a
      light `rewrite -> decompose` pre-clean pass could be tried as a
      fallback, not the default. Revisit only with failing examples.
- [ ] No caching of repeated queries — every call re-hits the LLM
- [ ] Rewrite fan-out is sequential (`REWRITE_CONCURRENCY = 1` in `config.py`)
      to spread Groq TPM usage — raise if p99 latency matters more than
      rate-limit headroom (quality is identical either way: calls are
      independent, same prompt/model, `temperature=0`)
- [ ] Decide with the retrieval owner what (if anything) goes in `metadata`
- [ ] Groq free tier rate limits (~30 req/min, 8000 TPM) — fine for solo
      dev with throttled eval runs, watch for this if both teammates test
      simultaneously or under any sustained load