# Query Service

> **Branch note — baseline docs + multi-strategy work.**
> This branch (`checkpoint/router-only`) started as a frozen router-only
> baseline (see [Baseline checkpoint](#baseline-checkpoint-router-only-2026-09-26)
> below: the router picked **exactly one** strategy per query, no chains)
> and now also carries the multi-strategy session work on top: chained
> composition (`chain_map`, `run_many`, `force_strategies`), 429
> retry-then-429 handling, sequential rewrite fan-out, the hedge-word
> routing fix, and eval harness upgrades. The `Baseline checkpoint` section
> describes the frozen starting point; everything else in this file
> describes the current code. To see only the post-baseline work:
> `git diff bc84e34 -- query-ret-optimization`.

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

- [ ] Router occasionally misclassifies vague/indirect-reference queries
      (e.g. "tell me about the thing that replaced X") as passthrough —
      `CLASSIFY_PROMPT`'s few-shot examples don't yet cover this pattern
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
- [ ] Groq free tier rate limits (~30 req/min, 8000 TPM) — chains cost 2-5
      LLM calls per query now (router + stages + per-sub-query rewrites).
      `llm_client.py` retries twice honoring Groq's `try again in Xs` wait,
      then the API returns truthful 429 + `Retry-After` instead of 500.
      Eval defaults to `--delay 9` between requests; raise it if both
      teammates test simultaneously

## Baseline checkpoint (router-only, 2026-09-26)

This section documents the frozen state of this branch: a **single-strategy**
query pipeline. It is the reference point all multi-strategy work diffs
against.

### What was built in this baseline

A FastAPI service (`POST /process_query`, `GET /health`) that takes a raw
user query and returns retrieval-ready queries using **one** of four
strategies per request:

```
raw query  →  [router picks ONE strategy]  →  [that strategy runs once]  →  queries
```

- **Router** (`router.py`) — `HybridRouter` (default): free heuristic
  pre-filter, LLM fallback. Heuristics: multi-hop keywords (`compare`,
  `" vs "`, `difference between`, `" and why"`, `" and how"`) → `decompose`;
  ≤ 3 words (`SIMPLE_WORD_THRESHOLD`) → `passthrough`; everything else goes
  to `LLMRouter`, one `temperature=0` classification call against
  `CLASSIFY_PROMPT` (few-shot, JSON-mode structured output).
- **Strategies** (`strategies/`) — each implements
  `QueryStrategyBase.run(query: str) -> ProcessedQuery`: `passthrough`
  (identity), `rewrite` (1 LLM call → clarified query), `expand` (1 LLM call
  → original + 3 variants), `decompose` (1 LLM call → pronoun-resolved
  sub-questions).
- **Pipeline** (`pipeline.py`) — `QueryPipeline.process`: single entry
  decision (`force_strategy` or `router.decide`), runs that one strategy.
  No chaining, no fan-out, no result merging.
- **LLM client** (`llm_client.py`) — Groq `AsyncGroq` + `instructor` in JSON
  mode (tool-calling proved unreliable on gpt-oss), `openai/gpt-oss-20b`
  for everything, `max_retries=3` for malformed JSON. **No 429 handling.**
- **Contract** (`models.py`) — `QueryRequest(query, force_strategy?)`,
  `ProcessedQuery(original_query, strategy_used, queries, metadata={})`.
  Retrieval needs only `.queries`.
- **Eval** (`eval/`) — 17 labeled queries across all four strategies;
  `run_eval.py` hits the live service with a 3s throttle and saves
  timestamped results.

### What the baseline proved (eval evidence, run 2026-09-26)

Score **15/17** with single-strategy routing: all `passthrough`, `expand`,
and multi-hop `decompose` cases routed correctly with clean output
(pronouns resolved, e.g. `them` → `RNNs`). The architecture's core bet —
one public endpoint, no retrieval coupling, swappable router/strategies —
held up unchanged through testing.

### Limitations found (the reason for multi-strategy)

1. **Hedge-word misroute** — `how attention works kinda` routed
   `passthrough` instead of `rewrite`. 4 words bypass the ≤3-word shortcut,
   and `CLASSIFY_PROMPT` has no hedge-word (`kinda`/`sorta`) example, so the
   LLM reads it as a simple lookup. Vague phrasing slips through as-is.
2. **429 surfaces as 500** — the 3-subquery RAG question crashed with
   `Internal Server Error`. Server traceback proved TPM exhaustion
   (`Limit 8000, Used 7622, Requested 1034, retry in 4.92s`):
   `groq.RateLimitError` → `InstructorRetryException` → unhandled → 500.
   `max_retries=3` covers malformed JSON, not rate limits. Any real load
   (each request costs 1–2 LLM calls) hits this, and the 500 hides the cause.
3. **One strategy is not enough** — `decompose` output is split but
   unpolished (still vague); `rewrite` output is clear but narrow (no recall
   variants). Real queries need *combinations*: split, then clarify each
   part, then diversify for recall. The single-choice router cannot express
   that, and parallel fan-out (run all, merge) just multiplies noise.

### Where the multi-strategy work takes it (implemented on this branch, on top of the checkpoint)

- **Chained composition, split-before-polish order:**
  `decompose → rewrite → expand`. Decompose first (split signals + pronouns
  still visible in the raw query), rewrite each single-intent sub-query,
  expand last for recall. Reversed (`rewrite → decompose`) blends intents
  and erases split signals — documented as a fallback for garbled input
  only, never the default.
- **Bounded defaults:** only `decompose → [decompose, rewrite]` chains by
  default; `rewrite`/`expand` stay single-step (LLM cost), `expand` never
  auto-appends. Any explicit chain runs verbatim via `force_strategies`
  (router skipped) for testing — full `decompose → rewrite → expand`
  stays forced-only until retrieval-side measurement justifies a default.
- **Composable strategies:** `run(query) -> ProcessedQuery` becomes
  `run_many(list) -> list` so stages feed each other; per-query LLM calls
  run sequentially (`REWRITE_CONCURRENCY = 1`, tunable) to spread TPM
  instead of spiking it — quality is identical either way (independent
  calls, same prompt/model, `temperature=0`).
- **Contract evolution:** `strategies_used: list` (primary) +
  `strategy_used` kept as deprecated alias; `force_strategies: list`
  alongside legacy `force_strategy`; `metadata={"chain": [...]}` trace.
- **Reliability:** retry-twice on 429 honoring Groq's `try again in Xs`
  (+ jitter), then truthful HTTP 429 + `Retry-After` instead of 500; eval
  `--delay` flag (default ~9s, chains cost 2–5 calls now) and 429-aware
  wait-once-and-resend.
- **Routing accuracy:** `HEDGE_WORDS` shortcut (`kinda`, `sorta`, …) that
  short-circuits to `rewrite` before the word-count check, plus a hedge
  few-shot example in `CLASSIFY_PROMPT`.

Compare any of this against the frozen baseline code:
`git diff bc84e34 -- query-ret-optimization` shows all post-baseline work.
(The `checkpoint/router-only-baseline` marker branch, if it still exists,
points at the same frozen tree.)
