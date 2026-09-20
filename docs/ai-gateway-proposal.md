# Proposal: AI Gateway Layer (LiteLLM) for LLM Calls

**Status:** for discussion, not yet decided
**Owner:** query-ret-optimization
**Context:** the query service currently calls Groq directly via `instructor` +
the raw Groq SDK. This works, but we're already hitting the limits of that
approach — this doc lays out the problem, the proposed fix, and what it
would take to adopt.

---

## The problem we're hitting

The query service is pinned to a single provider (Groq) and a single free
API key. In practice this means:

- **Rate limits stop the service outright.** Groq's free tier is 8000
  tokens/minute — a handful of concurrent requests exhausts it, and every
  provider call in the service currently has no backoff/retry on `429`, so
  a rate limit surfaces as a `500` to whoever's calling `/process_query`.
- **One provider is a single point of failure.** If Groq has an outage, or
  we hit our quota mid-demo, the whole service goes down — there's no
  fallback.
- **Every model reference is hardcoded** in `config.py` and `llm_client.py`
  (`openai/gpt-oss-20b`, Groq-specific client setup). Swapping providers or
  models means touching code, not config.
- **No shared visibility** into how much we're spending or calling, per
  strategy or per teammate, as this scales past one person's local dev.

None of this is a code-quality problem in `query_service` itself — it's a
missing infrastructure layer underneath it.

## The proposed fix: an AI gateway

Instead of the service talking to Groq's SDK directly, it talks to a
**gateway** — a thin, self-hosted proxy that sits between our code and
every LLM provider, and speaks one consistent API regardless of which
provider is actually handling the request underneath.

```
query_service  →  AI Gateway  →  Groq / OpenAI / Anthropic / ...
```

**[LiteLLM](https://github.com/BerriAI/litellm)** is the common open-source
choice for this — it wraps 100+ providers behind one OpenAI-compatible
interface, and it's something we'd host ourselves (not a shared public
service — see "Is this shared with anyone outside our team?" below).

### What it actually gets us

- **Automatic failover.** If Groq rate-limits or errors, the gateway can
  retry against a second provider/key without our code knowing anything
  changed.
- **Model swapping via config, not code.** Model choice becomes a string
  (`model="fast-router"`) mapped in a YAML file, not a hardcoded SDK client.
  Swapping Groq → Claude → OpenAI for any strategy is a config edit.
- **Centralized retry/backoff.** One place to handle `429`s properly,
  instead of every call site reimplementing it.
- **Optional semantic caching.** Repeated/similar queries can be served
  from cache instead of re-hitting the LLM — directly addresses the
  "no caching" limitation already in our README.
- **Per-key budgets, if we ever have more than one contributor hitting
  paid APIs** — not urgent now, but relevant once this isn't just solo
  free-tier dev.

### What it costs us

- **One more moving piece to run and maintain** — the gateway itself needs
  to be up (Docker container, or hosted) for the service to work at all.
- **An extra network hop** — marginal latency added per call.
- **Migration effort** — `llm_client.py` changes from a Groq-specific
  client to a LiteLLM-based one; not a large rewrite, but not zero.
- **New thing for the team to learn** — config file conventions, how
  virtual keys/budgets work, how to debug a failure that's now one layer
  removed from the raw provider error.

## Is this shared with anyone outside our team?

No — this would be self-hosted infrastructure, not a public/shared
service. We'd run the LiteLLM proxy ourselves (Docker container on our own
infra), and our own provider keys (Groq, etc.) live only in that gateway's
environment, never in individual developers' local `.env` files. If we
want per-developer isolation later, the gateway can issue **virtual keys**
per person/service with their own budget caps — but that's an optional
later step, not something required to get the basic failover/config
benefits.

## What a minimal version looks like

```yaml
# config.yaml (lives with the gateway, not in query_service)
model_list:
  - model_name: fast-router
    litellm_params:
      model: groq/openai-gpt-oss-20b
      api_key: "os.environ/GROQ_API_KEY"
  - model_name: fast-router-fallback
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: "os.environ/OPENAI_API_KEY"
```

```python
# llm_client.py, roughly — swaps groq SDK for the unified one
from litellm import acompletion
import instructor

client = instructor.from_litellm(acompletion)

response = await client.chat.completions.create(
    model="fast-router",
    messages=[...],
    response_model=schema,
    fallbacks=["fast-router-fallback"],
)
```

## Discussion points for the team

1. **Is this worth doing now, or after the current rate-limit workaround
   (throttling in `run_eval.py`) is stable enough to unblock demo/testing?**
   The throttle is a short-term patch; the gateway is closer to the real
   fix if rate limits keep being a recurring problem.
2. **Do we self-host the gateway per-workstream (just for query_service),
   or is this something worth proposing at the whole-project level**, since
   other workstreams (model-training, emb-vector-db) likely hit similar
   provider-dependency issues eventually?
3. **What's our fallback provider?** Groq is free but rate-limited;
   OpenAI/Anthropic cost money per call — worth deciding budget before
   wiring in a paid fallback.
4. **Who owns running the gateway** if we adopt it — does it live in this
   repo, or as separate shared infra?

## Recommendation

Given we're still early and solo-dev on this workstream, my suggestion is:
ship the throttling/backoff fix we already have as the near-term patch,
and treat the gateway as the right next step once rate limiting becomes a
recurring blocker rather than an occasional annoyance — or once more than
one of us is hitting the same API key regularly. Flagging it now so it's
on the team's radar before we're forced into it under time pressure.
