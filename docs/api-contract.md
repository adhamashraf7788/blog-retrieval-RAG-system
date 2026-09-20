# API Contract

Shared contract for services in the blog-retrieval RAG pipeline. Each
service's section below defines its public interface — the request/response
shapes other services can depend on. Internals of any service can change
freely as long as its section here stays accurate.

---

## Query Service

Takes a raw user query, returns one or more retrieval-ready queries via
query understanding (passthrough / rewrite / expand / decompose). Does not
call retrieval and does not merge/rerank results — retrieval owns that.

**Owner:** query/retrieval-optimization workstream
**Code:** `query_service/`

### `POST /process_query`

**Request**

| Field            | Type              | Required | Description |
|------------------|-------------------|----------|-------------|
| `query`          | `string`          | yes      | Raw user query |
| `force_strategy` | `string \| null`  | no       | Override the router. One of: `"passthrough"`, `"rewrite"`, `"expand"`, `"decompose"` |

```json
{
  "query": "How does attention differ from RNNs, and why did transformers replace them?",
  "force_strategy": null
}
```

**Response — `200 OK`**

| Field            | Type       | Description |
|------------------|------------|-------------|
| `original_query` | `string`   | The input query, unmodified |
| `strategy_used`  | `string`   | Which strategy actually ran: `"passthrough"`, `"rewrite"`, `"expand"`, or `"decompose"` |
| `queries`        | `string[]` | One or more retrieval-ready queries. **This is the only field retrieval strictly needs.** |
| `metadata`       | `object`   | Reserved for optional future context. Currently always `{}`. Retrieval must work correctly even if this is ignored entirely. |

```json
{
  "original_query": "How does attention differ from RNNs, and why did transformers replace them?",
  "strategy_used": "decompose",
  "queries": [
    "How does attention differ from RNNs?",
    "Why did transformers replace RNNs?"
  ],
  "metadata": {}
}
```

**Errors**

| Status | Cause |
|--------|-------|
| `422`  | Malformed request body (e.g. missing `query`, invalid `force_strategy` value) |
| `500`  | Upstream LLM call failed after retries |

### `GET /health`

Liveness check.

```json
{ "status": "ok" }
```

### Consumer notes

- Treat `queries` as an ordered list — for `decompose`, order is not
  guaranteed to reflect any particular priority; for `expand`, the first
  entry is always the original query, followed by variants.
- `queries` can be length 1 (`passthrough`, `rewrite`) or length N
  (`expand`, `decompose`). Consumers should not assume a fixed count.
- No retries or fusion of results across `queries` happens in this service —
  if you retrieve per-query, deduplication/merging is on you.

---

## [Next service — add its section here]