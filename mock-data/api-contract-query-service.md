## Query Service

Takes a user query and returns one or more queries to run against retrieval, using one of four strategies: `passthrough`, `rewrite`, `expand`, `decompose`.

Mock data: `/mock-data/query-service/`

**Endpoint:** `POST /<TODO: path>`

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | The raw user query |
| `force_strategy` | string \| null | no | One of `passthrough`, `rewrite`, `expand`, `decompose`. If `null`, the router picks the strategy |

```json
{
  "query": "vector databases",
  "force_strategy": "expand"
}
```

### Response

| Field | Type | Description |
|-------|------|-------------|
| `original_query` | string | The query as received |
| `strategy_used` | string | Strategy that was applied |
| `queries` | string[] | Queries to send to retrieval (1 for passthrough/rewrite, N for expand/decompose) |
| `metadata` | object | Currently always `{}` |

```json
{
  "original_query": "vector databases",
  "strategy_used": "expand",
  "queries": [
    "vector databases",
    "vector database systems",
    "embedding storage and retrieval systems",
    "vector search databases"
  ],
  "metadata": {}
}
```

### Error responses

<TODO: document error cases, e.g. 422 for invalid `force_strategy` or empty `query`>

### Assumptions
- `queries` always contains at least one item.
- For `expand`, the original query is included as the first item.
- For `decompose`, sub-queries are independent and pronoun-resolved.
