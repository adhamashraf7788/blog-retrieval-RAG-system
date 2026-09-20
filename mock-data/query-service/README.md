# Query Service Mock Data

Example requests and responses for the query service.
Contract: see `/docs/api-contract.md`.

Each file in `requests/` has a matching file in `responses/` with the same name.

| File | Strategy | Description |
|------|----------|-------------|
| `passthrough` | passthrough | Simple, unambiguous lookup |
| `rewrite` | rewrite | Vague/casual phrasing clarified into a single query |
| `expand` | expand | Single clear intent, broadened with related-term variants for recall |
| `decompose_two_part` | decompose | Multi-hop query split into independent, pronoun-resolved sub-queries |
| `decompose_three_part` | decompose | Three-part query |
| `force_strategy_expand` | expand | `force_strategy` override bypasses the router |

## Notes
- `force_strategy: null` means the router picks the strategy.
- `metadata` is currently always `{}`.
