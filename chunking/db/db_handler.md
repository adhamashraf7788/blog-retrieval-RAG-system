# db_handler.py

## Overview
This module is the database layer of the pipeline: it owns the Postgres connection, the schema (`websites`, `authors`, `pages`, `metadata`, `content`), and the batch-insert logic that takes a list of extracted-record dicts (from `extractors.py`, staged via `pipeline/stream_to_db.py`) and writes them into normalized tables.

It does not:
- Parse WARC files or extract HTML (see `parsers.py` / `extractors.py`)
- Decide what counts as a "valid" record (word-count filtering, language detection, etc. all happen upstream) — this module just stores whatever dicts it's handed
- Run any Hugging Face models

## Dependencies
- **psycopg2** (+ `psycopg2.extras.execute_values`) — Postgres driver and efficient multi-row insert helper.
- Standard library: `urllib.parse.urlparse` (domain extraction), `json` (serializing the `headings` list into JSONB).

## Schema

| Table | Key columns | Relationship |
|---|---|---|
| `websites` | `domain` (unique) | One row per distinct domain. |
| `authors` | `name` (unique) | One row per distinct author name. |
| `pages` | `warc_record_id` (unique), `website_id` → websites, `author_id` → authors | One row per article/page. |
| `metadata` | `page_id` (PK, → pages) | `title`, `published_date`, `language`, `arabic_dialect`, `word_count`, `char_count`, `links_count`, `headings` (JSONB). One-to-one with `pages`. |
| `content` | `page_id` (PK, → pages) | `cleaned_text`. Kept in its own table (rather than in `metadata`) so large text bodies don't bloat metadata-only queries. |

## Functions

| Function | Input | Output | Description |
|---|---|---|---|
| `get_connection()` | — | psycopg2 connection | Opens a new connection using the hardcoded `DB_CONFIG` dict at the top of the file. |
| `init_db()` | — | `None` | Creates all 5 tables if they don't exist, then runs migration `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements for columns added after the tables may have already existed (`pages.author_id`, `metadata.arabic_dialect`) so it's safe to re-run against an older database without losing data. Call this once before any inserts (both `stream_to_db.py` and running this file directly do so). |
| `extract_domain(url)` | a URL string | domain string, or `"unknown_domain"` | Parses the URL, lower-cases the host, and strips a leading `www.`. Falls back to `"unknown_domain"` on any parsing failure or empty URL. Used to group pages under `websites`. |
| `save_batch_records(records_list, conn=None)` | a list of record dicts (see keys below), optional existing DB connection | `None` (raises on failure) | The main batch-insert entry point. See step-by-step below. If `conn` is passed in, it's reused and left open for the caller to manage; if omitted, a fresh connection is opened and closed internally. Wraps everything in a single transaction — any error triggers a full rollback of the batch (nothing partially written). |

### `save_batch_records` step-by-step
1. **Websites** — collects the distinct domains in the batch, inserts any new ones (`ON CONFLICT DO NOTHING`), then re-selects to build a `domain → id` map.
2. **Authors** — same pattern for distinct non-empty author names → `name → id` map.
3. **Pages** — builds one tuple per record `(website_id, author_id, warc_record_id, url)` and inserts them all in one `execute_values(..., fetch=True)` call, using `ON CONFLICT (warc_record_id) DO UPDATE SET url = ...` so re-processing the same record updates rather than duplicates it. The `RETURNING warc_record_id, id` clause gives back the Postgres-assigned `id` for every row (new or updated), which is captured into `page_id_map`.
4. **Metadata & Content** — for each record, looks up its `page_id` via `page_id_map`; if found, appends a row to `metadata_tuples` and `content_tuples`. Both are inserted with `ON CONFLICT (page_id) DO NOTHING` (so metadata/content are written once and never overwritten by a re-run).
5. Commits the whole batch; on any exception, rolls back and re-raises so the caller (`stream_to_db.py`) can log it and keep going with the next batch.

## Expected keys in each record dict
`save_batch_records` reads these keys via `.get(...)` with safe defaults, so missing keys won't crash it — they'll just be stored as empty/default values:

`record_id`, `url`, `author`, `title`, `published_date`, `language`, `arabic_dialect`, `word_count`, `char_count`, `links_count`, `headings`, `cleaned_text`.

## Notes
- **Bug fix — `execute_values(..., fetch=True)` return value.** `execute_values` with `fetch=True` already runs `fetchall()` internally and returns the collected rows as its own return value. The original code ignored that and called `cur.fetchall()` again afterward, which found nothing left to fetch and silently returned `[]` — so `page_id_map` was always empty and **every** record failed its `if p_id:` check with no error at all. `pages` filled up fine; `metadata`/`content` stayed empty forever. Fixed by capturing `execute_values(...)`'s return value directly instead of calling `cur.fetchall()` after it.
- **Bug fix — `.strip()` on `None` author.** `r.get("author")` can be `None` (key present, value `None`), not just missing. The original `r.get("author", "N/A").strip()` would crash with `AttributeError` in that case. Fixed to `(r.get("author") or "").strip()` everywhere an author name is read.
- **Migration statements matter here.** If you ever add a new column to `metadata` or `pages` again, remember `CREATE TABLE IF NOT EXISTS` will silently do nothing on a table that already exists — you must add a matching `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to `init_db()`'s `migration_query`, or existing databases will start throwing `column ... does not exist` on every insert.
- **Connection reuse**: `save_batch_records` accepts an existing `conn` specifically so `pipeline/stream_to_db.py` can open one connection for an entire run instead of one per batch — opening/closing a connection per batch was previously the single biggest performance bottleneck in the pipeline.

## Used by
- `pipeline/stream_to_db.py` → `init_db`, `save_batch_records`, `get_connection`
- `db_summary.py` → `get_connection` (read-only reporting, doesn't call `save_batch_records`)
