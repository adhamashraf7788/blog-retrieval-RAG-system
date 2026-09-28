# `stream_to_db.py`

## Overview

This is the **main production pipeline**. It:

1. Fetches the CC-NEWS WARC file manifest for a given month/year from
   Common Crawl.
2. Streams each WARC file directly over HTTP (no local download step).
3. Extracts HTML article fields per record (cheap, no ML).
4. Runs Hugging Face language + Arabic dialect detection in **batches**.
5. Keeps only Arabic/English/French records.
6. Writes finished records into PostgreSQL in **batches**.

It combines three things that previously lived in separate scripts: CC-NEWS
network streaming (previously only in a preview/print script), batched
extraction + language filtering, and batched DB writes (previously only
wired up for local WARC files).

## Configuration

All configuration lives as module-level constants at the top of the file:

| Constant | Meaning |
|---|---|
| `YEAR`, `MONTH` | Which CC-NEWS crawl manifest to pull (`crawl-data/CC-NEWS/{YEAR}/{MONTH}/warc.paths.gz`) |
| `NUM_FILES` | How many WARC files from that manifest to process (each file is a separate HTTP download) |
| `DB_BATCH_SIZE` | How many finished records get written to Postgres per `INSERT` batch (default 500) |
| `HF_BATCH_SIZE` | How many texts get sent through the HF models per forward pass (default 32) |

## Dependencies

- `parsers.parsers.parse_warc_stream_fastwarc` — parses WARC data from a live HTTP stream (see `parsers.md`).
- `extractors.extractors.extract_html_fields`, `detect_languages_batch`, `detect_dialects_batch` — see `extractors.md`.
- `db.db_handler.init_db`, `save_batch_records`, `get_connection` — see `db_handler.md`.
- `utils.is_target_language` — returns `True` only for `{"ar", "en", "fr"}`.

## Functions

| Function | Description |
|---|---|
| `make_record_id(record, url, text)` | Returns `record["record_id"]` if present, else a deterministic MD5-based fallback id (`"gen-<hash>"`) so records never collide in the DB. |
| `get_cc_news_paths()` | Downloads + decompresses the CC-NEWS `warc.paths.gz` manifest and returns the first `NUM_FILES` full HTTPS URLs. |
| `flush_hf_batch(pending, db_batch)` | Runs batched language detection (and, for Arabic hits, batched dialect detection) on `pending`, keeps only ar/en/fr records, appends them to `db_batch`, clears `pending`. Returns count of records dropped by the language filter. |
| `run_streaming_pipeline()` | Orchestrates the full run: manifest → per-file streaming → per-record extraction → batched HF detection → batched DB writes → final summary. Entry point when run as a script. |

## Pipeline flow (per record)

```
WARC record (from CC-NEWS HTTP stream)
        │
        ▼
extract_html_fields()          ── reject if word_count < 80 or empty text
        │
        ▼
queued in pending_hf list
        │  (once len == HF_BATCH_SIZE)
        ▼
flush_hf_batch()
   ├─ detect_languages_batch()          → language field set
   ├─ detect_dialects_batch()  (ar only) → arabic_dialect field set
   └─ is_target_language() filter        → drop non ar/en/fr
        │
        ▼
queued in db_batch list
        │  (once len == DB_BATCH_SIZE)
        ▼
save_batch_records()  → PostgreSQL (websites, authors, pages, metadata, content)
```

## Error handling

Every stage is wrapped so a single failure doesn't kill the whole run:

- **Extraction failure** (`extract_html_fields` raises) → record skipped, logged as `[EXTRACT ERROR]`, counted in `skipped`.
- **Streaming failure** (a WARC file can't be downloaded/parsed) → that file is skipped entirely, logged as `[STREAM ERROR]`, loop continues with the next file.
- **DB write failure** (`save_batch_records` raises) → batch is dropped, logged as `[DB ERROR]`, counted in `db_errors`, loop continues.
- **Manifest fetch failure** (`get_cc_news_paths` raises) → the whole run aborts early with a printed error (no partial run attempted).

Final printed summary includes: total saved records, total skipped at
extraction, total skipped by language filter, number of failed batches,
and total execution time.

## Notes / Gotchas

- A **single PostgreSQL connection is reused for the entire run** (opened once via `get_connection()`, closed once in `finally`), not reopened per batch — this was an earlier fix for a major performance issue.
- Each CC-NEWS WARC file is fully buffered into memory before parsing (see `parse_warc_stream_fastwarc`), so very large `NUM_FILES` values increase peak RAM usage and total run time.
- The language filter (`is_target_language`) is applied **after** the HF model already ran on the whole batch — there's no way to know the language before running the model, so this doesn't save inference time, only storage.
- `record["record_id"]` (i.e. WARC-Record-ID) is what gives `ON CONFLICT (warc_record_id) DO UPDATE` in `db_handler.save_batch_records` its effect — safe re-runs won't duplicate `pages` rows, only `metadata`/`content` (which use `DO NOTHING`) won't be overwritten on conflict.

## Relationship to other pipeline scripts

- Distinct from the older **local-file** pipeline (which read from a `.warc.gz` file on disk via `parse_warc_fastwarc` instead of streaming from CC-NEWS).
- Distinct from the **CC-NEWS preview script**, which also streams from CC-NEWS but only prints sample records to the console — it does not write to the database and does not batch HF calls (uses `extract_selectolax`, one record at a time).

## Used by

- Run directly as the main data-collection entry point:
  ```bash
  python pipeline/stream_to_db.py
  ```
