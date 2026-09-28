# `db_summary.py`

## Overview

A small command-line diagnostic script that connects to the pipeline's
PostgreSQL database and prints:

1. Row counts for all five tables: `websites`, `authors`, `pages`,
   `metadata`, `content`.
2. Up to 5 sample rows (joined across `pages` → `websites` → `authors` →
   `metadata`) showing domain, author, language, word count, and title.

Used purely for manual verification/debugging — e.g. to quickly check
"did the pipeline actually write anything?" without opening `psql`.

## Dependencies

- `db.db_handler.get_connection` — same DB connection used by the rest of the pipeline.

## Usage

```bash
python db/db_summary.py
```

## Functions

| Function | Description |
|---|---|
| `print_db_summary()` | Connects to the DB, prints table row counts, and prints a small formatted sample of joined records. No return value — everything is printed to stdout. |

## Behavior details

- **Empty database handling**: if the sample query returns no rows (i.e.
  `pages`/`metadata` are empty), the script prints an explicit message
  telling you to check `stream_to_db.py`'s console output for
  `[DB ERROR]` / `[EXTRACT ERROR]` lines, instead of silently printing an
  empty table. This was a **fix** over the original version, which gave no
  hint at all when nothing had been stored yet.
- **NULL-safety fix**: `domain`, `author`, `lang`, `title` could be `NULL`
  in the database. The original code called `len()` / slicing on these
  values directly, which raised `TypeError` the moment any field was
  `NULL` and crashed the whole summary. This version coerces each to
  `"N/A"` (and `word_count` to `0`) before formatting.
- **Sample query joins**: `pages` is joined to `websites` and `metadata`
  with a plain `JOIN` (row excluded from the sample if missing), but to
  `authors` with a `LEFT JOIN` (author is optional per the schema).

## Example output

<img width="992" height="455" alt="WhatsApp Image 2026-08-24 at 3 41 47 AM" src="https://github.com/user-attachments/assets/d9a75afc-6ca1-42d3-847f-891e325f15c0" />

## Notes 

- This script only **reads** — it never modifies the database.
- It relies on the same `DB_CONFIG` / connection setup as `db/db_handler.py`;
  if that config is wrong, this script will fail to connect the same way
  any other DB script would.
- Not imported anywhere else in the pipeline — it's a standalone diagnostic
  entry point (`if __name__ == "__main__":`).

## Used by

- Run manually  after pipeline runs, to sanity-check that data
  landed correctly (e.g. after the `execute_values(..., fetch=True)` bug
  fix in `db_handler.py`, this script was what surfaced that `metadata`/
  `content` were empty while `pages` wasn't).
