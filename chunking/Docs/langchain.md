# `langchain.py`

## Overview

This script loads finished records out of PostgreSQL and converts them into
[LangChain](https://www.langchain.com/) `Document` objects, ready to be
chunked for embedding / RAG pipelines. It:
just a sample code test
1. Queries `pages` (joined with `websites`, `authors`, `metadata`, `content`)
   to pull the full text + metadata for every page.
2. Wraps each row into a LangChain `Document` (`page_content` + `metadata`
   dict).
3. Splits the documents into smaller overlapping chunks using
   `RecursiveCharacterTextSplitter`.

It's a downstream consumer of the data written by `stream_to_db.py` — it
doesn't fetch or write anything itself, it only reads what's already in the
DB.

## Dependencies

- `db.db_handler.get_connection` — same connection helper used by the main pipeline (see `db_handler.md`).
- `langchain_core.documents.Document`
- `langchain_text_splitters.RecursiveCharacterTextSplitter`

## Functions

| Function | Description |
|---|---|
| `load_documents_from_db()` | Runs the SQL join across `pages`, `websites`, `authors`, `metadata`, `content`, and returns a `List[Document]`. Skips rows where `cleaned_text` is `NULL`. |
| `split_documents_into_chunks(documents, chunk_size=800, chunk_overlap=100)` | Splits a list of `Document`s into smaller chunks with `RecursiveCharacterTextSplitter`, preserving each chunk's parent metadata. |

## Query

```sql
SELECT p.id, c.cleaned_text, p.url, m.title, w.domain, a.name AS author,
       m.language, m.arabic_dialect, m.published_date, p.warc_record_id
FROM pages p
LEFT JOIN websites w ON p.website_id = w.id
LEFT JOIN authors a ON p.author_id = a.id
LEFT JOIN metadata m ON p.id = m.page_id
LEFT JOIN content c ON p.id = c.page_id
```

- `authors` is joined explicitly because `metadata` has no `author` column
  in this schema — the author name lives in `authors.name`, reached via
  `pages.author_id`.
- All joins are `LEFT JOIN` so a `pages` row is never dropped just because
  it's missing a related `metadata`/`content`/`author` row.

## Document shape

Each `Document` has:

```python
Document(
    page_content=text,
    metadata={
        "doc_id": ...,
        "source": url,
        "title": title or "N/A",
        "publisher": source_domain or "Unknown",
        "author": author or source_domain or "Unknown",
        "domain": source_domain or "",
        "language": lang or "",
        "arabic_dialect": arabic_dialect or "N/A",
        "published_date": pub_date or "N/A",
        "warc_record_id": warc_record_id or "",
    },
)
```

`author` falls back to `source_domain`, then `"Unknown"`, so it's never
`None` downstream.

## Pipeline flow

```
PostgreSQL (pages ⨝ websites ⨝ authors ⨝ metadata ⨝ content)
        │
        ▼
load_documents_from_db()
   ├─ skip row if cleaned_text is NULL
   └─ wrap row → Document(page_content, metadata)
        │
        ▼
split_documents_into_chunks()
   └─ RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""])
        │
        ▼
List[Document] chunks, metadata preserved per chunk
```

## Error handling

- **Missing `cleaned_text`**: if a `pages` row has no matching `content`
  row (or `cleaned_text` is `NULL`), that row is silently skipped instead
  of producing an empty `Document`.
- **DB connection**: `conn.close()` runs in a `finally` block, so the
  connection is released even if the query/fetch raises.
- No retry logic — this is a one-shot batch load, not a long-running
  stream, so a query failure just raises and stops the script.

## Notes 

- This script does a **single unbounded `SELECT *`-style fetch** (no
  pagination, no batching) — fine for the current table sizes, but will
  need chunked fetching (e.g. server-side cursor or `LIMIT`/`OFFSET`) if
  the `pages` table grows large enough that the full result set doesn't
  fit comfortably in memory.
- `chunk_size=800` / `chunk_overlap=100` are hardcoded as function
  defaults but exposed as parameters, so callers can override them per
  use case (e.g. different values for embedding-model context limits).
- Every chunk keeps the **same metadata as its parent `Document`**
  (LangChain's splitter copies metadata onto each chunk), so `doc_id` /
  `warc_record_id` can be used to trace a chunk back to its source page.

## Used by

Run directly as a standalone smoke test — loads all documents, chunks
them, and prints a preview of the first chunk:

```bash
python /langchain.py
```

Intended to be imported by an embedding/indexing script that calls
`load_documents_from_db()` and `split_documents_into_chunks()` and then
pushes the resulting chunks into a vector store.

<img width="1268" height="398" alt="WhatsApp Image 2026-08-25 at 3 44 24 PM" src="https://github.com/user-attachments/assets/f828f13b-0ed2-4f91-a3dc-acf46ce9449c" />

