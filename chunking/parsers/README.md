# `parsers.py`

## Overview

This module is the **entry point of the pipeline**: it reads raw WARC
(Web ARChive) data — either from a local file or a live HTTP stream (e.g.
Common Crawl / CC-NEWS) — and yields only `text/html` **response** records
as plain dicts, ready for the extraction stage (`extractors/extractors.py`).

It does **not**:
- Decode or clean HTML (see `utils.decode_and_validate`)
- Extract titles/authors/text/language (see `extractors/extractors.py`)
- Touch the database

It only knows how to walk a WARC archive and hand back raw bytes + metadata
for the records that matter.

## Dependencies

- [`fastwarc`](https://pypi.org/project/fastwarc/) — fast, C++-backed WARC
  parser. Used by `parse_warc_fastwarc` and `parse_warc_stream_fastwarc`.
- [`warcio`](https://pypi.org/project/warcio/) — pure-Python WARC parser.
  Used by `parse_warc_warcio` as an alternative implementation.

## Functions

| Function | Input source | Description |
|---|---|---|
| `parse_warc_fastwarc(file_path)` | Local `.warc` / `.warc.gz` file | Main parser used by the local-file streaming pipeline (`pipeline/stream_to_db.py`). Fast (FastWARC-backed). |
| `parse_warc_warcio(file_path)` | Local `.warc` / `.warc.gz` file | Alternative parser using `warcio` instead of FastWARC. Slower, kept for comparison |
| `parse_warc_stream_fastwarc(stream)` | Open network stream (e.g. `urllib.request.urlopen(...)` response) | Used for streaming WARC files directly from Common Crawl / CC-NEWS without saving them to disk first. |

### Common output shape

All three functions are **generators** that yield one `dict` per matching
HTML response record:

```python
{
    "url": str | None,          # WARC-Target-URI
    "warc_date": str | None,    # WARC-Date (crawl time, NOT publish date)
    "content_type": str,        # HTTP Content-Type header
    "record_id": str | None,    # WARC-Record-ID (used as DB unique key)
    "raw_bytes": bytes,         # raw, still-encoded HTTP response body
}
```

Only `response` records whose `Content-Type` contains `text/html` are
yielded — everything else (WARC `request`/`metadata` records, non-HTML
responses like images/PDFs/JSON) is skipped silently.

## Notes 

- **Bug fix in `parse_warc_warcio`**: warcio record objects expose
  `.rec_headers` (WARC-level headers) and `.http_headers` (HTTP-level
  headers) — there is **no** `.headers` attribute. The original code called
  `record.headers.get('WARC-Record-ID')`, which raised `AttributeError` on
  every record. Fixed to use `record.rec_headers.get_header('WARC-Record-ID')`.
- **Memory usage in `parse_warc_stream_fastwarc`**: the entire input stream
  is buffered into memory (`io.BytesIO(stream.read())`) before parsing,
  because FastWARC's `ArchiveIterator` requires a seekable stream and raw
  HTTP response objects usually aren't. This is fine for individual CC-NEWS
  WARC segments but means peak RAM usage scales with file size.
- `raw_bytes` is **not decoded** — it must be passed through
  `utils.decode_and_validate()` (or equivalent) before any text processing.

## Used by

- `pipeline/stream_to_db.py` (local file mode) → `parse_warc_fastwarc`
- CC-NEWS streaming scripts (`cc_news_preview.py`, CC-NEWS DB pipeline) →
  `parse_warc_stream_fastwarc`
