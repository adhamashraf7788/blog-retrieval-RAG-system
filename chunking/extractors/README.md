# extractors.py

## Overview
This module is the extraction stage of the pipeline. It takes the raw dicts produced by `parsers.py` (`url`, `warc_date`, `record_id`, `raw_bytes`) and turns them into clean, structured article data: decoded HTML → title, author, publish date, clean text, word/char counts, links, headings — plus language and Arabic-dialect detection via Hugging Face models.

It does not:
- Read WARC files (see `parsers/parsers.py`)
- Decode/validate raw bytes itself — it delegates to `utils.decode_and_validate`
- Touch the database (see `db/db_handler.py`)

It only knows how to turn one (or many) raw HTML records into structured metadata + text.

Two extraction modes are provided:
 
- **Batched mode** (used by the optimized DB streaming pipeline): HTML
  parsing (`extract_html_fields`) is separated from language/dialect
  detection (`detect_languages_batch` / `detect_dialects_batch`), so the HF
  models can be run once per *batch* of records instead of once per record.
- **Single-record mode** (`extract_selectolax`): HTML parsing + language +
  dialect detection all in one call. Used by scripts that process one
  record at a time (e.g. the CC-NEWS preview script).


## Dependencies
- **selectolax** — fast HTML parser (`HTMLParser`). Used by the main extraction functions.
- **beautifulsoup4 (bs4) + lxml** — used only by the legacy/benchmark `extract_bs4_lxml`.
- **transformers** — Hugging Face `pipeline()`, used for language detection and Arabic dialect detection. Imported lazily (only when a detector is first needed), and models are downloaded/cached on first use.
- **utils.py** — `decode_and_validate` (safe byte→text decoding) and `is_valid_arabic_text` (sanity-check on Arabic script ratio).

## Language / dialect models (lazy-loaded)

| Function | Model | Purpose |
|---|---|---|
| `get_lang_detector()` | `papluca/xlm-roberta-base-language-detection` | General language classification (ar / en / fr / other) |
| `get_dialect_detector()` | `CAMeL-Lab/bert-base-arabic-camelbert-mix-did` | Arabic dialect classification, only run on text already detected as Arabic |

Both are loaded once into module-level globals (`hf_lang_detector`, `arabic_dialect_detector`) the first time they're needed, and reused after that — they are **not** reloaded per record.

## Functions

### Language / dialect detection

| Function | Input | Description |
|---|---|---|
| `detect_language_hf(text)` | single string | Detects language for **one** text. Pre-filters (skips the model entirely if text is too short or has no Arabic/Latin characters), then runs one HF forward pass. Returns `"ar"`, `"en"`, `"fr"`, `"corrupted"` (looked Arabic but failed the script-ratio check), `"ignored"` (no Arabic/Latin script at all), or `"unknown"` (model error / anything else). |
| `detect_arabic_dialect(text)` | single string | Runs the dialect model on **one** Arabic text. Returns the predicted dialect label, or `"MSA"` as a safe fallback on error/empty text. |
| `detect_languages_batch(texts)` | `list[str]` | Batched version of `detect_language_hf`. Applies the same cheap pre-filter per text, then sends everything that survives it through **one** HF call (internally chunked by `HF_BATCH_SIZE`) instead of one call per text. Returns a list of labels in the same order as the input. This is the function the streaming pipeline uses for real speed. |
| `detect_dialects_batch(texts)` | `list[str]` (Arabic texts only) | Batched version of `detect_arabic_dialect`. Meant to be called only with the subset of texts that `detect_languages_batch` already flagged as `"ar"`. |

### HTML metadata helpers (used internally by the extractors below)

| Function | Description |
|---|---|
| `extract_publication_date(tree)` | Tries, in order: known `<meta>` tags (`article:published_time`, `pubdate`, `date`, etc.) → JSON-LD `datePublished`/`dateCreated` → `<time datetime=...>` / `<time>` text. Returns `"N/A"` if nothing is found. |
| `extract_date_from_text_fallback(tree)` | Last-resort fallback: regex-scans the visible body text for a `YYYY-MM-DD`-like pattern. Used when `extract_publication_date` comes back `"N/A"`. |
| `extract_author(tree)` | Tries known `<meta>` tags (`author`, `article:author`, `dc.creator`, etc.), then falls back to JSON-LD `author` (handles both a plain string and an `{"name": ...}` object/list). Returns `"N/A"` if nothing is found. |

### Main extractors

| Function | Input | Output | Description |
|---|---|---|---|
| `extract_html_fields(record)` | one record dict (from `parsers.py`) | dict — HTML fields only, **no** `language`/`arabic_dialect` keys | **Phase 1** extractor. Decodes the HTML, strips unwanted tags (`script`, `style`, `nav`, etc.), builds `clean_text`, and pulls title/author/date/links/headings. Deliberately does **not** call any HF model — cheap enough to run per-record in a tight loop. Returns early with `word_count`/`clean_text` only if decoding fails (`language: "corrupted"`) or the article is under 80 words (`language: "too_short"`). Used together with `detect_languages_batch` / `detect_dialects_batch` by the streaming pipeline (`pipeline/stream_to_db.py`). |
| `extract_selectolax(record)` | one record dict | dict — same fields as `extract_html_fields` **plus** `language` and `arabic_dialect` | The original all-in-one extractor: does the HTML parsing **and** calls `detect_language_hf`/`detect_arabic_dialect` inline, one record at a time. Kept for callers that process a single record in isolation and don't benefit from batching (e.g. the CC-NEWS preview script). |
| `extract_bs4_lxml(record)` | one record dict | dict — `title`, `clean_text`, `word_count`, `text_length`, `links_count`, `headings_sample`, no language detection | Legacy/benchmark extractor built on BeautifulSoup + lxml instead of selectolax. Not used by the production pipeline; kept around for speed/accuracy comparisons against `extract_selectolax`. |

## Notes
- **Why two extraction paths exist**: `extract_html_fields` + `detect_*_batch` is the fast path (batched HF calls). `extract_selectolax` is the slow-but-simple path (one HF call per record). Both share the same HTML-parsing logic, kept in sync manually — if you change how `clean_text`/title/author/etc. are computed in one, mirror it in the other.
- **80-word threshold** is enforced inside the extractor itself (`extract_html_fields` and `extract_selectolax` both return early with `word_count`/no text if under 80 words), not just in the calling pipeline.
- Arabic language results go through a second check (`is_valid_arabic_text`) even after the model says `"ar"` — if the script ratio doesn't hold up, the record is downgraded to `"corrupted"` rather than trusted as Arabic.
- All HF calls are wrapped in `try/except` and degrade gracefully (`"unknown"` for language, `"MSA"` for dialect) rather than raising — a model/network failure will never crash the pipeline, it'll just mislabel that batch.

## Used by
- `pipeline/stream_to_db.py` → `extract_html_fields`, `detect_languages_batch`, `detect_dialects_batch`
- CC-NEWS preview script → `extract_selectolax`
