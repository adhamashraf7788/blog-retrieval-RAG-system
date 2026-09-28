# utils.py

## Overview
Shared, low-level helper functions used across the whole pipeline: safe byte-to-text decoding for raw HTML, encoding-quality/Mojibake checks, Arabic-script validation, console-friendly Arabic text rendering, a simple language allow-list, and a process memory helper. Nothing in this file is pipeline-stage-specific — `parsers.py`, `extractors.py`, and the CC-NEWS preview script all import from it.

It does not:
- Parse WARC files (see `parsers.py`)
- Parse/extract HTML structure (see `extractors.py`)
- Know anything about the database

## Dependencies
- **charset_normalizer** (`from_bytes`) — encoding detection fallback when UTF-8 decoding fails.
- **arabic_reshaper** + **python-bidi** (`get_display`) — reshape and reorder Arabic text so it renders correctly (not disconnected letters / reversed) in a plain terminal.
- **psutil** — process memory inspection.
- Standard library: `os`, `re`.

## Functions

| Function | Input | Output | Description |
|---|---|---|---|
| `decode_and_validate(raw_bytes)` | raw HTTP response bytes | `(text \| None, status)` | The main decoding entry point, used by both extractors. Tries, in order: (1) strict UTF-8, (2) `charset_normalizer.from_bytes` best-guess encoding (rejecting a low-confidence Latin-1/ISO-8859-1 guess since that's usually a false positive on non-Latin pages), (3) Windows-1256 (common for older Arabic pages). If nothing decodes, returns `(None, "decode_failed")`. If it decodes but fails the `is_clean_encoding` check, returns `(None, "encoding_corrupted")`. On success, returns `(text, "success")`. |
| `decode_and_clean_html(raw_bytes)` | raw bytes | `str` | Thin legacy wrapper around `decode_and_validate` that just returns the text (or `""` on any failure), dropping the status. Kept for backwards compatibility with older callers. |
| `is_clean_encoding(text)` | decoded text | `bool` | Rejects text containing the Unicode replacement character (`\ufffd`, a sign decoding already went wrong) or with more than 3% of characters being common Mojibake artifacts (stray Latin-1 accented/symbol characters that show up when text was decoded with the wrong codec). |
| `is_valid_arabic_text(text)` | text | `bool` | Sanity check used after a model says "this is Arabic": looks at the first 500 characters and requires at least 40% of alphabetic characters to be in the Arabic Unicode ranges. Guards against short/mixed text being mislabeled as Arabic. Returns `False` for text under 20 characters. |
| `is_target_language(lang_code)` | language code string | `bool` | Simple allow-list check: `True` only for `"ar"`, `"en"`, `"fr"` (case-insensitive). Used by the CC-NEWS preview script to decide what to print. |
| `fix_arabic_console_text(text)` | text | `str` | Reshapes Arabic letters into their connected forms and applies the BiDi algorithm so right-to-left text prints correctly in a left-to-right terminal. Returns the original text unchanged on any error, and `""` for empty input. Purely cosmetic — only affects how text looks when printed, never what's stored in the database. |
| `get_rss_memory_mb()` | — | `float` | Returns the current process's Resident Set Size (actual physical RAM in use) in megabytes, via `psutil`. Used for diagnostics/logging (for comparison in benchmark_runner). |

## Notes
- **Decoding order matters**: strict UTF-8 is tried first because it's unambiguous when it succeeds; `charset_normalizer` is the general-purpose fallback; Windows-1256 is tried last specifically because it's a strong signal for older/mis-served Arabic pages that `charset_normalizer` sometimes misidentifies as Latin-1.
- `is_clean_encoding`'s Mojibake character set is a fixed list of Latin-1 "leftover" characters — pages that legitimately use lots of accented Latin characters (e.g. French text) could in theory trip this if it's ever very dense, but the 3% threshold is deliberately loose to avoid false positives on normal French/Spanish content.
- `fix_arabic_console_text` is purely for terminal/console display — it should never be applied to text before it's saved to the database, since `arabic_reshaper`/`get_display` change the character order for visual rendering, not for storage or search.

## Used by
- `parsers.py` — (indirectly, no direct import, but its output feeds `decode_and_validate`)
- `extractors.py` → `decode_and_validate`, `is_valid_arabic_text`
- CC-NEWS preview script → `is_target_language`, `fix_arabic_console_text`
