# stream_pipeline.py

## Overview
A standalone demo/inspection script — not part of the database pipeline. It streams live WARC files directly from Common Crawl's **CC-NEWS** dataset over HTTP (no local download, no disk writes), extracts a handful of articles per language, and pretty-prints them to the console so you can eyeball extraction quality (title, author, dates, language, word count, etc.) without touching Postgres.

It does not:
- Write anything to disk or to the database
- Process a whole WARC file's worth of records — it stops once it has printed `MAX_SAMPLES_PER_LANG` samples for each target language
- Use batched Hugging Face detection — it calls the single-record extractor, so it's fine for a quick look but not meant for bulk processing

## Dependencies
- `parsers.parse_warc_stream_fastwarc` — parses WARC records directly from an open HTTP response stream.
- `extractors.extract_selectolax` — the single-record, all-in-one extractor (HTML parsing + HF language/dialect detection together).
- `utils.is_target_language` — filters to only `ar` / `en` / `fr`.
- `utils.fix_arabic_console_text` — reshapes + applies BiDi so Arabic prints correctly (not garbled/reversed) in a terminal.
- Standard library: `urllib.request` (HTTP), `gzip` (decompressing the WARC paths manifest).

## Configuration (module-level constants)

| Constant | Meaning |
|---|---|
| `YEAR`, `MONTH` | Which CC-NEWS crawl month to pull the WARC file list from. |
| `NUM_FILES` | How many WARC files (from the start of that month's manifest) to stream through. Currently `1`. |
| `MAX_SAMPLES_PER_LANG` | Stop printing samples for a language once this many have been shown (per run, across all files). Currently `3`. |
| `PATHS_URL` | The `warc.paths.gz` manifest listing every WARC file for `YEAR`/`MONTH`. |
| `HEADERS` | A browser-like `User-Agent`, sent with every request to Common Crawl. |

## Functions

| Function | Description |
|---|---|
| `run_streaming_pipeline()` | The only function in the script. 1) Downloads and decompresses the CC-NEWS WARC path manifest for `YEAR`/`MONTH`. 2) Takes the first `NUM_FILES` paths. 3) For each path, opens a live HTTP stream and passes it to `parse_warc_stream_fastwarc`. 4) For every yielded record, runs `extract_selectolax`, and if the language is one of the target languages **and** the article is ≥80 words, prints a formatted sample block (title, author, dates, word/char counts, links, headings, text preview) — up to `MAX_SAMPLES_PER_LANG` per language. 5) Prints a final summary (`total_processed_articles`, `lang_sample_counts`). Network/parse errors on a given WARC file are caught and printed per-file so one bad file doesn't stop the whole run. |

## Notes
- **This script uses the slow, single-record extraction path on purpose.** It's meant to preview a handful of articles, not process a whole file — batching (`extract_html_fields` + `detect_languages_batch`) wouldn't make sense here since it only ever needs up to `3 × 3 = 9` printed samples.
- `total_processed_articles` counts **every** record that passed the language + word-count filter, even after `MAX_SAMPLES_PER_LANG` has already been hit for that language — so it can end up much higher than the number of samples actually printed.
- If a WARC file fails to stream (network error, bad gzip, etc.), the script logs `Error streaming file {path}: {e}` and moves on to the next file rather than crashing.
- Arabic output is passed through `fix_arabic_console_text` before printing (title, author, headings, text preview) — English/French text is printed as-is.

## Used by
Standalone script — run directly (`python stream_pipeline.py`). Not imported by anything else in the project.
<img width="1248" height="823" alt="image" src="https://github.com/user-attachments/assets/62e63bc6-5ef5-4db8-a138-d0468fd47d63" />
<img width="1272" height="756" alt="image" src="https://github.com/user-attachments/assets/b60803af-fb37-44cd-a77d-5fd278fc2146" />


