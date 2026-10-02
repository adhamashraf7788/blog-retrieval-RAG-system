"""
stream_to_db_parallel.py

Parallel CC-NEWS -> PostgreSQL streamer (N worker processes, one WARC each).

Changes in this revision (less noise in the DB):
    - is_article_url(): home pages, tag/category/search/video pages,
      advertorial URLs and blocked domains are skipped BEFORE extraction
      (also saves CPU).
    - Near-duplicate articles (same first ~400 characters, e.g. wire copy
      syndicated on many sites) are stored once per WARC file.
    - extractors.py no longer falls back to raw <body> text and now keeps
      paragraph breaks; its global floor is 50 words, so --min-words-ar 50
      really works. --min-words-ar below 50 has no effect.

Earlier changes (goal: balanced languages, ~1000 chunks each):
    - EN down-sampling: only a fraction of English articles is kept
      (--en-keep-rate, default 0.1) so the DB fills with Arabic/French
      instead of tens of thousands of English chunks nobody needs.
    - Lower word threshold for Arabic (--min-words-ar, default 50).
      Non-Arabic articles still need --min-words (default 80).
      Language is only known after detection, so the pre-filter uses the
      lower value and non-Arabic short articles are dropped afterwards.
    - Per-language "queued for DB" counters printed at the end, so you can
      see how many ar / en / fr articles this run added.
    - FIX (Windows): settings are now passed to workers as function
      arguments. Before, they were module globals set in the parent, which
      spawned worker processes on Windows do NOT inherit (they re-import
      the module and get the defaults).
    - Resume is still file-level via .processed_warc_files.json.
      Already-processed WARC files are skipped, so to add data, raise
      --num-files (new files are picked up automatically).
"""

import argparse
import re
import sys
import os
import time
import json
import hashlib
import gzip
import random
import urllib.request
import urllib.error
from urllib.parse import urlparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from parsers.parsers import parse_warc_stream_fastwarc
from extractors.extractors import (
    extract_html_fields,
    get_domain,
    detect_languages_batch,
    detect_dialects_batch,
)
from db.db_handler import init_db, save_batch_records, get_connection
from utils import is_target_language

# ---------------------------------------------------------------------------
# Defaults (overridden by CLI / env)
# ---------------------------------------------------------------------------
DEFAULT_YEAR = os.environ.get("CC_NEWS_YEAR", "2026")
DEFAULT_MONTH = os.environ.get("CC_NEWS_MONTH", "05")
DEFAULT_NUM_FILES = int(os.environ.get("CC_NEWS_NUM_FILES", "1"))
DEFAULT_MAX_WORKERS = int(os.environ.get("CC_NEWS_MAX_WORKERS", "1"))
DEFAULT_TIMEOUT = int(os.environ.get("CC_NEWS_TIMEOUT", "120"))  # seconds
DEFAULT_EN_KEEP_RATE = float(os.environ.get("CC_NEWS_EN_KEEP_RATE", "0.1"))
DEFAULT_MIN_WORDS = int(os.environ.get("CC_NEWS_MIN_WORDS", "80"))
DEFAULT_MIN_WORDS_AR = int(os.environ.get("CC_NEWS_MIN_WORDS_AR", "50"))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DB_BATCH_SIZE = 500
HF_BATCH_SIZE = 32

STATE_FILE = os.path.join(BASE_DIR, ".processed_warc_files.json")

# Domains that are never useful for this project (SEO / promo content).
BLOCKED_DOMAINS = {
    "sixactualites.fr", "lg.com",
    "realting.com",        # real-estate UI text
    "kinoafisha.info",     # cinema ticket-booking pages
}

# Listing pages, media pages and sponsored content are not articles.
_NON_ARTICLE_URL_RE = re.compile(
    r"/(?:tags?|categor(?:y|ies)|topics?|authors?|search|page|videos?"
    r"|galler(?:y|ies)|horoscope\w*|live|podcasts?|forums?)(?:/|$)"
    r"|/publi-|/sponsored|advertorial|/partner-content|[?&](?:s|q|search)=",
    re.IGNORECASE,
)


def is_article_url(url: str) -> bool:
    if not url or get_domain(url) in BLOCKED_DOMAINS:
        return False
    p = urlparse(url)
    if p.path in ("", "/"):          # site home pages
        return False
    return not _NON_ARTICLE_URL_RE.search(
        p.path + ("?" + p.query if p.query else "")
    )


def make_record_id(record, url, text):
    rid = record.get("record_id", "")
    if rid:
        return rid
    basis = (url or "") + (text[:200] if text else "")
    return "gen-" + hashlib.md5(basis.encode("utf-8", errors="ignore")).hexdigest()


def get_cc_news_paths(year: str, month: str, num_files: int, timeout: int):
    paths_url = (
        f"https://data.commoncrawl.org/crawl-data/CC-NEWS/"
        f"{year}/{month}/warc.paths.gz"
    )
    print(f"Fetching CC-NEWS WARC paths manifest for {year}/{month}...")
    req = urllib.request.Request(paths_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            gz_bytes = response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download WARC paths manifest: {e}") from e

    paths_list = gzip.decompress(gz_bytes).decode("utf-8").splitlines()
    selected_paths = paths_list[:num_files]
    return [f"https://data.commoncrawl.org/{p}" for p in selected_paths]


def flush_hf_batch(pending, db_batch, en_keep_rate, min_words, lang_counter):
    """
    Detect languages and apply filters for the pending batch and move wanted items to db_batch.

    Drops:
      - languages that are not target languages
      - English articles beyond en_keep_rate (random down-sampling)
      - non-Arabic articles shorter than min_words (Arabic already passed
        its own, lower, threshold in the pre-filter)
    Returns the number of dropped items.
    """
    if not pending:
        return 0
    texts = [p["clean_text"] for p in pending]
    languages = detect_languages_batch(texts)

    arabic_positions = [i for i, lang in enumerate(languages) if lang == "ar"]
    if arabic_positions:
        arabic_texts = [texts[i] for i in arabic_positions]
        dialects = detect_dialects_batch(arabic_texts)
        for pos, dialect in zip(arabic_positions, dialects):
            pending[pos]["arabic_dialect"] = dialect

    filtered_out = 0
    for item, lang in zip(pending, languages):
        item["language"] = lang

        if not is_target_language(lang):
            filtered_out += 1
            continue
        if lang != "ar" and item["word_count"] < min_words:
            filtered_out += 1
            continue
        if lang == "en" and random.random() > en_keep_rate:
            filtered_out += 1
            continue

        db_batch.append(item)
        lang_counter[lang] += 1

    pending.clear()
    return filtered_out


def _empty_result(file_url, stream_error):
    return {
        "file_url": file_url,
        "stored": 0,
        "skipped": 0,
        "skipped_lang": 0,
        "db_errors": 0,
        "extract_errors": 0,
        "records_seen": 0,
        "lang_counts": {},
        "stream_error": stream_error,
    }


def process_one_warc_file(
    file_url: str,
    timeout: int,
    max_workers: int,
    en_keep_rate: float,
    min_words: int,
    min_words_ar: int,
) -> dict:
    """
    Process a single WARC file inside a worker process.
    All settings arrive as arguments (module globals are NOT shared with
    spawned workers on Windows).
    """
    try:
        import torch
        torch.set_num_threads(max(1, cpu_count() // max(1, max_workers)))
    except ImportError:
        pass

    # Pre-filter uses the lowest threshold, because language is unknown yet.
    prefilter_words = min(min_words, min_words_ar)

    tag = file_url.rsplit("/", 1)[-1][:30]
    print(f"[{tag}] Worker started: opening DB connection...")

    conn = get_connection()
    pending_hf = []
    db_batch = []
    seen_fp = set()        # near-duplicate filter (per WARC file)
    lang_counter = Counter()
    stored = 0
    skipped = 0
    skipped_lang = 0
    db_errors = 0
    extract_errors = 0
    records_seen = 0
    last_print = time.perf_counter()
    stream_error = None

    try:
        stream_req = urllib.request.Request(file_url, headers=HEADERS)
        try:
            stream_response = urllib.request.urlopen(stream_req, timeout=timeout)
        except urllib.error.URLError as e:
            stream_error = f"network: {e}"
            print(f"[STREAM ERROR] {file_url}: {stream_error}")
            return _empty_result(file_url, stream_error)

        with stream_response:
            try:
                record_iter = parse_warc_stream_fastwarc(stream_response)
            except Exception as e:
                stream_error = f"parse: {e}"
                print(f"[STREAM ERROR] {file_url}: {stream_error}")
                return _empty_result(file_url, stream_error)

            for record in record_iter:
                records_seen += 1
                if not is_article_url(record.get("url", "")):
                    skipped += 1
                    continue
                try:
                    fields = extract_html_fields(record)
                except Exception as e:
                    extract_errors += 1
                    skipped += 1
                    if extract_errors <= 3:
                        print(f"[{tag}] EXTRACT error: {e}")
                    continue

                word_count = fields.get("word_count", 0)
                clean_text = fields.get("clean_text", "")
                if word_count < prefilter_words or not clean_text:
                    skipped += 1
                    continue

                fp = hashlib.md5(
                    re.sub(r"\W+", "", clean_text[:400]).lower()
                    .encode("utf-8", "ignore")
                ).hexdigest()
                if fp in seen_fp:
                    skipped += 1
                    continue
                seen_fp.add(fp)

                url = record.get("url", "")
                pending_hf.append({
                    "record_id": make_record_id(record, url, clean_text),
                    "url": url,
                    "title": fields.get("title", "N/A"),
                    "author": fields.get("author", "N/A"),
                    "published_date": fields.get("published_date", "N/A"),
                    "cleaned_text": clean_text,
                    "clean_text": clean_text,
                    "word_count": word_count,
                    "char_count": fields.get("char_count", 0),
                    "links_count": fields.get("links_count", 0),
                    "headings": fields.get("headings_sample", []),
                    "arabic_dialect": "N/A",
                })

                if len(pending_hf) >= HF_BATCH_SIZE:
                    skipped_lang += flush_hf_batch(
                        pending_hf, db_batch, en_keep_rate, min_words, lang_counter
                    )

                if len(db_batch) >= DB_BATCH_SIZE:
                    try:
                        save_batch_records(db_batch, conn=conn)
                        stored += len(db_batch)
                    except Exception as e:
                        db_errors += 1
                        print(f"[DB ERROR] {tag}: {e}")
                    finally:
                        db_batch = []

                now = time.perf_counter()
                if now - last_print >= 10:
                    print(
                        f"[{tag}] records seen: {records_seen:,} | "
                        f"stored: {stored:,} | skipped: {skipped:,} | "
                        f"queued by lang: {dict(lang_counter)}"
                    )
                    last_print = now

        # Final flush for this file
        skipped_lang += flush_hf_batch(
            pending_hf, db_batch, en_keep_rate, min_words, lang_counter
        )
        if db_batch:
            try:
                save_batch_records(db_batch, conn=conn)
                stored += len(db_batch)
            except Exception as e:
                db_errors += 1
                print(f"[DB ERROR] {tag} (final batch): {e}")

    except Exception as e:
        stream_error = f"unexpected: {e}"
        print(f"[STREAM ERROR] {file_url}: {stream_error}")
    finally:
        conn.close()

    return {
        "file_url": file_url,
        "stored": stored,
        "skipped": skipped,
        "skipped_lang": skipped_lang,
        "db_errors": db_errors,
        "extract_errors": extract_errors,
        "records_seen": records_seen,
        "lang_counts": dict(lang_counter),
        "stream_error": stream_error,
    }


def load_processed_files() -> set:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def mark_file_processed(file_url: str, processed: set):
    processed.add(file_url)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)


def run_parallel_pipeline(
    year: str,
    month: str,
    num_files: int,
    max_workers: int,
    timeout: int,
    en_keep_rate: float,
    min_words: int,
    min_words_ar: int,
):
    init_db()

    try:
        warc_urls = get_cc_news_paths(year, month, num_files, timeout)
    except Exception as e:
        print(f"Failed to fetch CC-NEWS manifest: {e}")
        return

    processed = load_processed_files()
    remaining = [u for u in warc_urls if u not in processed]
    skipped_already_done = len(warc_urls) - len(remaining)

    print(
        f"Selected {len(warc_urls)} CC-NEWS WARC files "
        f"({skipped_already_done} already done, {len(remaining)} remaining)."
    )
    print(
        f"Workers={max_workers} | timeout={timeout}s | year={year} month={month}\n"
        f"EN keep rate={en_keep_rate} | min words={min_words} "
        f"(Arabic: {min_words_ar})"
    )

    if not remaining:
        print("Nothing to do -- all requested files were already processed. "
              "Raise --num-files to pull new files.")
        return

    start_time = time.perf_counter()
    total_stored = 0
    total_skipped = 0
    total_skipped_lang = 0
    total_db_errors = 0
    total_extract_errors = 0
    total_lang = Counter()
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_warc_file,
                url, timeout, max_workers, en_keep_rate, min_words, min_words_ar,
            ): url
            for url in remaining
        }

        for i, future in enumerate(as_completed(futures), 1):
            file_url = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"[WORKER CRASHED] {file_url}: {e}")
                continue

            total_stored += result["stored"]
            total_skipped += result["skipped"]
            total_skipped_lang += result["skipped_lang"]
            total_db_errors += result["db_errors"]
            total_extract_errors += result.get("extract_errors", 0)
            total_lang.update(result.get("lang_counts", {}))
            completed_count += 1

            if result["records_seen"] > 0 and not result.get("stream_error"):
                mark_file_processed(file_url, processed)
            else:
                print(
                    f"[WARNING] {file_url}: records_seen={result['records_seen']} "
                    f"stream_error={result.get('stream_error')!r} -- "
                    f"NOT marked as processed, will retry next run"
                )

            elapsed = time.perf_counter() - start_time
            avg_per_file = elapsed / completed_count
            files_left = len(remaining) - completed_count
            eta_seconds = avg_per_file * files_left
            eta_str = (
                time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                if eta_seconds < 86400
                else f"{eta_seconds / 3600:.1f}h"
            )

            print(
                f"[{i}/{len(remaining)}] Done: {file_url} "
                f"(+{result['stored']:,} stored, {result['skipped']:,} skipped, "
                f"{result['skipped_lang']:,} lang-filtered, "
                f"{result['db_errors']} db errors, "
                f"by lang: {result.get('lang_counts', {})}) | "
                f"Running total stored: {total_stored:,} | "
                f"Avg/file: {avg_per_file:.1f}s | ETA remaining: {eta_str}"
            )

    end_time = time.perf_counter()
    print("\nCompleted Processing!")
    print(f"Total Saved Records     : {total_stored:,}")
    print(f"  by language (queued)  : {dict(total_lang)}")
    print(f"Total Skipped (extract) : {total_skipped:,}")
    print(f"Total Skipped (lang/EN) : {total_skipped_lang:,}")
    print(f"Extract Errors          : {total_extract_errors}")
    print(f"Batches Failed (DB)     : {total_db_errors}")
    print(f"Total Execution Time    : {end_time - start_time:.2f} seconds")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream CC-NEWS WARC files into PostgreSQL (parallel)."
    )
    p.add_argument("--year", default=DEFAULT_YEAR,
                   help=f"CC-NEWS year (default: {DEFAULT_YEAR})")
    p.add_argument("--month", default=DEFAULT_MONTH,
                   help=f"CC-NEWS month zero-padded (default: {DEFAULT_MONTH})")
    p.add_argument("--num-files", type=int, default=DEFAULT_NUM_FILES,
                   help=f"How many WARC files in total to cover "
                        f"(default: {DEFAULT_NUM_FILES}); already-done files "
                        f"are skipped")
    p.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help=f"Parallel worker processes (default: {DEFAULT_MAX_WORKERS})")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"HTTP timeout seconds (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--en-keep-rate", type=float, default=DEFAULT_EN_KEEP_RATE,
                   help="Fraction of English articles to keep, 0-1 "
                        f"(default: {DEFAULT_EN_KEEP_RATE})")
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                   help=f"Min words for non-Arabic (default: {DEFAULT_MIN_WORDS})")
    p.add_argument("--min-words-ar", type=int, default=DEFAULT_MIN_WORDS_AR,
                   help=f"Min words for Arabic (default: {DEFAULT_MIN_WORDS_AR})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_parallel_pipeline(
        year=args.year,
        month=args.month,
        num_files=args.num_files,
        max_workers=args.workers,
        timeout=args.timeout,
        en_keep_rate=args.en_keep_rate,
        min_words=args.min_words,
        min_words_ar=args.min_words_ar,
    )