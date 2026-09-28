"""
stream_to_db_parallel.py

Parallel version of stream_to_db.py: spins up N worker processes and hands
each one a different CC-NEWS WARC file to stream, extract, and write to
PostgreSQL.

Changes in this revision:
    - NUM_FILES / MAX_WORKERS / YEAR / MONTH configurable via CLI or env
      (no more hardcoded 1/1 for production runs).
    - Network timeouts on manifest + WARC downloads (default 120s) so a
      hung connection cannot block a worker forever.
    - Clearer error categories: STREAM (network/parse), EXTRACT, DB.
    - Resume still file-level via .processed_warc_files.json; a file is
      only marked done if records_seen > 0 (crash mid-file retries next run).
    - torch.set_num_threads capped per worker to avoid CPU oversubscription.
"""

import argparse
import sys
import os
import time
import json
import hashlib
import gzip
import urllib.request
import urllib.error
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from parsers.parsers import parse_warc_stream_fastwarc
from extractors.extractors import (
    extract_html_fields,
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

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DB_BATCH_SIZE = 500
HF_BATCH_SIZE = 32

STATE_FILE = os.path.join(BASE_DIR, ".processed_warc_files.json")

# Module-level copies set in main() so worker processes can see them
# (ProcessPoolExecutor pickles the function; these are read inside the worker).
YEAR = DEFAULT_YEAR
MONTH = DEFAULT_MONTH
NUM_FILES = DEFAULT_NUM_FILES
MAX_WORKERS = DEFAULT_MAX_WORKERS
REQUEST_TIMEOUT = DEFAULT_TIMEOUT


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


def flush_hf_batch(pending, db_batch):
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
        if is_target_language(lang):
            db_batch.append(item)
        else:
            filtered_out += 1

    pending.clear()
    return filtered_out


def process_one_warc_file(file_url: str) -> dict:
    """
    Process a single WARC file inside a worker process.
    Returns a summary dict for the parent to aggregate.
    """
    # Cap PyTorch intra-op threads so parallel workers don't oversubscribe CPUs.
    try:
        import torch
        torch.set_num_threads(max(1, cpu_count() // max(1, MAX_WORKERS)))
    except ImportError:
        pass

    tag = file_url.rsplit("/", 1)[-1][:30]
    print(f"[{tag}] Worker started: opening DB connection...")

    conn = get_connection()
    pending_hf = []
    db_batch = []
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
            stream_response = urllib.request.urlopen(
                stream_req, timeout=REQUEST_TIMEOUT
            )
        except urllib.error.URLError as e:
            stream_error = f"network: {e}"
            print(f"[STREAM ERROR] {file_url}: {stream_error}")
            return {
                "file_url": file_url,
                "stored": 0,
                "skipped": 0,
                "skipped_lang": 0,
                "db_errors": 0,
                "extract_errors": 0,
                "records_seen": 0,
                "stream_error": stream_error,
            }

        with stream_response:
            try:
                record_iter = parse_warc_stream_fastwarc(stream_response)
            except Exception as e:
                stream_error = f"parse: {e}"
                print(f"[STREAM ERROR] {file_url}: {stream_error}")
                return {
                    "file_url": file_url,
                    "stored": 0,
                    "skipped": 0,
                    "skipped_lang": 0,
                    "db_errors": 0,
                    "extract_errors": 0,
                    "records_seen": 0,
                    "stream_error": stream_error,
                }

            for record in record_iter:
                records_seen += 1
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
                if word_count < 80 or not clean_text:
                    skipped += 1
                    continue

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
                    skipped_lang += flush_hf_batch(pending_hf, db_batch)

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
                        f"stored: {stored:,} | skipped: {skipped:,}"
                    )
                    last_print = now

        # Final flush for this file
        skipped_lang += flush_hf_batch(pending_hf, db_batch)
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
):
    global YEAR, MONTH, NUM_FILES, MAX_WORKERS, REQUEST_TIMEOUT
    YEAR = year
    MONTH = month
    NUM_FILES = num_files
    MAX_WORKERS = max_workers
    REQUEST_TIMEOUT = timeout

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
        f"Workers={max_workers} | timeout={timeout}s | "
        f"year={year} month={month}"
    )

    if not remaining:
        print("Nothing to do -- all requested files were already processed.")
        return

    start_time = time.perf_counter()
    total_stored = 0
    total_skipped = 0
    total_skipped_lang = 0
    total_db_errors = 0
    total_extract_errors = 0
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_warc_file, url): url
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
            completed_count += 1

            # Only mark done if we actually saw records (partial/crash retries).
            if result["records_seen"] > 0 and not result.get("stream_error"):
                mark_file_processed(file_url, processed)
            else:
                print(
                    f"[WARNING] {file_url}: records_seen={result['records_seen']} "
                    f"stream_error={result.get('stream_error')!r} — "
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
                f"{result['db_errors']} db errors) | "
                f"Running total stored: {total_stored:,} | "
                f"Avg/file: {avg_per_file:.1f}s | ETA remaining: {eta_str}"
            )

    end_time = time.perf_counter()
    print(f"\nCompleted Processing!")
    print(f"Total Saved Records     : {total_stored:,}")
    print(f"Total Skipped (extract) : {total_skipped:,}")
    print(f"Total Skipped (lang)    : {total_skipped_lang:,}")
    print(f"Extract Errors          : {total_extract_errors}")
    print(f"Batches Failed (DB)     : {total_db_errors}")
    print(f"Total Execution Time    : {end_time - start_time:.2f} seconds")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream CC-NEWS WARC files into PostgreSQL (parallel)."
    )
    p.add_argument(
        "--year", default=DEFAULT_YEAR,
        help=f"CC-NEWS year (default: {DEFAULT_YEAR})",
    )
    p.add_argument(
        "--month", default=DEFAULT_MONTH,
        help=f"CC-NEWS month zero-padded (default: {DEFAULT_MONTH})",
    )
    p.add_argument(
        "--num-files", type=int, default=DEFAULT_NUM_FILES,
        help=f"How many WARC files to process this run (default: {DEFAULT_NUM_FILES})",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Parallel worker processes (default: {DEFAULT_MAX_WORKERS})",
    )
    p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout seconds for downloads (default: {DEFAULT_TIMEOUT})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_parallel_pipeline(
        year=args.year,
        month=args.month,
        num_files=args.num_files,
        max_workers=args.workers,
        timeout=args.timeout,
    )