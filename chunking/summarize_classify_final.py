"""
summarize_classify_per_chunk.py (Per-Chunk Version - Groq Only)

Pipeline:
    PostgreSQL
      -> Documents (one per article)
      -> Chunks (split for the LLM)
      -> Summary + Category PER CHUNK (using Groq)  <-- no aggregation step
      -> JSON (one result per chunk)

Difference vs summarize_classify_final.py:
    The old version summarized each chunk (summary only, no category), then
    grouped all chunks of the same article and combined them into ONE final
    summary + ONE category per article (a "synthesis" pass).

    This version removes that synthesis pass entirely. Each chunk gets its
    own summary AND its own category directly, independently of every other
    chunk -- even chunks that belong to the same article are never combined.

CHANGELOG (this version):
    1. CATEGORIES: added "Crime" (police/court/fraud content had nowhere good
       to go before -- it was landing in Society or Other).
    2. CHUNK_ANALYSIS_PROMPT: made explicit per-category guidance instead of
       leaving the model to guess.
    3. analyze_chunks(): fixed a resume bug. Previously, any chunk that
       failed permanently was still written with status="ERROR", and the old
       resume logic treated "any row already in the file" as done. Now ERROR
       rows are re-queued. Additionally, resume is now KEY-BASED on
       (doc_id, chunk_index) instead of fragile list indices -- this stays
       correct even if chunk order changes across runs.
    4. select_chunks_per_language() / per_language_limit: guarantees balanced
       language coverage.
    5. _is_contaminated_summary() / tag capping: catch leaked structured-
       output syntax and hard-cap tags at 6.
    6. SECURITY: API key is no longer hardcoded; must come from the
       environment (GROQ_API_KEY).
    7. Pre-filter: drop very short chunks and obvious boilerplate
       (e.g. Legal Disclaimer blocks) before spending tokens on them.

NOTE ON DAILY TOKEN LIMITS (TPD):
    Groq's free/on-demand tier for qwen/qwen3.8-27b caps you at 200,000
    tokens/day. Each chunk call costs ~2,900-3,000 tokens, so expect roughly
    65-70 successful chunks per day. Re-running this script on a later day
    will correctly pick up where it left off, including retrying any chunk
    that failed because of the daily cap.
"""

import importlib.util
import json
import os
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple

import httpx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# API key: must be set in the environment -- never hardcode secrets in source.
# ---------------------------------------------------------------------------
if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit(
        "GROQ_API_KEY is not set. Export it before running, e.g.:\n"
        "  export GROQ_API_KEY='gsk_...'"
    )

# Windows fix: stops httpx crashing on Arabic-locale proxy settings
_no_proxy_client = httpx.Client(trust_env=False, timeout=60.0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# Load langchain.py manually -- to get PostgreSQL loader functions
# ---------------------------------------------------------------------------
_loader_path = os.path.join(BASE_DIR, "langchain.py")
_spec = importlib.util.spec_from_file_location("project_langchain_loader", _loader_path)
_loader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader_module)

load_documents_from_db = _loader_module.load_documents_from_db
split_documents_into_chunks = _loader_module.split_documents_into_chunks


# Added "Crime" -- police/court/fraud/legal-case content had no good home
# before (real-data eval showed it landing in Society or Other).
CATEGORIES = Literal[
    "Politics", "Economy", "Sports", "Technology",
    "Culture", "Health", "Society", "Religion",
    "Weather", "Food", "Crime", "Other",
]

MAX_TAGS = 6

# Drop chunks that are too short or are pure boilerplate before spending
# tokens on them.
MIN_CHUNK_CHARS = 100
BOILERPLATE_PREFIXES = (
    "legal disclaimer",
    "menafn provides the information",
    "© copyright",
    "all rights reserved",
)


class ChunkAnalysis(BaseModel):
    summary: str = Field(
        description="A concise 1-2 sentence summary of just THIS piece of "
        "text, written in the SAME language as the original text."
    )
    category: CATEGORIES = Field(
        description="The single best topic category for just THIS piece of "
        "text, based only on what this chunk itself discusses."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="3-6 short keyword tags for retrieval (RAG) purposes: "
        "named entities (people, places, organizations), and specific "
        "topical terms mentioned in THIS chunk. Written in the SAME "
        "language as the original text. No generic single-word tags like "
        "'news' or 'article'."
    )


DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
CHUNK_MAX_TOKENS = 450

# reasoning_effort="none" saves a large amount of tokens but can rarely
# cause structured-output syntax to leak into the summary field. That
# case is detected by _is_contaminated_summary() and retried.
GROQ_MODEL_KWARGS = {"reasoning_effort": "none"}

LANG_MODEL_MAP = {
    "ar": ChatGroq(
        model=DEFAULT_GROQ_MODEL,
        temperature=0,
        max_tokens=CHUNK_MAX_TOKENS,
        api_key=os.environ["GROQ_API_KEY"],
        http_client=_no_proxy_client,
        reasoning_effort="none",
    ),
    "en": ChatGroq(
        model=DEFAULT_GROQ_MODEL,
        temperature=0,
        max_tokens=CHUNK_MAX_TOKENS,
        api_key=os.environ["GROQ_API_KEY"],
        http_client=_no_proxy_client,
        reasoning_effort="none",
    ),
    "fr": ChatGroq(
        model=DEFAULT_GROQ_MODEL,
        temperature=0,
        max_tokens=CHUNK_MAX_TOKENS,
        api_key=os.environ["GROQ_API_KEY"],
        http_client=_no_proxy_client,
        reasoning_effort="none",
    ),
}

DEFAULT_MODEL = ChatGroq(
    model=DEFAULT_GROQ_MODEL,
    temperature=0,
    max_tokens=CHUNK_MAX_TOKENS,
    api_key=os.environ["GROQ_API_KEY"],
    http_client=_no_proxy_client,
    reasoning_effort="none",
)


def get_model_for_language(lang: Optional[str]):
    return LANG_MODEL_MAP.get((lang or "").lower(), DEFAULT_MODEL)


LANGUAGE_NAMES = {"ar": "Arabic", "en": "English", "fr": "French"}


CHUNK_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a multilingual news analyst. You will be given ONE "
            "piece of a longer article -- not the whole thing. Based ONLY "
            "on this piece: (1) write a concise summary of it, in the SAME "
            "language as the input text, (2) pick the single best "
            "topic category for it, and (3) extract 3-6 keyword tags for "
            "a retrieval system (RAG). Be factual, neutral, and concise. Do "
            "not guess at parts of the article you have not seen, and do "
            "not assume the category of the rest of the article -- judge "
            "only what this piece of text itself is about.\n\n"
            "IMPORTANT: the `summary` field must contain ONLY the plain-"
            "language summary sentence(s) -- never include JSON syntax, "
            "field names, XML/tool tags, or any part of your own output "
            "format inside the summary text itself.\n\n"
            "Tag guidance -- these tags will be used to help a search "
            "system find this chunk later, so favor SPECIFIC, searchable "
            "terms over generic ones:\n"
            "- Prioritize named entities actually mentioned in this chunk: "
            "people, organizations, places, companies, laws, events.\n"
            "- Include specific topical terms (e.g. 'F-35 fighter jets', "
            "not just 'military'; 'pension law amendment', not just "
            "'economy').\n"
            "- Do NOT include generic words like 'news', 'article', "
            "'report', or the category name itself.\n"
            "- Do NOT invent an entity that isn't in this chunk's text.\n"
            "- Maximum 6 tags. If more than 6 entities are mentioned, pick "
            "the 6 most central to what this chunk is about.\n\n"
            "Category guidance -- pick the MOST SPECIFIC category that "
            "applies, do not default to a general one when a specific one "
            "fits:\n"
            "- Weather: forecasts, temperatures, storms, weather warnings. "
            "Do NOT classify these as Other.\n"
            "- Food: recipes, cooking instructions, restaurant/dish "
            "coverage. Do NOT classify these as Other.\n"
            "- Religion: prayers, adhkar/duaa, fiqh rulings, religious "
            "holidays, sermons, scripture discussion. Do NOT classify "
            "these as Culture -- Culture is for arts, entertainment, "
            "celebrities, film/TV/music, and general cultural affairs, "
            "not religious practice.\n"
            "- Crime: police investigations, arrests, court cases, fraud, "
            "and criminal trials. Do NOT classify these as Society or "
            "Other -- use Society only for family/social-issue content "
            "that is NOT primarily about a crime or legal proceeding.\n"
            "- Health: medical news, public health, disease, healthcare "
            "policy.\n"
            "- Other: use this ONLY when nothing above fits -- e.g. site "
            "navigation menus, mixed headline lists, or content with no "
            "clear single topic.",
        ),
        (
            "human",
            "The input language is {language}. Write the summary in {language}.\n\n"
            "Text:\n\n{text}",
        ),
    ]
)


RETRYABLE_ERROR_SUBSTRINGS = (
    "429",
    "rate_limit",
    "connection error",
    "connection reset",
    "timeout",
    "timed out",
    "remote end closed",
    "server disconnected",
    "503",
    "502",
    "500",
    "leaked structured-output syntax",
)


class DailyQuotaExceeded(Exception):
    """Raised when Groq's DAILY token quota (TPD) is hit."""
    pass


def _is_daily_quota_error(error_str: str) -> bool:
    low = error_str.lower()
    return "tokens per day" in low or "(tpd)" in low


_SUMMARY_CONTAMINATION_MARKERS = (
    "</summary>",
    "<summary>",
    "<parameter",
    "</parameter",
    '"category":',
    '"tags":',
    '"summary":',
    "```json",
    "<|",
    "|>",
)


def _is_contaminated_summary(summary: Optional[str]) -> bool:
    if not summary:
        return False
    return any(marker in summary for marker in _SUMMARY_CONTAMINATION_MARKERS)


def _cap_tags(tags: Optional[List[str]], max_tags: int = MAX_TAGS) -> List[str]:
    if not tags:
        return []
    return tags[:max_tags]


def _chunk_key(doc_id, chunk_index) -> Tuple:
    """Stable unique key for a chunk across runs."""
    return (doc_id, chunk_index)


def _is_boilerplate_chunk(text: str) -> bool:
    """True if the chunk looks like pure boilerplate / disclaimer noise."""
    if not text:
        return True
    head = text.strip()[:120].lower()
    return any(head.startswith(p) or p in head for p in BOILERPLATE_PREFIXES)


def filter_chunks(chunks: List[Document]) -> List[Document]:
    """Drop chunks that are too short or pure boilerplate before analysis."""
    kept = []
    dropped = 0
    for c in chunks:
        text = (c.page_content or "").strip()
        if len(text) < MIN_CHUNK_CHARS or _is_boilerplate_chunk(text):
            dropped += 1
            continue
        kept.append(c)
    if dropped:
        print(f"[FILTER] Dropped {dropped} short/boilerplate chunk(s); "
              f"{len(kept)} remaining.")
    return kept


def analyze_chunk(chunk: Document, max_retries: int = 5) -> dict:
    lang = chunk.metadata.get("language", "")
    model = get_model_for_language(lang)
    structured_model = model.with_structured_output(ChunkAnalysis)
    chain = CHUNK_ANALYSIS_PROMPT | structured_model

    language_name = LANGUAGE_NAMES.get(
        lang.lower(), lang or "the same language as the text"
    )
    doc_id = chunk.metadata.get("doc_id")
    chunk_index = chunk.metadata.get("chunk_index")

    delay = 5
    last_error = None
    for attempt in range(max_retries):
        try:
            result: ChunkAnalysis = chain.invoke(
                {"text": chunk.page_content, "language": language_name}
            )

            if _is_contaminated_summary(result.summary):
                raise ValueError(
                    "Model leaked structured-output syntax into summary "
                    f"field: {result.summary[:150]!r}"
                )

            return {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "source": chunk.metadata.get("source"),
                "title": chunk.metadata.get("title"),
                "language": lang,
                "model_used": DEFAULT_GROQ_MODEL,
                "status": "SUCCESS",
                "summary": result.summary,
                "category": result.category,
                "tags": _cap_tags(result.tags),
                "chunk_text_preview": chunk.page_content[:120],
                "chunk_text_full": chunk.page_content,
            }
        except Exception as e:
            last_error = str(e)
            if _is_daily_quota_error(last_error):
                raise DailyQuotaExceeded(last_error) from e
            if any(s in last_error.lower() for s in RETRYABLE_ERROR_SUBSTRINGS):
                print(
                    f"    [RETRYABLE ERROR] doc_id={doc_id} chunk_index={chunk_index} "
                    f"attempt {attempt + 1}/{max_retries}, waiting {delay}s... "
                    f"({last_error[:120]})"
                )
                time.sleep(delay)
                delay *= 2
                continue
            break

    print(f"[ERROR] doc_id={doc_id} chunk_index={chunk_index} "
          f"analysis failed permanently: {last_error}")
    return {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "source": chunk.metadata.get("source"),
        "title": chunk.metadata.get("title"),
        "language": lang,
        "model_used": DEFAULT_GROQ_MODEL,
        "status": "ERROR",
        "error_message": last_error,
        "summary": None,
        "category": None,
        "tags": None,
        "chunk_text_preview": chunk.page_content[:120],
        "chunk_text_full": chunk.page_content,
    }


LANGUAGE_ORDER = {"ar": 0, "en": 1, "fr": 2}


def select_chunks_per_language(
    chunks: List[Document],
    languages: List[str],
    per_language: int,
    done_success_keys: Optional[set] = None,
    success_count_by_lang: Optional[Dict[str, int]] = None,
) -> List[Document]:
    """
    Pick valid chunks so TOTAL SUCCESS counts stay balanced across languages.

    After a quota stop mid-run, one language may be ahead. The next run
    catches lagging languages up first:

        target = min(success_per_lang) + per_language
        need[lang] = max(0, target - success[lang])

    So if ar=32, en=6, fr=0 and per_language=30:
        target=30 → ar needs 0, en needs 24, fr needs 30.

    Also skips already-SUCCESS keys and short/boilerplate chunks.
    """
    done_success_keys = done_success_keys or set()
    success_count_by_lang = success_count_by_lang or {}

    counts = {lang: int(success_count_by_lang.get(lang, 0)) for lang in languages}
    min_success = min(counts.values()) if counts else 0
    target = min_success + per_language

    print(
        f"[BALANCE] SUCCESS so far: "
        + ", ".join(f"{l}={counts[l]}" for l in languages)
        + f" → target this run: {target} each "
        f"(min={min_success} + {per_language})"
    )

    selected: List[Document] = []

    for lang in languages:
        need = max(0, target - counts[lang])
        matched = [
            c for c in chunks
            if (c.metadata.get("language") or "").lower() == lang
        ]
        pending = []
        skipped_noise = 0
        for c in matched:
            if len(pending) >= need:
                break
            key = _chunk_key(
                c.metadata.get("doc_id"), c.metadata.get("chunk_index")
            )
            if key in done_success_keys:
                continue
            text = (c.page_content or "").strip()
            if len(text) < MIN_CHUNK_CHARS or _is_boilerplate_chunk(text):
                skipped_noise += 1
                continue
            pending.append(c)

        print(
            f"[SELECT] {lang}: need={need} → {len(pending)} valid pending "
            f"(of {len(matched)} total; {counts[lang]} already SUCCESS; "
            f"{skipped_noise} short/boilerplate skipped)"
        )
        if need > 0 and len(pending) < need:
            print(
                f"[WARNING] Only {len(pending)} valid pending '{lang}' "
                f"chunk(s), fewer than needed ({need})."
            )
        selected.extend(pending)

    return selected


def _load_results_by_key(output_path: str) -> Dict[Tuple, dict]:
    """Load existing results indexed by (doc_id, chunk_index)."""
    by_key: Dict[Tuple, dict] = {}
    if not os.path.exists(output_path):
        return by_key
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return by_key
        for r in rows:
            if not r:
                continue
            key = _chunk_key(r.get("doc_id"), r.get("chunk_index"))
            by_key[key] = r
        print(f"[RESUME] Loaded {len(by_key)} previously saved result(s) "
              f"from '{output_path}'.")
    except Exception as e:
        print(f"[RESUME] Could not read existing output file, starting "
              f"fresh: {e}")
    return by_key


def _save_results(output_path: str, by_key: Dict[Tuple, dict],
                  target_order: List[Tuple]) -> None:
    """Write all known results. Prefer target_order, then remaining keys."""
    rows = []
    seen = set()
    for key in target_order:
        if key in by_key:
            rows.append(by_key[key])
            seen.add(key)
    for key, row in by_key.items():
        if key not in seen:
            rows.append(row)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def analyze_chunks(
    chunks: List[Document],
    output_path: str,
    limit: Optional[int] = None,
    per_language_limit: Optional[int] = None,
    languages: Optional[List[str]] = None,
    save_every: int = 20,
) -> List[dict]:
    """
    Resume-safe, key-based driver that advances through the corpus.

    Resume key is (doc_id, chunk_index).
    - SUCCESS keys are never selected again (quota advances).
    - ERROR keys are re-queued.
    - Never-seen keys are processed as new.

    Requires chunk_index to be set on every chunk (see langchain.py).
    """
    # Load prior results FIRST so selection can skip already-SUCCESS keys.
    by_key = _load_results_by_key(output_path)
    done_success_keys = {
        k for k, r in by_key.items() if r.get("status") == "SUCCESS"
    }

    # Warn early if chunk_index is missing — that collapses all chunks of a
    # doc into one key and silently drops the rest.
    sample = chunks[:50] if chunks else []
    null_idx = sum(1 for c in sample if c.metadata.get("chunk_index") is None)
    if sample and null_idx == len(sample):
        print(
            "[CRITICAL] All sampled chunks have chunk_index=None. "
            "Update langchain.py so split_documents_into_chunks assigns "
            "chunk_index, otherwise only one chunk per doc_id will be kept."
        )

    if per_language_limit is not None:
        languages = languages or ["ar", "en", "fr"]
        success_count_by_lang: Dict[str, int] = {l: 0 for l in languages}
        for r in by_key.values():
            if r and r.get("status") == "SUCCESS":
                lang = (r.get("language") or "").lower()
                if lang in success_count_by_lang:
                    success_count_by_lang[lang] += 1
        # Catch up lagging languages so totals stay balanced after quota stops.
        target = select_chunks_per_language(
            chunks,
            languages,
            per_language_limit,
            done_success_keys,
            success_count_by_lang,
        )
    else:
        sorted_chunks = sorted(
            chunks,
            key=lambda c: LANGUAGE_ORDER.get(
                (c.metadata.get("language") or "").lower(), 99
            ),
        )
        pending = [
            c for c in sorted_chunks
            if _chunk_key(c.metadata.get("doc_id"), c.metadata.get("chunk_index"))
            not in done_success_keys
        ]
        target = pending[:limit] if limit else pending
        # Flat-limit path still needs an explicit filter pass.
        target = filter_chunks(target)

    # Build ordered list of keys + map key -> Document
    target_keys: List[Tuple] = []
    key_to_chunk: Dict[Tuple, Document] = {}
    for c in target:
        key = _chunk_key(c.metadata.get("doc_id"), c.metadata.get("chunk_index"))
        if key in key_to_chunk:
            # Real duplicate only if chunk_index is set; if None, this is the
            # silent data-loss case — still skip but we already warned above.
            continue
        target_keys.append(key)
        key_to_chunk[key] = c

    # Also re-queue ERROR keys that are still in the corpus (may not be in
    # this target window if selection moved past them — pull them explicitly).
    error_keys_in_file = [
        k for k, r in by_key.items() if r.get("status") == "ERROR"
    ]
    # Map error keys back to chunks if present in full list
    all_key_to_chunk = {
        _chunk_key(c.metadata.get("doc_id"), c.metadata.get("chunk_index")): c
        for c in chunks
    }
    for k in error_keys_in_file:
        if k not in key_to_chunk and k in all_key_to_chunk:
            target_keys.append(k)
            key_to_chunk[k] = all_key_to_chunk[k]

    retry_keys = [
        k for k in target_keys
        if k in by_key and by_key[k].get("status") == "ERROR"
    ]
    new_keys = [k for k in target_keys if k not in by_key]
    # Preserve order: retries first, then new
    to_process = list(dict.fromkeys(retry_keys + new_keys))

    if not to_process:
        print(
            "[RESUME] Nothing left to process for this quota window — "
            "all selected chunks already SUCCESS, or no pending chunks remain."
        )
        _save_results(output_path, by_key, list(by_key.keys()))
        return list(by_key.values())

    if retry_keys:
        print(f"[RESUME] Re-queuing {len(retry_keys)} previously-failed "
              f"chunk(s) for another attempt.")
    if new_keys:
        print(f"[RESUME] {len(new_keys)} new chunk(s) not yet attempted.")

    error_count = sum(
        1 for k in by_key if by_key[k].get("status") == "ERROR"
    )
    current_lang = None
    processed_this_run = 0

    for count, key in enumerate(to_process, 1):
        chunk = key_to_chunk[key]
        lang = chunk.metadata.get("language", "?")
        if lang != current_lang:
            print(f"\n--- Now processing language: {lang} ---")
            current_lang = lang
        tag = "RETRY" if key in retry_keys else "NEW"
        print(
            f"[{count}/{len(to_process)}] ({tag}) Summarizing + classifying "
            f"doc_id={key[0]} chunk_index={key[1]} (language={lang})..."
        )

        was_error_before = (
            key in by_key and by_key[key].get("status") == "ERROR"
        )

        try:
            chunk_result = analyze_chunk(chunk)
        except DailyQuotaExceeded as e:
            _save_results(output_path, by_key, list(by_key.keys()) + target_keys)
            n_success = sum(
                1 for r in by_key.values() if r.get("status") == "SUCCESS"
            )
            print(
                f"\n[DAILY QUOTA HIT] Stopping after {processed_this_run} "
                f"chunk(s) this run. Total SUCCESS saved: {n_success}. "
                f"Re-run once the quota resets — selection will advance "
                f"automatically.\n    {e}"
            )
            return list(by_key.values())

        by_key[key] = chunk_result
        processed_this_run += 1

        if was_error_before and chunk_result["status"] == "SUCCESS":
            error_count -= 1
        elif not was_error_before and chunk_result["status"] == "ERROR":
            error_count += 1

        if processed_this_run % save_every == 0:
            _save_results(output_path, by_key, list(by_key.keys()))
            n_success = sum(
                1 for r in by_key.values() if r.get("status") == "SUCCESS"
            )
            print(f"    [CHECKPOINT] Total SUCCESS so far: {n_success}")

        time.sleep(1.5)

    _save_results(output_path, by_key, list(by_key.keys()))

    n_success = sum(1 for r in by_key.values() if r.get("status") == "SUCCESS")
    print(
        f"\n--- Done this run: processed {processed_this_run} | "
        f"total SUCCESS in file: {n_success} | "
        f"still ERROR: {error_count} ---"
    )
    return list(by_key.values())



# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(" Connecting to PostgreSQL and loading raw documents...")
    raw_documents = load_documents_from_db()
    print(f" Loaded {len(raw_documents)} raw documents from DB.")

    print(" Splitting documents into chunks...")
    chunks = split_documents_into_chunks(
        raw_documents, chunk_size=800, chunk_overlap=100
    )
    print(f" Generated {len(chunks)} total chunks.")

    output_path = os.path.join(
        BASE_DIR, "chunk_analysis_results_last_version_multilingual_chunks.json"
    )

    # per_language_limit=30 => 30 ar + 30 en + 30 fr (90 total), then
    # filter_chunks may drop a few short/boilerplate ones.
    print("\n Running per-chunk summarization + classification...")
    results = analyze_chunks(
        chunks,
        output_path=output_path,
        per_language_limit=30,
        languages=["ar", "en", "fr"],
        save_every=20,
    )

    print(f"\n Saved {len(results)} chunk result(s) to: {output_path}")
    if results:
        last_success = next(
            (r for r in reversed(results) if r and r.get("status") == "SUCCESS"),
            None,
        )
        if last_success:
            print("\n Sample Output Result:")
            print(json.dumps(last_success, ensure_ascii=False, indent=2))