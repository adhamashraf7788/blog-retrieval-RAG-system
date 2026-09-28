
"""
compare_keyword_methods.py

Keyword Method Comparison

Benchmark YAKE / KeyBERT / TF-IDF against each other on the same chunks
(from summarize_classify SUCCESS rows — identical text for a fair comparison).

Compares the three methods only (timing + pairwise Jaccard overlap).
Does NOT score against LLM tags.
"""

import json
import os
import sys
import time
from collections import defaultdict

import yake
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Same chunks the LLM already processed (text source only).
LLM_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "chunk_analysis_results_last_version_multilingual_chunks.json",
)

OUTPUT_FILE = os.path.join(BASE_DIR, "keyword_methods_comparison.json")

TOP_N = 5
LANGUAGES = ["ar", "en", "fr"]
MAX_PER_LANGUAGE = None  # e.g. 30 to limit; None = all SUCCESS chunks

print("\nLoading KeyBERT model...")
keybert_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
print("KeyBERT model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def load_chunks(path: str):
    """Load SUCCESS rows — text only (for fair same-chunk comparison)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Point LLM_RESULTS_FILE at your chunk_analysis JSON."
        )

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")

    chunks = []
    for r in rows:
        if not r or r.get("status") != "SUCCESS":
            continue
        text = clean_text(r.get("chunk_text_full") or r.get("chunk_text_preview") or "")
        if not text:
            continue
        lang = (r.get("language") or "").lower()
        if lang not in LANGUAGES:
            continue
        chunks.append({
            "doc_id": r.get("doc_id"),
            "chunk_index": r.get("chunk_index"),
            "language": lang,
            "source": r.get("source"),
            "title": r.get("title"),
            "text": text,
        })
    return chunks


def select_by_language(chunks, max_per_language=None):
    by_lang = defaultdict(list)
    for c in chunks:
        by_lang[c["language"]].append(c)

    print("\n" + "=" * 70)
    print("AVAILABLE CHUNKS")
    print("=" * 70)
    for lang in LANGUAGES:
        print(f"{lang.upper():<5}: {len(by_lang[lang])} chunks")

    selected = []
    for lang in LANGUAGES:
        group = by_lang[lang]
        group.sort(
            key=lambda x: (
                x["doc_id"] is None,
                x["doc_id"],
                x["chunk_index"] is None,
                x["chunk_index"],
            )
        )
        if max_per_language is not None:
            group = group[:max_per_language]
        selected.extend(group)

    # Drop exact duplicate keys (doc_id, chunk_index)
    seen = set()
    unique = []
    for c in selected:
        key = (c["doc_id"], c["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    print("\n" + "=" * 70)
    print("SELECTED FOR BENCHMARK")
    print("=" * 70)
    counts = defaultdict(int)
    for c in unique:
        counts[c["language"]] += 1
    for lang in LANGUAGES:
        print(f"{lang.upper():<5}: {counts[lang]} chunks")
    print(f"TOTAL: {len(unique)}")

    if not unique:
        raise SystemExit("No SUCCESS chunks found. Run summarize_classify first.")
    return unique


def normalize_kw(keywords):
    return [str(t).strip().lower() for t in keywords if t and str(t).strip()]


def jaccard(a, b):
    sa, sb = set(normalize_kw(a)), set(normalize_kw(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_yake(text, language, top_n=TOP_N):
    text = clean_text(text)
    if not text:
        return []
    lan = language if language in ("ar", "en", "fr") else "en"
    extractor = yake.KeywordExtractor(
        lan=lan,
        n=3,
        dedupLim=0.8,
        top=top_n,
        features=None,
    )
    return [kw for kw, _ in extractor.extract_keywords(text)]


def extract_keybert(text, top_n=TOP_N):
    text = clean_text(text)
    if not text:
        return []
    kws = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        use_mmr=True,
        diversity=0.5,
        top_n=top_n,
    )
    return [kw for kw, _ in kws]


def extract_tfidf(text, corpus, top_n=TOP_N):
    text = clean_text(text)
    if not text:
        return []
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names_out()
    scores = vectorizer.transform([text]).toarray()[0]
    ranked = scores.argsort()[::-1]
    keywords = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        keywords.append(feature_names[idx])
        if len(keywords) >= top_n:
            break
    return keywords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("YAKE vs KeyBERT vs TF-IDF (same chunks)")
    print("=" * 70)
    print(f"Source : {LLM_RESULTS_FILE}")
    print(f"Top N  : {TOP_N}")
    print(f"Output : {OUTPUT_FILE}")

    selected = select_by_language(load_chunks(LLM_RESULTS_FILE), MAX_PER_LANGUAGE)
    corpus = [c["text"] for c in selected]

    results = {
        "configuration": {
            "source_file": os.path.basename(LLM_RESULTS_FILE),
            "total_chunks": len(selected),
            "top_n_keywords": TOP_N,
            "languages": LANGUAGES,
            "max_per_language": MAX_PER_LANGUAGE,
            "keybert_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "note": "Compares YAKE / KeyBERT / TF-IDF only on the same chunks.",
        },
        "methods": {},
        "language_performance": {},
        "keyword_overlap": {},
        "chunks": [],
    }

    method_times = {"YAKE": 0.0, "KeyBERT": 0.0, "TF-IDF": 0.0}

    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    for i, c in enumerate(selected, start=1):
        text = c["text"]
        lang = c["language"]
        print(
            f"\n[{i}/{len(selected)}] lang={lang} "
            f"doc_id={c['doc_id']} chunk_index={c['chunk_index']}"
        )

        t0 = time.perf_counter()
        yake_kw = extract_yake(text, lang, TOP_N)
        yake_t = time.perf_counter() - t0
        method_times["YAKE"] += yake_t

        t0 = time.perf_counter()
        keybert_kw = extract_keybert(text, TOP_N)
        keybert_t = time.perf_counter() - t0
        method_times["KeyBERT"] += keybert_t

        t0 = time.perf_counter()
        tfidf_kw = extract_tfidf(text, corpus, TOP_N)
        tfidf_t = time.perf_counter() - t0
        method_times["TF-IDF"] += tfidf_t

        j_yake_keybert = jaccard(yake_kw, keybert_kw)
        j_yake_tfidf = jaccard(yake_kw, tfidf_kw)
        j_keybert_tfidf = jaccard(keybert_kw, tfidf_kw)

        results["chunks"].append({
            "chunk_number": i,
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "language": lang,
            "source": c.get("source"),
            "title": c.get("title"),
            "text_preview": text[:300],
            "YAKE": {
                "keywords": yake_kw,
                "time_seconds": round(yake_t, 4),
            },
            "KeyBERT": {
                "keywords": keybert_kw,
                "time_seconds": round(keybert_t, 4),
            },
            "TF-IDF": {
                "keywords": tfidf_kw,
                "time_seconds": round(tfidf_t, 4),
            },
            "pairwise_jaccard": {
                "YAKE_vs_KeyBERT": round(j_yake_keybert, 4),
                "YAKE_vs_TF-IDF": round(j_yake_tfidf, 4),
                "KeyBERT_vs_TF-IDF": round(j_keybert_tfidf, 4),
            },
        })

        print(f"  YAKE   : {yake_kw}")
        print(f"  KeyBERT: {keybert_kw}")
        print(f"  TF-IDF : {tfidf_kw}")
        print(
            f"  overlap YAKE↔KeyBERT={j_yake_keybert:.2f}  "
            f"YAKE↔TF-IDF={j_yake_tfidf:.2f}  "
            f"KeyBERT↔TF-IDF={j_keybert_tfidf:.2f}"
        )

    n = len(selected)

    print("\n" + "=" * 70)
    print("OVERALL METHOD PERFORMANCE")
    print("=" * 70)
    for method, total in method_times.items():
        avg = total / n if n else 0
        results["methods"][method] = {
            "total_execution_time_seconds": round(total, 4),
            "average_time_per_chunk_seconds": round(avg, 4),
            "average_keywords_per_chunk": TOP_N,
        }
        print(f"{method}: total={total:.4f}s  avg/chunk={avg:.4f}s")

    lang_times = defaultdict(lambda: defaultdict(float))
    lang_counts = defaultdict(int)
    for r in results["chunks"]:
        lang = r["language"]
        lang_counts[lang] += 1
        for m in ("YAKE", "KeyBERT", "TF-IDF"):
            lang_times[lang][m] += r[m]["time_seconds"]

    for lang in LANGUAGES:
        results["language_performance"][lang] = {}
        if lang_counts[lang] == 0:
            continue
        print(f"\n{lang.upper()}")
        for m in ("YAKE", "KeyBERT", "TF-IDF"):
            total = lang_times[lang][m]
            avg = total / lang_counts[lang]
            results["language_performance"][lang][m] = {
                "total_time_seconds": round(total, 4),
                "average_time_per_chunk_seconds": round(avg, 4),
            }
            print(f"  {m:<8}: {avg:.4f} s/chunk")

    print("\n" + "=" * 70)
    print("KEYWORD OVERLAP (methods vs each other)")
    print("=" * 70)
    pairs = {
        "YAKE_vs_KeyBERT": ("YAKE", "KeyBERT"),
        "YAKE_vs_TF-IDF": ("YAKE", "TF-IDF"),
        "KeyBERT_vs_TF-IDF": ("KeyBERT", "TF-IDF"),
    }
    for name, (a, b) in pairs.items():
        scores = [
            jaccard(r[a]["keywords"], r[b]["keywords"])
            for r in results["chunks"]
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        results["keyword_overlap"][name] = round(avg, 4)
        print(f"{name:<25}: {avg:.4f}")

    results["keyword_overlap_by_language"] = {}
    for lang in LANGUAGES:
        lang_rows = [r for r in results["chunks"] if r["language"] == lang]
        if not lang_rows:
            continue
        results["keyword_overlap_by_language"][lang] = {}
        print(f"\n  {lang.upper()}:")
        for name, (a, b) in pairs.items():
            scores = [
                jaccard(r[a]["keywords"], r[b]["keywords"])
                for r in lang_rows
            ]
            avg = sum(scores) / len(scores)
            results["keyword_overlap_by_language"][lang][name] = round(avg, 4)
            print(f"    {name:<25}: {avg:.4f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()