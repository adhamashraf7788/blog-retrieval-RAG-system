"""
compare_ner_models.py

NER Model Comparison (News Tagging)

Benchmark:
  - Davlan/xlm-roberta-base-ner-hrl
  - GLiNER Multi (urchade/gliner_multi-v2.1)

On the same SUCCESS chunks from summarize_classify.
Compares timing + pairwise entity overlap (Jaccard).
"""

import json
import os
import sys
import time
from collections import defaultdict

from transformers import pipeline
from gliner import GLiNER


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

LLM_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "chunk_analysis_results_v1_4_multilingual_chunks.json",
)

OUTPUT_FILE = os.path.join(BASE_DIR, "ner_models_comparison_v1_4.json")

TOP_N = 10                    # max entities to keep per method (after aggregation)
LANGUAGES = ["ar", "en", "fr"]
MAX_PER_LANGUAGE = None       # e.g. 30 to limit; None = all SUCCESS chunks

# Entity types for GLiNER (you can change them)
GLINER_LABELS = ["person", "organization", "location", "event", "product"]

print("\nLoading NER models...")

# 1. Davlan XLM-R Base
print("  → Davlan/xlm-roberta-base-ner-hrl")
davlan_pipe = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    tokenizer="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple",
    device=-1,         # CPU only; set to 0 for GPU
    trust_remote_code=False,
)

# 2. GLiNER Multi
print("  → urchade/gliner_multi-v2.1")
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

print("Models loaded successfully.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def load_chunks(path: str):
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

    print("=" * 70)
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

    # Drop exact duplicates
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


def normalize_entities(entities):
    """Normalize entity texts for Jaccard comparison."""
    return {str(e).strip().lower() for e in entities if e and str(e).strip()}


def jaccard(a, b):
    sa, sb = normalize_entities(a), normalize_entities(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_davlan(text, top_n=TOP_N):
    text = clean_text(text)
    if not text:
        return []
    try:
        results = davlan_pipe(text)
        # Keep unique entity texts, preserve order roughly by score
        entities = []
        seen = set()
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            ent = r.get("word", "").strip()
            if ent and ent.lower() not in seen:
                entities.append(ent)
                seen.add(ent.lower())
            if len(entities) >= top_n:
                break
        return entities
    except Exception as e:
        print(f"    [Davlan error] {e}")
        return []


def extract_gliner(text, top_n=TOP_N):
    text = clean_text(text)
    if not text:
        return []
    try:
        results = gliner_model.predict_entities(text, GLINER_LABELS, threshold=0.4)
        entities = []
        seen = set()
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            ent = r.get("text", "").strip()
            if ent and ent.lower() not in seen:
                entities.append(ent)
                seen.add(ent.lower())
            if len(entities) >= top_n:
                break
        return entities
    except Exception as e:
        print(f"    [GLiNER error] {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Davlan XLM-R  vs  GLiNER Multi  (same chunks)")
    print("=" * 70)
    print(f"Source : {LLM_RESULTS_FILE}")
    print(f"Top N  : {TOP_N}")
    print(f"Output : {OUTPUT_FILE}")

    selected = select_by_language(load_chunks(LLM_RESULTS_FILE), MAX_PER_LANGUAGE)

    results = {
        "configuration": {
            "source_file": os.path.basename(LLM_RESULTS_FILE),
            "total_chunks": len(selected),
            "top_n_entities": TOP_N,
            "languages": LANGUAGES,
            "max_per_language": MAX_PER_LANGUAGE,
            "davlan_model": "Davlan/xlm-roberta-base-ner-hrl",
            "gliner_model": "urchade/gliner_multi-v2.1",
            "gliner_labels": GLINER_LABELS,
            "note": "Compares Davlan NER vs GLiNER on the same chunks (timing + Jaccard).",
        },
        "methods": {},
        "language_performance": {},
        "entity_overlap": {},
        "chunks": [],
    }

    method_times = {"Davlan": 0.0, "GLiNER": 0.0}

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

        # Davlan
        t0 = time.perf_counter()
        davlan_ents = extract_davlan(text, TOP_N)
        davlan_t = time.perf_counter() - t0
        method_times["Davlan"] += davlan_t

        # GLiNER
        t0 = time.perf_counter()
        gliner_ents = extract_gliner(text, TOP_N)
        gliner_t = time.perf_counter() - t0
        method_times["GLiNER"] += gliner_t

        j_score = jaccard(davlan_ents, gliner_ents)

        results["chunks"].append({
            "chunk_number": i,
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "language": lang,
            "source": c.get("source"),
            "title": c.get("title"),
            "text_preview": text[:300],
            "Davlan": {
                "entities": davlan_ents,
                "time_seconds": round(davlan_t, 4),
            },
            "GLiNER": {
                "entities": gliner_ents,
                "time_seconds": round(gliner_t, 4),
            },
            "pairwise_jaccard": {
                "Davlan_vs_GLiNER": round(j_score, 4),
            },
        })

        print(f"  Davlan : {davlan_ents}")
        print(f"  GLiNER : {gliner_ents}")
        print(f"  overlap Davlan↔GLiNER = {j_score:.2f}")

    n = len(selected)

    # Overall timing
    print("\n" + "=" * 70)
    print("OVERALL METHOD PERFORMANCE")
    print("=" * 70)
    for method, total in method_times.items():
        avg = total / n if n else 0
        results["methods"][method] = {
            "total_execution_time_seconds": round(total, 4),
            "average_time_per_chunk_seconds": round(avg, 4),
        }
        print(f"{method}: total={total:.4f}s  avg/chunk={avg:.4f}s")

    # Per language timing
    lang_times = defaultdict(lambda: defaultdict(float))
    lang_counts = defaultdict(int)
    for r in results["chunks"]:
        lang = r["language"]
        lang_counts[lang] += 1
        for m in ("Davlan", "GLiNER"):
            lang_times[lang][m] += r[m]["time_seconds"]

    for lang in LANGUAGES:
        results["language_performance"][lang] = {}
        if lang_counts[lang] == 0:
            continue
        print(f"\n{lang.upper()}")
        for m in ("Davlan", "GLiNER"):
            total = lang_times[lang][m]
            avg = total / lang_counts[lang]
            results["language_performance"][lang][m] = {
                "total_time_seconds": round(total, 4),
                "average_time_per_chunk_seconds": round(avg, 4),
            }
            print(f"  {m:<8}: {avg:.4f} s/chunk")

    # Overall overlap
    print("\n" + "=" * 70)
    print("ENTITY OVERLAP (Davlan vs GLiNER)")
    print("=" * 70)
    scores = [r["pairwise_jaccard"]["Davlan_vs_GLiNER"] for r in results["chunks"]]
    avg_overlap = sum(scores) / len(scores) if scores else 0.0
    results["entity_overlap"]["Davlan_vs_GLiNER"] = round(avg_overlap, 4)
    print(f"Overall Jaccard: {avg_overlap:.4f}")

    # Per language overlap
    results["entity_overlap_by_language"] = {}
    for lang in LANGUAGES:
        lang_rows = [r for r in results["chunks"] if r["language"] == lang]
        if not lang_rows:
            continue
        scores = [r["pairwise_jaccard"]["Davlan_vs_GLiNER"] for r in lang_rows]
        avg = sum(scores) / len(scores)
        results["entity_overlap_by_language"][lang] = round(avg, 4)
        print(f"  {lang.upper()}: {avg:.4f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()