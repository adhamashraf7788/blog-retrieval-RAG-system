"""
compare_ner_models.py

NER Model Comparison (News Tagging)

Benchmark:
  - Davlan/xlm-roberta-base-ner-hrl
  - GLiNER Multi (urchade/gliner_multi-v2.1)

"""

import json
import os
import re
import time
from collections import defaultdict
from gliner import GLiNER

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "chunk_analysis_results_v1_4_multilingual_chunks.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "best_ner_stable_results.json")

TOP_N = 6
THRESHOLD = 0.35
LANGUAGES = ["ar", "en", "fr"]

GLINER_LABELS = [
    "person",
    "politician",
    "organization",
    "government",
    "military",
    "location",
    "country",
    "city",
    "event",
    "sports team",
    "sports competition",
    "company",
    "media outlet",
    "product",
]

print("\nLoading stable model: urchade/gliner_multi-v2.1 ...")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
print("Model loaded successfully!\n")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def post_process(entities: list) -> list:
    cleaned = []
    seen = set()

    bad_words = {
        "و", "ال", "لل", "في", "من", "على", "إلى", "أن", "إن", "كان", "كانت",
        "a", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "le", "la", "les", "de", "du", "des", "un", "une"
    }

    for ent in entities:
        ent = ent.strip()

        # شروط الرفض
        if len(ent) < 3:
            continue
        if ent.lower() in bad_words:
            continue
        if re.fullmatch(r"[\W\d_]+", ent):
            continue
        if len(ent.split()) > 8:          
            continue

        key = ent.lower()
        if key not in seen:
            cleaned.append(ent)
            seen.add(key)

    return cleaned[:TOP_N]


def extract_entities(text: str) -> list:
    text = clean_text(text)
    if not text:
        return []

    try:
        results = model.predict_entities(text, GLINER_LABELS, threshold=THRESHOLD)
        entities = [r["text"].strip() for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True)]
        return post_process(entities)
    except Exception as e:
        print(f"  Error: {e}")
        return []


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    results = []
    total_time = 0.0
    count = 0

    print("=" * 70)
    print("Processing chunks...")
    print("=" * 70)

    for row in rows:
        if row.get("status") != "SUCCESS":
            continue

        lang = (row.get("language") or "").lower()
        if lang not in LANGUAGES:
            continue

        text = clean_text(row.get("chunk_text_full") or row.get("chunk_text_preview") or "")
        if not text:
            continue

        count += 1
        print(f"[{count}] lang={lang} | doc_id={row.get('doc_id')}")

        t0 = time.perf_counter()
        entities = extract_entities(text)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        results.append({
            "doc_id": row.get("doc_id"),
            "chunk_index": row.get("chunk_index"),
            "language": lang,
            "title": row.get("title"),
            "entities": entities,
            "time_seconds": round(elapsed, 4)
        })

        print(f"  → {entities}")
        print(f"  time: {elapsed:.3f}s\n")

    # حفظ
    output = {
        "model": "urchade/gliner_multi-v2.1",
        "threshold": THRESHOLD,
        "labels": GLINER_LABELS,
        "total_chunks": len(results),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_chunk": round(total_time / len(results), 4) if results else 0,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total chunks : {len(results)}")
    print(f"Total time   : {total_time:.2f}s")
    print(f"Avg per chunk: {total_time / len(results):.3f}s")


if __name__ == "__main__":
    main()