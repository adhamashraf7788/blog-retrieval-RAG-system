# ============================================================
# HYBRID RETRIEVER
#
# Query
#   ↓
# Retrieval Router
#   ↓
# ┌───────────────────────┐
# ↓                       ↓
# Multilingual        Translation
# Search              Search
# ↓                       ↓
# Top-K                   Top-K
# └───────────┬───────────┘
#             ↓
#            RRF
#             ↓
#      Hybrid Candidates
# ============================================================


import psycopg2
import torch

from sentence_transformers import SentenceTransformer

from config import (
    DB_CONFIG,
    EMBEDDING_MODEL,
    TOP_K
)

from translator import translate_to_english


# ------------------------------------------------------------
# Load Embedding Model
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading embedding model on", device)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=device
)


# ------------------------------------------------------------
# Multilingual Search
# ------------------------------------------------------------

def multilingual_search(query: str, top_k: int = TOP_K):

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    conn = psycopg2.connect(**DB_CONFIG)

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    page_id,
                    chunk_index,
                    language,
                    chunk_text,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    query_vector,
                    query_vector,
                    top_k
                )
            )

            return cursor.fetchall()

    finally:

        conn.close()


# ------------------------------------------------------------
# Translation Search
# ------------------------------------------------------------

def translation_search(
    query: str,
    source_language: str,
    top_k: int = TOP_K
):

    translated_query = translate_to_english(
                                         query,
                                         source_language
)

    query_vector = embedding_model.encode(
        translated_query,
        normalize_embeddings=True
    ).tolist()

    conn = psycopg2.connect(**DB_CONFIG)

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    page_id,
                    chunk_index,
                    language,
                    chunk_text,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    query_vector,
                    query_vector,
                    top_k
                )
            )

            results = cursor.fetchall()

            return translated_query, results

    finally:

        conn.close()


# ------------------------------------------------------------
# RRF
# ------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists,
    k: int = 60
):

    fused = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1
        ):

            document_id = result[0]

            if document_id not in fused:

                fused[document_id] = {
                    "result": result,
                    "rrf_score": 0.0
                }

            fused[document_id]["rrf_score"] += (
                1.0 / (k + rank)
            )

    ranked_results = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked_results


# ------------------------------------------------------------
# Hybrid Search
# ------------------------------------------------------------

def hybrid_search(
    query: str,
    route: str = "both",
    source_language: str = "en",
    top_k: int = TOP_K
):

    result_lists = []

    translated_query = None

    # ========================================================
    # Multilingual Branch
    # ========================================================

    if route in ["multilingual", "both"]:

        print("\nRunning: Multilingual Search")

        multilingual_results = multilingual_search(
            query,
            top_k
        )

        result_lists.append(
            multilingual_results
        )

    # ========================================================
    # Translation Branch
    # ========================================================

    if route == "both":

        print("Running: Translation Search")

        translated_query, translation_results = (
            translation_search(
                query=query,
                source_language=source_language,
                top_k=top_k
    )
)

        result_lists.append(
            translation_results
        )

    # ========================================================
    # RRF
    # ========================================================

    if not result_lists:

        return translated_query, []

    fused_results = reciprocal_rank_fusion(
        result_lists
    )

    return translated_query, fused_results


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    query = (
        "كيف تساعد قواعد البيانات المتجهة "
        "في تحسين البحث الدلالي؟"
    )

    translated_query, results = hybrid_search(
        query=query,
        route="multilingual"
    )

    print("\nQuery:")
    print(query)

    if translated_query:

        print("\nTranslated Query:")
        print(translated_query)

    print("\nResults:")
    print("=" * 80)

    for rank, item in enumerate(
        results,
        start=1
    ):

        result = item["result"]

        print(f"Rank: {rank}")
        print(f"ID: {result[0]}")
        print(f"Language: {result[3]}")
        print(f"Similarity: {result[5]:.4f}")
        print(f"RRF Score: {item['rrf_score']:.6f}")
        print(f"Text: {result[4]}")

        print("-" * 80)