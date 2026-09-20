# Query
#   ↓
# Hybrid Retrieval
#   ↓
# RRF Top-K
#   ↓
# BGE Reranker v2 M3
#   ↓
# Multilingual Relevance Scores
#   ↓
# Sort
#   ↓
# Final-K


from sentence_transformers import CrossEncoder

from config import FINAL_K


# --------------------------------------------------
# Load Reranker Model
# --------------------------------------------------

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
print("Loading reranker model...")

reranker = CrossEncoder(RERANKER_MODEL)


# --------------------------------------------------
# Rerank Function
# --------------------------------------------------

def rerank(
    query: str,
    results: list,
    final_k: int = FINAL_K
):
    """
    Rerank retrieved documents using a Cross-Encoder.
    """

    if not results:
        return []

    # ----------------------------------------------
    # 1. Extract document texts
    # ----------------------------------------------

    pairs = []

    for item in results:

        result = item["result"]

        text = result[4]

        pairs.append(
            [query, text]
        )

    # ----------------------------------------------
    # 2. Calculate Cross-Encoder scores
    # ----------------------------------------------

    scores = reranker.predict(pairs)

    # ----------------------------------------------
    # 3. Attach reranker scores
    # ----------------------------------------------

    reranked_results = []

    for item, score in zip(results, scores):

        item = item.copy()

        item["reranker_score"] = float(score)

        reranked_results.append(item)

    # ----------------------------------------------
    # 4. Sort by reranker score
    # ----------------------------------------------

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # ----------------------------------------------
    # 5. Return Final-K
    # ----------------------------------------------

    return reranked_results[:final_k]


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    from hybrid_retriever import hybrid_search

    query = "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟"

    translated_query, hybrid_results = hybrid_search(
        query=query,
        source_language="ar"
    )

    final_results = rerank(
        query=query,
        results=hybrid_results
    )

    print("\nOriginal Query:")
    print(query)

    print("\nReranked Results:")
    print("=" * 80)

    for rank, item in enumerate(
        final_results,
        start=1
    ):

        result = item["result"]

        (
            record_id,
            page_id,
            chunk_index,
            language,
            text,
            _
        ) = result

        print(f"Rank: {rank}")
        print(f"ID: {record_id}")
        print(f"Language: {language}")
        print(
            f"RRF Score: "
            f"{item['rrf_score']:.6f}"
        )
        print(
            f"Reranker Score: "
            f"{item['reranker_score']:.4f}"
        )
        print(f"Text: {text}")

        print("-" * 80)