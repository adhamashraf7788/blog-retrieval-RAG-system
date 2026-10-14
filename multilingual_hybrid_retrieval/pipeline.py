# ============================================================
# COMPLETE RETRIEVAL PIPELINE
#
# Query
#   ↓
# Query Analyzer
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
#       Candidate Docs
#             ↓
#    Multilingual Reranker
#             ↓
#          Final-K
# ============================================================


from query_analyzer import analyze_query
from router import route_query
from hybrid_retriever import hybrid_search
from reranker import rerank


def run_pipeline(query: str):

    # --------------------------------------------------------
    # 1. Analyze Query
    # --------------------------------------------------------

    analysis = analyze_query(query)

    # --------------------------------------------------------
    # 2. Decide Retrieval Route
    # --------------------------------------------------------

    routing = route_query(query)

    print("\n" + "=" * 80)
    print("QUERY ANALYSIS")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Languages: {analysis['languages']}")
    print(f"Primary Language: {analysis['primary_language']}")
    print(f"Mixed Language: {analysis['mixed_language']}")
    print(f"Route: {routing['route']}")
    print(f"Branches: {routing['branches']}")

    # --------------------------------------------------------
    # 3. Retrieval + Fusion
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETRIEVAL")
    print("=" * 80)

    print(f"Running branches: {routing['branches']}")

    translated_query, retrieval_results = hybrid_search(
        query=query,
        route=routing["route"],
        source_language=analysis["primary_language"]
    )

    # --------------------------------------------------------
    # 4. Reranking
    # --------------------------------------------------------

    print("\nRunning Multilingual Reranker...")

    final_results = rerank(
        query=query,
        results=retrieval_results
    )

    # --------------------------------------------------------
    # 5. Final Output
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for rank, item in enumerate(
        final_results,
        start=1
    ):

        result = item["result"]

        record_id = result[0]
        language = result[3]
        text = result[4]

        print(f"\nRank: {rank}")
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

    # --------------------------------------------------------
    # 6. Return Complete Pipeline Information
    # --------------------------------------------------------
    #
    # This is important for evaluation.
    #
    # The evaluator can now inspect:
    #
    # - Query analysis
    # - Router decision
    # - Selected branches
    # - Translation
    # - Retrieval results
    # - RRF scores
    # - Reranker scores
    # - Final results
    #
    # --------------------------------------------------------

    return {
        "query": query,

        "analysis": analysis,

        "routing": routing,

        "translated_query": translated_query,

        "retrieval_results": retrieval_results,

        "final_results": final_results,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "ما هو semantic search في artificial intelligence؟"
    )

    run_pipeline(query)