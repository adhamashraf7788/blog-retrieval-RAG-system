import time

from multilingual_retriever import multilingual_search
from hybrid_retriever import translation_search
from pipeline import run_pipeline


# ============================================================
# Evaluation Dataset
# ============================================================

EVALUATION_QUERIES = [

    # --------------------------------------------------------
    # Arabic
    # --------------------------------------------------------

    {
        "id": "ar_01",
        "query": "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",
        "language": "ar",
        "expected_route": "multilingual",
        "relevant_ids": [4],
    },

    {
        "id": "ar_02",
        "query": "ما هو البحث الدلالي؟",
        "language": "ar",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "ar_03",
        "query": "ما هي فائدة قواعد البيانات المتجهة؟",
        "language": "ar",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },


    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    {
        "id": "en_01",
        "query": "How do vector databases improve semantic search?",
        "language": "en",
        "expected_route": "multilingual",
        "relevant_ids": [5],
    },

    {
        "id": "en_02",
        "query": "What is semantic search?",
        "language": "en",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "en_03",
        "query": "What are vector databases used for?",
        "language": "en",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },


    # --------------------------------------------------------
    # French
    # --------------------------------------------------------

    {
        "id": "fr_01",
        "query": "Comment les bases de données vectorielles améliorent-elles la recherche sémantique ?",
        "language": "fr",
        "expected_route": "multilingual",
        "relevant_ids": [6],
    },

    {
        "id": "fr_02",
        "query": "Qu'est-ce que la recherche sémantique ?",
        "language": "fr",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "fr_03",
        "query": "À quoi servent les bases de données vectorielles ?",
        "language": "fr",
        "expected_route": "multilingual",
        "relevant_ids": [4, 5, 6],
    },


    # --------------------------------------------------------
    # Mixed Arabic + English
    # --------------------------------------------------------

    {
        "id": "mix_01",
        "query": "ما هو semantic search في artificial intelligence؟",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_02",
        "query": "عايز أبحث عن vector databases و semantic search",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_03",
        "query": "كيف يعمل RAG مع vector databases؟",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_04",
        "query": "اشرحلي semantic search باستخدام vector database",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_05",
        "query": "ما الفرق بين semantic search و keyword search؟",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_06",
        "query": "عايز أعرف how vector databases improve semantic search",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

    {
        "id": "mix_07",
        "query": "اشرحلي how semantic search works",
        "language": "ar+en",
        "expected_route": "both",
        "relevant_ids": [4, 5, 6],
    },

]


# ============================================================
# Metrics
# ============================================================

def hit_at_k(retrieved_ids, relevant_ids, k=3):

    retrieved = retrieved_ids[:k]

    return int(
        any(doc_id in relevant_ids for doc_id in retrieved)
    )


def recall_at_k(retrieved_ids, relevant_ids, k=3):

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):

    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def calculate_metrics(results, relevant_ids, k=3):

    retrieved_ids = []

    for result in results[:k]:

        # multilingual_search / translation_search
        # return tuples.
        #
        # Full pipeline returns dictionaries.

        if isinstance(result, dict):

            row = result["result"]
            record_id = row[0]

        else:

            record_id = result[0]

        retrieved_ids.append(record_id)

    return {

        "retrieved_ids": retrieved_ids,

        "hit": hit_at_k(
            retrieved_ids,
            relevant_ids,
            k
        ),

        "recall": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k
        ),

        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_ids
        ),

    }


# ============================================================
# Multilingual Only
# ============================================================

def evaluate_multilingual(query, relevant_ids):

    start = time.perf_counter()

    results = multilingual_search(
        query=query,
        top_k=3
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    metrics = calculate_metrics(
        results,
        relevant_ids
    )

    metrics["latency"] = latency

    return metrics


# ============================================================
# Translation Only
# ============================================================

def evaluate_translation(
    query,
    source_language,
    relevant_ids
):

    start = time.perf_counter()

    translated_query, results = translation_search(
        query=query,
        source_language=source_language,
        top_k=3
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    metrics = calculate_metrics(
        results,
        relevant_ids
    )

    metrics["latency"] = latency
    metrics["translated_query"] = translated_query

    return metrics


# ============================================================
# Full Algorithm
# ============================================================

def evaluate_full_pipeline(
    query,
    relevant_ids
):

    start = time.perf_counter()

    result = run_pipeline(query)

    latency = (
        time.perf_counter() - start
    ) * 1000

    final_results = result.get(
        "final_results",
        []
    )

    routing = result.get(
        "routing",
        {}
    )

    metrics = calculate_metrics(
        final_results,
        relevant_ids
    )

    metrics["latency"] = latency

    metrics["route"] = routing.get(
        "route"
    )

    metrics["branches"] = routing.get(
        "branches",
        []
    )

    return metrics


# ============================================================
# Main Evaluation
# ============================================================

def evaluate():

    print("=" * 100)
    print("COMPARATIVE RETRIEVAL EVALUATION")
    print("=" * 100)

    systems = {

        "multilingual": {
            "hit": 0,
            "recall": 0.0,
            "mrr": 0.0,
            "latency": 0.0,
        },

        "translation": {
            "hit": 0,
            "recall": 0.0,
            "mrr": 0.0,
            "latency": 0.0,
        },

        "full_pipeline": {
            "hit": 0,
            "recall": 0.0,
            "mrr": 0.0,
            "latency": 0.0,
        },

    }

    router_correct = 0

    total_queries = len(
        EVALUATION_QUERIES
    )


    # ========================================================
    # Query Loop
    # ========================================================

    for i, item in enumerate(
        EVALUATION_QUERIES,
        start=1
    ):

        query = item["query"]
        language = item["language"]
        relevant_ids = item["relevant_ids"]

        print("\n")
        print("=" * 100)
        print(
            f"QUERY {i}/{total_queries}"
        )
        print("=" * 100)

        print(
            f"ID: {item['id']}"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Language: {language}"
        )

        print(
            f"Expected Route: "
            f"{item['expected_route']}"
        )


        # ====================================================
        # 1. Multilingual Only
        # ====================================================

        print("\n" + "-" * 100)
        print("MULTILINGUAL ONLY")
        print("-" * 100)

        multi = evaluate_multilingual(
            query,
            relevant_ids
        )

        print(
            f"Retrieved: "
            f"{multi['retrieved_ids']}"
        )

        print(
            f"Hit@3: "
            f"{multi['hit']}"
        )

        print(
            f"Recall@3: "
            f"{multi['recall']:.4f}"
        )

        print(
            f"MRR: "
            f"{multi['mrr']:.4f}"
        )

        print(
            f"Latency: "
            f"{multi['latency']:.2f} ms"
        )


        # ====================================================
        # 2. Translation Only
        # ====================================================

        print("\n" + "-" * 100)
        print("TRANSLATION ONLY")
        print("-" * 100)

        # For mixed queries, the translation model
        # needs the actual source language used by
        # the query analyzer.
        #
        # We use Arabic as the source for ar+en
        # because the current router identifies
        # Arabic as the primary language for these
        # mixed queries in most cases.
        #
        # For pure language queries, language is used directly.

        if language == "ar+en":
            translation_source_language = "ar"
        else:
            translation_source_language = language

        translation = evaluate_translation(
            query,
            translation_source_language,
            relevant_ids
        )

        print(
            f"Source Language: "
            f"{translation_source_language}"
        )

        print(
            f"Translated Query: "
            f"{translation['translated_query']}"
        )

        print(
            f"Retrieved: "
            f"{translation['retrieved_ids']}"
        )

        print(
            f"Hit@3: "
            f"{translation['hit']}"
        )

        print(
            f"Recall@3: "
            f"{translation['recall']:.4f}"
        )

        print(
            f"MRR: "
            f"{translation['mrr']:.4f}"
        )

        print(
            f"Latency: "
            f"{translation['latency']:.2f} ms"
        )


        # ====================================================
        # 3. Full Algorithm
        # ====================================================

        print("\n" + "-" * 100)
        print("FULL ALGORITHM")
        print("-" * 100)

        full = evaluate_full_pipeline(
            query,
            relevant_ids
        )

        print(
            f"Actual Route: "
            f"{full['route']}"
        )

        print(
            f"Branches: "
            f"{full['branches']}"
        )

        print(
            f"Retrieved: "
            f"{full['retrieved_ids']}"
        )

        print(
            f"Hit@3: "
            f"{full['hit']}"
        )

        print(
            f"Recall@3: "
            f"{full['recall']:.4f}"
        )

        print(
            f"MRR: "
            f"{full['mrr']:.4f}"
        )

        print(
            f"Latency: "
            f"{full['latency']:.2f} ms"
        )


        # ====================================================
        # Router Evaluation
        # ====================================================

        if full["route"] == item["expected_route"]:

            router_correct += 1

            print(
                "\nRouter: CORRECT"
            )

        else:

            print(
                "\nRouter: WRONG"
            )


        # ====================================================
        # Accumulate Metrics
        # ====================================================

        results_map = {

            "multilingual": multi,

            "translation": translation,

            "full_pipeline": full,

        }

        for system, metrics in results_map.items():

            systems[system]["hit"] += (
                metrics["hit"]
            )

            systems[system]["recall"] += (
                metrics["recall"]
            )

            systems[system]["mrr"] += (
                metrics["mrr"]
            )

            systems[system]["latency"] += (
                metrics["latency"]
            )


    # ========================================================
    # Final Comparison
    # ========================================================

    print("\n\n")

    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)


    for system, metrics in systems.items():

        print(
            "\n" + "-" * 100
        )

        if system == "multilingual":

            title = "MULTILINGUAL ONLY"

        elif system == "translation":

            title = "TRANSLATION ONLY"

        else:

            title = "FULL ALGORITHM"


        print(title)

        print(
            "-" * 100
        )

        print(
            f"Hit@3:           "
            f"{metrics['hit'] / total_queries:.4f}"
        )

        print(
            f"Recall@3:        "
            f"{metrics['recall'] / total_queries:.4f}"
        )

        print(
            f"MRR:              "
            f"{metrics['mrr'] / total_queries:.4f}"
        )

        print(
            f"Average Latency: "
            f"{metrics['latency'] / total_queries:.2f} ms"
        )


    # ========================================================
    # Router Performance
    # ========================================================

    router_accuracy = (
        router_correct /
        total_queries
    )

    print("\n" + "=" * 100)
    print("ROUTER PERFORMANCE")
    print("=" * 100)

    print(
        f"Router Accuracy: "
        f"{router_accuracy:.4f}"
    )

    print(
        f"Correct Routes: "
        f"{router_correct}/"
        f"{total_queries}"
    )

    print("=" * 100)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    evaluate()