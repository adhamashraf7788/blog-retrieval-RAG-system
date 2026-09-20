#Evaluation Queries
#        ↓
#multilingual_search()
#        ↓
#Top-K Results
#        ↓
#Compare page_id
#        ↓
#Recall@K
#MRR
#Hit@K
#Latency

import time

from evaluation_queries import EVALUATION_QUERIES
from multilingual_retriever import multilingual_search


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

def hit_at_k(retrieved_ids, relevant_ids, k):
    """
    Returns 1 if at least one relevant document
    appears in the top-k results.
    """

    retrieved_top_k = retrieved_ids[:k]

    return int(
        any(doc_id in relevant_ids for doc_id in retrieved_top_k)
    )


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Recall@K =
    number of relevant documents retrieved
    / total number of relevant documents
    """

    retrieved_top_k = set(retrieved_ids[:k])

    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    return len(retrieved_top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Reciprocal Rank =
    1 / rank of the first relevant result
    """

    relevant_ids = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

def evaluate():

    print("=" * 80)
    print("MULTILINGUAL RETRIEVAL EVALUATION")
    print("=" * 80)

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0

    number_of_queries = len(EVALUATION_QUERIES)

    for index, item in enumerate(EVALUATION_QUERIES, start=1):

        query = item["query"]
        relevant_chunks = item["relevant_chunks"]

        # ------------------------------------------
        # Search
        # ------------------------------------------

        start_time = time.perf_counter()

        results = multilingual_search(
            query,
            top_k=3
        )

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        total_latency += latency_ms

        # ------------------------------------------
        # Convert database results to logical IDs
        #
        # page_id:
        # 1 -> ar_vector
        # 2 -> en_vector
        # 3 -> fr_vector
        # ------------------------------------------

        page_to_chunk = {
            1: "ar_vector",
            2: "en_vector",
            3: "fr_vector"
        }

        retrieved_ids = [
            page_to_chunk.get(result[1])
            for result in results
        ]

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        hit = hit_at_k(
            retrieved_ids,
            relevant_chunks,
            3
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_chunks,
            3
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_chunks
        )

        total_hit += hit
        total_recall += recall
        total_mrr += rr

        # ------------------------------------------
        # Print Query Result
        # ------------------------------------------

        print("\n" + "-" * 80)

        print(f"Query #{index}")
        print(f"Query: {query}")
        print(f"Language: {item['language']}")

        print(f"\nRelevant:")
        print(relevant_chunks)

        print(f"\nRetrieved:")
        print(retrieved_ids)

        print(f"\nHit@3: {hit}")
        print(f"Recall@3: {recall:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print(f"Latency: {latency_ms:.2f} ms")

    # --------------------------------------------------
    # Final Metrics
    # --------------------------------------------------

    average_hit = total_hit / number_of_queries
    average_recall = total_recall / number_of_queries
    average_mrr = total_mrr / number_of_queries
    average_latency = total_latency / number_of_queries

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    print(f"Number of queries: {number_of_queries}")

    print(f"Hit@3: {average_hit:.4f}")

    print(f"Recall@3: {average_recall:.4f}")

    print(f"MRR: {average_mrr:.4f}")

    print(f"Average Latency: {average_latency:.2f} ms")

    print("=" * 80)


if __name__ == "__main__":
    evaluate()