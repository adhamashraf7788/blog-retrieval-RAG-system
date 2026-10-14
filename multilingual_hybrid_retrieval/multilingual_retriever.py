# User Query
#     ↓
# BAAI/bge-m3
#     ↓
# Normalized Query Vector (1024)
#     ↓
# PostgreSQL / pgvector
#     ↓
# HNSW Search
#     ↓
# Top-K Chunks


from embeddings import generate_embeddings
from database import get_connection
from config import TOP_K


def multilingual_search(query: str, top_k: int = TOP_K):
    """
    Multilingual semantic retrieval using BGE-M3.

    The query and document chunks are represented
    in the same multilingual embedding space.
    """

    # --------------------------------------------------
    # 1. Generate normalized BGE-M3 embedding
    # --------------------------------------------------
    query_embedding = generate_embeddings([query])[0]

    vector = query_embedding.tolist()

    # --------------------------------------------------
    # 2. Connect to PostgreSQL
    # --------------------------------------------------
    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            # --------------------------------------------------
            # 3. Global multilingual vector search
            # --------------------------------------------------
            #
            # BGE-M3 embeddings are normalized.
            #
            # <#> = negative inner product
            #
            # Therefore:
            #
            # higher similarity
            #        ↓
            # smaller negative inner product
            #
            query_sql = """
                SELECT
                    id,
                    page_id,
                    chunk_index,
                    language,
                    chunk_text,

                    -(embedding <#> %s::vector)
                    AS similarity

                FROM document_embeddings

                ORDER BY embedding <#> %s::vector

                LIMIT %s;
            """

            cursor.execute(
                query_sql,
                (
                    vector,
                    vector,
                    top_k
                )
            )

            results = cursor.fetchall()

        return results

    finally:
        conn.close()


# ------------------------------------------------------
# Test
# ------------------------------------------------------

if __name__ == "__main__":

    queries = [

        # Arabic
        "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",

        # English
        "How do vector databases improve semantic search?",

        # French
        "Comment les bases de données vectorielles améliorent-elles la recherche sémantique ?"

    ]

    for query in queries:

        print("\n" + "=" * 80)

        print("Query:")
        print(query)

        print("=" * 80)

        results = multilingual_search(query)

        for rank, result in enumerate(results, start=1):

            (
                record_id,
                page_id,
                chunk_index,
                language,
                text,
                similarity
            ) = result

            print(f"\nRank: {rank}")
            print(f"ID: {record_id}")
            print(f"Language: {language}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Text: {text}")

            print("-" * 80)