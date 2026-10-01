# User Query
#      ↓
# Language Detection
#      ↓
# NLLB-200
#      ↓
# English Query
#      ↓
# Qwen3 Embedding
#      ↓
# PostgreSQL / pgvector
#      ↓
# Top-K Results

from translator import translate_to_english
from embeddings import generate_embeddings
from database import get_connection
from config import TOP_K


def translation_search(
    query: str,
    source_language: str,
    top_k: int = TOP_K
):
    """
    Translate the query to English, generate its embedding,
    then perform vector similarity search.
    """

    # 1. Translate query to English
    translated_query = translate_to_english(
        query,
        source_language=source_language
    )

    # 2. Generate embedding for translated query
    query_embedding = generate_embeddings(
        [translated_query]
    )[0]

    vector = query_embedding.tolist()

    # 3. Connect to PostgreSQL
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            # 4. Vector similarity search
            query_sql = """
                SELECT
                    id,
                    page_id,
                    chunk_index,
                    language,
                    chunk_text,
                    -(embedding <#> %s::vector) AS similarity
                FROM document_embeddings
                ORDER BY embedding <#> %s::vector
                LIMIT %s;
            """

            cursor.execute(
                query_sql,
                (vector, vector, top_k)
            )

            results = cursor.fetchall()

        return translated_query, results

    finally:
        conn.close()


if __name__ == "__main__":

    query = "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟"

    translated_query, results = translation_search(
        query=query,
        source_language="ar"
    )

    print("\nOriginal Query:")
    print(query)

    print("\nTranslated Query:")
    print(translated_query)

    print("\nResults:")
    print("-" * 80)

    for result in results:

        (
            record_id,
            page_id,
            chunk_index,
            language,
            text,
            similarity
        ) = result

        print(f"ID: {record_id}")
        print(f"Language: {language}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Text: {text}")
        print("-" * 80)