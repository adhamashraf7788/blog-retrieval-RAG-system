# Chunks
#   ↓
# BAAI/bge-m3
#   ↓
# 1024-dimensional normalized vectors
#   ↓
# PostgreSQL / pgvector

from psycopg2.extras import execute_values

from database import get_connection
from embeddings import generate_embeddings


def clear_embeddings():
    """
    Remove existing document embeddings
    so the database contains only BGE-M3 vectors.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE document_embeddings;")

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def insert_chunks(chunks):
    """
    Generate BGE-M3 embeddings for chunks
    and insert them into PostgreSQL.
    """

    texts = [chunk["text"] for chunk in chunks]

    # Generate normalized 1024-dimensional embeddings
    embeddings = generate_embeddings(texts)

    records = [
        (
            chunk["page_id"],
            chunk["chunk_index"],
            chunk["language"],
            chunk["text"],
            embedding.tolist()
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            query = """
                INSERT INTO document_embeddings
                (
                    page_id,
                    chunk_index,
                    language,
                    chunk_text,
                    embedding
                )
                VALUES %s
            """

            execute_values(
                cursor,
                query,
                records
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":

    sample_chunks = [

        {
            "page_id": 1,
            "chunk_index": 0,
            "language": "ar",
            "text": "تساعد قواعد البيانات المتجهة في تحسين كفاءة أنظمة البحث الدلالي باستخدام الذكاء الاصطناعي."
        },

        {
            "page_id": 2,
            "chunk_index": 0,
            "language": "en",
            "text": "Vector databases enable fast semantic search in artificial intelligence applications."
        },

        {
            "page_id": 3,
            "chunk_index": 0,
            "language": "fr",
            "text": "Les bases de données vectorielles permettent une recherche sémantique rapide."
        }

    ]

    print("Clearing old embeddings...")
    clear_embeddings()

    print("Inserting new BGE-M3 embeddings...")
    insert_chunks(sample_chunks)

    print("Chunks inserted successfully.")