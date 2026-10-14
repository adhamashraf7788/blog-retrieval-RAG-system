import psycopg2
from pgvector.psycopg2 import register_vector

from config import DB_CONFIG


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def initialize_database():
    conn = get_connection()

    try:
        # 1. Enable pgvector extension first
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

        conn.commit()

        # 2. Now register pgvector
        register_vector(conn)

        # 3. Create table and indexes
        with conn.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    page_id INTEGER,
                    chunk_index INTEGER,
                    language VARCHAR(10),
                    chunk_text TEXT,
                    embedding VECTOR(1024)
                );
            """)

            # Language index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_lang
                ON document_embeddings(language);
            """)

            # HNSW vector index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
                ON document_embeddings
                USING hnsw (embedding vector_ip_ops);
            """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")