# import psycopg2
# from psycopg2.extras import execute_values
# from urllib.parse import urlparse
# import json

# DB_CONFIG = {
#     "dbname": "warc_db",
#     "user": "warc_user",
#     "password": "my_secure_password",
#     "host": "localhost",
#     "port": "5432"
# }

# def get_connection():
#     return psycopg2.connect(**DB_CONFIG)

# def init_db():
#     create_tables_query = """
#     -- 1. WEBSITES TABLE
#     CREATE TABLE IF NOT EXISTS websites (
#         id SERIAL PRIMARY KEY,
#         domain VARCHAR(255) UNIQUE NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );

#     -- 2. AUTHORS TABLE
#     CREATE TABLE IF NOT EXISTS authors (
#         id SERIAL PRIMARY KEY,
#         name VARCHAR(255) UNIQUE NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );

#     -- 3. PAGES TABLE (References Website & Author)
#     CREATE TABLE IF NOT EXISTS pages (
#         id SERIAL PRIMARY KEY,
#         website_id INT REFERENCES websites(id) ON DELETE CASCADE,
#         author_id INT REFERENCES authors(id) ON DELETE SET NULL,
#         warc_record_id VARCHAR(255) UNIQUE,
#         url TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );

#     -- 4. METADATA TABLE (One Metadata -> One Page)
#     CREATE TABLE IF NOT EXISTS metadata (
#         page_id INT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
#         title TEXT,
#         published_date VARCHAR(100),
#         language VARCHAR(50),
#         arabic_dialect VARCHAR(50),
#         word_count INT,
#         char_count INT,
#         links_count INT,
#         headings JSONB,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );

#     -- 5. CONTENT TABLE (One Content -> One Page)
#     CREATE TABLE IF NOT EXISTS content (
#         page_id INT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
#         cleaned_text TEXT,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );
#     """
#     # FIX: CREATE TABLE IF NOT EXISTS above will NOT add new columns to
#     # tables that already exist from an earlier run of this project (e.g.
#     # a "pages" table created before author_id existed, or a "metadata"
#     # table created before arabic_dialect existed). These migrations make
#     # init_db() safe to re-run against an existing database without
#     # losing data or hitting "column ... does not exist" errors.
#     migration_query = """
#     ALTER TABLE pages ADD COLUMN IF NOT EXISTS author_id INT REFERENCES authors(id) ON DELETE SET NULL;
#     ALTER TABLE metadata ADD COLUMN IF NOT EXISTS arabic_dialect VARCHAR(50);
#     """

#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute(create_tables_query)
#     cur.execute(migration_query)
#     conn.commit()
#     cur.close()
#     conn.close()

# def extract_domain(url):
#     try:
#         parsed = urlparse(url)
#         domain = parsed.netloc or parsed.path.split('/')[0]
#         return domain.lower().replace('www.', '') if domain else "unknown_domain"
#     except Exception:
#         return "unknown_domain"

# def save_batch_records(records_list, conn=None):
#     """
#     Insert a batch of records.

#     FIX: now accepts an optional existing `conn`. If the caller passes one,
#     it is reused (and NOT closed here) instead of opening/closing a brand new
#     DB connection on every single batch, which was slowing the whole
#     pipeline down badly on large WARC files.
#     """
#     if not records_list:
#         return

#     own_connection = conn is None
#     if own_connection:
#         conn = get_connection()
#     cur = conn.cursor()

#     try:
#         # A. Processing Websites
#         domains = list(set(extract_domain(r.get("url", "")) for r in records_list))
#         execute_values(
#             cur,
#             "INSERT INTO websites (domain) VALUES %s ON CONFLICT (domain) DO NOTHING;",
#             [(d,) for d in domains]
#         )
#         cur.execute("SELECT domain, id FROM websites WHERE domain = ANY(%s);", (domains,))
#         domain_map = dict(cur.fetchall())

#         # B. Processing Authors
#         # FIX: r.get("author") can be None (not just missing), which used to
#         # crash .strip() with "AttributeError: 'NoneType' object has no
#         # attribute 'strip'". Using `(r.get("author") or "")` guards that.
#         authors = list(set((r.get("author") or "").strip() for r in records_list if (r.get("author") or "").strip()))
#         if authors:
#             execute_values(
#                 cur,
#                 "INSERT INTO authors (name) VALUES %s ON CONFLICT (name) DO NOTHING;",
#                 [(a,) for a in authors]
#             )
#             cur.execute("SELECT name, id FROM authors WHERE name = ANY(%s);", (authors,))
#             author_map = dict(cur.fetchall())
#         else:
#             author_map = {}

#         # C. Insert Pages & Retrieve ID accurately (handling conflicts properly)
#         pages_tuples = [
#             (
#                 domain_map.get(extract_domain(r.get("url", ""))),
#                 author_map.get((r.get("author") or "").strip()),
#                 r.get("record_id", ""),
#                 r.get("url", "")
#             )
#             for r in records_list
#         ]

#         insert_pages_query = """
#         INSERT INTO pages (website_id, author_id, warc_record_id, url)
#         VALUES %s
#         ON CONFLICT (warc_record_id) DO UPDATE 
#         SET url = EXCLUDED.url
#         RETURNING warc_record_id, id;
#         """
    
#         inserted_pages = execute_values(cur, insert_pages_query, pages_tuples, fetch=True)   
#         page_id_map = {rec_id: p_id for rec_id, p_id in inserted_pages}

#         # D. Insert METADATA & CONTENT
#         metadata_tuples = []
#         content_tuples = []

#         for r in records_list:
#             p_id = page_id_map.get(r.get("record_id", ""))
#             if p_id:
#                 metadata_tuples.append((
#                     p_id,
#                     r.get("title", ""),
#                     r.get("published_date", ""),
#                     r.get("language", "unknown"),
#                     r.get("arabic_dialect", "N/A"),
#                     r.get("word_count", 0),
#                     r.get("char_count", 0),
#                     r.get("links_count", 0),
#                     json.dumps(r.get("headings", []))
#                 ))
#                 content_tuples.append((
#                     p_id,
#                     r.get("cleaned_text", "")
#                 ))

#         if metadata_tuples:
#             execute_values(
#                 cur,
#                 """INSERT INTO metadata 
#                 (page_id, title, published_date, language, arabic_dialect, word_count, char_count, links_count, headings) 
#                 VALUES %s 
#                 ON CONFLICT (page_id) DO NOTHING;""",
#                 metadata_tuples
#             )

#         if content_tuples:
#             execute_values(
#                 cur,
#                 """INSERT INTO content (page_id, cleaned_text) 
#                 VALUES %s 
#                 ON CONFLICT (page_id) DO NOTHING;""",
#                 content_tuples
#             )

#         conn.commit()

#     except Exception as e:
#         conn.rollback()
#         raise e
#     finally:
#         cur.close()
#         if own_connection:
#             conn.close()

# if __name__ == "__main__":
#     init_db()

"""
db_handler.py

PostgreSQL helpers for the WARC pipeline.

Credentials come from the environment (or a .env file if python-dotenv is
installed). Never hardcode passwords in source.
"""

import os
import json
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import execute_values

# Optional: load .env from project root / cwd if python-dotenv is available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _db_config():
    """Build connection kwargs from environment variables."""
    # Prefer a single DATABASE_URL if provided.
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return {"dsn": database_url}

    return {
        "dbname": os.environ.get("POSTGRES_DB", "warc_db"),
        "user": os.environ.get("POSTGRES_USER", "warc_user"),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }


def get_connection():
    cfg = _db_config()
    if "dsn" in cfg:
        return psycopg2.connect(cfg["dsn"])
    return psycopg2.connect(**cfg)


def init_db():
    create_tables_query = """
    -- 1. WEBSITES TABLE
    CREATE TABLE IF NOT EXISTS websites (
        id SERIAL PRIMARY KEY,
        domain VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 2. AUTHORS TABLE
    CREATE TABLE IF NOT EXISTS authors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 3. PAGES TABLE (References Website & Author)
    CREATE TABLE IF NOT EXISTS pages (
        id SERIAL PRIMARY KEY,
        website_id INT REFERENCES websites(id) ON DELETE CASCADE,
        author_id INT REFERENCES authors(id) ON DELETE SET NULL,
        warc_record_id VARCHAR(255) UNIQUE,
        url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 4. METADATA TABLE (One Metadata -> One Page)
    CREATE TABLE IF NOT EXISTS metadata (
        page_id INT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
        title TEXT,
        published_date VARCHAR(100),
        language VARCHAR(50),
        arabic_dialect VARCHAR(50),
        word_count INT,
        char_count INT,
        links_count INT,
        headings JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 5. CONTENT TABLE (One Content -> One Page)
    CREATE TABLE IF NOT EXISTS content (
        page_id INT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
        cleaned_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    # CREATE TABLE IF NOT EXISTS will NOT add new columns to tables that
    # already exist. These migrations keep init_db() safe to re-run.
    migration_query = """
    ALTER TABLE pages ADD COLUMN IF NOT EXISTS author_id INT REFERENCES authors(id) ON DELETE SET NULL;
    ALTER TABLE metadata ADD COLUMN IF NOT EXISTS arabic_dialect VARCHAR(50);
    """

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(create_tables_query)
        cur.execute(migration_query)
        conn.commit()
    finally:
        cur.close()
        conn.close()


def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        return domain.lower().replace("www.", "") if domain else "unknown_domain"
    except Exception:
        return "unknown_domain"


def save_batch_records(records_list, conn=None):
    """
    Insert a batch of records.

    Accepts an optional existing `conn`. If the caller passes one, it is
    reused (and NOT closed here). If not, a connection is opened and closed
    inside this function.
    """
    if not records_list:
        return

    own_connection = conn is None
    if own_connection:
        conn = get_connection()
    cur = conn.cursor()

    try:
        # A. Websites
        domains = list(
            set(extract_domain(r.get("url", "")) for r in records_list)
        )
        execute_values(
            cur,
            "INSERT INTO websites (domain) VALUES %s ON CONFLICT (domain) DO NOTHING;",
            [(d,) for d in domains],
        )
        cur.execute(
            "SELECT domain, id FROM websites WHERE domain = ANY(%s);",
            (domains,),
        )
        domain_map = dict(cur.fetchall())

        # B. Authors — guard against None
        authors = list(
            set(
                (r.get("author") or "").strip()
                for r in records_list
                if (r.get("author") or "").strip()
            )
        )
        if authors:
            execute_values(
                cur,
                "INSERT INTO authors (name) VALUES %s ON CONFLICT (name) DO NOTHING;",
                [(a,) for a in authors],
            )
            cur.execute(
                "SELECT name, id FROM authors WHERE name = ANY(%s);",
                (authors,),
            )
            author_map = dict(cur.fetchall())
        else:
            author_map = {}

        # C. Pages
        pages_tuples = [
            (
                domain_map.get(extract_domain(r.get("url", ""))),
                author_map.get((r.get("author") or "").strip()),
                r.get("record_id", ""),
                r.get("url", ""),
            )
            for r in records_list
        ]

        insert_pages_query = """
        INSERT INTO pages (website_id, author_id, warc_record_id, url)
        VALUES %s
        ON CONFLICT (warc_record_id) DO UPDATE
        SET url = EXCLUDED.url
        RETURNING warc_record_id, id;
        """
        inserted_pages = execute_values(
            cur, insert_pages_query, pages_tuples, fetch=True
        )
        page_id_map = {rec_id: p_id for rec_id, p_id in inserted_pages}

        # D. Metadata & Content
        metadata_tuples = []
        content_tuples = []

        for r in records_list:
            p_id = page_id_map.get(r.get("record_id", ""))
            if p_id:
                metadata_tuples.append((
                    p_id,
                    r.get("title", ""),
                    r.get("published_date", ""),
                    r.get("language", "unknown"),
                    r.get("arabic_dialect", "N/A"),
                    r.get("word_count", 0),
                    r.get("char_count", 0),
                    r.get("links_count", 0),
                    json.dumps(r.get("headings", [])),
                ))
                content_tuples.append((
                    p_id,
                    r.get("cleaned_text", ""),
                ))

        if metadata_tuples:
            execute_values(
                cur,
                """INSERT INTO metadata
                (page_id, title, published_date, language, arabic_dialect,
                 word_count, char_count, links_count, headings)
                VALUES %s
                ON CONFLICT (page_id) DO NOTHING;""",
                metadata_tuples,
            )

        if content_tuples:
            execute_values(
                cur,
                """INSERT INTO content (page_id, cleaned_text)
                VALUES %s
                ON CONFLICT (page_id) DO NOTHING;""",
                content_tuples,
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        if own_connection:
            conn.close()


if __name__ == "__main__":
    init_db()
    print("DB initialized successfully.")