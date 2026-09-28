import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from db.db_handler import get_connection


def print_db_summary():
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM websites;")
        total_websites = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM authors;")
        total_authors = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM pages;")
        total_pages = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM metadata;")
        total_metadata = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM content;")
        total_content = cur.fetchone()[0]

        cur.execute("""
            SELECT p.id, w.domain, COALESCE(a.name, 'N/A') AS author, m.language, m.word_count, m.title
            FROM pages p
            JOIN websites w ON p.website_id = w.id
            LEFT JOIN authors a ON p.author_id = a.id
            JOIN metadata m ON p.id = m.page_id
            LIMIT 5;
        """)
        sample_records = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    print("\n" + "=" * 85)
    print("                STRUCTURED DATABASE SUMMARY (WITH AUTHORS SCHEMA)")
    print("=" * 85)
    print(f" Total Websites (WEBSITES) : {total_websites:,}")
    print(f" Total Authors  (AUTHORS)  : {total_authors:,}")
    print(f" Total Pages    (PAGES)    : {total_pages:,}")
    print(f" Total Metadata (METADATA) : {total_metadata:,}")
    print(f" Total Content  (CONTENT)  : {total_content:,}")
    print("=" * 85)

    if not sample_records:
        # FIX: if pages/metadata are empty this used to just print an empty
        # table with no explanation. Now it says so explicitly, which is a
        # much faster hint that "the data isn't going in" than a blank table.
        print(" No rows found yet in pages/metadata — the pipeline hasn't")
        print(" successfully stored anything. Check stream_to_db.py output")
        print(" for [DB ERROR] / [EXTRACT ERROR] lines.")
        print("=" * 85 + "\n")
        return

    print(" SAMPLE DATA (JOINED TABLES WITH AUTHORS):")
    print("-" * 85)
    print(f"{'Page ID':<8} | {'Domain':<18} | {'Author':<15} | {'Lang':<5} | {'Words':<6} | {'Title'}")
    print("-" * 85)
    for r in sample_records:
        page_id, domain, author, lang, word_count, title = r

        # FIX: title/domain/author/lang could theoretically be None
        # (NULL in DB) and the old code called len()/slicing on them
        # directly, which would raise TypeError and crash the whole
        # summary instead of just printing "N/A".
        domain = domain or "N/A"
        author = author or "N/A"
        lang = lang or "N/A"
        title = title or "N/A"
        word_count = word_count if word_count is not None else 0

        title_preview = (title[:20] + "...") if len(title) > 20 else title
        author_preview = (author[:13] + "..") if len(author) > 13 else author

        print(f"{page_id:<8} | {domain[:18]:<18} | {author_preview:<15} | {lang:<5} | {word_count:<6} | {title_preview}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    print_db_summary()