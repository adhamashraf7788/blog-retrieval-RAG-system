import sys
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from db.db_handler import get_connection


def load_documents_from_db(
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Document]:
    """Fetch PostgreSQL records and convert them into LangChain Documents.

    Optional `limit`/`offset` enable pagination so callers don't have to
    load the entire corpus into memory at once.
    """

    # FIX: added JOIN with `authors` table -- `metadata` has no `author`
    # column in our schema, author name lives in authors.name via
    # pages.author_id.
    query = """
        SELECT p.id, c.cleaned_text, p.url, m.title, w.domain, a.name AS author,
               m.language, m.arabic_dialect, m.published_date, p.warc_record_id
        FROM pages p
        LEFT JOIN websites w ON p.website_id = w.id
        LEFT JOIN authors a ON p.author_id = a.id
        LEFT JOIN metadata m ON p.id = m.page_id
        LEFT JOIN content c ON p.id = c.page_id
        WHERE c.cleaned_text IS NOT NULL
        ORDER BY p.id
    """
    if limit is not None:
        query += f" LIMIT {int(limit)} OFFSET {int(offset)}"

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        conn.close()

    documents = []
    for row in rows:
        (
            doc_id,
            text,
            url,
            title,
            source_domain,
            author,
            lang,
            arabic_dialect,
            pub_date,
            warc_record_id,
        ) = row

        # cleaned_text can still be empty string even if not NULL
        if not text or not text.strip():
            continue

        doc = Document(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "source": url,
                "title": title or "N/A",
                "publisher": source_domain or "Unknown",
                "author": author or source_domain or "Unknown",
                "domain": source_domain or "",
                "language": lang or "",
                "arabic_dialect": arabic_dialect or "N/A",
                "published_date": pub_date or "N/A",
                "warc_record_id": warc_record_id or "",
            },
        )
        documents.append(doc)

    print(
        f"[LangChain Loader] Successfully created {len(documents)} LangChain Document objects."
    )
    return documents


def split_documents_into_chunks(
    documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> List[Document]:
    """Splits documents into smaller chunks while preserving metadata.

    Each resulting chunk gets a stable `chunk_index` (0-based, per parent
    document) so downstream resume logic can key on (doc_id, chunk_index)
    instead of a fragile list position.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Assign a per-document chunk_index. RecursiveCharacterTextSplitter
    # preserves parent metadata but does not add an index, so we do it
    # here. Index resets for each new doc_id so the pair (doc_id,
    # chunk_index) is a stable unique key across runs.
    current_doc_id = None
    per_doc_counter = 0
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id")
        if doc_id != current_doc_id:
            current_doc_id = doc_id
            per_doc_counter = 0
        chunk.metadata["chunk_index"] = per_doc_counter
        per_doc_counter += 1

    print(
        f"[Chunking] Generated {len(chunks)} text chunks from {len(documents)} original documents."
    )
    return chunks


if __name__ == "__main__":
    raw_documents = load_documents_from_db()

    chunks = split_documents_into_chunks(
        raw_documents, chunk_size=800, chunk_overlap=100
    )

    if chunks:
        print("\n" + "=" * 65)
        print("                   SAMPLE CHUNK PREVIEW                        ")
        print("=" * 65)
        print(f"Chunk #1 Metadata:\n{chunks[0].metadata}")
        print("-" * 65)
        print(f"Chunk Text Preview:\n{chunks[0].page_content[:300]}...")
        print("=" * 65)