# Text
#   ↓
# BAAI/bge-m3
#   ↓
# Normalized Embedding
#   ↓
# 1024-dimensional vector

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


# Load embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")

model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu"
)


def generate_embeddings(texts: list[str]):
    """
    Generate normalized embeddings using BGE-M3.

    Parameters
    ----------
    texts : list[str]
        List of text chunks or queries.

    Returns
    -------
    numpy.ndarray
        Normalized 1024-dimensional embeddings.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    texts = [
        "How do vector databases improve semantic search?",
        "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",
        "Comment les bases de données vectorielles améliorent-elles la recherche sémantique ?"
    ]

    embeddings = generate_embeddings(texts)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])