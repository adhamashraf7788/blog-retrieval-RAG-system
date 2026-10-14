# Evaluation Queries
# Shared test set for comparing multilingual retrieval systems


EVALUATION_QUERIES = [

    {
        "id": "ar_vector_search",
        "query": "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",
        "language": "ar",
        "relevant_chunks": ["ar_vector"]
    },

    {
        "id": "en_vector_search",
        "query": "How do vector databases improve semantic search?",
        "language": "en",
        "relevant_chunks": ["en_vector"]
    },

    {
        "id": "fr_vector_search",
        "query": "Comment les bases de données vectorielles améliorent-elles la recherche sémantique ?",
        "language": "fr",
        "relevant_chunks": ["fr_vector"]
    },

    {
        "id": "ar_semantic_search",
        "query": "ما هو البحث الدلالي؟",
        "language": "ar",
        "relevant_chunks": [
            "ar_vector",
            "en_vector",
            "fr_vector"
        ]
    },

    {
        "id": "en_semantic_search",
        "query": "What is semantic search?",
        "language": "en",
        "relevant_chunks": [
            "ar_vector",
            "en_vector",
            "fr_vector"
        ]
    },

    {
        "id": "fr_semantic_search",
        "query": "Qu'est-ce que la recherche sémantique ?",
        "language": "fr",
        "relevant_chunks": [
            "ar_vector",
            "en_vector",
            "fr_vector"
        ]
    },

]