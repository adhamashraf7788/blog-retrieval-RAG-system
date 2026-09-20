from query_analyzer import analyze_query


def route_query(query: str) -> dict:
    """
    Decide which retrieval branches should be executed.
    """

    analysis = analyze_query(query)

    route = analysis["route"]

    if route == "multilingual":
        branches = ["multilingual"]

    elif route == "both":
        branches = [
            "multilingual",
            "translation"
        ]

    else:
        # Safety fallback
        branches = [
            "multilingual",
            "translation"
        ]

    return {
        "query": query,
        "languages": analysis["languages"],
        "primary_language": analysis["primary_language"],
        "mixed_language": analysis["mixed_language"],
        "route": route,
        "branches": branches
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [

        "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",

        "How do vector databases improve semantic search?",

        "عايز أبحث عن vector databases و semantic search",

        "ما هو semantic search في artificial intelligence؟"
    ]

    for query in test_queries:

        result = route_query(query)

        print("=" * 80)

        print(f"Query: {query}")
        print(f"Languages: {result['languages']}")
        print(f"Primary Language: {result['primary_language']}")
        print(f"Mixed Language: {result['mixed_language']}")
        print(f"Route: {result['route']}")
        print(f"Branches: {result['branches']}")