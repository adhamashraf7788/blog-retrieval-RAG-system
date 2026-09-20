# Query
#   ↓
# Tokenization
#   ↓
# Token-Level Script Detection
#   ↓
# Language Analysis
#   ↓
# Routing Decision


import re
from langdetect import detect, LangDetectException


SUPPORTED_LANGUAGES = ["ar", "en", "fr"]

# Minimum percentage of a language required
# to consider the query mixed.
MIXED_LANGUAGE_THRESHOLD = 0.20


# --------------------------------------------------
# Arabic Detection
# --------------------------------------------------

def is_arabic(token: str) -> bool:
    """
    Check whether a token contains Arabic characters.
    """

    return bool(
        re.search(r"[\u0600-\u06FF]", token)
    )


# --------------------------------------------------
# Latin Detection
# --------------------------------------------------

def is_latin(token: str) -> bool:
    """
    Check whether a token contains Latin characters.
    """

    return bool(
        re.search(r"[A-Za-zÀ-ÿ]", token)
    )


# --------------------------------------------------
# Tokenize Query
# --------------------------------------------------

def tokenize(query: str) -> list:
    """
    Extract word-like tokens from the query.
    """

    return re.findall(
        r"\S+",
        query
    )


# --------------------------------------------------
# Detect Languages
# --------------------------------------------------

def detect_languages(query: str) -> dict:
    """
    Detect languages at token level.

    Arabic is detected using Unicode script.
    Latin text is analyzed using langdetect.
    """

    tokens = tokenize(query)

    arabic_tokens = []
    latin_tokens = []

    for token in tokens:

        # Remove punctuation
        clean_token = re.sub(
            r"[^\wÀ-ÿ\u0600-\u06FF]",
            "",
            token
        )

        if not clean_token:
            continue

        if is_arabic(clean_token):

            arabic_tokens.append(clean_token)

        elif is_latin(clean_token):

            latin_tokens.append(clean_token)

    languages = {}

    # --------------------------------------------------
    # Arabic
    # --------------------------------------------------

    if arabic_tokens:

        languages["ar"] = len(arabic_tokens)

    # --------------------------------------------------
    # Latin Languages
    # --------------------------------------------------

    if latin_tokens:

        latin_text = " ".join(latin_tokens)

        try:

            detected = detect(latin_text)

            if detected in SUPPORTED_LANGUAGES:

                languages[detected] = (
                    languages.get(detected, 0)
                    + len(latin_tokens)
                )

            else:

                # Unknown Latin text → assume English
                languages["en"] = (
                    languages.get("en", 0)
                    + len(latin_tokens)
                )

        except LangDetectException:

            languages["en"] = (
                languages.get("en", 0)
                + len(latin_tokens)
            )

    return languages


# --------------------------------------------------
# Analyze Query
# --------------------------------------------------

def analyze_query(query: str) -> dict:
    """
    Analyze query and determine retrieval route.
    """

    language_counts = detect_languages(query)

    # --------------------------------------------------
    # No language detected
    # --------------------------------------------------

    if not language_counts:

        return {
            "languages": [],
            "primary_language": "unknown",
            "mixed_language": False,
            "route": "both"
        }

    # --------------------------------------------------
    # Calculate total tokens
    # --------------------------------------------------

    total_tokens = sum(
        language_counts.values()
    )

    language_ratios = {
        lang: count / total_tokens
        for lang, count in language_counts.items()
    }

    # --------------------------------------------------
    # Keep languages above threshold
    # --------------------------------------------------

    detected_languages = [
        lang
        for lang, ratio in language_ratios.items()
        if ratio >= MIXED_LANGUAGE_THRESHOLD
    ]

    # --------------------------------------------------
    # Primary Language
    # --------------------------------------------------

    primary_language = max(
        language_counts,
        key=language_counts.get
    )

    # --------------------------------------------------
    # Mixed Language
    # --------------------------------------------------

    mixed_language = len(
        detected_languages
    ) > 1

    # --------------------------------------------------
    # Routing
    # --------------------------------------------------

    if mixed_language:

        route = "both"

    else:

        route = "multilingual"

    return {
        "languages": detected_languages,
        "primary_language": primary_language,
        "mixed_language": mixed_language,
        "language_ratios": language_ratios,
        "route": route
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [

        "كيف تساعد قواعد البيانات المتجهة في تحسين البحث الدلالي؟",

        "How do vector databases improve semantic search?",

        "Comment les bases de données vectorielles améliorent-elles la recherche sémantique ?",

        "عايز أبحث عن vector databases و semantic search",

        "ما هو semantic search في artificial intelligence؟",

        "كيف يعمل RAG مع vector databases؟"
    ]

    for query in test_queries:

        analysis = analyze_query(query)

        print("=" * 80)

        print(f"Query: {query}")

        print(
            f"Languages: "
            f"{analysis['languages']}"
        )

        print(
            f"Primary Language: "
            f"{analysis['primary_language']}"
        )

        print(
            f"Mixed Language: "
            f"{analysis['mixed_language']}"
        )

        print(
            f"Language Ratios: "
            f"{analysis.get('language_ratios', {})}"
        )

        print(
            f"Route: "
            f"{analysis['route']}"
        )