"""
Final generation step — NO LoRA, NO training. Prompting only.

Takes retrieved context (whatever the chunking / vector-DB retrieval step
hands over) and returns a summary written in the SAME language as that
context: Arabic (Modern Standard Arabic, not dialect), English, or French.

This is the prompting-based alternative to loading a LoRA adapter
(see inference/inference_with_rag.py for the adapter-based version).
Chosen model: openai/gpt-oss-120b — the only model of the three tested
(Qwen3.8-27b, GPT-OSS-120b, ALLaM-2-7b) that kept every output in the
correct language under both prompts. See test_language_detection.py.

Setup:
    pip install groq
    $env:GROQ_API_KEY="your-key-here"     # PowerShell
    # or: export GROQ_API_KEY="..."       # bash/Linux

    Optional, for French-vs-English verification:
    pip install langdetect

Use from the RAG pipeline:
    from generate_response_groq import generate_response
    summary = generate_response(retrieved_context)

Quick manual check:
    python generate_response_groq.py
"""

import os
import re
import time
from groq import Groq

MODEL_ID = os.environ.get("GROQ_MODEL_ID", "openai/gpt-oss-120b")
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a news summarization assistant. First, detect which
language the article is written in — it will be Arabic, English, or French.
Then write a 2-3 sentence summary in that SAME language. If the article is in
Arabic, respond in formal Modern Standard Arabic (not a regional dialect).
Do not translate the article into a different language under any circumstances —
the summary language must match the article's original language exactly.
Output only the summary itself, with no preamble and no notes."""

_client = None


def _get_client():
    # Created lazily so importing this module never fails just because the
    # key isn't set yet (e.g. during unit tests of other pipeline pieces).
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


# ── Cheap language check, used to catch the exact failure seen in testing ──
ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")
LATIN_CHARS = re.compile(r"[A-Za-z]")


def script_of(text):
    """Returns 'arabic' or 'latin' based on which script dominates."""
    arabic = len(ARABIC_CHARS.findall(text))
    latin = len(LATIN_CHARS.findall(text))
    return "arabic" if arabic >= latin else "latin"


def detect_language(text):
    """
    Best-effort language of `text`: 'ar', 'en', 'fr', or None if unsure.
    Arabic is detected by script alone (reliable). English vs French uses
    langdetect if installed; without it, returns None for Latin-script text
    and the caller falls back to the script-only check.
    """
    if script_of(text) == "arabic":
        return "ar"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
        code = detect(text)
        return code if code in ("en", "fr") else None
    except Exception:
        return None


def _languages_match(source_text, output_text):
    source_lang = detect_language(source_text)
    output_lang = detect_language(output_text)
    if source_lang is not None and output_lang is not None:
        return source_lang == output_lang
    # Fall back to the script-only check when langdetect can't decide.
    return script_of(source_text) == script_of(output_text)


def generate_response(retrieved_context, max_attempts=MAX_ATTEMPTS):
    """
    Summarize `retrieved_context` in its own language.

    Retries (up to max_attempts) if the output comes back in a different
    language than the input — the main failure mode seen in testing — and
    raises RuntimeError if it never matches, rather than silently returning
    a wrong-language answer.
    """
    if not retrieved_context or not retrieved_context.strip():
        raise ValueError("retrieved_context is empty")

    last_output = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = _get_client().chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": retrieved_context},
                ],
                temperature=0.3,   # lower than the 0.7 used in testing: less
                                   # run-to-run variation for production use
                max_tokens=400,
            )
        except Exception as exc:
            if attempt == max_attempts:
                raise
            time.sleep(2 * attempt)  # simple backoff on rate limits / blips
            continue

        last_output = response.choices[0].message.content.strip()
        if _languages_match(retrieved_context, last_output):
            return last_output

    raise RuntimeError(
        f"Output language never matched input after {max_attempts} attempts. "
        f"Last output was: {last_output!r}"
    )


if __name__ == "__main__":
    samples = [
        "أعلنت وزارة التجارة اليوم عن اتفاقية جديدة لخفض الرسوم الجمركية "
        "على المنتجات الزراعية بين الدول المشاركة، على أن يبدأ التنفيذ الشهر المقبل.",
        "The Ministry of Trade announced a new agreement today to reduce tariffs "
        "on agricultural products between participating countries, with "
        "implementation set to begin next month.",
        "Le ministère du Commerce a annoncé aujourd'hui un nouvel accord visant "
        "à réduire les droits de douane sur les produits agricoles entre les "
        "pays participants, dont la mise en œuvre débutera le mois prochain.",
    ]
    for text in samples:
        print("INPUT :", text)
        print("OUTPUT:", generate_response(text))
        print("-" * 60)
    print("NOTE: judge Arabic output in a browser or editor, not PowerShell —")
    print("the terminal often renders right-to-left text in the wrong order.")
