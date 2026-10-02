"""
summarize_classify_final.py  (v2.4.1 - Groq only, balanced + domain-diverse pool)

v2.4.1: NPR host-bio chunks ("See stories by ...", "Copyright 20xx NPR") are
filtered before the LLM call (_NPR_BIO_RE, used by _is_author_bio).

Pipeline:
    PostgreSQL -> Documents -> CLEAN (cut "Read more >>" tails) -> Chunks
      -> filter noise -> balanced, domain-diverse POOL
      -> Summary + Category + Tags per chunk (Groq) -> JSON

What changed in v2.4 (vs v2.3):
    A. Self-flagged noise is caught regardless of category. Before, a chunk
       was only marked NOISE if the model answered category "Other"; pages
       such as NPR headers or cinema booking pages came back as Sports /
       Culture and slipped through as SUCCESS. Now:
         1. the prompt asks the model to START the summary with NOT_ARTICLE:
            for non-article text (most reliable signal), and
         2. a regex safety net (is_noise_summary) reads the summary when the
            model describes the text as a menu / header / booking page.
    B. Prompt: summaries must not open with "The text ..." / "Le texte ..." /
       "This is ..."; never add a year, date or number that is not in the text.
    C. Boilerplate prefixes: "skip to main content" (NPR member stations etc.)
       and "n'assume aucune responsabilite".
    D. Blocked domains: + lg.com, realting.com, kinoafisha.info.
       Forum URLs (/forum/, /forums/) are dropped as non-article.
    E. LLM_PER_LANGUAGE = 40 (matches the runs you actually did) and the
       output file is chunk_analysis_results_v1_4_multilingual_chunks.json so
       existing results keep resuming. Use clean_results.py once to turn the
       leaked noise rows already in that file into NOISE.

What changed in v2.3 (vs v2.2):
    A. Prompt: never take names/roles/facts from the title; quotes only go to
       the speaker the TEXT names; tags must come from the text.
    B. _filter_tags(): drops tags whose words do not appear in the chunk text
       (title / publisher tags such as a site name).
    C. New categories: Science, Environment. Food now covers food labeling
       and restaurant rules.
    D. Advertorial / sponsored URLs (/publi-, /sponsored, advertorial) are
       dropped as noise.
    E. Cleaner: strips the site-menu header before "Advertisement", cuts the
       "Sawt al-Hijaz" newspaper footer, removes trailing "Copyright 20xx ..."
       lines. Menus with "SECTION (218)" counters are treated as nav lists.
    F. Chunks the model itself flags as non-article (category Other + summary
       says navigation / bio / donation appeal ...) are stored with
       status "NOISE" (kept for bookkeeping, never retried, not counted as
       SUCCESS). Filter them out downstream: status == "SUCCESS".

What changed in v2.2 and earlier:
    1. MAX_PER_DOMAIN: the cap is now per DOMAIN (netloc), not per full URL.
       Before, every URL was its own "source", so the cap never triggered.
    2. Round-robin over domains after shuffling, so the first N chunks the
       LLM sees come from as many different sites as possible.
    3. clean_article_text(): cuts headtopics-style tails ("Read more >>",
       "Similar News:", ...) BEFORE chunking, so link lists never become chunks.
    4. _is_noise(): one place for all noise rules (short, boilerplate,
       link-list, author bio, blocked domain, garbled, headline aggregator).
    5. Garbled detection rewritten: script-ratio check + real mojibake
       patterns (the old char set contained normal Arabic letters).
    6. Prompt: no outcomes that are not in the text, upcoming != finished,
       keep who-did-what exact, title given as context only.
    7. Retry uses a higher temperature (temperature=0 gave identical output).
    8. Chains built once (cached), not per chunk. Retry regex no longer
       matches random numbers. Save in try/finally (safe on Ctrl+C).
    9. Two new categories: Lifestyle, Security.

NOTE: the output file name changed (v2) so old, noisier results are not mixed
in. Delete/rename it if you want a fresh start.
"""

import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit(
        "GROQ_API_KEY is not set. Export it before running, e.g.:\n"
        "  Windows (PowerShell): $env:GROQ_API_KEY='gsk_...'\n"
        "  Linux/macOS         : export GROQ_API_KEY='gsk_...'"
    )

# Windows fix: stops httpx crashing on Arabic-locale proxy settings
_no_proxy_client = httpx.Client(trust_env=False, timeout=60.0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

_loader_path = os.path.join(BASE_DIR, "langchain.py")
_spec = importlib.util.spec_from_file_location("project_langchain_loader", _loader_path)
_loader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader_module)

load_documents_from_db = _loader_module.load_documents_from_db
split_documents_into_chunks = _loader_module.split_documents_into_chunks


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CATEGORIES = Literal[
    "Politics", "Economy", "Sports", "Technology",
    "Culture", "Health", "Society", "Religion",
    "Weather", "Food", "Crime", "Lifestyle", "Security",
    "Science", "Environment", "Other",
]

MAX_TAGS = 6
MIN_CHUNK_CHARS = 250

BOILERPLATE_PREFIXES = (
    "legal disclaimer",
    "menafn provides the information",
    "© copyright",
    "all rights reserved",
    "disclaimer:",
    "this article is meant for information purposes only",
    "this article and information do not constitute",
    "forward-looking statements contained in this press release",
    "securities and exchange commission",
    "skip to main content",
    "skip to content",
    "aller au contenu",
    "passer au contenu",
    "n'assume aucune responsabilit",
    "n’assume aucune responsabilit",
)

LANGUAGES = ["ar", "en", "fr"]

# Pool settings
POOL_MAX = 1500
MAX_PER_DOC = 5
MAX_PER_DOMAIN = 40          # per DOMAIN (e.g. sa.headtopics.com), None = no cap
POOL_SEED = 42
LLM_PER_LANGUAGE = 40

# Domains whose content is generated / low-value for this project.
BLOCKED_DOMAINS = {
    "sixactualites.fr", "lg.com",
    "realting.com",        # real-estate UI text
    "kinoafisha.info",     # cinema ticket-booking pages
}

# Sponsored / advertorial pages and forums are not news articles.
_NON_ARTICLE_URL_RE = re.compile(
    r"/publi-|/sponsored|advertorial|/partner-content|/forums?/",
    re.IGNORECASE,
)

# A site-menu header ends with "Advertisement" right before the article.
_SITE_HEADER_MARKERS = ("Privacy Policy", "\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644",
                        "Se connecter")
_SITE_HEADER_MAX = 2500

# Trailing "Copyright 2026 NPR" style lines.
_COPYRIGHT_TAIL_RE = re.compile(
    r"\s*(?:\u00a9|Copyright)\s*(?:19|20)\d{2}\b[^\n]{0,60}$", re.IGNORECASE
)

# ---- Self-flagged noise (the model says the chunk is not article content) ----
# 1) Sentinel: the prompt asks the model to start the summary with NOT_ARTICLE:
_NOT_ARTICLE_TAG_RE = re.compile(r"^\W*NOT[_ ]ARTICLE\b\W*", re.IGNORECASE)

# 2) Weak signals, only trusted when the model also answered category "Other".
_SELF_FLAGGED_NOISE_RE = re.compile(
    r"navigation|not (?:part of|a coherent)|no substantive|contains no "
    r"(?:substantive|article)|(?:author|reviewer) bio|donation|cookie|"
    r"menu de navigation|pas (?:un )?(?:article|de contenu)|aucun contenu|"
    r"\u0642\u0627\u0626\u0645\u0629 (?:\u062a\u0646\u0642\u0644|\u0631\u0648\u0627\u0628\u0637|\u0623\u0642\u0633\u0627\u0645)|\u0644\u0627 \u064a\u062d\u062a\u0648\u064a|\u0646\u0628\u0630\u0629 \u0639\u0646 \u0627\u0644\u0643\u0627\u062a\u0628",
    re.IGNORECASE,
)

# 3) Strong signals, trusted in ANY category, but only when the summary opens
#    by describing the text itself ("The text is a ...", "Ce texte est ...").
_META_OPENING_RE = re.compile(
    r"^\W*(?:the (?:provided |given |above |following )?"
    r"(?:text|excerpt|passage|content|chunk)"
    r"|this (?:text|excerpt|passage|content|is|page)"
    r"|(?:ce|le) (?:texte|passage|contenu)(?: fourni)?"
    r"|cet extrait|il s.agit"
    r"|\u0647\u0630\u0627 \u0627\u0644\u0646\u0635|\u0627\u0644\u0646\u0635|\u064a\u064f\u0639\u062f\u0651 \u0647\u0630\u0627 \u0627\u0644\u0646\u0635|\u064a\u0639\u062f \u0647\u0630\u0627 \u0627\u0644\u0646\u0635)",
    re.IGNORECASE,
)
_NON_ARTICLE_TERMS_RE = re.compile(
    r"not (?:the |an? )?(?:actual )?article|not (?:part of|a coherent)|"
    r"no substantive|site navigation|website navigation|"
    r"navigation (?:menu|links?|block|elements?|bar)|"
    r"(?:ticket|seat)[- ]booking|booking page|"
    r"(?:station|site|page) header|metadata block|"
    r"forum (?:interface|header|page)|website interface|web interface|"
    r"interface (?:text|elements?|excerpt)|copyright notice|"
    r"donation appeal|promotional blurb|comments? policy|brief news headline|"
    r"extrait d.interface|interface utilisateur|liens? de navigation|"
    r"menu de navigation|ne contient (?:aucun|pas) (?:de )?(?:contenu|article)|"
    r"aucun contenu|pas (?:un )?article|"
    r"\u062a\u0631\u0648\u064a\u0633|\u0648\u0627\u062c\u0647\u0629 (?:\u0627\u0644\u0645\u0648\u0642\u0639|\u0645\u0648\u0642\u0639|\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645)|\u0639\u0646\u0627\u0635\u0631 (?:\u0627\u0644\u062a\u0646\u0642\u0644|\u062a\u0646\u0642\u0644)|"
    r"\u0642\u0627\u0626\u0645\u0629 (?:\u0627\u0644\u062a\u0646\u0642\u0644|\u062a\u0646\u0642\u0644|\u0631\u0648\u0627\u0628\u0637|\u0623\u0642\u0633\u0627\u0645)|"
    r"\u0644\u0627 \u064a\u062d\u062a\u0648\u064a (?:\u0639\u0644\u0649 )?(?:\u0645\u062d\u062a\u0648\u0649|\u0645\u0642\u0627\u0644|\u0646\u0635)|\u0644\u064a\u0633 \u0645\u0642\u0627\u0644",
    re.IGNORECASE,
)


def is_noise_summary(summary: Optional[str], category: Optional[str]) -> bool:
    """True if the model's own summary says the chunk is not article content.

    NOTE: clean_results.py has a copy of these rules - keep both in sync.
    """
    s = (summary or "").strip()
    if not s:
        return False
    if _NOT_ARTICLE_TAG_RE.match(s):
        return True
    if category == "Other" and _SELF_FLAGGED_NOISE_RE.search(s):
        return True
    return bool(_META_OPENING_RE.match(s) and _NON_ARTICLE_TERMS_RE.search(s[:300]))

# Text after these markers is a list of unrelated links, not article body.
_CUT_MARKERS = (
    "Read more »",
    "Similar News:",
    "United States Latest News",
    "\u0635\u0648\u062a \u0627\u0644\u062d\u062c\u0627\u0632 \u0623\u0648\u0644 \u062c\u0631\u064a\u062f\u0629",
)

# Minimum share of letters that must belong to the expected script.
MIN_SCRIPT_RATIO = {"ar": 0.50, "en": 0.70, "fr": 0.70}

# Mojibake: Arabic ذ/ر followed directly by a non-Arabic symbol (typical of
# Cyrillic UTF-8 decoded as cp1256), or classic Latin-1/UTF-8 double-decoding
# fragments, or the replacement character.
_MOJIBAKE_RE = re.compile(
    r"[ذر][^\u0600-\u06FF\s\d.,;:!?()«»\"'\-/%]"
    r"|[ØÙÐÑÃÂ][\u0080-\u00BF\u2018-\u203A\u0152\u0153\u0160\u0161\u0178\u017D\u017E\u0192\u02C6\u02DC]"
    r"|ط[§¨©ª«¬®¯°±]|ظ[„…†‡]"
    r"|\ufffd"
)
MOJIBAKE_RATIO_THRESHOLD = 0.03
GARBLED_MIN_LEN_TO_CHECK = 40

_HEADLINE_AGGREGATOR_PATTERN = re.compile(
    r"\d{2}:\d{2}\s*\|\s*\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}\s*\|\s*\d{4}-\d{2}-\d{2}"
    r"|(?:\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+){2}"
    r"|(?:Lebanon 24\s+){2,}",
    re.DOTALL,
)

_AUTHOR_BIO_RE = re.compile(
    r"(?:\bis an? [\w\s,-]{0,40}(?:journalist|reporter|editor|writer|correspondent)\b"
    r".{0,120}(?:years? of experience|covers|covering|when she|when he))"
    r"|(?:^[\W_]*(?:she|he) (?:also )?(?:covers|writes about|reports on)\b)",
    re.IGNORECASE | re.DOTALL,
)

# NPR member-station pages end with host bios ("See stories by ...") and a
# "Copyright 2026 NPR" line. Those chunks are noise; dropping them before the
# LLM call saves about 1,300 tokens of daily quota each.
_NPR_BIO_RE = re.compile(
    r"\bSee stories by\b|\bCopyright\s+(?:19|20)\d{2}\s+NPR\b", re.IGNORECASE
)

DEBUG_GARBLED = False   # print a few dropped "garbled" samples to verify the filter


class ChunkAnalysis(BaseModel):
    summary: str = Field(
        description="A concise 1-2 sentence summary of just THIS piece of "
        "text, written in the SAME language as the original text."
    )
    category: CATEGORIES = Field(
        description="The single best topic category for just THIS piece of "
        "text, based only on what this chunk itself discusses."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="3-6 short keyword tags for retrieval (RAG): named "
        "entities and specific topical terms that appear in THIS chunk, "
        "in the SAME language as the text. No generic tags."
    )


DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
CHUNK_MAX_TOKENS = 600
RETRY_TEMPERATURE = 0.3


@lru_cache(maxsize=None)
def _get_model(temperature: float) -> ChatGroq:
    return ChatGroq(
        model=DEFAULT_GROQ_MODEL,
        temperature=temperature,
        max_tokens=CHUNK_MAX_TOKENS,
        api_key=os.environ["GROQ_API_KEY"],
        http_client=_no_proxy_client,
        reasoning_effort="none",
    )


LANGUAGE_NAMES = {"ar": "Arabic", "en": "English", "fr": "French"}
LANGUAGE_ORDER = {"ar": 0, "en": 1, "fr": 2}


CHUNK_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a multilingual news analyst. You get ONE piece of a "
            "longer article. Based ONLY on it: write a concise 1-2 sentence "
            "summary in the input language, pick ONE category, and extract "
            "3-6 tags for a retrieval (RAG) system.\n\n"
            "Rules:\n"
            "- State only what the text says; never guess about unseen parts.\n"
            "- Do not present upcoming/undecided things as finished, and keep "
            "who-did-what exact (winners/losers, subjects/objects).\n"
            "- The title is context only: never take names, roles, titles or "
            "facts from it. Attribute a quote only to the person the TEXT "
            "names as the speaker; if the text does not say who, do not "
            "guess.\n"
            "- Tags must come from the text itself, never from the title, the "
            "publisher name or copyright lines.\n"
            "- The summary is plain sentences only: no JSON, XML or field "
            "names. Start directly with the subject (who did what, what "
            "happened). Never open with 'The text', 'This text', 'This is', "
            "'The article', 'Le texte', 'Ce texte' or 'هذا النص'.\n"
            "- Never add a year, date, name or number that is not in the "
            "text.\n"
            "- If the text is NOT article content (unreadable, site "
            "navigation or header, a list of unrelated headlines/links, a "
            "cookie notice, an author bio, a forum/booking/product-page "
            "interface, a newsletter or donation appeal): start the summary "
            "with the exact English word NOT_ARTICLE: followed by one short "
            "sentence saying what it is, use category 'Other', and take no "
            "tags from menus or link lists. Never invent a plausible summary "
            "from noise.\n"
            "- Tags: specific named entities (people, organizations, places, "
            "laws, events) and specific topical terms that appear in the "
            "text; no generic words, no category name, no invented entities; "
            "max 6.\n\n"
            "Categories (pick the most specific): Weather; Food (recipes, "
            "dishes, food labeling and restaurant rules); Religion (prayer, fiqh, religious holidays, sermons; "
            "not Culture); Culture (arts, entertainment, celebrities, books, "
            "festivals); Crime (police, arrests, courts, fraud; Society only "
            "for social issues that are not mainly crime); Health (medicine, "
            "public health, nutrition science); Lifestyle (beauty, "
            "horoscopes, fashion, relationship advice, consumer tips, "
            "deals); Security (military operations, battlefield events, "
            "weapons, terrorism, cyber security; diplomacy and government "
            "positions on a war are Politics); Politics; Economy; Sports; "
            "Technology; Science (space, physics, astronomy, research "
            "findings); Environment (climate, pollution, forests, rivers, "
            "wildlife); Society; Other (only if nothing fits).",
        ),
        (
            "human",
            "The input language is {language}. Write the summary in {language}.\n"
            "Article title (context only): {title}\n\n"
            "Text:\n\n{text}",
        ),
    ]
)


@lru_cache(maxsize=None)
def _get_chain(temperature: float):
    return CHUNK_ANALYSIS_PROMPT | _get_model(temperature).with_structured_output(
        ChunkAnalysis
    )


RETRYABLE_ERROR_SUBSTRINGS = (
    "rate_limit",
    "rate limit",
    "connection error",
    "connection reset",
    "timeout",
    "timed out",
    "remote end closed",
    "server disconnected",
    "leaked structured-output syntax",
    "bad summary",
)
_RETRYABLE_STATUS_RE = re.compile(r"error code:\s*(429|500|502|503|504)")


def _is_retryable(error_str: str) -> bool:
    low = error_str.lower()
    return any(s in low for s in RETRYABLE_ERROR_SUBSTRINGS) or bool(
        _RETRYABLE_STATUS_RE.search(low)
    )


class DailyQuotaExceeded(Exception):
    """Raised when Groq's DAILY token quota (TPD) is hit."""


def _is_daily_quota_error(error_str: str) -> bool:
    low = error_str.lower()
    return "tokens per day" in low or "(tpd)" in low


_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?\s*([\d.]+)s", re.IGNORECASE)
MAX_AUTO_WAIT = 15 * 60   # seconds; longer waits stop the run instead


def _quota_wait_seconds(err: str) -> Optional[float]:
    m = _WAIT_RE.search(err)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2)) + 5


_SUMMARY_CONTAMINATION_MARKERS = (
    "</summary>", "<summary>", "<parameter", "</parameter",
    '"category":', '"tags":', '"summary":', "```json", "<|", "|>",
)


def _is_contaminated_summary(summary: Optional[str]) -> bool:
    if not summary:
        return False
    return any(m in summary for m in _SUMMARY_CONTAMINATION_MARKERS)


def _cap_tags(tags: Optional[List[str]], max_tags: int = MAX_TAGS) -> List[str]:
    return (tags or [])[:max_tags]


_TAG_TOKEN_RE = re.compile(r"\w{3,}", re.UNICODE)


def _norm_token(t: str) -> str:
    t = t.lower()
    return t[2:] if t.startswith("\u0627\u0644") and len(t) > 4 else t


def _filter_tags(tags: Optional[List[str]], text: str,
                 min_keep: int = 2) -> List[str]:
    """Keep tags with at least one word that appears in the chunk text.

    Drops tags the model took from the title or the publisher name. If fewer
    than min_keep tags survive, the original list is kept (better than none).
    """
    tags = tags or []
    hay = (text or "").lower()
    kept = []
    for tag in tags:
        toks = [_norm_token(t) for t in _TAG_TOKEN_RE.findall(tag)]
        if toks and any(t in hay for t in toks):
            kept.append(tag)
    return kept if len(kept) >= min_keep else tags


def _chunk_key(doc_id, chunk_index) -> Tuple:
    return (doc_id, chunk_index)


def _key_of(c: Document) -> Tuple:
    return _chunk_key(c.metadata.get("doc_id"), c.metadata.get("chunk_index"))


def _domain(url: Optional[str]) -> str:
    host = (urlparse(url or "").netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


# ---------------------------------------------------------------------------
# CLEANING + NOISE RULES
# ---------------------------------------------------------------------------
def _strip_site_header(text: str) -> str:
    """Drop a site-menu header that ends with 'Advertisement'."""
    i = text.find("Advertisement", 0, _SITE_HEADER_MAX)
    if i != -1 and any(m in text[:i] for m in _SITE_HEADER_MARKERS):
        return text[i + len("Advertisement"):].lstrip()
    return text


def clean_article_text(text: str) -> str:
    """Strip the menu header, cut at the first 'link list' marker, drop
    a trailing copyright line."""
    if not text:
        return text
    text = _strip_site_header(text)
    cut = len(text)
    for m in _CUT_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut].rstrip()
    return _COPYRIGHT_TAIL_RE.sub("", text).rstrip()


def clean_documents(docs: List[Document]) -> List[Document]:
    changed = 0
    for d in docs:
        new = clean_article_text(d.page_content or "")
        if new != d.page_content:
            d.page_content = new
            changed += 1
    print(f"[CLEAN] Trimmed link-list tails from {changed} document(s).")
    return docs


def _is_boilerplate_chunk(text: str) -> bool:
    if not text:
        return True
    head = text.strip()[:160].lower()
    return any(head.startswith(p) or p in head for p in BOILERPLATE_PREFIXES)


def _script_ratio(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "ar":
        ok = sum(
            "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F"
            or "\uFB50" <= c <= "\uFEFF"
            for c in letters
        )
    else:
        ok = sum(c.isascii() or "\u00C0" <= c <= "\u024F" for c in letters)
    return ok / len(letters)


def _is_garbled_text(text: str, lang: Optional[str] = None) -> bool:
    """Mojibake, or text whose script does not match the labelled language."""
    if not text:
        return True
    sample = text[:600]
    if len(sample) < GARBLED_MIN_LEN_TO_CHECK:
        return False
    if len(_MOJIBAKE_RE.findall(sample)) / len(sample) > MOJIBAKE_RATIO_THRESHOLD:
        return True
    lang = (lang or "").lower()
    if lang in MIN_SCRIPT_RATIO and _script_ratio(sample, lang) < MIN_SCRIPT_RATIO[lang]:
        return True
    return False


def _is_headline_aggregator(text: str) -> bool:
    return bool(text) and bool(_HEADLINE_AGGREGATOR_PATTERN.search(text[:500]))


_SENT_END_RE = re.compile(r"[.!?\u061f\u3002](?:\s|$)")


def _looks_like_headline_list(text: str) -> bool:
    """Long text with almost no sentences/commas = nav menu or news ticker."""
    if text.count("\u00b0C") >= 3:            # city temperature strips
        return True
    if len(re.findall(r"\(\d+\)", text)) >= 5:  # "SECTION NAME (218)" menus
        return True
    return (len(text) >= 500
            and len(_SENT_END_RE.findall(text)) <= 1
            and len(re.findall(r"[,\u060c]", text)) <= 2)


def _is_author_bio(text: str) -> bool:
    return bool(_AUTHOR_BIO_RE.search(text[:400]) or _NPR_BIO_RE.search(text))


def _noise_reason(c: Document) -> Optional[str]:
    """Return why a chunk should be dropped, or None if it is usable."""
    text = (c.page_content or "").strip()
    lang = (c.metadata.get("language") or "").lower()
    if len(text) < MIN_CHUNK_CHARS or _is_boilerplate_chunk(text):
        return "short"
    if _domain(c.metadata.get("source")) in BLOCKED_DOMAINS:
        return "blocked"
    if _NON_ARTICLE_URL_RE.search(c.metadata.get("source") or ""):
        return "non_article_url"
    if any(m in text for m in _CUT_MARKERS):
        return "linklist"
    if _is_author_bio(text):
        return "bio"
    if _looks_like_headline_list(text):
        return "headlines"
    if _is_garbled_text(text, lang):
        return "garbled"
    if _is_headline_aggregator(text):
        return "aggregator"
    return None


def filter_chunks(chunks: List[Document]) -> List[Document]:
    kept, dropped = [], Counter()
    for c in chunks:
        reason = _noise_reason(c)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(c)
    if dropped:
        parts = " + ".join(f"{v} {k}" for k, v in dropped.items())
        print(f"[FILTER] Dropped {parts}; {len(kept)} remaining.")
    if DEBUG_GARBLED and dropped["garbled"]:
        samples = [c.page_content[:100].replace("\n", " ") for c in chunks
                   if _noise_reason(c) == "garbled"][:5]
        for smp in samples:
            print(f"    [DEBUG garbled] {smp!r}")
    return kept


# ---------------------------------------------------------------------------
# BALANCED, DOMAIN-DIVERSE POOL
# ---------------------------------------------------------------------------
def build_pool(
    chunks: List[Document],
    languages: List[str] = LANGUAGES,
    max_pool: int = POOL_MAX,
    max_per_doc: int = MAX_PER_DOC,
    max_per_domain: Optional[int] = MAX_PER_DOMAIN,
    seed: int = POOL_SEED,
) -> List[Document]:
    """
    Per language: filter -> deterministic sort -> seeded shuffle -> cap per
    article and per domain -> ROUND-ROBIN across domains, so the head of the
    list is as diverse as possible. Final size = min(max_pool, smallest lang).
    """
    rng = random.Random(seed)
    per_lang: Dict[str, List[Document]] = {}

    for lang in languages:
        matched = [
            c for c in chunks
            if (c.metadata.get("language") or "").lower() == lang
        ]
        valid = filter_chunks(matched)
        valid.sort(key=lambda c: (str(c.metadata.get("doc_id")),
                                  str(c.metadata.get("chunk_index"))))
        rng.shuffle(valid)

        per_doc: Dict = defaultdict(int)
        by_dom: Dict[str, List[Document]] = defaultdict(list)
        for c in valid:
            d = c.metadata.get("doc_id")
            dom = _domain(c.metadata.get("source")) or "?"
            if per_doc[d] >= max_per_doc:
                continue
            if max_per_domain and len(by_dom[dom]) >= max_per_domain:
                continue
            per_doc[d] += 1
            by_dom[dom].append(c)

        doms = list(by_dom)
        rng.shuffle(doms)
        picked: List[Document] = []
        depth = max((len(v) for v in by_dom.values()), default=0)
        for i in range(depth):
            for dom in doms:
                if i < len(by_dom[dom]):
                    picked.append(by_dom[dom][i])
        per_lang[lang] = picked

    sizes = {l: len(v) for l, v in per_lang.items()}
    n = min([max_pool] + list(sizes.values()))
    limiting = min(sizes, key=sizes.get)
    print(f"\n[POOL] available per language (max_per_doc={max_per_doc}, "
          f"max_per_domain={max_per_domain}): {sizes}")
    print(f"[POOL] balanced size per language = {n} (limited by '{limiting}')")
    if n == 0:
        raise SystemExit("[POOL] A language has 0 valid chunks. Stream more data.")
    if n < max_pool:
        print(f"[WARNING] Pool is smaller than the {max_pool} target. "
              f"More '{limiting}' data (or more domains) would help.")

    pool: List[Document] = []
    for lang in languages:
        sel = per_lang[lang][:n]
        docs = {c.metadata.get("doc_id") for c in sel}
        doms = Counter(_domain(c.metadata.get("source")) for c in sel)
        top = ", ".join(f"{s}:{k}" for s, k in doms.most_common(3))
        print(f"[POOL] {lang}: {len(sel)} chunks | {len(docs)} articles | "
              f"{len(doms)} domains | top domains -> {top}")
        pool.extend(sel)
    return pool


# ---------------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------------
def _base_row(chunk: Document, lang: str) -> dict:
    return {
        "doc_id": chunk.metadata.get("doc_id"),
        "chunk_index": chunk.metadata.get("chunk_index"),
        "source": chunk.metadata.get("source"),
        "title": chunk.metadata.get("title"),
        "language": lang,
        "model_used": DEFAULT_GROQ_MODEL,
        "chunk_text_preview": chunk.page_content[:120],
        "chunk_text_full": chunk.page_content,
    }


_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0E00-\u0E7F\u1100-\u11FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)
_TERMINALS = tuple(".!?\u2026\u00bb\u201d\")\u3002\u061f")


def _summary_problem(summary: str, lang: str) -> Optional[str]:
    """Return a reason if the summary is unusable (triggers a retry)."""
    s = (summary or "").strip()
    if not s:
        return "empty summary"
    if _is_contaminated_summary(s):
        return "leaked structured-output syntax"
    if _FOREIGN_SCRIPT_RE.search(s):
        return "foreign script in summary"
    if lang.lower() in ("en", "fr") and re.search(r"[\u0600-\u06FF]", s):
        return "Arabic script in non-Arabic summary"
    if not s.endswith(_TERMINALS):
        return "truncated summary"
    return None


def analyze_chunk(chunk: Document, max_retries: int = 5) -> dict:
    lang = chunk.metadata.get("language", "")
    language_name = LANGUAGE_NAMES.get(
        lang.lower(), lang or "the same language as the text"
    )
    doc_id = chunk.metadata.get("doc_id")
    chunk_index = chunk.metadata.get("chunk_index")
    title = chunk.metadata.get("title") or "(none)"

    delay = 5
    last_error = None
    for attempt in range(max_retries):
        temperature = 0.0 if attempt == 0 else RETRY_TEMPERATURE
        try:
            result: ChunkAnalysis = _get_chain(temperature).invoke(
                {"text": chunk.page_content, "language": language_name,
                 "title": title}
            )
            problem = _summary_problem(result.summary, lang)
            if problem:
                raise ValueError(
                    f"bad summary ({problem}): {result.summary[:150]!r}"
                )
            if is_noise_summary(result.summary, result.category):
                row = _base_row(chunk, lang)
                row.update({
                    "status": "NOISE",
                    "summary": _NOT_ARTICLE_TAG_RE.sub("", result.summary).strip(),
                    "category": None, "tags": None,
                })
                return row
            row = _base_row(chunk, lang)
            row.update({
                "status": "SUCCESS",
                "summary": result.summary,
                "category": result.category,
                "tags": _cap_tags(_filter_tags(result.tags, chunk.page_content)),
            })
            return row
        except Exception as e:
            last_error = str(e)
            if _is_daily_quota_error(last_error):
                raise DailyQuotaExceeded(last_error) from e
            if _is_retryable(last_error):
                print(f"    [RETRYABLE ERROR] doc_id={doc_id} "
                      f"chunk_index={chunk_index} attempt {attempt + 1}/"
                      f"{max_retries}, waiting {delay}s... ({last_error[:120]})")
                time.sleep(delay)
                delay *= 2
                continue
            break

    print(f"[ERROR] doc_id={doc_id} chunk_index={chunk_index} "
          f"analysis failed permanently: {last_error}")
    row = _base_row(chunk, lang)
    row.update({"status": "ERROR", "error_message": last_error,
                "summary": None, "category": None, "tags": None})
    return row


# ---------------------------------------------------------------------------
# SELECTION (language-balanced catch-up)
# ---------------------------------------------------------------------------
def select_chunks_per_language(
    chunks: List[Document],
    languages: List[str],
    per_language: int,
    done_success_keys: Optional[set] = None,
    success_count_by_lang: Optional[Dict[str, int]] = None,
) -> List[Document]:
    done_success_keys = done_success_keys or set()
    success_count_by_lang = success_count_by_lang or {}

    counts = {l: int(success_count_by_lang.get(l, 0)) for l in languages}
    min_success = min(counts.values()) if counts else 0
    target = min_success + per_language

    print("[BALANCE] SUCCESS so far: "
          + ", ".join(f"{l}={counts[l]}" for l in languages)
          + f" -> target this run: {target} each "
          f"(min={min_success} + {per_language})")

    selected: List[Document] = []
    for lang in languages:
        need = max(0, target - counts[lang])
        matched = [c for c in chunks
                   if (c.metadata.get("language") or "").lower() == lang]
        pending, skipped = [], 0
        for c in matched:
            if len(pending) >= need:
                break
            if _key_of(c) in done_success_keys:
                continue
            if _noise_reason(c):
                skipped += 1
                continue
            pending.append(c)

        print(f"[SELECT] {lang}: need={need} -> {len(pending)} valid pending "
              f"(of {len(matched)} in pool; {counts[lang]} already SUCCESS; "
              f"{skipped} noise skipped)")
        if need > 0 and len(pending) < need:
            print(f"[WARNING] Only {len(pending)} valid pending '{lang}' "
                  f"chunk(s), fewer than needed ({need}).")
        selected.extend(pending)
    return selected


# ---------------------------------------------------------------------------
# RESULTS FILE HELPERS
# ---------------------------------------------------------------------------
def _load_results_by_key(output_path: str) -> Dict[Tuple, dict]:
    by_key: Dict[Tuple, dict] = {}
    if not os.path.exists(output_path):
        return by_key
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return by_key
        for r in rows:
            if r:
                by_key[_chunk_key(r.get("doc_id"), r.get("chunk_index"))] = r
        print(f"[RESUME] Loaded {len(by_key)} previously saved result(s) "
              f"from '{output_path}'.")
    except Exception as e:
        print(f"[RESUME] Could not read existing output file, starting "
              f"fresh: {e}")
    return by_key


def _save_results(output_path: str, by_key: Dict[Tuple, dict],
                  target_order: List[Tuple]) -> None:
    rows, seen = [], set()
    for key in target_order:
        if key in by_key and key not in seen:
            rows.append(by_key[key])
            seen.add(key)
    for key, row in by_key.items():
        if key not in seen:
            rows.append(row)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)   # atomic: never leaves a half-written file


def _count_success(by_key: Dict[Tuple, dict]) -> int:
    return sum(1 for r in by_key.values() if r.get("status") == "SUCCESS")


def _print_distribution(by_key: Dict[Tuple, dict]) -> None:
    dist: Dict[str, Counter] = defaultdict(Counter)
    for r in by_key.values():
        if r.get("status") == "SUCCESS":
            dist[(r.get("language") or "?").lower()][r.get("category")] += 1
    if not dist:
        return
    print("\n[DISTRIBUTION] categories per language (SUCCESS rows):")
    for lang in LANGUAGES:
        if lang in dist:
            items = ", ".join(f"{k}:{v}" for k, v in dist[lang].most_common())
            print(f"  {lang}: {items}")


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def analyze_chunks(
    chunks: List[Document],
    output_path: str,
    limit: Optional[int] = None,
    per_language_limit: Optional[int] = None,
    languages: Optional[List[str]] = None,
    save_every: int = 20,
) -> List[dict]:
    """Resume-safe driver. Resume key is (doc_id, chunk_index)."""
    by_key = _load_results_by_key(output_path)
    # NOISE rows count as "done" so they are never re-sent to the LLM.
    done_success_keys = {k for k, r in by_key.items()
                         if r.get("status") in ("SUCCESS", "NOISE")}

    sample = chunks[:50] if chunks else []
    if sample and all(c.metadata.get("chunk_index") is None for c in sample):
        print("[CRITICAL] All sampled chunks have chunk_index=None. Update "
              "langchain.py so split_documents_into_chunks assigns chunk_index.")

    if per_language_limit is not None:
        languages = languages or LANGUAGES
        counts = {l: 0 for l in languages}
        for r in by_key.values():
            if r.get("status") == "SUCCESS":
                lang = (r.get("language") or "").lower()
                if lang in counts:
                    counts[lang] += 1
        target = select_chunks_per_language(
            chunks, languages, per_language_limit, done_success_keys, counts
        )
    else:
        ordered = sorted(chunks, key=lambda c: LANGUAGE_ORDER.get(
            (c.metadata.get("language") or "").lower(), 99))
        pending = [c for c in ordered if _key_of(c) not in done_success_keys]
        target = filter_chunks(pending[:limit] if limit else pending)

    target_keys: List[Tuple] = []
    key_to_chunk: Dict[Tuple, Document] = {}
    for c in target:
        k = _key_of(c)
        if k not in key_to_chunk:
            target_keys.append(k)
            key_to_chunk[k] = c

    all_key_to_chunk = {_key_of(c): c for c in chunks}
    for k, r in by_key.items():
        if r.get("status") == "ERROR" and k not in key_to_chunk \
                and k in all_key_to_chunk:
            target_keys.append(k)
            key_to_chunk[k] = all_key_to_chunk[k]

    retry_keys = [k for k in target_keys
                  if k in by_key and by_key[k].get("status") == "ERROR"]
    new_keys = [k for k in target_keys if k not in by_key]
    to_process = list(dict.fromkeys(retry_keys + new_keys))

    if not to_process:
        print("[RESUME] Nothing left to process for this quota window.")
        _save_results(output_path, by_key, list(by_key.keys()))
        return list(by_key.values())

    if retry_keys:
        print(f"[RESUME] Re-queuing {len(retry_keys)} failed chunk(s).")
    if new_keys:
        print(f"[RESUME] {len(new_keys)} new chunk(s) not yet attempted.")

    current_lang = None
    processed = 0
    retry_set = set(retry_keys)

    try:
        for count, key in enumerate(to_process, 1):
            chunk = key_to_chunk[key]
            lang = chunk.metadata.get("language", "?")
            if lang != current_lang:
                print(f"\n--- Now processing language: {lang} ---")
                current_lang = lang
            tag = "RETRY" if key in retry_set else "NEW"
            print(f"[{count}/{len(to_process)}] ({tag}) doc_id={key[0]} "
                  f"chunk_index={key[1]} (language={lang})...")

            by_key[key] = analyze_chunk(chunk)
            processed += 1
            if by_key[key].get("status") == "NOISE":
                print("    [NOISE] model flagged this chunk as non-article; "
                      "excluded from SUCCESS rows.")

            if processed % save_every == 0:
                _save_results(output_path, by_key, list(by_key.keys()))
                print(f"    [CHECKPOINT] Total SUCCESS so far: "
                      f"{_count_success(by_key)}")
            time.sleep(1.5)

    except DailyQuotaExceeded as e:
        print(f"\n[DAILY QUOTA HIT] Stopping after {processed} chunk(s) this "
              f"run. Total SUCCESS saved: {_count_success(by_key)}. Re-run "
              f"after the quota resets.\n    {e}")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving progress before exit...")
    finally:
        _save_results(output_path, by_key, list(by_key.keys()) + target_keys)

    n_err = sum(1 for r in by_key.values() if r.get("status") == "ERROR")
    n_noise = sum(1 for r in by_key.values() if r.get("status") == "NOISE")
    if n_noise:
        print(f"[NOISE] {n_noise} chunk(s) flagged by the model as non-article "
              f"(status='NOISE' in the file).")
    print(f"\n--- Done this run: processed {processed} | total SUCCESS in "
          f"file: {_count_success(by_key)} | still ERROR: {n_err} ---")
    _print_distribution(by_key)
    return list(by_key.values())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(" Connecting to PostgreSQL and loading raw documents...")
    raw_documents = load_documents_from_db()
    print(f" Loaded {len(raw_documents)} raw documents from DB.")

    print(" Cleaning documents (cutting link-list tails)...")
    raw_documents = clean_documents(raw_documents)

    print(" Splitting documents into chunks...")
    chunks = split_documents_into_chunks(
        raw_documents, chunk_size=800, chunk_overlap=100
    )
    print(f" Generated {len(chunks)} total chunks.")

    print("\n Building balanced pool...")
    pool = build_pool(chunks)

    output_path = os.path.join(
        BASE_DIR, "chunk_analysis_results_v1_4_multilingual_chunks.json"
    )

    print("\n Running per-chunk summarization + classification...")
    results = analyze_chunks(
        pool,
        output_path=output_path,
        per_language_limit=LLM_PER_LANGUAGE,
        languages=LANGUAGES,
        save_every=20,
    )

    print(f"\n Saved {len(results)} chunk result(s) to: {output_path}")
    last_success = next(
        (r for r in reversed(results) if r and r.get("status") == "SUCCESS"),
        None,
    )
    if last_success:
        print("\n Sample Output Result:")
        print(json.dumps(last_success, ensure_ascii=False, indent=2))