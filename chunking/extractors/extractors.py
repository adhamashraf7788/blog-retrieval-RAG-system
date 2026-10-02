# extractors.py -- active version (post-fix, noise-reduction revision)
# Changes in this revision (less noise in the DB):
#   A. No blind fallback to raw <body> text: trafilatura output (or the
#      cleaned fallback) must pass looks_like_article() (>=3 sentences).
#      Menus, link lists, tickers and UI chrome are dropped at the source.
#   B. The hard-coded 80-word floor is now MIN_WORDS_EXTRACT (50). Before it,
#      --min-words-ar 50 in the streamer could never take effect (everything
#      under 80 words was returned as "too_short").
#   C. Paragraph breaks are kept in clean_text (better chunking downstream).
#   D. Extra boilerplate cuts: headtopics / albiladdaily footers, the site
#      menu header before "Advertisement", trailing "Copyright 20xx ..." lines.
#   E. Language detection samples the middle of the article (not the menu
#      header) and returns "unknown" when confidence < LANG_MIN_PROB.

import os

# FIX: the old code did `os.environ["HF_HOME"] = "D:/huggingface_cache"`.
# That is a Windows-only, and NOT a true absolute path ("D:/..." only means
# a drive root on Windows -- on Linux/mac it's just a relative folder named
# "D:"). Every time this script ran from a different working directory (or
# on a different machine/OS), transformers couldn't find the previous
# cache, so it silently re-downloaded both models from scratch.
#
# Fix: build a stable, absolute, cross-platform cache path once, next to
# this project, and only set HF_HOME if the user/environment hasn't
# already configured one themselves.
_DEFAULT_HF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.makedirs(_DEFAULT_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)

import re
import json
import hashlib
import trafilatura
from collections import defaultdict
from typing import Dict, Any, List
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser
from utils import (
    decode_and_validate,
    is_valid_arabic_text
)

ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')

# ============================================================
# LANGUAGE DETECTION -- backed by `langdetect` instead of a transformer.
# ============================================================

from langdetect import detect_langs, DetectorFactory, LangDetectException
DetectorFactory.seed = 0  # deterministic results across runs

hf_lang_detector = None
arabic_dialect_detector = None

HF_BATCH_SIZE = 32

LANG_MIN_PROB = 0.90


def _lang_sample(text: str) -> str:
    """Sample from the body of the article, not the site-menu header."""
    t = text[:2000]
    if len(t) > 800:
        start = len(t) // 3
        return t[start:start + 500]
    return t[:500]


def get_lang_detector():
    """Kept for API compatibility. langdetect needs no model loading."""
    global hf_lang_detector
    if hf_lang_detector is None:
        hf_lang_detector = True
    return hf_lang_detector


def get_dialect_detector():
    """Lazy init for Arabic Dialect Detector (CAMeL BERT)."""
    global arabic_dialect_detector
    if arabic_dialect_detector is None:
        from transformers import pipeline
        arabic_dialect_detector = pipeline(
            "text-classification",
            model="CAMeL-Lab/bert-base-arabic-camelbert-mix-did",
            top_k=1,
            truncation=True,
            max_length=128,
            batch_size=HF_BATCH_SIZE
        )
    return arabic_dialect_detector

UNWANTED_TAGS = ["script", "style", "noscript", "header", "footer", "svg", "nav"]


def detect_language(text: str) -> str:
    """Language detection using langdetect.

    (Previously named detect_language_hf; renamed because the backend is
    now langdetect, not a Hugging Face model.)
    """
    if not text or len(text) < 20:
        return "unknown"

    has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
    has_latin = bool(LATIN_PATTERN.search(text[:300]))

    if not (has_arabic or has_latin):
        return "ignored"

    try:
        langs = detect_langs(_lang_sample(text))
        if not langs or langs[0].prob < LANG_MIN_PROB:
            return "unknown"
        lang_code = langs[0].lang

        if lang_code == "ar":
            if is_valid_arabic_text(text):
                return "ar"
            else:
                return "corrupted"
        elif lang_code == "en":
            return "en"
        elif lang_code == "fr":
            return "fr"
        return lang_code
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


# Backward-compatible alias
detect_language_hf = detect_language


def detect_arabic_dialect(text: str) -> str:
    if not text:
        return "MSA"
    try:
        detector = get_dialect_detector()
        sample_text = text[:250]
        prediction = detector(sample_text)[0][0]['label']
        return prediction
    except Exception:
        return "MSA"


def detect_languages_batch(texts: List[str]) -> List[str]:
    if not texts:
        return []

    results = ["unknown"] * len(texts)
    run_idx = []
    samples = []
    for i, text in enumerate(texts):
        if not text or len(text) < 20:
            continue
        has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
        has_latin = bool(LATIN_PATTERN.search(text[:300]))
        if not (has_arabic or has_latin):
            results[i] = "ignored"
            continue
        run_idx.append(i)
        samples.append(_lang_sample(text))

    for i, sample in zip(run_idx, samples):
        try:
            langs = detect_langs(sample)
            if not langs or langs[0].prob < LANG_MIN_PROB:
                results[i] = "unknown"
                continue
            lang_code = langs[0].lang
        except LangDetectException:
            results[i] = "unknown"
            continue
        except Exception:
            results[i] = "unknown"
            continue

        if lang_code == "ar":
            results[i] = "ar" if is_valid_arabic_text(texts[i]) else "corrupted"
        elif lang_code == "en":
            results[i] = "en"
        elif lang_code == "fr":
            results[i] = "fr"
        else:
            results[i] = lang_code

    return results


def detect_dialects_batch(texts: List[str]) -> List[str]:
    if not texts:
        return []
    try:
        detector = get_dialect_detector()
        samples = [t[:250] if t else "" for t in texts]
        predictions = detector(samples)
        return [p[0]['label'] if p else "MSA" for p in predictions]
    except Exception:
        return ["MSA"] * len(texts)


# ============================================================
# DOMAIN-LEVEL BOILERPLATE STRIPPING
# ============================================================

BOILERPLATE_CUT_PATTERNS = {
    "almanar.com.lb": [
        # Arabic
        r"آخر الأخبار",
        r"أحدث الأخبار",
        r"مواضيع ساخنة",
        r"المزيد أحدث الأخبار",
        # English
        r"Latest News",
        r"Related News",
        r"Read More",
        r"Hot Topics",
        # French 
        r"Dernières nouvelles",
        r"Actualités récentes",
        r"Derniers articles",
        r"Articles récents",
        r"Les Plus Récents",
        r"Sujets Tendances",
        r"Sujets brûlants",
        r"Spécial notre site",
        r"Article traduit",
        r"A la une",
        r"À la une",
    ],
}

MASTHEAD_PATTERNS = {
    "almanar.com.lb": re.compile(
        r"(?is)^.*?(?:Beyrouth|بيروت)\s*\d{1,2}:\d{2}\s*"
        r"(?:ع\s*)?(?:Ar|En|Fr|Es)(?:\s*(?:Ar|En|Fr|Es))*\s*"
        r"(?:عاجل|Breaking|Urgent|Dernière heure)?\s*",
    ),
}

# More domain-specific link-list / footer cuts (same ones the summarizer used).
BOILERPLATE_CUT_PATTERNS.update({
    "headtopics.com":   [r"Read more \u00bb", r"Similar News:", r"United States Latest News"],
    "albiladdaily.com": [r"\u0635\u0648\u062a \u0627\u0644\u062d\u062c\u0627\u0632 \u0623\u0648\u0644 \u062c\u0631\u064a\u062f\u0629 \u0633\u0639\u0648\u062f\u064a\u0629"],
})

# Trailing "Copyright 2026 NPR"-style lines.
_COPYRIGHT_TAIL_RE = re.compile(
    r"\s*(?:\u00a9|Copyright)\s*(?:19|20)\d{2}\b[^\n]{0,60}$", re.IGNORECASE
)

# A site-menu header that ends with "Advertisement" right before the article.
_HEADER_MARKERS = ("Privacy Policy", "\u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644", "Se connecter")


def _strip_site_header(text: str) -> str:
    i = text.find("Advertisement", 0, 2500)
    if i != -1 and any(m in text[:i] for m in _HEADER_MARKERS):
        return text[i + len("Advertisement"):].lstrip()
    return text


_domain_tail_hashes = defaultdict(lambda: defaultdict(int))
BOILERPLATE_REPEAT_THRESHOLD = 3
TAIL_CHARS = 150


_MULTI_PART_TLDS = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.sa", "com.eg", "com.lb", "com.ae", "com.jo", "com.kw",
    "co.il", "org.il",
    "co.jp", "com.au", "co.za",
})


def get_domain(url: str) -> str:
    """Returns the bare registrable domain (no 'www.', no subdomain).

    Handles common multi-part TLDs (e.g. .co.uk, .com.sa).
    """
    if not url:
        return ""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = netloc.split(".")
    if len(parts) >= 3:
        last_two = ".".join(parts[-2:])
        if last_two in _MULTI_PART_TLDS:
            return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return netloc


def strip_domain_boilerplate(text: str, url: str) -> str:
    if not text or not url:
        return text

    domain = get_domain(url)
    text = _strip_site_header(text)

    masthead_re = MASTHEAD_PATTERNS.get(domain)
    if masthead_re:
        stripped = masthead_re.sub("", text, count=1)
        if stripped and len(stripped) < len(text):
            text = stripped

    cut_patterns = BOILERPLATE_CUT_PATTERNS.get(domain)
    if cut_patterns:
        earliest_cut = None
        for pattern in cut_patterns:
            m = re.search(pattern, text)
            if m and (earliest_cut is None or m.start() < earliest_cut):
                earliest_cut = m.start()
        if earliest_cut is not None:
            text = text[:earliest_cut].strip()

    return _COPYRIGHT_TAIL_RE.sub("", text).strip()


def detect_and_strip_repeated_tail(text: str, domain: str) -> str:
    if not text or len(text) < TAIL_CHARS or not domain:
        return text

    tail = text[-TAIL_CHARS:]
    tail_hash = hashlib.md5(tail.encode("utf-8", errors="ignore")).hexdigest()

    _domain_tail_hashes[domain][tail_hash] += 1
    count = _domain_tail_hashes[domain][tail_hash]

    if count >= BOILERPLATE_REPEAT_THRESHOLD:
        anchor = tail[:40]
        cutoff = text.rfind(anchor)
        if cutoff > 0:
            return text[:cutoff].strip()

    return text


def extract_publication_date(tree: HTMLParser) -> str:
    meta_selectors = [
        ("meta[property='article:published_time']", "content"),
        ("meta[name='pubdate']", "content"),
        ("meta[name='publishdate']", "content"),
        ("meta[name='date']", "content"),
        ("meta[name='DC.date.issued']", "content"),
        ("meta[name='parsely-pub-date']", "content")
    ]

    for selector, attr in meta_selectors:
        node = tree.css_first(selector)
        if node and node.attributes.get(attr):
            date_val = node.attributes.get(attr).strip()
            if date_val:
                return date_val

    for script_node in tree.css("script[type='application/ld+json']"):
        raw = script_node.text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict):
                date_val = item.get("datePublished") or item.get("dateCreated")
                if date_val:
                    return str(date_val).strip()

    time_node = tree.css_first("time")
    if time_node:
        datetime_val = time_node.attributes.get("datetime")
        if datetime_val:
            return datetime_val.strip()
        text_val = time_node.text(strip=True)
        if text_val:
            return text_val

    return "N/A"


def extract_date_from_text_fallback(tree: HTMLParser) -> str:
    body_text = tree.body.text() if tree.body else tree.text()
    date_match = re.search(
        r'\b(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b',
        body_text
    )
    if date_match:
        return date_match.group(0)
    return "N/A"


def extract_author(tree: HTMLParser) -> str:
    author_selectors = [
        ("meta[name='author']", "content"),
        ("meta[property='article:author']", "content"),
        ("meta[name='parsely-author']", "content"),
        ("meta[name='dc.creator']", "content"),
        ("meta[name='author_name']", "content")
    ]

    for selector, attr in author_selectors:
        node = tree.css_first(selector)
        if node and node.attributes.get(attr):
            val = node.attributes.get(attr).strip()
            if val:
                return val

    for script_node in tree.css("script[type='application/ld+json']"):
        raw = script_node.text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if isinstance(item, dict) and "author" in item:
                    author_data = item["author"]
                    if isinstance(author_data, dict) and "name" in author_data:
                        return str(author_data["name"]).strip()
                    elif isinstance(author_data, list) and len(author_data) > 0:
                        first_author = author_data[0]
                        if isinstance(first_author, dict):
                            return str(first_author.get("name", "N/A")).strip()
                        return str(first_author).strip()
                    elif isinstance(author_data, str):
                        return author_data.strip()
        except (json.JSONDecodeError, ValueError):
            continue

    return "N/A"


def extract_main_content(html_str: str) -> str:
    extracted = trafilatura.extract(
        html_str,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
    )
    return extracted


# ============================================================
# ARTICLE-QUALITY GATE
# ============================================================
# Language-specific limits (e.g. 50 Arabic / 80 other) are applied later in
# the streamer; this is only the global floor.
MIN_WORDS_EXTRACT = 50

_SENT_END_RE = re.compile(r"[.!?\u061f\u2026](?:\s|$)")

_FALLBACK_DROP = UNWANTED_TAGS + [
    "aside", "form", "button", "iframe",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    "[class*='comment']", "[class*='sidebar']", "[class*='related']",
    "[class*='newsletter']", "[class*='cookie']",
]


def looks_like_article(text: str) -> bool:
    """Menus, link lists and tickers have almost no sentences."""
    words = len(text.split())
    if words < MIN_WORDS_EXTRACT:
        return False
    sentences = len(_SENT_END_RE.findall(text))
    return sentences >= 3 and words / sentences <= 80


def normalize_paragraphs(text: str) -> str:
    """Collapse whitespace inside a paragraph but KEEP paragraph breaks."""
    paras = (" ".join(p.split()) for p in text.split("\n"))
    return "\n\n".join(p for p in paras if p)


def extract_html_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: pure HTML parsing/cleaning (no language detection).
    trafilatura first, then domain boilerplate stripping.
    """
    html_str, status = decode_and_validate(record["raw_bytes"])

    if status != "success" or not html_str:
        return {"language": "corrupted", "word_count": 0, "clean_text": ""}

    tree = HTMLParser(html_str)

    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else "N/A"

    main_text = extract_main_content(html_str)
    clean_text = normalize_paragraphs(main_text) if main_text else ""

    if not looks_like_article(clean_text):
        # Fallback: page text after a stronger clean-up. It is accepted only
        # if it still looks like an article (no more raw menu/footer dumps).
        for tag in tree.css(", ".join(_FALLBACK_DROP)):
            tag.decompose()
        body = tree.body
        raw_text = (body.text(separator="\n", strip=True) if body
                    else tree.text(separator="\n", strip=True))
        clean_text = normalize_paragraphs(raw_text)
        if not looks_like_article(clean_text):
            return {"language": "too_short",
                    "word_count": len(clean_text.split()), "clean_text": ""}

    url = record.get("url", "")
    domain = get_domain(url)
    clean_text = strip_domain_boilerplate(clean_text, url)
    clean_text = detect_and_strip_repeated_tail(clean_text, domain)

    word_cnt = len(clean_text.split())
    if word_cnt < MIN_WORDS_EXTRACT:
        return {"language": "too_short", "word_count": word_cnt, "clean_text": ""}

    pub_date = extract_publication_date(tree)
    if pub_date == "N/A":
        pub_date = extract_date_from_text_fallback(tree)

    author = extract_author(tree)

    return {
        "title": title,
        "author": author,
        "url": record["url"],
        "warc_date": record["warc_date"],
        "published_date": pub_date,
        "clean_text": clean_text,
        "word_count": word_cnt,
        "char_count": len(clean_text),
        "html_size_bytes": len(record["raw_bytes"]),
        "links_count": len(tree.css("a")),
        "headings_sample": [node.text(strip=True) for node in tree.css("h1, h2, h3")][:3]
    }


def extract_selectolax(record: Dict[str, Any]) -> Dict[str, Any]:
    """Single-record extractor with language detection.

    Reuses extract_html_fields() so trafilatura + domain boilerplate
    stripping are applied the same way as in the batch pipeline.
    """
    fields = extract_html_fields(record)

    if fields.get("language") in ("corrupted", "too_short") or not fields.get("clean_text"):
        return fields

    clean_text = fields["clean_text"]
    language = detect_language(clean_text)

    arabic_dialect = "N/A"
    if language == "ar":
        if not is_valid_arabic_text(clean_text):
            return {
                "language": "corrupted",
                "word_count": fields.get("word_count", 0),
                "clean_text": "",
            }
        arabic_dialect = detect_arabic_dialect(clean_text)

    fields["language"] = language
    fields["arabic_dialect"] = arabic_dialect
    return fields


def extract_bs4_lxml(record: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy/Benchmark Extractor fallback."""
    html_str, status = decode_and_validate(record["raw_bytes"])
    if not html_str:
        return {"clean_text": "", "word_count": 0, "text_length": 0}

    soup = BeautifulSoup(html_str, 'lxml')

    title = soup.title.get_text(strip=True) if soup.title else "N/A"

    for tag in soup(UNWANTED_TAGS):
        tag.decompose()

    body = soup.body
    text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
    clean_text = " ".join(text.split())
    headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
    links_count = len(soup.find_all('a'))

    return {
        "title": title,
        "url": record["url"],
        "warc_date": record["warc_date"],
        "clean_text": clean_text,
        "text_length": len(clean_text),
        "word_count": len(clean_text.split()),
        "html_size_bytes": len(record["raw_bytes"]),
        "links_count": links_count,
        "headings_sample": headings[:3]
    }