# import os

# # FIX: the old code did `os.environ["HF_HOME"] = "D:/huggingface_cache"`.
# # That is a Windows-only, and NOT a true absolute path ("D:/..." only means
# # a drive root on Windows -- on Linux/mac it's just a relative folder named
# # "D:"). Every time this script ran from a different working directory (or
# # on a different machine/OS), transformers couldn't find the previous
# # cache, so it silently re-downloaded both models from scratch. That is
# # almost certainly the real reason the pipeline "takes forever to load".
# #
# # Fix: build a stable, absolute, cross-platform cache path once, next to
# # this project, and only set HF_HOME if the user/environment hasn't
# # already configured one themselves.
# _DEFAULT_HF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
# os.makedirs(_DEFAULT_HF_CACHE, exist_ok=True)
# os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)

# import re
# import json
# from typing import Dict, Any, List
# from bs4 import BeautifulSoup
# from selectolax.parser import HTMLParser
# from utils import (
#     decode_and_validate,
#     is_valid_arabic_text
# )

# ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
# LATIN_PATTERN = re.compile(r'[a-zA-Z]')

# hf_lang_detector = None
# arabic_dialect_detector = None

# HF_BATCH_SIZE = 32  # how many texts the HF pipeline groups into one forward pass

# def get_lang_detector():
#     """Lazy initialization for Language Detector.
#     NOTE: batch_size lets the pipeline internally chunk a list of texts
#     into forward passes instead of doing one text at a time - this is
#     what makes detect_languages_batch() below actually fast.
#     """
#     global hf_lang_detector
#     if hf_lang_detector is None:
#         from transformers import pipeline
#         hf_lang_detector = pipeline(
#             "text-classification",
#             model="papluca/xlm-roberta-base-language-detection",
#             top_k=1,
#             truncation=True,
#             max_length=128,
#             batch_size=HF_BATCH_SIZE
#         )
#     return hf_lang_detector

# def get_dialect_detector():
#     """Lazy initialization for Arabic Dialect Detector. See note above."""
#     global arabic_dialect_detector
#     if arabic_dialect_detector is None:
#         from transformers import pipeline
#         arabic_dialect_detector = pipeline(
#             "text-classification",
#             model="CAMeL-Lab/bert-base-arabic-camelbert-mix-did",
#             top_k=1,
#             truncation=True,
#             max_length=128,
#             batch_size=HF_BATCH_SIZE
#         )
#     return arabic_dialect_detector

# UNWANTED_TAGS = ["script", "style", "noscript", "header", "footer", "svg", "nav"]

# def detect_language_hf(text: str) -> str:
#     """Language detection using Hugging Face with pre-filtering."""
#     if not text or len(text) < 20:
#         return "unknown"

#     has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
#     has_latin = bool(LATIN_PATTERN.search(text[:300]))

#     if not (has_arabic or has_latin):
#         return "ignored"

#     try:
#         detector = get_lang_detector()
#         sample_text = text[:250]
#         prediction = detector(sample_text)[0][0]['label'].lower()

#         if prediction.startswith(("ar", "arabic")):
#             if is_valid_arabic_text(text):
#                 return "ar"
#             else:
#                 return "corrupted"
#         elif prediction.startswith(("en", "english")):
#             return "en"
#         elif prediction.startswith(("fr", "french")):
#             return "fr"
#         return prediction
#     except Exception:
#         return "unknown"

# def detect_arabic_dialect(text: str) -> str:
#     """Detects Arabic dialect using Hugging Face."""
#     if not text:
#         return "MSA"
#     try:
#         detector = get_dialect_detector()
#         sample_text = text[:250]
#         prediction = detector(sample_text)[0][0]['label']
#         return prediction
#     except Exception:
#         return "MSA"

# def detect_languages_batch(texts: List[str]) -> List[str]:
#     """
#     Batched version of detect_language_hf(): runs ONE HF forward pass
#     (internally chunked by HF_BATCH_SIZE) for a whole list of texts
#     instead of one pipeline call per document. This is the main speedup
#     for the streaming pipeline.
#     """
#     if not texts:
#         return []

#     results = ["unknown"] * len(texts)

#     # Same cheap pre-filter as the single-text version: skip the model
#     # entirely for text that's too short or has no Arabic/Latin script.
#     run_idx = []
#     samples = []
#     for i, text in enumerate(texts):
#         if not text or len(text) < 20:
#             continue
#         has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
#         has_latin = bool(LATIN_PATTERN.search(text[:300]))
#         if not (has_arabic or has_latin):
#             results[i] = "ignored"
#             continue
#         run_idx.append(i)
#         samples.append(text[:250])

#     if not samples:
#         return results

#     try:
#         detector = get_lang_detector()
#         predictions = detector(samples)
#     except Exception:
#         for i in run_idx:
#             results[i] = "unknown"
#         return results

#     for pos, i in enumerate(run_idx):
#         try:
#             label = predictions[pos][0]['label'].lower()
#         except Exception:
#             results[i] = "unknown"
#             continue
#         if label.startswith(("ar", "arabic")):
#             results[i] = "ar" if is_valid_arabic_text(texts[i]) else "corrupted"
#         elif label.startswith(("en", "english")):
#             results[i] = "en"
#         elif label.startswith(("fr", "french")):
#             results[i] = "fr"
#         else:
#             results[i] = label

#     return results

# def detect_dialects_batch(texts: List[str]) -> List[str]:
#     """Batched version of detect_arabic_dialect() for a list of Arabic texts."""
#     if not texts:
#         return []
#     try:
#         detector = get_dialect_detector()
#         samples = [t[:250] if t else "" for t in texts]
#         predictions = detector(samples)
#         return [p[0]['label'] if p else "MSA" for p in predictions]
#     except Exception:
#         return ["MSA"] * len(texts)

# def extract_publication_date(tree: HTMLParser) -> str:
#     """Extracts publication date via Meta tags, JSON-LD, or <time> tag."""
#     meta_selectors = [
#         ("meta[property='article:published_time']", "content"),
#         ("meta[name='pubdate']", "content"),
#         ("meta[name='publishdate']", "content"),
#         ("meta[name='date']", "content"),
#         ("meta[name='DC.date.issued']", "content"),
#         ("meta[name='parsely-pub-date']", "content")
#     ]

#     for selector, attr in meta_selectors:
#         node = tree.css_first(selector)
#         if node and node.attributes.get(attr):
#             date_val = node.attributes.get(attr).strip()
#             if date_val:
#                 return date_val

#     for script_node in tree.css("script[type='application/ld+json']"):
#         raw = script_node.text(strip=True)
#         if not raw:
#             continue
#         try:
#             data = json.loads(raw)
#         except (json.JSONDecodeError, ValueError):
#             continue
#         candidates = data if isinstance(data, list) else [data]
#         for item in candidates:
#             if isinstance(item, dict):
#                 date_val = item.get("datePublished") or item.get("dateCreated")
#                 if date_val:
#                     return str(date_val).strip()

#     time_node = tree.css_first("time")
#     if time_node:
#         datetime_val = time_node.attributes.get("datetime")
#         if datetime_val:
#             return datetime_val.strip()
#         text_val = time_node.text(strip=True)
#         if text_val:
#             return text_val

#     return "N/A"

# def extract_date_from_text_fallback(tree: HTMLParser) -> str:
#     """Fallback: Regex search over clean body text for date patterns."""
#     body_text = tree.body.text() if tree.body else tree.text()
#     date_match = re.search(
#         r'\b(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b',
#         body_text
#     )
#     if date_match:
#         return date_match.group(0)
#     return "N/A"

# def extract_author(tree: HTMLParser) -> str:
#     """Extracts author information via Meta tags or JSON-LD."""
#     author_selectors = [
#         ("meta[name='author']", "content"),
#         ("meta[property='article:author']", "content"),
#         ("meta[name='parsely-author']", "content"),
#         ("meta[name='dc.creator']", "content"),
#         ("meta[name='author_name']", "content")
#     ]

#     for selector, attr in author_selectors:
#         node = tree.css_first(selector)
#         if node and node.attributes.get(attr):
#             val = node.attributes.get(attr).strip()
#             if val:
#                 return val

#     for script_node in tree.css("script[type='application/ld+json']"):
#         raw = script_node.text(strip=True)
#         if not raw:
#             continue
#         try:
#             data = json.loads(raw)
#             candidates = data if isinstance(data, list) else [data]
#             for item in candidates:
#                 if isinstance(item, dict) and "author" in item:
#                     author_data = item["author"]
#                     if isinstance(author_data, dict) and "name" in author_data:
#                         return str(author_data["name"]).strip()
#                     elif isinstance(author_data, list) and len(author_data) > 0:
#                         first_author = author_data[0]
#                         if isinstance(first_author, dict):
#                             return str(first_author.get("name", "N/A")).strip()
#                         return str(first_author).strip()
#                     elif isinstance(author_data, str):
#                         return author_data.strip()
#         except (json.JSONDecodeError, ValueError):
#             continue

#     return "N/A"

# def extract_html_fields(record: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Phase 1 of extraction: pure HTML parsing/cleaning (title, author, dates,
#     links, clean_text, word_count). Deliberately does NOT call any Hugging
#     Face model, so it's cheap and safe to run per-record in a tight loop.
#     Language/dialect detection is done separately, in batches, by
#     detect_languages_batch() / detect_dialects_batch() below - see
#     extract_selectolax() for the old single-record, all-in-one version
#     (still used by scripts that process one record at a time, e.g. the
#     CC-NEWS preview script).
#     """
#     html_str, status = decode_and_validate(record["raw_bytes"])

#     if status != "success" or not html_str:
#         return {"language": "corrupted", "word_count": 0, "clean_text": ""}

#     tree = HTMLParser(html_str)

#     title_node = tree.css_first("title")
#     title = title_node.text(strip=True) if title_node else "N/A"

#     for tag in tree.css(", ".join(UNWANTED_TAGS)):
#         tag.decompose()

#     body = tree.body
#     raw_text = body.text(separator=' ', strip=True) if body else tree.text(separator=' ', strip=True)
#     clean_text = " ".join(raw_text.split())

#     word_cnt = len(clean_text.split())
#     if word_cnt < 80:
#         return {"language": "too_short", "word_count": word_cnt, "clean_text": ""}

#     pub_date = extract_publication_date(tree)
#     if pub_date == "N/A":
#         pub_date = extract_date_from_text_fallback(tree)

#     author = extract_author(tree)

#     return {
#         "title": title,
#         "author": author,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "published_date": pub_date,
#         "clean_text": clean_text,
#         "word_count": word_cnt,
#         "char_count": len(clean_text),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": len(tree.css("a")),
#         "headings_sample": [node.text(strip=True) for node in tree.css("h1, h2, h3")][:3]
#     }

# def extract_selectolax(record: Dict[str, Any]) -> Dict[str, Any]:
#     """Extractor with quality filtering and Hugging Face language detection.
#     Kept for callers that process one record at a time (no batching)."""
#     html_str, status = decode_and_validate(record["raw_bytes"])

#     if status != "success" or not html_str:
#         return {"language": "corrupted", "word_count": 0, "clean_text": ""}

#     tree = HTMLParser(html_str)

#     title_node = tree.css_first("title")
#     title = title_node.text(strip=True) if title_node else "N/A"

#     for tag in tree.css(", ".join(UNWANTED_TAGS)):
#         tag.decompose()

#     body = tree.body
#     raw_text = body.text(separator=' ', strip=True) if body else tree.text(separator=' ', strip=True)
#     clean_text = " ".join(raw_text.split())

#     word_cnt = len(clean_text.split())
#     if word_cnt < 80:
#         return {"language": "too_short", "word_count": word_cnt, "clean_text": ""}

#     language = detect_language_hf(clean_text)

#     arabic_dialect = "N/A"
#     if language == "ar":
#         if not is_valid_arabic_text(clean_text):
#             return {"language": "corrupted", "word_count": word_cnt, "clean_text": ""}
#         arabic_dialect = detect_arabic_dialect(clean_text)

#     pub_date = extract_publication_date(tree)
#     if pub_date == "N/A":
#         pub_date = extract_date_from_text_fallback(tree)

#     author = extract_author(tree)

#     return {
#         "title": title,
#         "author": author,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "published_date": pub_date,
#         "clean_text": clean_text,
#         "language": language,
#         "arabic_dialect": arabic_dialect,
#         "word_count": word_cnt,
#         "char_count": len(clean_text),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": len(tree.css("a")),
#         "headings_sample": [node.text(strip=True) for node in tree.css("h1, h2, h3")][:3]
#     }

# def extract_bs4_lxml(record: Dict[str, Any]) -> Dict[str, Any]:
#     """Legacy/Benchmark Extractor fallback."""
#     html_str, status = decode_and_validate(record["raw_bytes"])
#     if not html_str:
#         return {"clean_text": "", "word_count": 0, "text_length": 0}

#     soup = BeautifulSoup(html_str, 'lxml')

#     title = soup.title.get_text(strip=True) if soup.title else "N/A"

#     for tag in soup(UNWANTED_TAGS):
#         tag.decompose()

#     body = soup.body
#     text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
#     clean_text = " ".join(text.split())
#     headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
#     links_count = len(soup.find_all('a'))

#     return {
#         "title": title,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "clean_text": clean_text,
#         "text_length": len(clean_text),
#         "word_count": len(clean_text.split()),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": links_count,
#         "headings_sample": headings[:3]
#     }



###
# import os

# # FIX: the old code did `os.environ["HF_HOME"] = "D:/huggingface_cache"`.
# # That is a Windows-only, and NOT a true absolute path ("D:/..." only means
# # a drive root on Windows -- on Linux/mac it's just a relative folder named
# # "D:"). Every time this script ran from a different working directory (or
# # on a different machine/OS), transformers couldn't find the previous
# # cache, so it silently re-downloaded both models from scratch.
# #
# # Fix: build a stable, absolute, cross-platform cache path once, next to
# # this project, and only set HF_HOME if the user/environment hasn't
# # already configured one themselves.
# _DEFAULT_HF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
# os.makedirs(_DEFAULT_HF_CACHE, exist_ok=True)
# os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)

# import re
# import json
# import hashlib
# import trafilatura
# from collections import defaultdict
# from typing import Dict, Any, List
# from urllib.parse import urlparse
# from bs4 import BeautifulSoup
# from selectolax.parser import HTMLParser
# from utils import (
#     decode_and_validate,
#     is_valid_arabic_text
# )

# ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
# LATIN_PATTERN = re.compile(r'[a-zA-Z]')

# # ============================================================
# # LANGUAGE DETECTION -- backed by `langdetect` instead of a transformer.
# #
# # Why not fasttext: fasttext's official PyPI package (and the fasttext-
# # wheel fallback) both need to compile a C++ extension at install time,
# # which requires Microsoft C++ Build Tools on Windows -- a multi-GB
# # install most machines don't have. `langdetect` is pure Python (a port
# # of Google's language-detection library), so `pip install` just works,
# # no compiler needed anywhere.
# #
# # Why this still fixes the slowdown: the original bottleneck was running
# # `papluca/xlm-roberta-base-language-detection` (a ~270M-parameter
# # transformer) on almost every record. langdetect uses simple n-gram
# # statistics instead of a neural network, so even without fasttext's
# # specific speed, it is still dramatically faster than a transformer
# # forward pass -- there's no model weights to load, no PyTorch involved
# # for this step at all.
# #
# # Trade-off: langdetect has no native "classify this whole list at once"
# # batch API like fasttext does, so detect_languages_batch() below just
# # loops over the list calling detect_langs() once per text. This is still
# # fast because each call is lightweight pure-Python work, not because of
# # batching -- HF_BATCH_SIZE below is now only relevant to the (much
# # smaller) Arabic dialect model's batches, not to language ID anymore.
# #
# # If you later manage to get a real fasttext build working (e.g. by
# # installing MSVC Build Tools), swapping back in is a drop-in change:
# # only get_lang_detector()/detect_language_hf()/detect_languages_batch()
# # below need to change; nothing else in this file or in callers does.
# # ============================================================

# from langdetect import detect_langs, DetectorFactory, LangDetectException
# DetectorFactory.seed = 0  # deterministic results across runs

# hf_lang_detector = None          # kept as a flag/placeholder for API compatibility
# arabic_dialect_detector = None   # unchanged: still the HF CAMeL BERT model

# HF_BATCH_SIZE = 32  # still used for the (much smaller) Arabic dialect batches


# def get_lang_detector():
#     """Kept for API compatibility with any caller that still references
#     this function name. langdetect needs no model loading, so this is now
#     just a no-op that marks the detector as 'ready'."""
#     global hf_lang_detector
#     if hf_lang_detector is None:
#         hf_lang_detector = True
#     return hf_lang_detector


# def get_dialect_detector():
#     """Lazy initialization for Arabic Dialect Detector. Unchanged: this is
#     still a Hugging Face transformer, but it only ever runs on the much
#     smaller subset of texts already confirmed to be Arabic, so it isn't
#     the bottleneck the language detector was."""
#     global arabic_dialect_detector
#     if arabic_dialect_detector is None:
#         from transformers import pipeline
#         arabic_dialect_detector = pipeline(
#             "text-classification",
#             model="CAMeL-Lab/bert-base-arabic-camelbert-mix-did",
#             top_k=1,
#             truncation=True,
#             max_length=128,
#             batch_size=HF_BATCH_SIZE
#         )
#     return arabic_dialect_detector

# UNWANTED_TAGS = ["script", "style", "noscript", "header", "footer", "svg", "nav"]


# def detect_language_hf(text: str) -> str:
#     """Language detection using langdetect, with the same pre-filtering and
#     return contract as the original transformer-based version."""
#     if not text or len(text) < 20:
#         return "unknown"

#     has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
#     has_latin = bool(LATIN_PATTERN.search(text[:300]))

#     if not (has_arabic or has_latin):
#         return "ignored"

#     try:
#         sample_text = text[:250]
#         langs = detect_langs(sample_text)
#         lang_code = langs[0].lang if langs else "unknown"

#         if lang_code == "ar":
#             if is_valid_arabic_text(text):
#                 return "ar"
#             else:
#                 return "corrupted"
#         elif lang_code == "en":
#             return "en"
#         elif lang_code == "fr":
#             return "fr"
#         return lang_code
#     except LangDetectException:
#         return "unknown"
#     except Exception:
#         return "unknown"


# def detect_arabic_dialect(text: str) -> str:
#     """Detects Arabic dialect using Hugging Face. Unchanged from the
#     original -- still transformer-based, still only called on confirmed
#     Arabic text."""
#     if not text:
#         return "MSA"
#     try:
#         detector = get_dialect_detector()
#         sample_text = text[:250]
#         prediction = detector(sample_text)[0][0]['label']
#         return prediction
#     except Exception:
#         return "MSA"


# def detect_languages_batch(texts: List[str]) -> List[str]:
#     """
#     Language detection for a list of texts, now backed by langdetect
#     instead of the HF transformer pipeline. Same input/output contract as
#     before: a list of texts in, a list of language codes out. langdetect
#     has no native "classify a whole list at once" API, so this loops over
#     the texts -- but each call is lightweight pure-Python work (no model
#     weights, no PyTorch), so it's still dramatically faster than the
#     transformer version it replaces.
#     """
#     if not texts:
#         return []

#     results = ["unknown"] * len(texts)

#     # Same cheap pre-filter as before: skip detection entirely for text
#     # that's too short or has no Arabic/Latin script at all.
#     run_idx = []
#     samples = []
#     for i, text in enumerate(texts):
#         if not text or len(text) < 20:
#             continue
#         has_arabic = bool(ARABIC_PATTERN.search(text[:300]))
#         has_latin = bool(LATIN_PATTERN.search(text[:300]))
#         if not (has_arabic or has_latin):
#             results[i] = "ignored"
#             continue
#         run_idx.append(i)
#         samples.append(text[:250])

#     for i, sample in zip(run_idx, samples):
#         try:
#             langs = detect_langs(sample)
#             lang_code = langs[0].lang if langs else "unknown"
#         except LangDetectException:
#             results[i] = "unknown"
#             continue
#         except Exception:
#             results[i] = "unknown"
#             continue

#         if lang_code == "ar":
#             results[i] = "ar" if is_valid_arabic_text(texts[i]) else "corrupted"
#         elif lang_code == "en":
#             results[i] = "en"
#         elif lang_code == "fr":
#             results[i] = "fr"
#         else:
#             results[i] = lang_code

#     return results


# def detect_dialects_batch(texts: List[str]) -> List[str]:
#     """Batched version of detect_arabic_dialect() for a list of Arabic
#     texts. Unchanged: still the HF CAMeL BERT model."""
#     if not texts:
#         return []
#     try:
#         detector = get_dialect_detector()
#         samples = [t[:250] if t else "" for t in texts]
#         predictions = detector(samples)
#         return [p[0]['label'] if p else "MSA" for p in predictions]
#     except Exception:
#         return ["MSA"] * len(texts)


# # ============================================================
# # DOMAIN-LEVEL BOILERPLATE STRIPPING
# #
# # trafilatura handles most sites' nav/sidebar/footer noise correctly, but
# # some sites (e.g. almanar.com.lb) embed a repeated "latest news" ticker
# # INSIDE the same DOM container as the article body, so trafilatura can't
# # tell it apart from real content. Two complementary layers handle this:
# #
# # 1. Manual regex patterns per known-problem domain, in every language
# #    that domain publishes in (Arabic/English/French here) -- catches the
# #    exact tickers we've already seen in our own samples.
# # 2. Automatic repeated-tail detection, language-agnostic -- catches the
# #    SAME kind of problem on domains we haven't manually inspected yet,
# #    by noticing when the same trailing text keeps showing up across
# #    multiple articles from the same domain.
# #
# # Layer 1 is precise but needs manual upkeep per domain/language. Layer 2
# # needs no upkeep but only kicks in after it's seen a domain's repeated
# # tail a few times, so it won't catch a brand-new problem domain's very
# # first few articles. Together they cover more ground than either alone.
# # ============================================================

# BOILERPLATE_CUT_PATTERNS = {
#     "almanar.com.lb": [
#         # Arabic
#         r"آخر الأخبار",
#         r"أحدث الأخبار",
#         r"مواضيع ساخنة",
#         r"المزيد أحدث الأخبار",
#         # English
#         r"Latest News",
#         r"Related News",
#         r"Read More",
#         r"Hot Topics",
#         # French
#         r"Dernières nouvelles",
#         r"Actualités récentes",
#         r"Sujets brûlants",
#     ],
# }

# # The masthead block (date/hijri-date/city/time + language switcher) that
# # almanar.com.lb repeats verbatim at the top of every article, in every
# # language it publishes in.
# MASTHEAD_PATTERNS = {
#     "almanar.com.lb": re.compile(
#         r"^.*?Ar En Fr Es\s*(عاجل|Breaking|Urgent|Dernière heure)?\s*",
#         re.DOTALL
#     ),
# }

# # Automatic repeated-tail detection state. Keyed by domain -> {tail_hash: count}.
# # Process-local (not persisted), so it resets each run; that's fine since
# # it only needs to see a domain a few times within a single run to kick in.
# _domain_tail_hashes = defaultdict(lambda: defaultdict(int))
# BOILERPLATE_REPEAT_THRESHOLD = 3  # repeats of the same tail before we treat it as boilerplate
# TAIL_CHARS = 150  # how many trailing characters we fingerprint


# # def get_domain(url: str) -> str:
# #     """Returns the bare domain (no 'www.') for a URL, or '' if unparsable."""
# #     if not url:
# #         return ""
# #     netloc = urlparse(url).netloc
# #     return netloc[4:] if netloc.startswith("www.") else netloc


# def get_domain(url: str) -> str:
#     """Returns the bare domain (no 'www.') for a URL, or '' if unparsable."""
#     if not url:
#         return ""
#     netloc = urlparse(url).netloc.lower()
#     parts = netloc.split(".")
#     # ناخد آخر جزئين بس (اسم الدومين + الامتداد)، ونتجاهل أي subdomain
#     # ملاحظة: ده تبسيط - مش هيشتغل صح مع امتدادات مركبة زي .co.uk，
#     # بس كويس كفاية لحالتنا (.com, .lb, .ae...)
#     if len(parts) >= 2:
#         return ".".join(parts[-2:])
#     return netloc


# def strip_domain_boilerplate(text: str, url: str) -> str:
#     """Removes known boilerplate (repeated masthead / news-ticker) for
#     domains we've manually identified as having this problem. Returns the
#     text unchanged if the domain isn't in the list or nothing matches."""
#     if not text or not url:
#         return text

#     domain = get_domain(url)

#     # Strip a repeated masthead block from the start of the text.
#     masthead_re = MASTHEAD_PATTERNS.get(domain)
#     if masthead_re:
#         stripped = masthead_re.sub("", text, count=1)
#         if stripped and len(stripped) < len(text):
#             text = stripped

#     # Cut everything from the earliest boilerplate marker onward.
#     cut_patterns = BOILERPLATE_CUT_PATTERNS.get(domain)
#     if cut_patterns:
#         earliest_cut = None
#         for pattern in cut_patterns:
#             m = re.search(pattern, text)
#             if m and (earliest_cut is None or m.start() < earliest_cut):
#                 earliest_cut = m.start()
#         if earliest_cut is not None:
#             text = text[:earliest_cut].strip()

#     return text


# def detect_and_strip_repeated_tail(text: str, domain: str) -> str:
#     """
#     Language-agnostic fallback for domains we haven't manually inspected.
#     Fingerprints the trailing TAIL_CHARS of each article's text per
#     domain; once the same fingerprint has been seen
#     BOILERPLATE_REPEAT_THRESHOLD times for that domain, treats it as a
#     repeated ticker/footer and strips it from subsequent articles.

#     Note: this only helps within a single run and only after it has
#     already seen a domain's repeated tail a few times -- it will not
#     catch a new problem domain's very first few articles. It's a
#     complement to the manual patterns above, not a replacement.
#     """
#     if not text or len(text) < TAIL_CHARS or not domain:
#         return text

#     tail = text[-TAIL_CHARS:]
#     tail_hash = hashlib.md5(tail.encode("utf-8", errors="ignore")).hexdigest()

#     _domain_tail_hashes[domain][tail_hash] += 1
#     count = _domain_tail_hashes[domain][tail_hash]

#     if count >= BOILERPLATE_REPEAT_THRESHOLD:
#         # Find where this repeated tail begins in the current text and
#         # cut there, so we keep whatever unique content precedes it.
#         anchor = tail[:40]
#         cutoff = text.rfind(anchor)
#         if cutoff > 0:
#             return text[:cutoff].strip()

#     return text


# def extract_publication_date(tree: HTMLParser) -> str:
#     """Extracts publication date via Meta tags, JSON-LD, or <time> tag."""
#     meta_selectors = [
#         ("meta[property='article:published_time']", "content"),
#         ("meta[name='pubdate']", "content"),
#         ("meta[name='publishdate']", "content"),
#         ("meta[name='date']", "content"),
#         ("meta[name='DC.date.issued']", "content"),
#         ("meta[name='parsely-pub-date']", "content")
#     ]

#     for selector, attr in meta_selectors:
#         node = tree.css_first(selector)
#         if node and node.attributes.get(attr):
#             date_val = node.attributes.get(attr).strip()
#             if date_val:
#                 return date_val

#     for script_node in tree.css("script[type='application/ld+json']"):
#         raw = script_node.text(strip=True)
#         if not raw:
#             continue
#         try:
#             data = json.loads(raw)
#         except (json.JSONDecodeError, ValueError):
#             continue
#         candidates = data if isinstance(data, list) else [data]
#         for item in candidates:
#             if isinstance(item, dict):
#                 date_val = item.get("datePublished") or item.get("dateCreated")
#                 if date_val:
#                     return str(date_val).strip()

#     time_node = tree.css_first("time")
#     if time_node:
#         datetime_val = time_node.attributes.get("datetime")
#         if datetime_val:
#             return datetime_val.strip()
#         text_val = time_node.text(strip=True)
#         if text_val:
#             return text_val

#     return "N/A"


# def extract_date_from_text_fallback(tree: HTMLParser) -> str:
#     """Fallback: Regex search over clean body text for date patterns."""
#     body_text = tree.body.text() if tree.body else tree.text()
#     date_match = re.search(
#         r'\b(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b',
#         body_text
#     )
#     if date_match:
#         return date_match.group(0)
#     return "N/A"


# def extract_author(tree: HTMLParser) -> str:
#     """Extracts author information via Meta tags or JSON-LD."""
#     author_selectors = [
#         ("meta[name='author']", "content"),
#         ("meta[property='article:author']", "content"),
#         ("meta[name='parsely-author']", "content"),
#         ("meta[name='dc.creator']", "content"),
#         ("meta[name='author_name']", "content")
#     ]

#     for selector, attr in author_selectors:
#         node = tree.css_first(selector)
#         if node and node.attributes.get(attr):
#             val = node.attributes.get(attr).strip()
#             if val:
#                 return val

#     for script_node in tree.css("script[type='application/ld+json']"):
#         raw = script_node.text(strip=True)
#         if not raw:
#             continue
#         try:
#             data = json.loads(raw)
#             candidates = data if isinstance(data, list) else [data]
#             for item in candidates:
#                 if isinstance(item, dict) and "author" in item:
#                     author_data = item["author"]
#                     if isinstance(author_data, dict) and "name" in author_data:
#                         return str(author_data["name"]).strip()
#                     elif isinstance(author_data, list) and len(author_data) > 0:
#                         first_author = author_data[0]
#                         if isinstance(first_author, dict):
#                             return str(first_author.get("name", "N/A")).strip()
#                         return str(first_author).strip()
#                     elif isinstance(author_data, str):
#                         return author_data.strip()
#         except (json.JSONDecodeError, ValueError):
#             continue

#     return "N/A"


# def extract_main_content(html_str: str) -> str:
#     """
#     يرجع نص المقال الرئيسي فقط، بدون nav/sidebar/related-articles.
#     يرجع None لو فشل الاستخراج (نادر) فنرجع للطريقة القديمة كـ fallback.
#     """
#     extracted = trafilatura.extract(
#         html_str,
#         include_comments=False,
#         include_tables=False,
#         no_fallback=False,  
#         favor_precision=True,  
#     )
#     return extracted


# def extract_html_fields(record: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Phase 1 of extraction: pure HTML parsing/cleaning (title, author, dates,
#     links, clean_text, word_count). Deliberately does NOT call any
#     language-ID model, so it's cheap and safe to run per-record in a tight
#     loop. Language/dialect detection is done separately, in batches, by
#     detect_languages_batch() / detect_dialects_batch() below.

#     Main-content extraction goes through trafilatura first (falls back to
#     raw body text if trafilatura fails or returns too little), then
#     through domain-specific boilerplate stripping (manual patterns +
#     automatic repeated-tail detection) to catch tickers/footers that
#     trafilatura couldn't separate from the article body on some sites.
#     """
#     html_str, status = decode_and_validate(record["raw_bytes"])

#     if status != "success" or not html_str:
#         return {"language": "corrupted", "word_count": 0, "clean_text": ""}

#     tree = HTMLParser(html_str)

#     title_node = tree.css_first("title")
#     title = title_node.text(strip=True) if title_node else "N/A"

#     # --- Try to extract just the main article content first ---
#     main_text = extract_main_content(html_str)

#     if main_text and len(main_text.split()) >= 80:
#         clean_text = " ".join(main_text.split())
#     else:
#         # Fallback to the old whole-body method if trafilatura failed or
#         # returned too little text.
#         for tag in tree.css(", ".join(UNWANTED_TAGS)):
#             tag.decompose()
#         body = tree.body
#         raw_text = body.text(separator=' ', strip=True) if body else tree.text(separator=' ', strip=True)
#         clean_text = " ".join(raw_text.split())

#     # --- Strip domain-specific boilerplate that trafilatura missed ---
#     url = record.get("url", "")
#     domain = get_domain(url)
#     clean_text = strip_domain_boilerplate(clean_text, url)
#     clean_text = detect_and_strip_repeated_tail(clean_text, domain)

#     word_cnt = len(clean_text.split())
#     if word_cnt < 80:
#         return {"language": "too_short", "word_count": word_cnt, "clean_text": ""}

#     pub_date = extract_publication_date(tree)
#     if pub_date == "N/A":
#         pub_date = extract_date_from_text_fallback(tree)

#     author = extract_author(tree)

#     return {
#         "title": title,
#         "author": author,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "published_date": pub_date,
#         "clean_text": clean_text,
#         "word_count": word_cnt,
#         "char_count": len(clean_text),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": len(tree.css("a")),
#         "headings_sample": [node.text(strip=True) for node in tree.css("h1, h2, h3")][:3]
#     }


# def extract_selectolax(record: Dict[str, Any]) -> Dict[str, Any]:
#     """Extractor with quality filtering and language detection (fasttext).
#     Kept for callers that process one record at a time (no batching)."""
#     html_str, status = decode_and_validate(record["raw_bytes"])

#     if status != "success" or not html_str:
#         return {"language": "corrupted", "word_count": 0, "clean_text": ""}

#     tree = HTMLParser(html_str)

#     title_node = tree.css_first("title")
#     title = title_node.text(strip=True) if title_node else "N/A"

#     for tag in tree.css(", ".join(UNWANTED_TAGS)):
#         tag.decompose()

#     body = tree.body
#     raw_text = body.text(separator=' ', strip=True) if body else tree.text(separator=' ', strip=True)
#     clean_text = " ".join(raw_text.split())

#     word_cnt = len(clean_text.split())
#     if word_cnt < 80:
#         return {"language": "too_short", "word_count": word_cnt, "clean_text": ""}

#     language = detect_language_hf(clean_text)

#     arabic_dialect = "N/A"
#     if language == "ar":
#         if not is_valid_arabic_text(clean_text):
#             return {"language": "corrupted", "word_count": word_cnt, "clean_text": ""}
#         arabic_dialect = detect_arabic_dialect(clean_text)

#     pub_date = extract_publication_date(tree)
#     if pub_date == "N/A":
#         pub_date = extract_date_from_text_fallback(tree)

#     author = extract_author(tree)

#     return {
#         "title": title,
#         "author": author,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "published_date": pub_date,
#         "clean_text": clean_text,
#         "language": language,
#         "arabic_dialect": arabic_dialect,
#         "word_count": word_cnt,
#         "char_count": len(clean_text),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": len(tree.css("a")),
#         "headings_sample": [node.text(strip=True) for node in tree.css("h1, h2, h3")][:3]
#     }


# def extract_bs4_lxml(record: Dict[str, Any]) -> Dict[str, Any]:
#     """Legacy/Benchmark Extractor fallback."""
#     html_str, status = decode_and_validate(record["raw_bytes"])
#     if not html_str:
#         return {"clean_text": "", "word_count": 0, "text_length": 0}

#     soup = BeautifulSoup(html_str, 'lxml')

#     title = soup.title.get_text(strip=True) if soup.title else "N/A"

#     for tag in soup(UNWANTED_TAGS):
#         tag.decompose()

#     body = soup.body
#     text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
#     clean_text = " ".join(text.split())
#     headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
#     links_count = len(soup.find_all('a'))

#     return {
#         "title": title,
#         "url": record["url"],
#         "warc_date": record["warc_date"],
#         "clean_text": clean_text,
#         "text_length": len(clean_text),
#         "word_count": len(clean_text.split()),
#         "html_size_bytes": len(record["raw_bytes"]),
#         "links_count": links_count,
#         "headings_sample": headings[:3]
#     }



####

# extractors.py -- active version (post-fix)
# Changes vs previous:
#   1. detect_language() renamed from detect_language_hf (langdetect backend);
#      detect_language_hf kept as alias for backward compatibility.
#   2. get_domain() handles multi-part TLDs (.co.uk, .com.sa, ...).
#   3. extract_selectolax() now reuses extract_html_fields() so trafilatura
#      + boilerplate stripping run on the single-record path too.
#
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
        sample_text = text[:250]
        langs = detect_langs(sample_text)
        lang_code = langs[0].lang if langs else "unknown"

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
        samples.append(text[:250])

    for i, sample in zip(run_idx, samples):
        try:
            langs = detect_langs(sample)
            lang_code = langs[0].lang if langs else "unknown"
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

    return text


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

    if main_text and len(main_text.split()) >= 80:
        clean_text = " ".join(main_text.split())
    else:
        for tag in tree.css(", ".join(UNWANTED_TAGS)):
            tag.decompose()
        body = tree.body
        raw_text = body.text(separator=' ', strip=True) if body else tree.text(separator=' ', strip=True)
        clean_text = " ".join(raw_text.split())

    url = record.get("url", "")
    domain = get_domain(url)
    clean_text = strip_domain_boilerplate(clean_text, url)
    clean_text = detect_and_strip_repeated_tail(clean_text, domain)

    word_cnt = len(clean_text.split())
    if word_cnt < 80:
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