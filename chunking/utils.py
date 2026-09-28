import os
import re
import psutil
from typing import Optional, Tuple
from charset_normalizer import from_bytes
import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic_console_text(text: str) -> str:
    """Reshapes Arabic text and applies BiDi algorithms for clean console output."""
    if not text:
        return ""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except Exception:
        return text
    
ARABIC_CHAR_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

def get_rss_memory_mb() -> float:
    """Gets the total Resident Set Size (RSS) memory of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def is_target_language(lang_code: str) -> bool:
    """Checks if the detected language is English, French, or Arabic."""
    if not lang_code:
        return False
    lang = str(lang_code).lower().strip()
    return lang in {"ar", "en", "fr"}

def is_clean_encoding(text: str) -> bool:
    """Detects encoding corruption (Mojibake) and broken Unicode control characters."""
    if not text:
        return False

    if "\ufffd" in text:
        return False

    mojibake_chars = sum(1 for c in text if c in '¤§¼«¬¦¶±µ°¥¢£©®™¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ')
    if len(text) > 0 and (mojibake_chars / len(text)) > 0.03:
        return False

    return True

def is_valid_arabic_text(text: str) -> bool:
    """Ensures that text classified as Arabic contains a valid ratio of Arabic characters."""
    if not text or len(text) < 20:
        return False

    sample = text[:500]
    arabic_chars = len(ARABIC_CHAR_PATTERN.findall(sample))
    alpha_chars = sum(1 for c in sample if c.isalpha())

    if alpha_chars == 0:
        return False

    return (arabic_chars / alpha_chars) >= 0.40

def decode_and_validate(raw_bytes: bytes) -> Tuple[Optional[str], str]:
    """Safely decodes raw HTML bytes and validates the text encoding quality."""
    if not raw_bytes:
        return None, "empty_bytes"

    decoded_text = None

    try:
        decoded_text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass

    if not decoded_text:
        try:
            results = from_bytes(raw_bytes)
            best_match = results.best()
            if best_match and best_match.encoding:
                if best_match.encoding.lower() in ["iso-8859-1", "latin-1"] and best_match.coherence < 0.8:
                    pass
                else:
                    decoded_text = str(best_match)
        except Exception:
            pass

    if not decoded_text:
        try:
            decoded_text = raw_bytes.decode("windows-1256", errors="strict")
        except UnicodeDecodeError:
            pass

    if not decoded_text:
        return None, "decode_failed"

    if not is_clean_encoding(decoded_text):
        return None, "encoding_corrupted"

    return decoded_text, "success"

def decode_and_clean_html(raw_bytes: bytes) -> str:
    """Legacy helper function kept for backwards compatibility."""
    decoded, status = decode_and_validate(raw_bytes)
    return decoded if decoded else ""

