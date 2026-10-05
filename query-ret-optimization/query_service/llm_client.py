"""
Thin wrapper around whatever LLM SDK you use. Swap the internals freely —
nothing else in the service should import the underlying SDK directly.
"""

import asyncio
import os
import random
import re
from typing import TypeVar

import instructor
from groq import AsyncGroq, RateLimitError
from instructor.v2.core.errors import InstructorRetryException
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Retry-twice-then-fail on Groq 429s (see eval run_20260926: TPM 8000
# bucket, 3-subquery chains cost ~5 calls in seconds). Honors Groq's
# "try again in Xs" wait when present, else 5s/10s fallback + jitter.
MAX_RATE_LIMIT_RETRIES = 2
FALLBACK_WAIT_SECONDS = (5.0, 10.0)

_RETRY_IN_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def parse_retry_wait(message: str, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited call."""
    m = _RETRY_IN_RE.search(message or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    idx = min(attempt, len(FALLBACK_WAIT_SECONDS) - 1)
    return FALLBACK_WAIT_SECONDS[idx]


def is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, InstructorRetryException):
        msg = str(exc).lower()
        return "429" in msg or "rate_limit_exceeded" in msg or "rate limit" in msg
    return False


class LLMClient:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        # Built manually (not via instructor.from_provider) and pinned to
        # JSON mode explicitly — from_provider's `mode=` kwarg was silently
        # ignored for this provider/model, so it kept defaulting to
        # tool-calling mode, which gpt-oss on Groq doesn't handle reliably.
        raw_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        self.client = instructor.from_groq(
            raw_client,
            mode=instructor.Mode.JSON,
        )

    async def generate_structured(self, prompt: str, schema: type[T]) -> T:
        last_exc: BaseException | None = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,
                    # gpt-oss on Groq occasionally replies with a bare word, or a
                    # conversational "please provide a query" instead of JSON —
                    # retry a couple times before giving up.
                    max_retries=3,
                    # Deterministic, non-chatty output — this is a structured
                    # extraction task, not a conversation. Reduces the odds of the
                    # model wandering into a conversational reply instead of JSON.
                    temperature=0,
                )
            except Exception as exc:  # noqa: BLE001 — classified below
                if not is_rate_limit_error(exc) or attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise
                last_exc = exc
                wait = parse_retry_wait(str(exc), attempt) + random.uniform(0, 1.0)
                await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc
