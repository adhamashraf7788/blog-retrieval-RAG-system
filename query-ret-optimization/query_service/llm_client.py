"""
Thin wrapper around whatever LLM SDK you use. Swap the internals freely —
nothing else in the service should import the underlying SDK directly.
"""

import os
from typing import TypeVar

import instructor
from groq import AsyncGroq
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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