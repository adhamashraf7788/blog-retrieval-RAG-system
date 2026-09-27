import os
import instructor
from groq import AsyncGroq
from dotenv import load_dotenv

from pydantic import BaseModel
from .models import (
    Chunk,
    EvaluationResult,
    RetrievalDecision,
)
from .prompts import EVALUATION_PROMPT
load_dotenv()

class _EvaluationOutput(BaseModel):
    decision: RetrievalDecision
    reason: str

class CRAGEvaluator:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        raw_client = AsyncGroq(
            api_key=os.environ["GROQ_API_KEY"]
        )
        self.client = instructor.from_groq(
            raw_client,
            mode=instructor.Mode.JSON,
        )

    async def evaluate(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> EvaluationResult:
        context = self._format_chunks(chunks)
        prompt = EVALUATION_PROMPT.format(
            query=query,
            context=context,
        )
        result = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=_EvaluationOutput,
            temperature=0,
            max_retries=3,
        )
        return EvaluationResult(
            decision=result.decision,
            reason=result.reason,
        )

    @staticmethod
    def _format_chunks(chunks: list[Chunk]) -> str:
        if not chunks:
            return "[NO RETRIEVED CONTEXT]"

        formatted = []

        for chunk in chunks:
            formatted.append(
                f"[Chunk {chunk.id}]\n{chunk.content}"
            )
        return "\n\n".join(formatted)