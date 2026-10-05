from fastapi import FastAPI
from fastapi.responses import JSONResponse

from query_service.config import build_default_pipeline
from query_service.llm_client import is_rate_limit_error, parse_retry_wait
from query_service.models import ProcessedQuery, QueryRequest
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Query Service", version="0.1.0")
pipeline = build_default_pipeline()


@app.exception_handler(Exception)
async def unhandled_to_json(request, exc: Exception):  # noqa: ANN001, ANN201
    # Groq 429s that survived llm_client retries surface here as
    # RateLimitError or InstructorRetryException — return truthful 429
    # with Retry-After instead of a misleading 500.
    if is_rate_limit_error(exc):
        wait = parse_retry_wait(str(exc), 0)
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(int(wait) + 1)},
            content={
                "detail": "LLM rate limit exhausted, retry after a few seconds.",
                "retry_after_seconds": wait,
            },
        )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.post("/process_query", response_model=ProcessedQuery)
async def process_query(request: QueryRequest) -> ProcessedQuery:
    """
    Public contract: send a raw query, get back retrieval-ready queries.
    This service never calls retrieval and never merges results —
    that stays entirely on the retrieval side.
    """
    return await pipeline.process(request)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
