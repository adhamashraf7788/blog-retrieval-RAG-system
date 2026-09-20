from fastapi import FastAPI

from query_service.config import build_default_pipeline
from query_service.models import ProcessedQuery, QueryRequest
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Query Service", version="0.1.0")
pipeline = build_default_pipeline()


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
