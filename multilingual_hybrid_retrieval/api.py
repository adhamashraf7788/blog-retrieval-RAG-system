# ============================================================
# FastAPI wrapper around the existing retrieval pipeline.
#
# Does NOT touch pipeline.py / hybrid_retriever.py / router.py —
# it only imports run_pipeline() and shapes the response.
#
# Run:
#   pip install fastapi "uvicorn[standard]"
#   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
#
# Docs (free, auto-generated):
#   http://localhost:8000/docs
# ============================================================

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Lifespan: load every model ONCE, at startup — not on the
# first request. Importing pipeline cascades into
# hybrid_retriever.py, reranker.py and translator.py, each of
# which loads a heavy model at import time.
# ------------------------------------------------------------

run_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global run_pipeline
    print("Loading models (BGE-M3, NLLB, BGE-reranker)... this can take a while.")
    from pipeline import run_pipeline as _run_pipeline
    run_pipeline = _run_pipeline
    print("Models loaded. API ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="Multilingual Hybrid Retrieval API", lifespan=lifespan)

# Adjust for your actual frontend origin(s) before shipping this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's search query, any supported language.")


class ResultItem(BaseModel):
    rank: int
    id: int
    language: str
    text: str
    rrf_score: float
    reranker_score: float


class QueryResponse(BaseModel):
    query: str
    languages: List[str]
    primary_language: str
    mixed_language: bool
    route: str
    branches: List[str]
    translated_query: Optional[str] = None
    results: List[ResultItem]


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": run_pipeline is not None}


@app.post("/search", response_model=QueryResponse)
def search(payload: QueryRequest):
    """
    Runs the full pipeline: Analyzer -> Router -> Hybrid Retrieval
    (Multilingual / Translation) -> RRF -> Reranker.

    Defined as a plain `def` (not `async def`) on purpose: every
    step below is blocking (DB calls, model inference), and FastAPI
    automatically runs sync endpoints in a threadpool so one slow
    request doesn't stall the whole server.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        result = run_pipeline(query)
    except Exception as exc:
        # Replace with real logging in production.
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    routing = result["routing"]

    results = []
    for rank, item in enumerate(result["final_results"], start=1):
        row = item["result"]
        results.append(ResultItem(
            rank=rank,
            id=row[0],
            language=row[3],
            text=row[4],
            rrf_score=float(item["rrf_score"]),
            reranker_score=float(item["reranker_score"]),
        ))

    return QueryResponse(
        query=query,
        languages=routing["languages"],
        primary_language=routing["primary_language"],
        mixed_language=routing["mixed_language"],
        route=routing["route"],
        branches=routing["branches"],
        translated_query=result.get("translated_query"),
        results=results,
    )
