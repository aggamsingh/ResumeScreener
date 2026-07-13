"""
FastAPI application — main entrypoint.

Startup:
  - Loads the SentenceTransformer embedding model into app.state (once only)
  - Initializes the Qdrant client into app.state

Middleware:
  - API key authentication on every route except /health and docs

Routes:
  - GET  /health          → service status check
  - POST /api/v1/screen  → run a screening request
"""
from __future__ import annotations

import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api.models import HealthResponse, ScreeningRequest, ScreeningResponse
from api.retriever import retrieve_candidates, build_candidate_response
from api.reranker import rerank_candidates

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


# ── Settings ───────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    # Auth
    # min_length=1 ensures the service refuses to start with an empty API_KEY.
    # Generate a strong key with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    api_key: str = Field(min_length=1)

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "resumes"

    # Screening
    default_top_k: int = 10
    retrieval_top_n: int = 30
    embedding_model: str = "all-MiniLM-L6-v2"

    # Server
    allowed_origins: str = "*"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ── App Lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once on startup; clean up on shutdown."""
    settings: Settings = app.state.settings

    logger.info("=" * 55)
    logger.info("Resume Screener API — Starting up")
    logger.info("=" * 55)

    # Load embedding model (baked into Docker image, loads from disk cache ~1s)
    logger.info("Loading embedding model: %s", settings.embedding_model)
    app.state.model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model ready")

    # Initialize Qdrant client
    logger.info("Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
    app.state.qdrant = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=10,
    )
    logger.info("Qdrant client ready")

    logger.info("API is ready to serve requests")
    logger.info("=" * 55)

    yield  # ← service runs here

    logger.info("Resume Screener API — Shutting down")


# ── App Factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="Resume Screener API",
        description=(
            "RAG-powered resume screening microservice. "
            "Submit a job description, receive ranked candidates."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store settings on app so lifespan and routes can access them
    app.state.settings = settings

    # CORS
    origins = [o.strip() for o in settings.allowed_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    return app


app = create_app()

# ── Auth Middleware ────────────────────────────────────────────────────────────

_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    expected_key = app.state.settings.api_key

    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        logger.warning(
            "Rejected unauthenticated request: path=%s host=%s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    "Invalid or missing API key. "
                    "Send your key in the X-API-Key request header."
                )
            },
        )

    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
async def health_check(request: Request):
    """
    Returns service status. No authentication required.
    Use this endpoint to confirm the service is running before sending
    screening requests, or for automated monitoring.
    """
    qdrant_ok = False
    try:
        request.app.state.qdrant.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)

    model_ok = (
        hasattr(request.app.state, "model")
        and request.app.state.model is not None
    )

    is_healthy = qdrant_ok and model_ok
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content=HealthResponse(
            status="healthy" if is_healthy else "degraded",
            qdrant_connected=qdrant_ok,
            model_loaded=model_ok,
        ).model_dump(),
    )


@app.post(
    "/api/v1/screen",
    response_model=ScreeningResponse,
    tags=["Screening"],
    summary="Screen CVs against a job description",
)
async def screen_resumes(request: Request, body: ScreeningRequest):
    """
    Submit a job description and receive the top matching candidates.

    **Authentication:** Include your API key in the `X-API-Key` header.

    **Flow:**
    1. Job description is embedded using MiniLM (runs locally on the server).
    2. Top candidates retrieved from the vector database.
    3. Hard filters applied (experience, location, required skills).
    4. An LLM (Groq/Gemini) reranks remaining candidates with reasoning.
    5. Top K candidates returned with scores, reasoning, and metadata.

    **Typical response time:** 3–8 seconds (dominated by LLM API latency).
    """
    settings: Settings = request.app.state.settings
    top_k = body.top_k or settings.default_top_k

    has_filters = body.filters.is_active()
    logger.info(
        "Screening request: top_k=%d, jd=%d chars, filters=%s",
        top_k,
        len(body.job_description),
        body.filters.model_dump(exclude_none=True) if has_filters else "none",
    )

    # ── Step 1: Vector Retrieval + Hard Filtering ────────────────
    try:
        raw_candidates, n_filtered_out = retrieve_candidates(
            qdrant_client=request.app.state.qdrant,
            model=request.app.state.model,
            jd_text=body.job_description,
            collection_name=settings.qdrant_collection,
            top_n=settings.retrieval_top_n,
            filters=body.filters,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Vector retrieval failed. Is Qdrant running and the collection indexed?",
        )

    if not raw_candidates:
        msg = (
            "No candidates passed the applied filters."
            if has_filters and n_filtered_out > 0
            else "No candidates found. Has the CV collection been indexed?"
        )
        logger.warning(msg)
        return ScreeningResponse(candidates=[], total_filtered_out=n_filtered_out)

    # Reshape raw dicts into the format reranker + response builder expect
    candidates = [build_candidate_response(c) for c in raw_candidates]

    # ── Step 2: LLM Reranking ────────────────────────────────────
    ranked = rerank_candidates(
        jd_text=body.job_description,
        candidates=candidates,
        top_k=top_k,
    )

    logger.info(
        "Returning %d candidates (%d filtered out by hard filters)",
        len(ranked),
        n_filtered_out,
    )
    return ScreeningResponse(candidates=ranked, total_filtered_out=n_filtered_out)
