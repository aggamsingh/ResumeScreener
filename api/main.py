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

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ScreeningRequest,
    ScreeningResponse,
    SyncRequest,
    SyncResponse,
)
from api.retriever import retrieve_candidates, build_candidate_response
from api.reranker import rerank_candidates
from api.sharepoint import sync_sharepoint_resumes

# ── Logging ────────────────────────────────────────────────────────────────────

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

    # CV Storage
    cv_folder_path: str = "./cvs"

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
    try:
        logger.info("Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3)
        client.get_collections()
        app.state.qdrant = client
        logger.info("Connected to Qdrant server successfully")
    except Exception as e:
        logger.warning("Could not connect to Qdrant server at %s:%d (%s). Falling back to embedded local storage at ./data/qdrant_db", settings.qdrant_host, settings.qdrant_port, e)
        app.state.qdrant = QdrantClient(path="./data/qdrant_db")
        logger.info("Embedded local Qdrant database initialized")

    try:
        from indexer.embedder import ensure_collection
        vector_dim = app.state.model.get_sentence_embedding_dimension()
        ensure_collection(app.state.qdrant, settings.qdrant_collection, vector_dim)
    except Exception as exc:
        logger.warning("Could not auto-create collection '%s': %s", settings.qdrant_collection, exc)

    logger.info("API is ready to serve requests")
    logger.info("=" * 55)

    yield  # ← service runs here

    logger.info("Resume Screener API — Shutting down")


# ── App Factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = Settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

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

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Serve downloaded CVs statically so they can be embedded in frontend iframes without auth/SharePoint issues
    cv_folder = getattr(settings, "cv_folder_path", "./cvs")
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder, exist_ok=True)
    app.mount("/api/v1/cvs", StaticFiles(directory=cv_folder), name="cvs")

    return app


app = create_app()

# ── Auth Middleware ────────────────────────────────────────────────────────────

# Prefix-match so /docs, /docs/, and Swagger asset sub-paths all pass through.
# FastAPI redirects /docs → /docs/ internally; exact-match blocked the redirect target.
_PUBLIC_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json", "/api/v1/cvs")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or any(request.url.path.startswith(p) for p in _PUBLIC_PREFIXES):
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


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Local AI Recruiter Agent chat endpoint",
)
async def chat_agent(request: Request, body: ChatRequest):
    """
    Local AI Recruiter Copilot.
    Processes user query against indexed resumes using local vector retrieval
    and local NLP analysis without requiring external API keys.
    """
    settings: Settings = request.app.state.settings
    query_text = body.message.strip()

    if not query_text:
        return ChatResponse(reply="Please provide a query or question about your candidates.")

    # Retrieve candidates using local MiniLM embedding model + Qdrant vector index
    try:
        raw_candidates, _ = retrieve_candidates(
            qdrant_client=request.app.state.qdrant,
            model=request.app.state.model,
            jd_text=query_text,
            collection_name=settings.qdrant_collection,
            top_n=10,
        )
    except Exception as exc:
        logger.error("Chat candidate retrieval failed: %s", exc, exc_info=True)
        raw_candidates = []

    if not raw_candidates:
        reply_text = (
            f"Unfortunately, I couldn't find any candidates matching your query **\"{query_text}\"** "
            "in the current database search results.\n\n"
            "It appears that no profiles matching those specific qualifications were returned from the semantic vector database. "
            "Make sure your resume files have been indexed into the vector database."
        )
        return ChatResponse(reply=reply_text)

    # Format retrieved candidate details intelligently into a local AI recruiter response
    candidates = [build_candidate_response(c) for c in raw_candidates[:5]]

    reply_lines = [
        f"Based on your query **\"{query_text}\"**, I analyzed the candidate database and retrieved the top matching profiles:\n"
    ]

    for idx, cand in enumerate(candidates, 1):
        name = cand.get("name") or "Unknown Candidate"
        metadata = cand.get("metadata", {})
        exp = f"{metadata.get('experience_years')} years exp" if metadata.get('experience_years') is not None else "Experience N/A"
        loc = metadata.get("location") or "Location N/A"
        skills_list = metadata.get("skills", [])
        skills = ", ".join(skills_list[:8]) if skills_list else "Skills extracted in CV"
        
        # Use best_chunk_text or summary as reasoning fallback
        reason = cand.get("best_chunk_text") or "Candidate profile matched via semantic search."
        if len(reason) > 200:
            reason = reason[:200] + "..."

        reply_lines.append(
            f"### {idx}. **{name}**\n"
            f"- **Experience & Location:** {exp} | {loc}\n"
            f"- **Key Skills:** {skills}\n"
            f"- **Match Overview:** {reason}\n"
        )

    reply_lines.append("\nFeel free to ask me to filter further by specific skills, experience levels, or locations!")

    return ChatResponse(reply="\n".join(reply_lines))


def run_sharepoint_sync_task(
    body: SyncRequest,
    cv_folder_path: str,
    qdrant_client,
    sentence_transformer_model,
    collection_name: str,
):
    logger.info("Starting background SharePoint sync and indexing task...")
    try:
        from pathlib import Path
        from indexer.parser import parse_file
        from indexer.utils import get_candidate_id
        from indexer.embedder import embed_and_upsert, ensure_collection

        cv_dir = Path(cv_folder_path)
        existing_files = [
            f for f in cv_dir.glob("*")
            if f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt") and not f.name.startswith(".")
        ]

        if existing_files:
            logger.info("Found %d existing local files. Indexing sequentially...", len(existing_files))
            vector_dim = sentence_transformer_model.get_sentence_embedding_dimension()
            ensure_collection(qdrant_client, collection_name, vector_dim)

            for idx, cv_file in enumerate(existing_files, 1):
                try:
                    cand_id = get_candidate_id(cv_file)
                    cv_obj = parse_file(cv_file, cand_id)
                    if cv_obj:
                        embed_and_upsert(qdrant_client, sentence_transformer_model, cv_obj, collection_name)
                        if idx % 10 == 0 or idx == len(existing_files):
                            logger.info("Indexed CV progress: %d/%d files", idx, len(existing_files))
                except Exception as p_err:
                    logger.warning("Error parsing/indexing CV %s: %s", cv_file.name, p_err)

        # Instant callback to index newly downloaded files immediately
        def on_download(file_path: Path):
            try:
                cand_id = get_candidate_id(file_path)
                cv_obj = parse_file(file_path, cand_id)
                if cv_obj:
                    vector_dim = sentence_transformer_model.get_sentence_embedding_dimension()
                    ensure_collection(qdrant_client, collection_name, vector_dim)
                    embed_and_upsert(qdrant_client, sentence_transformer_model, cv_obj, collection_name)
                    logger.info("Instantly indexed newly downloaded CV: %s", file_path.name)
            except Exception as index_err:
                logger.warning("Instant index failed for %s: %s", file_path.name, index_err)

        files_processed, new_downloaded = sync_sharepoint_resumes(
            tenant_id=body.tenant_id,
            client_id=body.client_id,
            client_secret=body.client_secret,
            target_dir=cv_folder_path,
            on_file_downloaded=on_download,
        )

        logger.info("Background SharePoint sync completed. Files processed: %d, New downloaded: %d", files_processed, new_downloaded)
    except Exception as exc:
        logger.error("Background SharePoint sync task failed: %s", exc, exc_info=True)


@app.post(
    "/api/v1/sync",
    response_model=SyncResponse,
    tags=["Sync"],
    summary="Sync candidate resumes from SharePoint and index them into Qdrant",
)
async def sync_sharepoint_endpoint(request: Request, body: SyncRequest, background_tasks: BackgroundTasks):
    """
    Downloads candidate resumes directly from SharePoint via Microsoft Graph API,
    parses them, and indexes them into Qdrant using the local embedding model in the background.
    """
    settings: Settings = request.app.state.settings
    background_tasks.add_task(
        run_sharepoint_sync_task,
        body=body,
        cv_folder_path=getattr(settings, "cv_folder_path", "./cvs"),
        qdrant_client=request.app.state.qdrant,
        sentence_transformer_model=request.app.state.model,
        collection_name=settings.qdrant_collection,
    )
    return SyncResponse(
        status="success",
        files_processed=0,
        candidates_added=0,
        message="SharePoint sync and indexing started in the background. Resumes will appear in the system as they are downloaded and processed.",
    )
