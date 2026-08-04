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

# Import mock_grpc first to bypass blocked cygrpc DLL under restrictive environment policies
import mock_grpc

# Load .env into os.environ BEFORE importing anything that reads env vars at
# import time (api.reranker reads GROQ_API_KEY/LLM_PROVIDER, api.sharepoint reads
# SHAREPOINT_*, indexer.parser reads ENABLE_OCR). pydantic-settings only feeds
# .env into Settings fields — it never populates os.environ — so without this
# every os.getenv() call below silently fell back to its default when running
# outside Docker, e.g. reranking degraded to vector-only with no error.
from dotenv import load_dotenv

load_dotenv()

import hmac
import logging  
import os
import sys
import re
from pathlib import Path
from contextlib import asynccontextmanager
    
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
    JDRequest,
    JDResponse,
)
from api.retriever import retrieve_candidates, build_candidate_response
from api.reranker import rerank_candidates, generate_llm_response
from api import sharepoint
from api.sharepoint import get_graph_token, resolve_site_id, sync_sharepoint_resumes
from indexer.utils import setup_logging

# ── Logging ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Settings ───────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    # Auth
    # min_length=1 ensures the service refuses to start with an empty API_KEY.
    # Generate a strong key with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    api_key: str = Field(min_length=1)

    # Qdrant
    # "localhost", not "qdrant": docker-compose sets QDRANT_HOST=qdrant explicitly,
    # so the service-name default only ever applied to local runs — where it never
    # resolves. The API then fell back to embedded ./data/qdrant_db while
    # indexer/run.py (which defaults to localhost) wrote to the Qdrant server:
    # two different databases, so screening returned zero candidates after a
    # seemingly successful index.
    qdrant_host: str = "localhost"
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

    # Reuse the indexer's setup instead of a second copy of the same basicConfig:
    # it forces UTF-8 on stdout, without which Windows consoles raise
    # UnicodeEncodeError on any log line containing non-cp1252 characters.
    setup_logging(settings.log_level)

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

    # ALLOWED_ORIGINS was parsed into Settings but never reached the middleware:
    # every deployment ran with allow_origin_regex=r"https?://.*", i.e. any site
    # on the internet could call this API from a user's browser regardless of
    # what the operator configured. Honour the setting; "*" still opts out.
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        # Credentials cannot be combined with a "*" origin (browsers reject it),
        # and auth here is a header, not a cookie — so it is never needed.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
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


# ── Custom Static Files Router ──────────────────────────────────────────────────

@app.get("/api/v1/cvs/{filename:path}", tags=["CVs"], summary="Fetch local CV file")
async def get_cv_file(filename: str):
    """
    Serves CV files statically, supporting:
      1. Exact match
      2. URL decoded match
      3. Cleaned filename match (removes 34-char item_id prefix)
      4. Fallback search (finds file regardless of stored/local prefix discrepancy)
    """
    cv_folder = Path(app.state.settings.cv_folder_path)
    
    # Strip leading 'cvs/' or 'cvs\' prefix if present
    filename = re.sub(r"^cvs[/\\]", "", filename)
    
    # 1. Try exact match first
    exact_path = cv_folder / filename
    if exact_path.is_file():
        return FileResponse(exact_path)
        
    # 2. Try URL decoded exact match
    from urllib.parse import unquote
    decoded_filename = re.sub(r"^cvs[/\\]", "", unquote(filename))
    decoded_path = cv_folder / decoded_filename
    if decoded_path.is_file():
        return FileResponse(decoded_path)
        
    # 3. Try removing 34-character ID prefix from requested filename
    cleaned_filename = re.sub(r"(^|/|\\)([A-Z0-9]{34})_", r"\1", decoded_filename)
    cleaned_path = cv_folder / cleaned_filename
    if cleaned_path.is_file():
        return FileResponse(cleaned_path)
        
    # 4. Try removing ID prefix from actual files in the folder (fallback search)
    search_name = cleaned_filename.lower()
    for f in cv_folder.glob("**/*"):
        if f.is_file():
            f_cleaned = re.sub(r"([A-Z0-9]{34})_", "", f.name).lower()
            if f_cleaned == search_name or f.name.lower() == search_name:
                return FileResponse(f)
                
            # Handle mismatch in position prefixes (e.g., "Position_File.pdf" vs "File.pdf")
            f_base = os.path.splitext(f_cleaned)[0]
            search_base = os.path.splitext(search_name)[0]
            if len(f_base) > 5 and (search_base.endswith(f_base) or f_base.endswith(search_base)):
                return FileResponse(f)
                
    raise HTTPException(status_code=404, detail="CV file not found")


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
    and local NLP analysis, with fallback to standard template if LLM is unavailable.
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

    # Format candidate context details for the LLM
    candidates = [build_candidate_response(c) for c in raw_candidates[:8]]
    candidates_context = ""
    for idx, cand in enumerate(candidates, 1):
        name = cand.get("name") or "Unknown Candidate"
        metadata = cand.get("metadata", {})
        exp = f"{metadata.get('experience_years')} years" if metadata.get('experience_years') is not None else "N/A"
        loc = metadata.get("location") or "N/A"
        skills = ", ".join(metadata.get("skills", []))
        cv_excerpt = cand.get("best_chunk_text") or ""
        candidates_context += (
            f"Candidate {idx}:\n"
            f"ID: {cand.get('candidate_id')}\n"
            f"Name: {name}\n"
            f"Experience: {exp}\n"
            f"Location: {loc}\n"
            f"Skills: {skills}\n"
            f"Resume Excerpt:\n{cv_excerpt[:800]}\n"
            f"-----------------\n"
        )

    # Format history context
    history_context = ""
    if body.history:
        for item in body.history:
            role_name = "User" if item.role == "user" else "Assistant"
            history_context += f"{role_name}: {item.content}\n"

    # Construct LLM prompt
    prompt = f"""You are "Aryan", a smart and helpful AI Recruiter Copilot at TalentMatch.
You help recruitment and HR teams evaluate candidates, compare skillsets, check locations, draft screening emails/invitations, and answer recruiting questions.

Below is a list of the top candidate profiles retrieved from our vector database that match the user's current query:
{candidates_context}

INSTRUCTIONS:
1. When answering questions about candidates, use the candidate context provided above.
2. If the user asks for a template (like a screening invite email, reject email, etc.), draft a professional template using the details of the candidate they are talking about.
3. If the user's message is a general greeting or unrelated recruiting question, answer it professionally.
4. IMPORTANT: Always refer to candidates by their exact name as listed in the context.
5. Keep your tone professional, supportive, and conversational. Use markdown formatting.

Conversation History:
{history_context}
User: {query_text}
Assistant:"""

    # Check for dynamic client-supplied Gemini key header
    gemini_key_override = request.headers.get("X-Gemini-API-Key", "").strip() or None

    try:
        reply_text = generate_llm_response(prompt, gemini_api_key_override=gemini_key_override)
        if reply_text and reply_text.strip():
            return ChatResponse(reply=reply_text.strip())
    except Exception as llm_err:
        logger.warning("Failed to get conversational response from LLM: %s. Falling back to template.", llm_err)

    # Fallback to template response
    if not candidates:
        reply_text = (
            f"Unfortunately, I couldn't find any candidates matching your query **\"{query_text}\"** "
            "in the current database search results.\n\n"
            "It appears that no profiles matching those specific qualifications were returned from the semantic vector database. "
            "Make sure your resume files have been indexed into the vector database."
        )
        return ChatResponse(reply=reply_text)

    reply_lines = [
        f"Based on your query **\"{query_text}\"**, I analyzed the candidate database and retrieved the top matching profiles:\n"
    ]

    for idx, cand in enumerate(candidates[:5], 1):
        name = cand.get("name") or "Unknown Candidate"
        metadata = cand.get("metadata", {})
        exp = f"{metadata.get('experience_years')} years exp" if metadata.get('experience_years') is not None else "Experience N/A"
        loc = metadata.get("location") or "Location N/A"
        skills_list = metadata.get("skills", [])
        skills = ", ".join(skills_list[:8]) if skills_list else "Skills extracted in CV"
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
        # rglob, not glob: CVs dropped into subfolders of cvs/ (the usual shape
        # once SharePoint folders are mirrored) were previously never indexed.
        # Extensions come from the parser so we never queue a file it will drop.
        from indexer.parser import SUPPORTED_EXTENSIONS

        existing_files = [
            f for f in cv_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
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
            site_url=body.site_url,
            folder_path=body.folder_path,
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
    cv_folder_path = getattr(settings, "cv_folder_path", "./cvs")

    # Validate credentials, site and write access up front. The download loop
    # still runs in the background, but a bad tenant/secret/site_url now fails
    # the request instead of returning "success" while the real error is buried
    # in the server log where the caller never sees it.
    try:
        creds = (
            body.tenant_id or sharepoint.DEFAULT_TENANT_ID,
            body.client_id or sharepoint.DEFAULT_CLIENT_ID,
            body.client_secret or sharepoint.DEFAULT_CLIENT_SECRET,
        )
        if not all(creds):
            raise ValueError(
                "SharePoint Azure AD credentials are required. Send tenant_id, client_id and "
                "client_secret in the request body, or set SHAREPOINT_TENANT_ID / "
                "SHAREPOINT_CLIENT_ID / SHAREPOINT_CLIENT_SECRET in .env."
            )
        token = get_graph_token(*creds)
        resolve_site_id(
            {"Authorization": f"Bearer {token}"},
            body.site_url or sharepoint.DEFAULT_SITE_URL or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("SharePoint preflight failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Could not reach Microsoft Graph: {exc}")

    cv_dir = Path(cv_folder_path)
    try:
        cv_dir.mkdir(parents=True, exist_ok=True)
        probe = cv_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"CV folder '{cv_dir}' is not writable ({exc}). Downloaded resumes would be "
                "discarded. Check CV_FOLDER_PATH and that the folder is mounted read-write."
            ),
        )

    background_tasks.add_task(
        run_sharepoint_sync_task,
        body=body,
        cv_folder_path=cv_folder_path,
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


@app.post(
    "/api/v1/generate-jd",
    response_model=JDResponse,
    tags=["AI Utility"],
    summary="Generate Job Description and keywords offline",
)
async def generate_job_description(request: Request, payload: JDRequest):
    """
    Offline local generation of Job Descriptions and keywords using Qwen2.5-0.5B local LLM.
    """
    title = payload.job_title.strip()
    
    from api.local_llm import generate_local_jd
    
    logger.info("Generating local dynamic Job Description using Qwen-0.5B model for: %s", title)
    result = generate_local_jd(title)
    
    return JDResponse(
        job_description=result.get("job_description", ""),
        keywords=result.get("keywords", [])
    )
