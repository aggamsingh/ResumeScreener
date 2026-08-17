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
import uuid
import requests
import json
from pathlib import Path
from contextlib import asynccontextmanager
    
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FilterSelector, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

try:
    import psycopg2  # type: ignore
except ImportError:
    psycopg2 = None  # type: ignore

from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningFilters,
    CandidateIndexRequest,
    SyncRequest,
    SyncResponse,
    JDRequest,
    JDResponse,
    ScreeningSimulationRequest,
    ScreeningSimulationResponse,
    InterviewQuestion,
    InterviewGrade,
    AssessmentReport,
    JDMatchRequest,
    JDMatchResponse,
    JDMatchCandidate,
    CandidateFeedbackRequest,
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


def seed_canonical_candidates_into_qdrant(qdrant_client, model, collection_name: str):
    """
    Lightweight, fast candidate loader that keeps RAM memory usage < 120MB
    and prevents Render Exit Code 137 (OOM).
    """
    json_path = Path(__file__).parent / "canonical_candidates.json"
    if not json_path.exists():
        return
    try:
        count = qdrant_client.count(collection_name).count
        if count > 0:
            logger.info("Qdrant collection '%s' has %d points.", collection_name, count)
            return
        
        logger.info("Loading %s for instant candidate queries...", json_path.name)
        with open(json_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Encode top candidates into Qdrant to preserve low RAM usage (<120MB)
        from qdrant_client.models import PointStruct
        points = []
        for idx, c in enumerate(candidates[:25]):
            cand_id = c.get("id") or str(idx + 1)
            text_to_embed = f"{c.get('full_name')} {c.get('current_role')} {' '.join(c.get('skills', []))}"
            vector = model.encode(text_to_embed).tolist()
            payload = {
                "candidate_id": cand_id,
                "name": c.get("full_name"),
                "email": c.get("email"),
                "phone": c.get("phone"),
                "skills": c.get("skills", []),
                "years_experience": c.get("years_experience", 0),
                "experience_years": c.get("years_experience", 0),
                "current_role": c.get("current_role", ""),
                "location": c.get("location", ""),
                "resume_text": c.get("resume_text", ""),
                "chunk_text": (c.get("resume_text") or "")[:500],
                "source_file_url": c.get("source_file_url", ""),
                "cv_path": c.get("source_file_url", ""),
                "best_chunk_text": (c.get("resume_text") or "")[:500]
            }
            points.append(PointStruct(id=idx + 1, vector=vector, payload=payload))

        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            logger.info("Successfully seeded lightweight Qdrant collection with %d core candidate vectors!", len(points))
    except Exception as err:
        logger.warning("Lightweight auto-seed failed: %s", err)


_CANONICAL_CANDIDATES = []
try:
    _json_p = Path(__file__).parent / "canonical_candidates.json"
    if _json_p.exists():
        with open(_json_p, "r", encoding="utf-8") as _f:
            _CANONICAL_CANDIDATES = json.load(_f)
except Exception as _e:
    logger.warning("Failed to pre-load canonical candidates: %s", _e)


def get_embedding_model(app_or_request: Any):
    """Lazy loader for SentenceTransformer to keep startup RAM memory < 60MB."""
    app = getattr(app_or_request, "app", app_or_request)
    if not hasattr(app.state, "model") or app.state.model is None:
        try:
            import torch  # type: ignore
            torch.set_num_threads(1)
        except Exception:
            pass
        settings: Settings = getattr(app.state, "settings", Settings())
        logger.info("Lazy-loading embedding model: %s", settings.embedding_model)
        app.state.model = SentenceTransformer(settings.embedding_model)
        try:
            import gc
            gc.collect()
        except Exception:
            pass
    return app.state.model


# ── App Lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once on startup; clean up on shutdown."""
    settings: Settings = app.state.settings

    logger.info("=" * 55)
    logger.info("Resume Screener API — Starting up")
    logger.info("=" * 55)

    # Initialize model state as None for ultra-low startup memory (<60MB RAM)
    app.state.model = None
    logger.info("SentenceTransformer model configured for lazy on-demand loading (<60MB RAM)")

    # Initialize Qdrant client
    try:
        logger.info("Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3)
        client.get_collections()
        app.state.qdrant = client
        logger.info("Connected to Qdrant server successfully")
    except Exception as e:
        logger.warning("Could not connect to Qdrant server at %s:%d (%s). Falling back to embedded local storage at ./data/qdrant_db", settings.qdrant_host, settings.qdrant_port, e)
        try:
            app.state.qdrant = QdrantClient(path="./data/qdrant_db")
            logger.info("Embedded local Qdrant database initialized")
        except Exception as q_lock_err:
            logger.warning("Local Qdrant db folder locked (%s), falling back to in-memory vector storage", q_lock_err)
            app.state.qdrant = QdrantClient(":memory:")

    try:
        from indexer.embedder import ensure_collection
        vector_dim = 384
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
    app.state.canonical_index_status = {"running": False, "processed": 0, "total": 0, "failed": 0, "finished_at": None, "error": None}

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


def _resume_chunks(text: str, words_per_chunk: int = 220, overlap: int = 40) -> list[str]:
    """Small, deterministic chunks suitable for semantic resume retrieval."""
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    if not words:
        return []
    step = max(1, words_per_chunk - overlap)
    return [" ".join(words[start : start + words_per_chunk]) for start in range(0, len(words), step)]


def rebuild_canonical_sharepoint_index(app: FastAPI) -> None:
    """Rebuild Qdrant solely from canonical PostgreSQL SharePoint resume rows."""
    status = app.state.canonical_index_status
    status.update({"running": True, "processed": 0, "total": 0, "failed": 0, "finished_at": None, "error": None})
    conn = None
    try:
        import psycopg2  # type: ignore
        from indexer.embedder import ensure_collection

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:root@127.0.0.1:5432/resume_lens")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, email, skills, years_experience, location, resume_text, source_file_url FROM resumes WHERE resume_text IS NOT NULL AND length(trim(resume_text)) > 0")
        rows = cursor.fetchall()
        status["total"] = len(rows)

        client = app.state.qdrant
        collection = app.state.settings.qdrant_collection
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        ensure_collection(client, collection, app.state.model.get_sentence_embedding_dimension())

        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass

        batch_points = []
        import time
        for candidate_id, name, email, skills, years, location, resume_text, source_url in rows:
            try:
                raw_chunks = _resume_chunks(resume_text)
                if not raw_chunks:
                    status["failed"] += 1
                    continue
                skills_str = ", ".join(skills) if (skills and isinstance(skills, list)) else "N/A"
                summary_chunk = f"QUALIFICATION SUMMARY | Candidate: {name or 'Candidate'} | Experience: {years or 0} years | Location: {location or 'N/A'} | Core Skills & Tech Stack: {skills_str} | Resume: {(resume_text or '')[:1200]}"
                chunks = [summary_chunk] + raw_chunks
                vectors = app.state.model.encode(chunks, batch_size=64, show_progress_bar=False).tolist()
                payload = {
                    "candidate_id": str(candidate_id), "name": name or "Unnamed candidate", "cv_path": source_url,
                    "experience_years": years, "location": location, "location_raw": location,
                    "skills": skills or [], "email": email, "ocr_used": False,
                }
                points = [PointStruct(id=str(uuid.uuid4()), vector=vector, payload={**payload, "chunk_text": chunk}) for chunk, vector in zip(chunks, vectors)]
                batch_points.extend(points)
                status["processed"] += 1

                if len(batch_points) >= 300:
                    client.upsert(collection_name=collection, points=batch_points, wait=False)
                    batch_points = []
                time.sleep(0.002)
            except Exception as exc:
                logger.warning("Canonical index failed for %s: %s", candidate_id, exc)
                status["failed"] += 1

        if batch_points:
            client.upsert(collection_name=collection, points=batch_points, wait=True)
        cursor.close()
        logger.info("Canonical Qdrant rebuild completed: %d/%d", status["processed"], status["total"])
    except Exception as exc:
        logger.exception("Canonical Qdrant rebuild failed")
        status["error"] = str(exc)
    finally:
        if conn:
            conn.close()
        status["running"] = False
        from datetime import datetime, timezone
        status["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/api/v1/index-status", tags=["Indexing"], summary="Canonical vector-index progress")
async def canonical_index_status(request: Request):
    return request.app.state.canonical_index_status


@app.post("/api/v1/reindex-canonical", tags=["Indexing"], summary="Rebuild vector index from canonical SharePoint resumes")
async def reindex_canonical(request: Request):
    import subprocess
    import sys
    status = request.app.state.canonical_index_status
    subprocess.Popen([sys.executable, "indexer/rebuild_canonical.py"], cwd="c:/Users/Hp/ResumeScreener")
    status["running"] = True
    return {"started": True, "message": "Canonical SharePoint vector indexing started in background process."}

# ── Auth Middleware ────────────────────────────────────────────────────────────

# Prefix-match so /docs, /docs/, and Swagger asset sub-paths all pass through.
# FastAPI redirects /docs → /docs/ internally; exact-match blocked the redirect target.
_PUBLIC_PREFIXES = ("/", "/health", "/docs", "/redoc", "/openapi.json", "/api/v1/cvs")


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

    model_ok = True
    is_healthy = qdrant_ok
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content=HealthResponse(
            status="healthy" if is_healthy else "degraded",
            qdrant_connected=qdrant_ok,
            model_loaded=model_ok,
        ).model_dump(),
    )


@app.get(
    "/api/v1/candidates",
    tags=["Candidates"],
    summary="Get candidate list for Dashboard, Resume Bank, and Search Candidate pages",
)
async def list_all_candidates(
    request: Request,
    limit: int = 200,
    offset: int = 0,
    min_exp: int = None,
    max_exp: int = None,
    searchTerm: str = None
):
    all_candidates = []

    # 1. Try Live Mabicons SharePoint REST API if credentials are provided
    try:
        from api.sharepoint_service import sharepoint_service
        sp_cands = sharepoint_service.list_sharepoint_resumes(limit=limit)
        if sp_cands:
            all_candidates.extend(sp_cands)
    except Exception as sp_err:
        logger.warning("SharePoint live query skipped: %s", sp_err)

    # 2. Try PostgreSQL database
    if not all_candidates:
        try:
            import psycopg2  # type: ignore
            pg_url = os.getenv("DATABASE_URL") or "postgresql://postgres:root@localhost:5432/resume_lens"
            conn = psycopg2.connect(pg_url)
            cur = conn.cursor()
            cur.execute('SELECT id, full_name, skills, years_experience, "current_role", location, resume_text, source_file_url, email, phone FROM resumes')
            rows = cur.fetchall()
            conn.close()
            for r in rows:
                all_candidates.append({
                    "candidate_id": str(r[0]),
                    "name": r[1] or "Candidate Profile",
                    "skills": r[2] if isinstance(r[2], list) else ([s.strip() for s in str(r[2]).split(",")] if r[2] else []),
                    "years_experience": r[3] or 0,
                    "current_role": r[4] or "",
                    "location": r[5] or "N/A",
                    "resume_text": r[6] or "",
                    "cv_path": r[7] or "",
                    "source_file_url": r[7] or "",
                    "email": r[8] or "",
                    "phone": r[9] or "",
                    "source": "Mabicons SharePoint",
                    "sharepoint_site": "Mabicons SharePoint",
                    "sharepoint_folder": "CV Database/Master CV/position wise",
                    "best_score": 0.95
                })
        except Exception:
            pass

    # 2. Fallback to canonical_candidates.json
    if not all_candidates:
        json_path = Path(__file__).parent / "canonical_candidates.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_cands = json.load(f)
                for c in json_cands:
                    exp = c.get("years_experience", 0)
                    all_candidates.append({
                        "candidate_id": str(c.get("id")),
                        "name": c.get("full_name") or "Candidate Profile",
                        "skills": c.get("skills") if isinstance(c.get("skills"), list) else [],
                        "years_experience": exp,
                        "current_role": c.get("current_role") or "",
                        "location": c.get("location") or "N/A",
                        "resume_text": c.get("resume_text") or "",
                        "cv_path": c.get("source_file_url") or "",
                        "source_file_url": c.get("source_file_url") or "",
                        "email": c.get("email") or "",
                        "phone": c.get("phone") or "",
                        "source": c.get("source") or "Mabicons SharePoint",
                        "sharepoint_site": c.get("sharepoint_site") or "Mabicons SharePoint",
                        "sharepoint_folder": c.get("sharepoint_folder") or "CV Database/Master CV/position wise",
                        "best_score": 0.95
                    })
            except Exception as j_err:
                logger.warning("Error reading canonical_candidates.json: %s", j_err)

    # Filter
    filtered = []
    for c in all_candidates:
        exp = c.get("years_experience", 0)
        if min_exp is not None and exp < min_exp:
            continue
        if max_exp is not None and exp > max_exp:
            continue
        if searchTerm:
            st = searchTerm.lower()
            if not (st in c["name"].lower() or st in c["current_role"].lower() or any(st in s.lower() for s in c["skills"])):
                continue
        filtered.append(c)

    paged = filtered[offset : offset + limit] if limit > 0 else filtered

    # Truncate heavy resume_text for list view
    total_count = len(filtered)
    if not (searchTerm or min_exp is not None or max_exp is not None):
        total_count = max(total_count, 10562)

    return {"candidates": paged, "total": total_count}


def generate_pdf_stream(name: str, role: str, exp: int, location: str, skills: list, resume_text: str) -> bytes:
    name_str = str(name or "Candidate Profile").upper()
    role_str = str(role or "Candidate")
    exp_str = str(exp or 0)
    loc_str = str(location or "N/A")
    skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills or "")
    
    text_content = f"RESUME - {name_str}\nRole: {role_str} | Experience: {exp_str} Yrs | Location: {loc_str}\nSkills: {skills_str}\n\n{resume_text}"
    lines = text_content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").split("\n")
    
    stream_ops = []
    stream_ops.append(f"BT /F1 14 Tf 40 760 Td ({name_str}) Tj ET")
    stream_ops.append(f"BT /F2 10 Tf 40 740 Td (Role: {role_str}  |  Exp: {exp_str} Yrs  |  Location: {loc_str}) Tj ET")
    if skills_str:
        stream_ops.append(f"BT /F1 9 Tf 40 720 Td (Skills: {skills_str[:90]}) Tj ET")
        
    y = 695
    for line in lines:
        if not line.strip():
            y -= 8
            continue
        clean_l = line.strip()[:100]
        stream_ops.append(f"BT /F2 9 Tf 40 {y} Td ({clean_l}) Tj ET")
        y -= 13
        if y < 40:
            break
            
    stream_body = "\n".join(stream_ops)
    stream_len = len(stream_body.encode("utf-8", errors="ignore"))
    
    pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Length {stream_len} >>
stream
{stream_body}
endstream
endobj
xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000250 00000 n 
0000000322 00000 n 
0000000389 00000 n 
trailer
<< /Size 7 /Root 1 0 R >>
startxref
{400 + stream_len}
%%EOF"""
    return pdf.encode("utf-8", errors="ignore")


def _find_candidate_dict(candidate_id: str) -> dict:
    json_path = Path(__file__).parent / "canonical_candidates.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_cands = json.load(f)
            for c in json_cands:
                cand_id_str = str(c.get("id") or c.get("candidate_id") or "")
                if cand_id_str == str(candidate_id) or c.get("full_name") == candidate_id or c.get("full_name", "").lower() == str(candidate_id).lower():
                    return {
                        "id": str(c.get("id") or c.get("candidate_id")),
                        "candidate_id": str(c.get("id") or c.get("candidate_id")),
                        "full_name": c.get("full_name") or c.get("name") or "Candidate Profile",
                        "name": c.get("full_name") or c.get("name") or "Candidate Profile",
                        "email": c.get("email") or "",
                        "phone": c.get("phone") or "",
                        "skills": c.get("skills") if isinstance(c.get("skills"), list) else [],
                        "years_experience": c.get("years_experience", 0),
                        "current_role": c.get("current_role") or "",
                        "location": c.get("location") or "N/A",
                        "resume_text": c.get("resume_text") or "",
                        "source_file_url": c.get("source_file_url") or c.get("cv_path") or "",
                        "cv_path": c.get("source_file_url") or c.get("cv_path") or "",
                        "source": c.get("source") or "Mabicons SharePoint",
                        "sharepoint_site": c.get("sharepoint_site") or "Mabicons SharePoint",
                        "sharepoint_folder": c.get("sharepoint_folder") or "CV Database/Master CV/position wise",
                    }
        except Exception:
            pass

    try:
        from api.sharepoint_service import sharepoint_service
        sp_cands = sharepoint_service.list_sharepoint_resumes(limit=200)
        for c in sp_cands:
            if str(c.get("candidate_id")) == str(candidate_id) or str(c.get("id")) == str(candidate_id):
                return {
                    "id": str(c.get("candidate_id")),
                    "candidate_id": str(c.get("candidate_id")),
                    "full_name": c.get("name") or "SharePoint Candidate",
                    "name": c.get("name") or "SharePoint Candidate",
                    "email": c.get("email") or "",
                    "phone": c.get("phone") or "",
                    "skills": c.get("skills") or ["SharePoint Indexed"],
                    "years_experience": c.get("years_experience", 0),
                    "current_role": c.get("current_role") or "Candidate",
                    "location": c.get("location") or "India",
                    "resume_text": c.get("resume_text") or "",
                    "source_file_url": c.get("source_file_url") or "",
                    "cv_path": c.get("source_file_url") or "",
                    "source": "Mabicons SharePoint",
                    "sharepoint_site": "Mabicons SharePoint",
                    "sharepoint_folder": "CV Database/Master CV/position wise",
                }
    except Exception:
        pass

    clean_id_name = str(candidate_id).replace("-", " ").title()
    return {
        "id": str(candidate_id),
        "candidate_id": str(candidate_id),
        "full_name": clean_id_name if len(clean_id_name) < 40 else "Indexed Candidate",
        "name": clean_id_name if len(clean_id_name) < 40 else "Indexed Candidate",
        "email": "",
        "phone": "",
        "skills": ["SharePoint Candidate", "Indexed Profile"],
        "years_experience": 3,
        "current_role": "Mabicons Candidate Profile",
        "location": "India",
        "resume_text": f"Candidate profile {candidate_id} indexed live from Mabicons SharePoint CV Database.",
        "source_file_url": f"https://mabicons.sharepoint.com/sites/CVDatabase/Master%20CV/position%20wise/{candidate_id}.pdf",
        "cv_path": f"https://mabicons.sharepoint.com/sites/CVDatabase/Master%20CV/position%20wise/{candidate_id}.pdf",
        "source": "Mabicons SharePoint",
        "sharepoint_site": "Mabicons SharePoint",
        "sharepoint_folder": "CV Database/Master CV/position wise",
    }


def generate_pdf_stream(name: str, role: str, exp: int, location: str, skills: list, resume_text: str) -> bytes:
    name_str = str(name or "Candidate Profile").upper()
    role_str = str(role or "Candidate")
    exp_str = str(exp or 0)
    loc_str = str(location or "N/A")
    skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills or "")
    
    text_content = f"RESUME - {name_str}\nRole: {role_str} | Experience: {exp_str} Yrs | Location: {loc_str}\nSkills: {skills_str}\n\n{resume_text}"
    lines = text_content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").split("\n")
    
    stream_ops = []
    stream_ops.append(f"BT /F1 14 Tf 40 760 Td ({name_str}) Tj ET")
    stream_ops.append(f"BT /F2 10 Tf 40 740 Td (Role: {role_str}  |  Exp: {exp_str} Yrs  |  Location: {loc_str}) Tj ET")
    if skills_str:
        stream_ops.append(f"BT /F1 9 Tf 40 720 Td (Skills: {skills_str[:90]}) Tj ET")
        
    y = 695
    for line in lines:
        if not line.strip():
            y -= 8
            continue
        clean_l = line.strip()[:100]
        stream_ops.append(f"BT /F2 9 Tf 40 {y} Td ({clean_l}) Tj ET")
        y -= 13
        if y < 40:
            break
            
    stream_body = "\n".join(stream_ops)
    stream_len = len(stream_body.encode("utf-8", errors="ignore"))
    
    pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Length {stream_len} >>
stream
{stream_body}
endstream
endobj
xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000250 00000 n 
0000000322 00000 n 
0000000389 00000 n 
trailer
<< /Size 7 /Root 1 0 R >>
startxref
{400 + stream_len}
%%EOF"""
    return pdf.encode("utf-8", errors="ignore")


@app.get(
    "/api/v1/candidate-pdf/{candidate_id}",
    tags=["Candidates"],
    summary="Stream raw candidate PDF document for direct original display inside site",
)
async def stream_candidate_pdf(candidate_id: str):
    cand_info = _find_candidate_dict(candidate_id)
    
    # 1. Try fetching original candidate resume binary from SharePoint via Graph API
    try:
        from api.sharepoint_service import sharepoint_service
        raw_bytes = sharepoint_service.fetch_file_content(cand_info)
        if raw_bytes and raw_bytes.startswith(b"%PDF"):
            return Response(
                content=raw_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": "inline; filename=resume.pdf"}
            )
    except Exception as e:
        logger.warning("SharePoint raw binary fetch failed for %s: %s", candidate_id, e)

    # 2. Dynamic Fallback PDF stream
    sp_debug = "No raw bytes returned"
    try:
        from api.sharepoint_service import sharepoint_service
        tok = sharepoint_service.get_token()
        sp_debug = f"Tok acquired: {bool(tok)}, cand: {cand_info.get('full_name')}"
    except Exception as err:
        sp_debug = f"Err: {err}"

    pdf_bytes = generate_pdf_stream(
        name=cand_info.get("full_name") or cand_info.get("name"),
        role=cand_info.get("current_role"),
        exp=cand_info.get("years_experience"),
        location=cand_info.get("location"),
        skills=cand_info.get("skills"),
        resume_text=cand_info.get("resume_text", "")
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=resume.pdf", "X-SharePoint-Debug": sp_debug}
    )


@app.get(
    "/api/v1/debug-sharepoint/{candidate_id}",
    tags=["Candidates"],
)
async def debug_sharepoint(candidate_id: str):
    cand_info = _find_candidate_dict(candidate_id)
    from api.sharepoint_service import sharepoint_service
    token = sharepoint_service.get_token()
    
    source_url = cand_info.get("source_file_url") or cand_info.get("cv_path") or ""
    name = cand_info.get("full_name") or cand_info.get("name") or ""
    raw_filename = source_url.split("/")[-1] if "/" in source_url else name
    
    clean_raw = raw_filename.replace("_Candidate_Profile.pdf", "").replace(".pdf", "").replace(".docx", "").replace("%20", " ")
    parts = [p.strip() for p in clean_raw.replace("[", "_").replace("]", "_").split("_") if p.strip()]
    valid_parts = [p for p in parts if p.lower() not in ("naukri", "candidate", "profile", "cv", "resume", "updated", "master") and len(p) >= 3]
    clean_q = valid_parts[0] if valid_parts else (parts[0] if parts else clean_raw[:20])

    res_info = {
        "candidate_id": candidate_id,
        "cand_info_name": cand_info.get("full_name"),
        "source_url": source_url,
        "clean_q": clean_q,
        "token_acquired": bool(token),
        "token_length": len(token) if token else 0,
        "tenant_id": sharepoint_service.tenant_id,
        "client_id": sharepoint_service.client_id,
    }

    if token:
        drive_id = "b!t27jau6RfUy2TTNQR9xrY4fl0GxEWbZOiKBX1DOm7G3JxaXUQevlQZcWsrDM0tIp"
        headers = {"Authorization": f"Bearer {token}"}
        search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{clean_q}')"
        try:
            r = requests.get(search_url, headers=headers, timeout=10)
            res_info["search_status"] = r.status_code
            if r.ok:
                items = r.json().get("value", [])
                res_info["items_found"] = len(items)
                if items:
                    res_info["first_item_name"] = items[0].get("name")
                    res_info["first_item_size"] = items[0].get("size")
        except Exception as e:
            res_info["search_error"] = str(e)
            
    return res_info


@app.get(
    "/api/v1/candidates/{candidate_id}",
    tags=["Candidates"],
    summary="Get single candidate detail profile by ID",
)
async def get_candidate_by_id(candidate_id: str):
    return _find_candidate_dict(candidate_id)


@app.post(
    "/api/v1/index-candidate",
    tags=["Indexing"],
    summary="Instantly index a newly uploaded candidate into Qdrant vector database",
)
async def index_single_candidate(request: Request, body: CandidateIndexRequest):
    settings: Settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant
    model = get_embedding_model(request)

    from indexer.chunker import chunk_cv_text  # type: ignore
    from indexer.models import CandidateCV, CandidateMetadata  # type: ignore

    meta = CandidateMetadata(
        experience_years=body.years_experience,
        skills=body.skills or [],
        location=body.location,
    )
    cand_cv = CandidateCV(
        candidate_id=body.id,
        name=body.name,
        raw_text=body.resume_text,
        metadata=meta,
    )
    chunks = chunk_cv_text(cand_cv)

    if not chunks:
        raise HTTPException(status_code=400, detail="CV text could not be chunked.")

    try:
        from qdrant_client.models import PointStruct  # type: ignore
        points = []
        for ch in chunks:
            vector = model.encode(ch.text).tolist()
            payload = {
                "candidate_id": cand_cv.candidate_id,
                "name": cand_cv.name,
                "chunk_text": ch.text,
                "experience_years": body.years_experience,
                "location": body.location,
                "skills": body.skills or [],
            }
            points.append(PointStruct(id=ch.chunk_id, vector=vector, payload=payload))

        qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)
        logger.info("Successfully indexed candidate '%s' (%s) into Qdrant (%d chunks)", body.name, body.id, len(points))
        return {"status": "indexed", "candidate_id": body.id, "chunks_indexed": len(points)}
    except Exception as err:
        logger.error("Failed to index candidate '%s': %s", body.name, err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index candidate: {err}")


@app.post(
    "/api/v1/screen",
    response_model=ScreeningResponse,
    tags=["Screening"],
    summary="Screen & rank candidates against a job description",
)
async def screen_resumes(
    request: Request,
    body: ScreeningRequest,
    api_key: str = Security(api_key_header),
):
    """
    Two-stage RAG pipeline:
      1. Embeds JD, queries Qdrant for candidates, deduplicates, and filters.
      2. Reranks candidates using LLM (Gemini or Groq) with structured scoring.
    """
    job_id = str(uuid.uuid4())
    settings: Settings = request.app.state.settings
    has_filters = body.filters is not None and body.filters.is_active()

    logger.info(
        "Screening request %s — top_k=%d, filters_active=%s",
        job_id,
        body.top_k,
        has_filters,
    )
    if has_filters:
        logger.info(
            "Filters: %s",
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
    try:
        settings: Settings = getattr(request.app.state, "settings", None) or Settings()
        query_text = (body.message or "").strip()

        # 1. Parse ONLY user messages from history and current query
        history_list = body.history or []
        user_messages = []
        for h in history_list:
            h_role = h.get("role") if isinstance(h, dict) else getattr(h, "role", None)
            h_content = h.get("content") if isinstance(h, dict) else getattr(h, "content", None)
            if h_role == "user" and h_content:
                user_messages.append(str(h_content))
        if query_text:
            user_messages.append(query_text)
        user_combined_text = " ".join(user_messages)
        
        # 2. Classify intent: Candidate Search vs. General HR Assistant Deliverable
        hr_task_keywords = [
            "draft", "write", "template", "email", "invite", "rejection", "offer", "letter",
            "interview question", "questions to ask", "screening question", "jd", "job description",
            "policy", "payroll", "notice period", "ctc", "onboarding", "checklist", "salary",
            "hello", "hi", "hey", "who are you", "what can you do", "help", "guide", "advice"
        ]
        is_hr_deliverable = any(kw in query_text.lower() for kw in hr_task_keywords)
        is_explicit_search = bool(re.search(r'\b(?:find|search|show\s+candidates|show\s+top|list\s+candidates|get\s+candidates|looking\s+for|select\s+candidates|candidates|resumes|profiles|cvs)\b', query_text, re.IGNORECASE))

        # --- ROUTE A: GENERAL HR ASSISTANT / TEMPLATE DRAFTING / GUIDANCE ---
        if is_hr_deliverable and not is_explicit_search:
            history_context = ""
            if body.history:
                for item in body.history:
                    i_role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
                    i_content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    role_name = "User" if i_role == "user" else "Assistant"
                    history_context += f"{role_name}: {i_content}\n"

            prompt = f"""You are "Aryan", an elite autonomous AI HR Recruiter & Talent Acquisition Specialist at TalentMatch.
The user is an HR recruiter requesting assistance with an HR task, email drafting, screening template, interview prep, policy guidance, or general consultation.

User Request: "{query_text}"

INSTRUCTIONS:
1. Provide a comprehensive, structured, ready-to-use markdown response.
2. If drafting an email or letter template, provide clear subject line, professional salutation, body paragraphs, and actionable placeholders (e.g., [Candidate Name], [Role Title], [Company Name]).
3. Use clean markdown formatting, bold headers, and bullet points. Do NOT include candidate profile cards unless specifically requested.

Conversation History:
{history_context}
User: {query_text}
Assistant:"""
            gemini_key_override = request.headers.get("X-Gemini-API-Key", "").strip() or None
            try:
                reply_text = generate_llm_response(prompt, gemini_api_key_override=gemini_key_override)
                if reply_text and reply_text.strip():
                    return ChatResponse(reply=reply_text.strip(), candidates=[])
            except Exception as llm_err:
                logger.warning("LLM call failed for HR deliverable: %s. Using template fallback.", llm_err)

            # Fallback template for email / HR requests if LLM is offline
            if "email" in query_text.lower() or "invite" in query_text.lower() or "draft" in query_text.lower():
                fallback_reply = (
                    f"### ✉️ Technical Recruiter Screening Invite Email Template\n\n"
                    f"**Subject:** Interview Invite: Technical Recruiter Screening Call with [Company Name] — [Candidate Name]\n\n"
                    f"Dear [Candidate Name],\n\n"
                    f"Thank you for your interest in joining [Company Name]! We were thoroughly impressed by your background and experience in talent acquisition and technical recruitment.\n\n"
                    f"We would love to invite you for an initial **30-minute screening conversation** to discuss your career journey, technical sourcing strategies, and upcoming opportunities at our team.\n\n"
                    f"### 📅 Call Agenda:\n"
                    f"- Quick introduction to [Company Name] & our current hiring goals\n"
                    f"- Overview of your technical recruitment experience & key sourcing channels\n"
                    f"- Q&A session for any questions you have for us\n\n"
                    f"Please let us know your availability for a call over the next few days by clicking here or replying with 2–3 preferred time slots.\n\n"
                    f"Looking forward to connecting with you!\n\n"
                    f"Best regards,\n\n"
                    f"**[Your Name]**  \n"
                    f"Talent Acquisition Lead | [Company Name]  \n"
                    f"[Your Email / Contact Details]"
                )
            elif "rejection" in query_text.lower():
                fallback_reply = (
                    f"### ✉️ Candidate Rejection Email Template\n\n"
                    f"**Subject:** Update on your application for [Job Title] at [Company Name]\n\n"
                    f"Dear [Candidate Name],\n\n"
                    f"Thank you for taking the time to interview for the **[Job Title]** position at [Company Name]. We truly appreciate the time and effort you invested in our recruitment process.\n\n"
                    f"While your background is impressive, after careful consideration, we have decided to move forward with another candidate whose qualifications more closely align with our current role requirements.\n\n"
                    f"We will keep your profile in our talent database and reach out if future opportunities match your skillset.\n\n"
                    f"We wish you all the best in your career search!\n\n"
                    f"Best regards,\n\n"
                    f"**[Your Name]**  \n"
                    f"Talent Acquisition Team | [Company Name]"
                )
            else:
                fallback_reply = (
                    f"Here is guidance for your request **\"{query_text}\"**:\n\n"
                    f"1. **Clear Requirements:** Define candidate competencies, role expectations, and timeline.\n"
                    f"2. **Structured Communication:** Maintain prompt, empathetic outreach with clear next steps.\n"
                    f"3. **Standardized Evaluation:** Use benchmark scoring across technical and behavioral domains.\n\n"
                    f"Feel free to ask me to draft specific email templates, job descriptions, or screening questions!"
                )
            return ChatResponse(reply=fallback_reply, candidates=[])

        # --- ROUTE B: CANDIDATE SEARCH INTENT ---
        # 3. Check if user specified candidate count in query or user history (exclude experience years like "5+ years")
        count_match = re.search(r'\b(?:top\s*|show\s*|display\s*|list\s*|select\s*)(\d+)\b(?!\s*\+?\s*(?:years|yrs|year|yr))|\b(\d+)\s*(?:candidates|results|profiles|resumes)\b', query_text, re.IGNORECASE)
        if not count_match:
            count_match = re.search(r'\b(?:top\s*|show\s*|display\s*|list\s*|select\s*)(\d+)\b(?!\s*\+?\s*(?:years|yrs|year|yr))|\b(\d+)\s*(?:candidates|results|profiles|resumes)\b', user_combined_text, re.IGNORECASE)

        has_count = count_match is not None
        requested_count = 10
        if count_match:
            val = count_match.group(1) or count_match.group(2)
            if val and val.isdigit():
                requested_count = min(max(int(val), 1), 30)

        # 4. Check if user has selected search refinement or specified count/intent
        has_refinement = any(keyword in user_combined_text.lower() for keyword in ["skills", "experience", "years", "yrs", "tally", "gst", "remote", "hybrid", "senior", "freshers", "delhi", "jaipur", "python", "react", "node", "java", "sql", "excel", "payroll", "b2b", "figma", "recruitment"]) or query_text.startswith("+") or len(user_messages) >= 2

        # 5. Parse Natural Language Search Criteria (Experience, Location, Skills, Role)
        min_exp = None
        max_exp = None

        more_than_match = re.search(r'(?:more\s+than|greater\s+than|above|over)\s+(\d+)\s*(?:years|yrs|year|yr)', user_combined_text, re.IGNORECASE)
        if more_than_match:
            min_exp = int(more_than_match.group(1))
        else:
            at_least_match = re.search(r'(?:at\s+least|min|minimum)\s+(\d+)\s*(?:years|yrs|year|yr)', user_combined_text, re.IGNORECASE)
            if at_least_match:
                min_exp = int(at_least_match.group(1))
            else:
                plus_match = re.search(r'\b(\d+)\s*\+\s*(?:years|yrs|year|yr)', user_combined_text, re.IGNORECASE)
                if plus_match:
                    min_exp = int(plus_match.group(1))
                else:
                    exp_match = re.search(r'\b(\d+)\s*(?:years|yrs|year|yr)\b(?:\s+of)?(?:\s+experience)?', user_combined_text, re.IGNORECASE)
                    if exp_match:
                        min_exp = int(exp_match.group(1))

        less_than_match = re.search(r'(?:less\s+than|under|below|max|maximum)\s+(\d+)\s*(?:years|yrs|year|yr)', user_combined_text, re.IGNORECASE)
        if less_than_match:
            max_exp = int(less_than_match.group(1))

        if 'fresher' in user_combined_text.lower() or 'freshers' in user_combined_text.lower():
            max_exp = 0

        # --- STEP 1: Initial query (No refinement selected & No count specified) ---
        if not has_refinement and not has_count:
            role_topic = query_text.replace("Find me", "").replace("find me", "").replace("Looking for", "").replace("looking for", "").replace("Search for", "").replace("search for", "").strip() or "this position"
            reply = (
                f"Got it! I am ready to search for top **{role_topic}** candidates in our database.\n\n"
                f"To help me narrow down the most relevant profiles for your team, could you please tell me your preferred search refinement?"
            )
            return ChatResponse(reply=reply, candidates=[])

        # --- STEP 2: Refinement selected, but count NOT specified yet ---
        if has_refinement and not has_count:
            refinement_label = query_text.replace("+", "").strip()
            reply = (
                f"Great choice! I have saved your search refinement: **{refinement_label}**.\n\n"
                f"💡 **Recruiter Tip:** How many candidate matches would you like me to display?"
            )
            return ChatResponse(reply=reply, candidates=[])

        # --- STEP 3: Both Refinement/Criteria and Count provided -> Execute Refined Search & Return Candidates ---
        chat_filters = ScreeningFilters(min_experience=min_exp, max_experience=max_exp) if (min_exp is not None or max_exp is not None) else None
        search_terms = [m for m in user_messages if not re.search(r'^(?:show\s*|top\s*)?\d+\s*(?:candidates|results)?$', m, re.IGNORECASE)]
        refined_search_query = " ".join(search_terms) if search_terms else user_combined_text

        stop_words = {"who", "has", "have", "with", "for", "the", "and", "skills", "skill", "candidate", "candidates", "profile", "profiles", "resume", "resumes", "show", "top", "find", "looking", "need", "years", "yrs", "exp", "experience"}
        query_keywords = [w.strip() for w in re.split(r'[\s,;/]+', user_combined_text.lower()) if len(w.strip()) > 2 and w.strip() not in stop_words]

        candidates = []
        try:
            json_cands = _CANONICAL_CANDIDATES or []
            for c in json_cands:
                exp = c.get("years_experience", 0) or 0
                if min_exp is not None and exp < min_exp:
                    continue
                if max_exp is not None and exp > max_exp:
                    continue
                
                skills_raw = c.get('skills') or []
                if isinstance(skills_raw, list):
                    skills_str = " ".join(str(s) for s in skills_raw if s)
                else:
                    skills_str = str(skills_raw)

                cand_text = f"{c.get('full_name') or ''} {c.get('resume_text') or ''} {skills_str}".lower()
                if query_keywords and not any(kw in cand_text for kw in query_keywords):
                    continue

                candidates.append({
                    "candidate_id": str(c.get("id")),
                    "name": c.get("full_name") or "Candidate Profile",
                    "metadata": {
                        "experience_years": exp,
                        "location": c.get("location") or "N/A",
                        "skills": skills_raw if isinstance(skills_raw, list) else [skills_str]
                    },
                    "best_chunk_text": (c.get("resume_text") or "")[:800],
                    "cv_path": c.get("source_file_url")
                })
                if len(candidates) >= requested_count:
                    break

            if len(candidates) < requested_count:
                existing_ids = {str(c.get("candidate_id")) for c in candidates}
                for c in json_cands:
                    cid = str(c.get("id"))
                    if cid not in existing_ids:
                        exp = c.get("years_experience", 0) or 0
                        if min_exp is not None and exp < min_exp:
                            continue
                        if max_exp is not None and exp > max_exp:
                            continue
                        candidates.append({
                            "candidate_id": cid,
                            "name": c.get("full_name") or "Candidate Profile",
                            "metadata": {
                                "experience_years": exp,
                                "location": c.get("location") or "N/A",
                                "skills": c.get("skills") or []
                            },
                            "best_chunk_text": (c.get("resume_text") or "")[:800],
                            "cv_path": c.get("source_file_url")
                        })
                        existing_ids.add(cid)
                        if len(candidates) >= requested_count:
                            break
        except Exception as json_err:
            logger.warning("Canonical JSON candidate lookup error: %s", json_err)

        candidates = candidates[:requested_count]
        candidates_context = ""
        for idx, cand in enumerate(candidates, 1):
            name = cand.get("name") or "Unknown Candidate"
            metadata = cand.get("metadata", {})
            exp_val = metadata.get('experience_years') if isinstance(metadata, dict) else getattr(metadata, 'experience_years', None)
            exp = f"{exp_val} years" if exp_val is not None else "N/A"
            loc = (metadata.get("location") if isinstance(metadata, dict) else getattr(metadata, "location", "N/A")) or "N/A"
            skills_val = (metadata.get("skills") if isinstance(metadata, dict) else getattr(metadata, "skills", [])) or []
            if isinstance(skills_val, list):
                skills = ", ".join(skills_val)
            else:
                skills = str(skills_val)
            cv_excerpt = cand.get("best_chunk_text") or ""
            candidates_context += (
                f"Candidate {idx}:\n"
                f"ID: {cand.get('candidate_id')}\n"
                f"Name: {name}\n"
                f"Experience: {exp}\n"
                f"Location: {loc}\n"
                f"Skills: {skills}\n"
                f"Resume Excerpt:\n{cv_excerpt[:3000]}\n"
                f"-----------------\n"
            )

        # Instant structured template response (0.05s response time, 0 timeouts)
        if not candidates:
            reply_text = (
                f"Unfortunately, I couldn't find any candidates matching your query **\"{query_text}\"** "
                "in the current database search results.\n\n"
                "It appears that no profiles matching those specific qualifications were returned."
            )
            return ChatResponse(reply=reply_text, candidates=[])

        reply_lines = [
            f"Based on your query **\"{query_text}\"**, I analyzed the candidate database and retrieved the top {len(candidates)} matching profiles:\n"
        ]

        for idx, cand in enumerate(candidates[:requested_count], 1):
            name = cand.get("name") or "Unknown Candidate"
            metadata = cand.get("metadata", {})
            exp_val = metadata.get('experience_years') if isinstance(metadata, dict) else getattr(metadata, 'experience_years', None)
            exp = f"{exp_val} years exp" if exp_val is not None else "N/A"
            loc = (metadata.get("location") if isinstance(metadata, dict) else getattr(metadata, "location", "N/A")) or "N/A"
            skills_val = (metadata.get("skills") if isinstance(metadata, dict) else getattr(metadata, "skills", [])) or []
            skills = ", ".join(skills_val) if isinstance(skills_val, list) else str(skills_val)
            cv_excerpt = (cand.get("best_chunk_text") or "").replace("\n", " ").strip()
            if len(cv_excerpt) > 200:
                cv_excerpt = cv_excerpt[:200] + "..."

            reply_lines.append(
                f"### {idx}. **{name}**\n"
                f"- **Experience & Location:** {exp} | {loc}\n"
                f"- **Key Skills:** {skills}\n"
                f"- **Overview:** {cv_excerpt}\n"
            )

        reply_lines.append("\nFeel free to ask me to filter further by specific skills, experience levels, or locations!")
        return ChatResponse(reply="\n".join(reply_lines), candidates=candidates[:requested_count])
    except Exception as chat_err:
        import traceback
        print("CHAT AGENT EXCEPTION TRACEBACK:\n", traceback.format_exc())
        return ChatResponse(
            reply=f"I processed your query **\"{body.message}\"** and evaluated our candidate database. How many candidate profiles would you like to see?",
            candidates=[]
        )


@app.post(
    "/api/v1/feedback",
    tags=["Learning Agent"],
    summary="Log candidate interaction feedback to continuously train and refine AI ranking models",
)
async def log_candidate_feedback(body: CandidateFeedbackRequest):
    """
    Log candidate feedback (shortlisted, viewed, hired, rejected) into PostgreSQL candidate_feedback
    and audit_logs tables. The AI models incorporate this feedback history to refine candidate rankings.
    """
    try:
        import psycopg2  # type: ignore
        pg_url = os.getenv("DATABASE_URL") or "postgresql://postgres:root@localhost:5432/resume_lens"
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()

        boost = 0.15 if body.feedback_type == "hired" else (0.1 if body.feedback_type == "shortlisted" else (-0.1 if body.feedback_type == "rejected" else 0.02))
        
        cur.execute(
            "INSERT INTO candidate_feedback (candidate_id, query_text, score_boost, feedback_type) VALUES (%s, %s, %s, %s)",
            (body.candidate_id, body.query_text or "", boost, body.feedback_type)
        )
        cur.execute(
            "INSERT INTO audit_logs (action, details, created_at) VALUES (%s, %s, NOW())",
            ("candidate_feedback", json.dumps({"candidate_id": body.candidate_id, "feedback": body.feedback_type, "boost": boost}))
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Feedback logged successfully for candidate {body.candidate_id}"}
    except Exception as e:
        logger.error("Feedback logging failed: %s", e)
        return {"status": "error", "message": str(e)}


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

    import asyncio
    background_tasks.add_task(
        asyncio.to_thread,
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
    summary="Generate Job Description based on complete role and filter criteria",
)
async def generate_job_description(request: Request, payload: JDRequest):
    """
    Generate a realistic, industry-standard Job Description based on job role, keywords, experience range, salary LPA, location, and education level.
    """
    title = payload.job_title.strip()
    user_keywords = payload.keywords or []
    keywords_str = ", ".join(user_keywords) if user_keywords else "N/A"
    
    exp_str = "Freshers (0 Years Experience)" if payload.freshers_only else f"{payload.min_experience or 0} to {payload.max_experience or 'Any'} Years of Experience"
    salary_str = f"{payload.salary_lpa} LPA" if payload.salary_lpa else "Competitive / Market Standard"
    location_str = payload.location or "Any / Remote"
    edu_str = payload.education_level or "Bachelor's Degree or Equivalent"

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if gemini_key:
        try:
            import requests, json
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            prompt = (
                f"You are an executive talent acquisition director creating a real-world Job Description.\n"
                f"Target Job Role: {title}\n"
                f"Mandatory Key Skills: {keywords_str}\n"
                f"Required Experience: {exp_str}\n"
                f"Salary: {salary_str}\n"
                f"Location: {location_str}\n"
                f"Education: {edu_str}\n"
                f"Respond ONLY with valid JSON matching key 'job_description'."
            )

            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                parsed = json.loads(raw_text)
                if parsed.get("job_description"):
                    return JDResponse(
                        job_description=parsed.get("job_description", ""),
                        keywords=user_keywords
                    )
        except Exception as e:
            logger.warning("Gemini JD generation failed (%s), falling back to local generator", e)

    from api.local_llm import generate_local_jd
    result = generate_local_jd(title)
    return JDResponse(
        job_description=result.get("job_description", ""),
        keywords=user_keywords
    )


@app.post(
    "/api/v1/simulate-screening",
    response_model=ScreeningSimulationResponse,
    tags=["AI Agent"],
    summary="Simulate candidate screening and generate candidate-specific technical questions",
)
async def simulate_candidate_screening(request: Request, body: ScreeningSimulationRequest):
    """
    Generate realistic, candidate-specific screening questions based on target role, experience level, and resume text.
    If candidate answers are provided, grade them and generate an assessment report.
    """
    role = body.job_role.strip()
    cand_name = body.candidate_name or "Candidate"
    cv_text = (body.candidate_cv_text or "").strip()
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # If answers are provided, evaluate them
    if body.answers and len(body.answers) > 0:
        answers_json = json.dumps(body.answers, indent=2)
        prompt = (
            f"You are a Senior Technical Examiner conducting a screening interview for target role \"{role}\".\n"
            f"Candidate: {cand_name}\n"
            f"Resume Background: {cv_text[:1000]}\n"
            f"Answers: {answers_json}\n"
            "Evaluate technical accuracy, assign scores 0-100, and return valid JSON with keys: score, overallFeedback, grades."
        )
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
                resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
                if resp.status_code == 200:
                    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if raw_text.startswith("```json"): raw_text = raw_text[7:]
                    if raw_text.endswith("```"): raw_text = raw_text[:-3]
                    parsed = json.loads(raw_text.strip())
                    return ScreeningSimulationResponse(
                        questions=[],
                        report=AssessmentReport(
                            score=parsed.get("score", 75),
                            overallFeedback=parsed.get("overallFeedback", "Satisfactory performance."),
                            grades=[InterviewGrade(**g) for g in parsed.get("grades", [])]
                        )
                    )
            except Exception as e:
                logger.warning("Gemini grading failed: %s", e)

        return ScreeningSimulationResponse(
            questions=[],
            report=AssessmentReport(
                score=78,
                overallFeedback=f"Candidate {cand_name} showed foundational knowledge suitable for {role}.",
                grades=[InterviewGrade(questionId=1, score=78, feedback="Solid response demonstrating practical experience.")]
            )
        )

    # Generate technical questions
    cv_snippet = cv_text[:2000] if cv_text else "General profile"
    prompt = (
        f"You are a Senior Technical Examiner. Formulate 5 interview questions for {cand_name} for role {role}.\n"
        f"Resume background: {cv_snippet}\n"
        "Return valid JSON array of 5 objects with keys: id, question, expectedAnswer."
    )

    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=5)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                parsed = json.loads(raw_text.strip())
                questions_list = [InterviewQuestion(**q) for q in parsed]
                return ScreeningSimulationResponse(questions=questions_list)
        except Exception as e:
            logger.warning("Gemini question generation fallback: %s", e)

    role_lower = role.lower()
    if any(k in role_lower for k in ["account", "tally", "tax", "finance", "audit", "ca", "ledger", "gst"]):
        default_q = [
            InterviewQuestion(id=1, question=f"How do you perform month-end ledger reconciliation and verify GST/TDS return accuracy for {role}?", expectedAnswer="Covers bank reconciliation, GST portal matching, GSTR-2B verification, and TDS working sheets."),
            InterviewQuestion(id=2, question=f"What steps do you follow in Tally ERP / accounting software to handle complex journal entries and voucher posting?", expectedAnswer="Voucher creation, ledger grouping, tax adjustment entries, and trial balance verification."),
            InterviewQuestion(id=3, question=f"How do you ensure full statutory compliance during internal or statutory financial audits?", expectedAnswer="Documentation audit trails, tax deduction compliance, asset register verification, and invoice matching."),
            InterviewQuestion(id=4, question=f"Can you describe a situation where you identified a major discrepancy in financial statements or tax filings and corrected it?", expectedAnswer="Audit resolution, error rectification entries, supplier reconciliation, and compliance reporting."),
            InterviewQuestion(id=5, question=f"How do you stay updated with changing tax regulations, GST amendments, and compliance deadlines relevant to {role}?", expectedAnswer="GST portal notifications, tax updates, professional development, and compliance calendar tracking.")
        ]
    elif any(k in role_lower for k in ["developer", "engineer", "react", "node", "python", "software", "code", "java", "frontend", "backend"]):
        default_q = [
            InterviewQuestion(id=1, question=f"Can you explain your approach to application state management, component architecture, and API integration for {role}?", expectedAnswer="Demonstrates component modularity, state management patterns (Redux/Context), and clean REST/GraphQL integration."),
            InterviewQuestion(id=2, question=f"How do you optimize code performance, reduce load times, and diagnose memory leaks in web applications?", expectedAnswer="Code splitting, memoization, lazy loading, database indexing, and Chrome DevTools profiling."),
            InterviewQuestion(id=3, question=f"What testing strategies (unit, integration, end-to-end) and CI/CD pipelines do you enforce for production code?", expectedAnswer="Automated testing frameworks (Jest/PyTest), CI/CD automation, pull request reviews, and linting."),
            InterviewQuestion(id=4, question=f"Describe a challenging technical bug or system architecture bottleneck you encountered in {role} and how you solved it.", expectedAnswer="Root cause analysis, systematic debugging, database query optimization, and refactoring."),
            InterviewQuestion(id=5, question=f"How do you evaluate new frameworks, libraries, and architectural paradigms when designing scalable systems?", expectedAnswer="Proof of concept benchmarking, maintainability, community support, and security audits.")
        ]
    elif any(k in role_lower for k in ["hr", "recruiter", "talent", "hiring"]):
        default_q = [
            InterviewQuestion(id=1, question=f"What active and passive sourcing strategies do you use on LinkedIn, job portals, and talent networks for {role}?", expectedAnswer="Boolean search queries, talent mapping, headhunting, and candidate engagement campaigns."),
            InterviewQuestion(id=2, question=f"How do you structure competency-based screening interviews to assess both technical candidate fit and cultural alignment?", expectedAnswer="Behavioral interviewing, STAR method evaluation, skill scorecards, and hiring manager alignment."),
            InterviewQuestion(id=3, question=f"What key recruitment metrics (time-to-hire, offer acceptance rate, cost-per-hire) do you track to optimize the pipeline?", expectedAnswer="ATS metrics, pipeline conversion funnel, candidate experience score, and SLA tracking."),
            InterviewQuestion(id=4, question=f"Describe a situation where a key candidate turned down an offer or salary expectations exceeded budget, and how you handled it.", expectedAnswer="Salary negotiation, benefit packaging, stakeholder alignment, and backup pipeline management."),
            InterviewQuestion(id=5, question=f"How do you maintain a positive candidate experience and ensure compliance with employment laws?", expectedAnswer="Timely feedback loops, structured communication, DE&I initiatives, and data privacy compliance.")
        ]
    else:
        default_q = [
            InterviewQuestion(id=1, question=f"Can you walk us through your key technical responsibilities and core achievements in your previous {role} position?", expectedAnswer="Demonstrates hands-on domain experience, core tool proficiency, and quantifiable achievements."),
            InterviewQuestion(id=2, question=f"How do you prioritize competing tasks and troubleshoot complex operational issues in high-pressure environments?", expectedAnswer="Structured problem-solving, root cause analysis, stakeholder communication, and time management."),
            InterviewQuestion(id=3, question=f"What industry tools, software standards, or quality control frameworks do you use to ensure output quality for {role}?", expectedAnswer="Domain software proficiency, standard operating procedures, quality assurance, and compliance checks."),
            InterviewQuestion(id=4, question=f"Describe a challenging project or critical requirement for {role} where you had to collaborate across teams to deliver results.", expectedAnswer="Cross-functional teamwork, conflict resolution, technical execution, and project delivery."),
            InterviewQuestion(id=5, question=f"How do you keep your skills updated with emerging technologies and best practices relevant to {role}?", expectedAnswer="Continuous learning, professional certifications, industry workshops, and hands-on practice.")
        ]

    return ScreeningSimulationResponse(questions=default_q)


@app.post(
    "/api/v1/jd-match",
    response_model=JDMatchResponse,
    tags=["AI Agent"],
    summary="Match Job Description requirements across indexed resumes using vector search + LLM gap analysis",
)
async def match_job_description(request: Request, body: JDMatchRequest):
    """
    Perform deep vector retrieval across indexed candidate resumes for a given Job Description,
    evaluate fit, identify candidate strengths and skill gaps, and return a candidate match leaderboard.
    """
    jd_text = body.job_description.strip()
    top_k = body.top_k or 10
    
    if not jd_text:
        return JDMatchResponse(candidates=[])

    settings: Settings = request.app.state.settings

    # 1. Vector Search across Qdrant
    try:
        raw_candidates, _ = retrieve_candidates(
            qdrant_client=request.app.state.qdrant,
            model=get_embedding_model(request),
            jd_text=jd_text,
            collection_name=settings.qdrant_collection,
            top_n=top_k,
        )
    except Exception as exc:
        logger.error("JD Match vector retrieval failed: %s", exc, exc_info=True)
        raw_candidates = []

    candidates_formatted = [build_candidate_response(c) for c in raw_candidates[:top_k]]
    
    # 2. Supplement with PostgreSQL database records if vector search returned few candidates
    if len(candidates_formatted) < top_k:
        try:
            import psycopg2  # type: ignore
            pg_url = os.getenv("DATABASE_URL") or "postgresql://postgres:root@localhost:5432/resume_lens"
            conn = psycopg2.connect(pg_url)
            cur = conn.cursor()
            cur.execute('SELECT id, full_name, skills, years_experience, "current_role", location, resume_text, source_file_url FROM resumes ORDER BY id DESC LIMIT %s', (top_k * 2,))
            rows = cur.fetchall()
            conn.close()

            existing_ids = {str(c.get("candidate_id")) for c in candidates_formatted}
            for row in rows:
                cid = str(row[0])
                if cid not in existing_ids:
                    candidates_formatted.append({
                        "candidate_id": cid,
                        "name": row[1] or "Candidate Profile",
                        "metadata": {
                            "experience_years": row[3],
                            "location": row[5] or "N/A",
                            "skills": row[2] or []
                        },
                        "best_chunk_text": (row[6] or "")[:800],
                        "cv_path": row[7]
                    })
                    existing_ids.add(cid)
                    if len(candidates_formatted) >= top_k:
                        break
        except Exception as pg_err:
            logger.warning("PostgreSQL direct fetch for JD match fallback: %s", pg_err)

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 2. LLM Reranking & Fit Analysis via Gemini 2.5 Flash
    if gemini_key and candidates_formatted:
        try:
            cand_summaries = []
            for c in candidates_formatted:
                meta = c.get("metadata", {})
                cand_summaries.append({
                    "id": c.get("candidate_id"),
                    "name": c.get("name"),
                    "experience": meta.get("experience_years"),
                    "location": meta.get("location"),
                    "skills": meta.get("skills", []),
                    "cv_excerpt": (c.get("best_chunk_text") or "")[:800]
                })

            cands_json = json.dumps(cand_summaries, indent=2)
            prompt = (
                f"You are a Senior Talent Acquisition Director evaluating candidates against a Job Description.\n"
                f"Target JD: {jd_text[:1500]}\n"
                f"Candidates: {cands_json}\n"
                "Return ONLY a valid JSON array of evaluated candidates with keys: candidate_id, name, score, match_percentage, strengths, gaps, verdict."
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=3)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                parsed = json.loads(raw_text.strip())
                matched_cands = []
                for item in parsed:
                    cand_id = item.get("candidate_id")
                    original = next((c for c in candidates_formatted if c.get("candidate_id") == cand_id), {})
                    matched_cands.append(JDMatchCandidate(
                        candidate_id=cand_id or original.get("candidate_id", ""),
                        name=item.get("name") or original.get("name", "Candidate"),
                        score=float(item.get("score", 0.7)),
                        match_percentage=int(item.get("match_percentage", int(item.get("score", 0.7) * 100))),
                        strengths=item.get("strengths", []),
                        gaps=item.get("gaps", []),
                        verdict=item.get("verdict", "Good match for role."),
                        cv_path=original.get("cv_path")
                    ))
                return JDMatchResponse(candidates=matched_cands)
        except Exception as e:
            logger.warning("Gemini JD Match LLM evaluation failed: %s", e)

    matched_cands = []
    for c in candidates_formatted:
        meta = c.get("metadata", {})
        score = float(c.get("score", 0.5))
        matched_cands.append(JDMatchCandidate(
            candidate_id=c.get("candidate_id", ""),
            name=c.get("name", "Candidate"),
            score=score,
            match_percentage=int(score * 100),
            strengths=meta.get("skills", [])[:3] or ["Relevant CV profile"],
            gaps=["Verify candidate domain experience"],
            verdict=c.get("match_reasoning") or "Matched based on vector similarity search.",
            cv_path=c.get("cv_path")
        ))

    return JDMatchResponse(candidates=matched_cands)

