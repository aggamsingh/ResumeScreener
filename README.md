# Resume Screener — RAG-Powered CV Screening Microservice

> **For the incoming developer:** This document is written to be consumed directly by you and your agentic IDE. Every section is intentionally verbose. Read it top to bottom once before touching any code.

A production-ready, self-hosted REST microservice that accepts a **Job Description** and returns the **top N most relevant candidates** from a database of up to 20,000 CVs. Built on a Retrieval-Augmented Generation (RAG) pipeline. Delivered as a Docker Compose product that runs on a single server with no GPU required.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Project Structure](#2-project-structure)
3. [How It Works — Full Pipeline](#3-how-it-works--full-pipeline)
4. [Tech Stack & Dependency Decisions](#4-tech-stack--dependency-decisions)
5. [First-Time Setup (Operators)](#5-first-time-setup-operators)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Indexing CVs](#7-indexing-cvs)
8. [API Reference](#8-api-reference)
9. [ERP Integration Guide](#9-erp-integration-guide)
10. [Metadata Filtering System](#10-metadata-filtering-system)
11. [OCR Support for Scanned PDFs](#11-ocr-support-for-scanned-pdfs)
12. [Monitoring & Maintenance](#12-monitoring--maintenance)
13. [Updating & Re-indexing](#13-updating--re-indexing)
14. [Troubleshooting](#14-troubleshooting)
15. [Developer Handoff Notes](#15-developer-handoff-notes)
16. [FAQ](#16-faq)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT SERVER                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │   HR / ERP   │    │         Docker Compose Stack         │  │
│  │  (browser /  │───▶│                                      │  │
│  │   REST call) │    │  ┌──────────────┐  ┌──────────────┐ │  │
│  └──────────────┘    │  │   FastAPI    │  │    Qdrant    │ │  │
│                      │  │  (api svc)   │◀▶│  Vector DB   │ │  │
│  ┌──────────────┐    │  │  port 8000   │  │  port 6333   │ │  │
│  │   CV Files   │    │  └──────┬───────┘  └──────────────┘ │  │
│  │  (PDF/DOCX)  │    │         │                            │  │
│  │  on host FS  │    │  ┌──────▼───────┐                   │  │
│  └──────┬───────┘    │  │  Groq/Gemini │ (external, HTTPS) │  │
│         │            │  │   LLM API    │                   │  │
│         │            │  └──────────────┘                   │  │
│         ▼            │                                      │  │
│  ┌──────────────┐    │  ┌──────────────┐                   │  │
│  │   Indexer    │───▶│  │  data/ dir   │                   │  │
│  │ (run once,   │    │  │ index_state  │                   │  │
│  │  then as     │    │  │   .json      │                   │  │
│  │  needed)     │    │  └──────────────┘                   │  │
│  └──────────────┘    └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Three Docker services:**

| Service | Image | Role | Restart |
|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.9.2` | Vector database storing all CV embeddings | Always |
| `api` | `resume-screener:latest` (built from `Dockerfile`) | FastAPI server handling screening requests | Always |
| `indexer` | `resume-screener:latest` (same image) | One-shot CV indexer, run manually | Never (profile-gated) |

---

## 2. Project Structure

```
RAGScreeningResume/
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App factory, lifespan, routes, auth middleware
│   ├── models.py               # Pydantic request/response models (API contract)
│   ├── retriever.py            # Qdrant vector search + post-retrieval filtering
│   └── reranker.py             # Groq/Gemini LLM reranking with fallback
│
├── indexer/                    # CV indexing pipeline
│   ├── __init__.py
│   ├── run.py                  # Orchestrator: walks CV folder, manages state
│   ├── parser.py               # PDF (text + OCR) and DOCX text extraction
│   ├── embedder.py             # Batch MiniLM embedding + Qdrant upsert
│   ├── metadata.py             # Regex extraction: experience, location, skills, email
│   └── utils.py                # MD5 hashing, atomic state file, logging setup
│
├── cvs/                        # Drop CV files here (PDF or DOCX)
│   └── .gitkeep                # Keeps folder in git; actual CVs are gitignored
│
├── data/                       # Auto-generated runtime state
│   └── .gitkeep                # index_state.json appears here after first indexing
│
├── Dockerfile                  # CPU-only build; pre-bakes embedding model
├── docker-compose.yml          # Three-service orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template (copy to .env)
└── README.md                   # This file
```

---

## 3. How It Works — Full Pipeline

### 3a. Indexing Phase (run once, then as needed)

```
CV File (PDF or DOCX)
        │
        ▼
  [parser.py] ──── pdfplumber reads text layer
        │           If < 100 chars extracted AND ENABLE_OCR=true:
        │           └── pdf2image converts pages → Tesseract OCR
        │
        ▼
  Raw text extracted
        │
        ▼
  [metadata.py] ── Regex extracts:
        │           • experience_years (int | None)
        │           • location (normalized city name | None)
        │           • location_raw (original string | None)
        │           • skills (list[str] from 130+ taxonomy)
        │           • email (str | None)
        │
        ▼
  Text chunked into 400-word windows (50-word overlap)
        │
        ▼
  [embedder.py] ── all-MiniLM-L6-v2 encodes each chunk → 384-dim vector
        │
        ▼
  Qdrant upsert: each chunk stored as a point with:
    { vector: [384 floats], payload: {candidate_id, name, cv_path,
      chunk_text, experience_years, location, location_raw,
      skills, email, ocr_used} }
        │
        ▼
  [utils.py] ── index_state.json updated with MD5 hash of file
                 (enables incremental re-indexing: unchanged files skipped)
```

**Memory profile at 20,000 CVs:**
- ~8 chunks/CV × 20,000 = 160,000 vectors
- 384 floats × 4 bytes × 160,000 = ~245 MB in Qdrant
- Plus payload storage: ~400 MB total on disk

### 3b. Screening Phase (per API request)

```
POST /api/v1/screen
{ job_description: "...", top_k: 10, filters: {...} }
        │
        ▼
  [api/main.py] ── Auth middleware validates X-API-Key header
        │
        ▼
  [api/retriever.py]
        │
        ├── 1. Embed JD text → 384-dim vector (MiniLM, ~50ms CPU)
        │
        ├── 2. Qdrant cosine similarity search
        │      top_n = 30 normally
        │      top_n = 90 when filters are active (3× buffer)
        │      Returns top chunks across all candidates (~100-500ms)
        │
        ├── 3. Deduplicate by candidate_id
        │      Keep highest-scoring chunk per person
        │      Merge metadata from that chunk's payload
        │
        └── 4. Apply hard filters (Python, in-memory):
               • min_experience / max_experience  → experience_years range check
               • location                         → substring match on location + location_raw
               • required_skills                  → ALL skills must be in candidate's skill list
               • strict=false (default)           → unknown metadata passes through + flagged
               • strict=true                      → unknown metadata excluded
        │
        ▼
  [api/reranker.py]
        │
        ├── Build prompt: JD text + up to 10 candidate blocks
        │   Each block includes: name, experience, location, skills, CV excerpt
        │
        ├── Call LLM (Groq llama-3.1-8b-instant OR Gemini 1.5-flash)
        │   20-second timeout
        │
        ├── Parse JSON response: [{candidate_id, score, reasoning}]
        │
        └── On ANY failure (timeout, bad JSON, rate limit, network):
               → Fallback: sort by Qdrant vector score instead
               → match_reasoning = "Ranked by semantic similarity..."
               → Service never returns 500 due to LLM issues
        │
        ▼
  ScreeningResponse returned:
  {
    job_id, screened_at, total_filtered_out,
    candidates: [{ candidate_id, name, score, match_reasoning,
                   cv_path, metadata, filter_flags }]
  }
```

---

## 4. Tech Stack & Dependency Decisions

| Component | Choice | Why |
|---|---|---|
| API framework | FastAPI 0.111+ | Async, OpenAPI docs auto-generated, Pydantic validation |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | 80MB, 384-dim, runs on CPU in ~50ms, excellent semantic quality |
| Vector DB | Qdrant 1.9.2 | Docker-native, persistent, cosine search, no cloud dependency |
| LLM reranker | Groq (llama-3.1-8b-instant) | Free tier, 30 req/min, <1s latency; Gemini as alternative |
| PDF text extraction | pdfplumber | Better layout handling than PyPDF2; handles tables and columns |
| PDF OCR | pytesseract + pdf2image + Tesseract | Open-source, runs locally, English language pack included |
| DOCX extraction | python-docx | Official DOCX parser |
| Deep learning | torch (CPU wheel only) | Installed from `https://download.pytorch.org/whl/cpu` — avoids 2GB+ CUDA build |

**Why CPU-only torch?**
The client server has no GPU. The CPU wheel is ~250MB vs ~2GB for CUDA. MiniLM inference on CPU takes ~50ms per JD query, which is acceptable.

**Why not Qdrant native filters?**
Metadata extraction is best-effort — ~20-30% of CVs may have `null` for `experience_years` or `location`. Qdrant native range/match filters silently exclude nulls. Post-retrieval Python filtering lets us handle nulls explicitly: pass them through with a `filter_flag` (default) or exclude them (`strict=true`). This is a conscious design decision.

---

## 5. First-Time Setup (Operators)

### Prerequisites
- Docker Desktop (or Docker Engine + Docker Compose plugin) installed and running
- Git installed
- A Groq API key: [console.groq.com](https://console.groq.com) (free, 2 minutes to sign up)
- A folder of CV files in PDF or DOCX format

### Step 1 — Clone the repo

```bash
git clone https://github.com/aggamsingh/ResumeScreener.git
cd ResumeScreener
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` in any text editor and set **these three required values**:

```env
API_KEY=your-secret-key-here           # any strong random string, e.g. openssl rand -hex 32
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx  # from console.groq.com → API Keys
CV_FOLDER_PATH=/absolute/path/to/your/cvs  # folder containing PDF/DOCX files
```

All other values have sensible defaults. See [Section 6](#6-environment-variables-reference) for the full reference.

### Step 3 — Build the Docker image

```bash
docker compose build
```

This takes 10–20 minutes on first run (downloads Python packages and pre-bakes the embedding model). Subsequent builds use the Docker layer cache and take ~30 seconds unless `requirements.txt` changes.

### Step 4 — Index your CVs

```bash
docker compose run --rm indexer
```

This scans `CV_FOLDER_PATH`, parses every PDF and DOCX, embeds them, and stores them in Qdrant. You'll see progress logs like:

```
2024-01-15 10:23:01 | INFO | indexer.run | Scanning /app/cvs...
2024-01-15 10:23:01 | INFO | indexer.run | Found 847 CV files
2024-01-15 10:23:05 | INFO | indexer.run | [1/847] Indexing john_doe.pdf → 6 chunks
2024-01-15 10:24:12 | INFO | indexer.run | [2/847] Indexing jane_smith.docx → 8 chunks
...
2024-01-15 10:51:33 | INFO | indexer.run | ─────────────────────────────
2024-01-15 10:51:33 | INFO | indexer.run | Indexed: 847 | Skipped: 0 | Failed: 3
```

**Timing:** Roughly 1-3 minutes per 100 CVs on CPU. 20,000 CVs takes 3-6 hours on first run.

### Step 5 — Start the service

```bash
docker compose up -d
```

Verify everything is running:

```bash
docker compose ps
# Expected:
# resume_screener_qdrant   running (healthy)
# resume_screener_api      running (healthy)

curl http://localhost:8000/health
# Expected: {"status":"healthy","qdrant_connected":true,"model_loaded":true,"version":"1.0.0"}
```

---

## 6. Environment Variables Reference

All variables go in your `.env` file (copy from `.env.example`).

### Required

| Variable | Example | Description |
|---|---|---|
| `API_KEY` | `s3cr3t-k3y-abc123` | Secret key for API authentication. Send as `X-API-Key` header in every request. |
| `GROQ_API_KEY` | `gsk_xxxx...` | Groq API key for LLM reranking. Get free at console.groq.com. |
| `CV_FOLDER_PATH` | `/home/user/cvs` | Absolute path to the folder containing CV files on the host machine. |

### LLM Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM provider for reranking. Options: `groq`, `gemini`. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model. Alternatives: `llama-3.3-70b-versatile` (slower, more accurate). |
| `GEMINI_API_KEY` | _(empty)_ | Required only if `LLM_PROVIDER=gemini`. Get at aistudio.google.com. |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use. |

### Screening Behaviour

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_TOP_K` | `10` | How many candidates to return when `top_k` is not specified in the request. |
| `RETRIEVAL_TOP_N` | `30` | How many chunks to fetch from Qdrant before deduplication. Increase to 50 for higher accuracy at the cost of slightly more compute. When filters are active, this is automatically tripled. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | **Do not change** unless you know what you're doing. Changing requires a full re-index. |

### OCR

| Variable | Default | Description |
|---|---|---|
| `ENABLE_OCR` | `true` | Enable Tesseract OCR fallback for scanned/image-only PDFs. Set `false` if all your CVs are digital. |

### Server

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | Port the API is exposed on. Access via `http://server-ip:8000`. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `ALLOWED_ORIGINS` | `*` | CORS origins. Use `*` for open access or comma-separate domains: `https://erp.company.com,https://hr.company.com`. |

### Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `qdrant` | Docker service name. **Do not change** — this is overridden in docker-compose.yml automatically. |
| `QDRANT_PORT` | `6333` | Qdrant REST port. |
| `QDRANT_COLLECTION` | `resumes` | Qdrant collection name where CV vectors are stored. |

---

## 7. Indexing CVs

### Supported Formats

| Format | Text Layer | Scanned/Image |
|---|---|---|
| `.pdf` | ✅ pdfplumber (fast, ~10ms/page) | ✅ Tesseract OCR (slow, ~3s/page) if `ENABLE_OCR=true` |
| `.docx` | ✅ python-docx | ❌ Not applicable |
| `.doc` | ❌ Not supported | ❌ Not supported |
| `.png/.jpg` | ❌ Not supported | ❌ Not supported |

> **For `.doc` files:** Convert to `.docx` using LibreOffice: `libreoffice --headless --convert-to docx *.doc`

### Running the Indexer

```bash
# First time or after adding new CVs:
docker compose run --rm indexer

# Check what was indexed:
cat data/index_state.json | python -m json.tool | head -50
```

### How Incremental Indexing Works

The indexer stores an MD5 hash of each file in `data/index_state.json`. On subsequent runs:

- **File unchanged** → skipped instantly (hash match)
- **File modified** → re-indexed (hash mismatch)
- **New file** → indexed
- **File deleted** → NOT automatically removed from Qdrant (see [Section 13](#13-updating--re-indexing))

This means running the indexer on 20,000 CVs after adding 50 new ones takes ~2 minutes instead of 5 hours.

### What Gets Extracted Per CV

The indexer extracts and stores the following in Qdrant for every CV chunk:

| Field | Type | Source | Notes |
|---|---|---|---|
| `candidate_id` | UUID5 | Deterministic from file path | Stable across re-index runs |
| `name` | string | Heuristic: first capitalised 2-4 word line | Falls back to filename |
| `cv_path` | string | Absolute path on host | Used by ERP to link to file |
| `experience_years` | int \| null | Regex: "5 years of experience" patterns | null if not found |
| `location` | string \| null | City alias map (55+ cities) | Normalised, e.g. "Gurugram" → "Gurgaon" |
| `location_raw` | string \| null | Raw extracted location string | e.g. "New Delhi, India" |
| `skills` | string[] | 130+ skill taxonomy keyword match | e.g. ["Python", "Docker", "PostgreSQL"] |
| `email` | string \| null | Standard email regex | Lowercased |
| `ocr_used` | bool | Parser detection | True if Tesseract was used |
| `chunk_text` | string | 400-word window of CV text | Used as LLM context |

---

## 8. API Reference

**Base URL:** `http://your-server-ip:8000`

**Authentication:** All endpoints except `/health` require the `X-API-Key` header.

**Interactive docs:** Visit `http://your-server-ip:8000/docs` in a browser for the full Swagger UI.

---

### GET /health

Returns service status. **No authentication required.** Use for monitoring.

**Response 200 — Healthy:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Response 503 — Degraded:**
```json
{
  "status": "degraded",
  "qdrant_connected": false,
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### POST /api/v1/screen

Screen CVs against a job description. Returns ranked candidates.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

#### Request Body

```jsonc
{
  "job_description": "string (required, min 20 chars)",
  "top_k": 10,          // optional, 1-50, default 10
  "filters": {          // optional — all filter fields are optional
    "min_experience": 3,               // int, minimum years (inclusive)
    "max_experience": 10,              // int, maximum years (inclusive)
    "location": "Delhi",               // string, substring match (case-insensitive)
    "required_skills": ["Python", "Docker"],  // string[], ALL must be present
    "strict": false                    // bool, default false (see Section 10)
  }
}
```

**Full Request Field Reference:**

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `job_description` | string | ✅ Yes | — | Min 20 chars |
| `top_k` | integer | No | 10 | 1–50 |
| `filters.min_experience` | integer | No | null | ≥ 0 |
| `filters.max_experience` | integer | No | null | ≥ 0 |
| `filters.location` | string | No | null | Case-insensitive substring match |
| `filters.required_skills` | string[] | No | null | Must match taxonomy (case-insensitive) |
| `filters.strict` | boolean | No | false | See [Section 10](#10-metadata-filtering-system) |

#### Response Body

```jsonc
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID for this request
  "screened_at": "2024-01-15T10:23:01.123456+00:00",  // ISO 8601 UTC
  "total_filtered_out": 12,  // candidates excluded by hard filters
  "candidates": [
    {
      "candidate_id": "a1b2c3d4-...",  // stable UUID5 from file path
      "name": "Priya Sharma",
      "score": 0.87,                   // 0.0–1.0 fit score from LLM
      "match_reasoning": "Strong FastAPI and PostgreSQL background with 6 years matching the required seniority.",
      "cv_path": "/app/cvs/priya_sharma.pdf",
      "metadata": {
        "experience_years": 6,         // null if could not be extracted
        "location": "Bangalore",       // canonical city name, null if unknown
        "location_raw": "Bengaluru, Karnataka",  // as found in CV
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        "email": "priya.sharma@email.com"  // null if not found
      },
      "filter_flags": []               // empty = all checks passed cleanly
      // Non-empty example: ["experience_unknown"]
      // means experience filter was active but couldn't be extracted from CV
    }
  ]
}
```

**Full Response Field Reference:**

| Field | Type | Description |
|---|---|---|
| `job_id` | UUID | Unique ID for this screening run. Log this for audit trails. |
| `screened_at` | ISO 8601 string | UTC timestamp. |
| `total_filtered_out` | integer | Count of candidates excluded by hard filters before reranking. |
| `candidates[].candidate_id` | UUID5 | Stable — same CV file always gets same ID. |
| `candidates[].name` | string | Name extracted from CV text. May fall back to filename. |
| `candidates[].score` | float 0–1 | LLM-assigned fit score. Higher is better. |
| `candidates[].match_reasoning` | string | One-sentence AI explanation. |
| `candidates[].cv_path` | string | Server filesystem path. Map to your URL scheme (see [Section 9](#9-erp-integration-guide)). |
| `candidates[].metadata.experience_years` | int \| null | Extracted experience. `null` means unknown. |
| `candidates[].metadata.location` | string \| null | Normalised city. `null` means unknown. |
| `candidates[].metadata.location_raw` | string \| null | Raw location string from CV. |
| `candidates[].metadata.skills` | string[] | Matched skills from 130+ taxonomy. May not be exhaustive. |
| `candidates[].metadata.email` | string \| null | Contact email from CV. |
| `candidates[].filter_flags` | string[] | Non-empty if a filter was active but data was unavailable. Values: `experience_unknown`, `location_unknown`. |

**Error Responses:**

| HTTP Code | When | Response |
|---|---|---|
| 401 | Missing or wrong `X-API-Key` | `{"detail": "Invalid or missing API key..."}` |
| 422 | Validation error (e.g. `top_k > 50`) | Pydantic validation detail |
| 503 | Qdrant unreachable | `{"detail": "Vector retrieval failed..."}` |

---

## 9. ERP Integration Guide

This section is for the ERP development team. The API speaks standard HTTP/JSON — no special client library is needed.

### Authentication Pattern

Every request to `/api/v1/screen` must include:
```
X-API-Key: <value of API_KEY from .env>
```

Store this key in your ERP's secret management (environment variable, secrets vault, etc.). Never hardcode it.

### Mapping cv_path to a Download URL

`cv_path` in the response is the server-side filesystem path (e.g. `/app/cvs/john_doe.pdf`). Your ERP needs to translate this to a URL that HR can click to open the CV.

**Option A — Shared network drive / NFS mount:**
If the CV folder is on a shared drive accessible from both the server and the ERP, strip the `/app/cvs/` prefix and prepend your file server path.

```javascript
const cvUrl = candidate.cv_path.replace('/app/cvs/', '\\\\fileserver\\cvs\\');
```

**Option B — Serve CVs via nginx:**
Add a static file server to docker-compose or your nginx config:
```nginx
location /cvs/ {
    alias /path/to/your/cvs/;
    add_header Content-Disposition inline;
}
```
Then: `cv_path.replace('/app/cvs/', 'http://server-ip/cvs/')`

**Option C — ERP already manages CV storage:**
If CVs are stored in your ERP's document system, `cv_path` contains the original filename. Match on filename: `path.basename(cv_path)` → look up in your ERP's document table.

### Integration Code Examples

#### cURL
```bash
curl -X POST http://your-server:8000/api/v1/screen \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "job_description": "Senior Python Developer with FastAPI experience...",
    "top_k": 10,
    "filters": {
      "min_experience": 3,
      "location": "Delhi",
      "required_skills": ["Python"]
    }
  }'
```

#### Python (requests)
```python
import requests

SCREENER_URL = "http://your-server:8000"
API_KEY = "your-api-key"

def screen_candidates(job_description: str, top_k: int = 10, filters: dict = None):
    response = requests.post(
        f"{SCREENER_URL}/api/v1/screen",
        headers={"X-API-Key": API_KEY},
        json={
            "job_description": job_description,
            "top_k": top_k,
            "filters": filters or {},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

# Usage
result = screen_candidates(
    job_description="Looking for a senior Python developer...",
    top_k=5,
    filters={"min_experience": 5, "required_skills": ["Python", "Docker"]},
)

for candidate in result["candidates"]:
    print(f"{candidate['name']} — Score: {candidate['score']:.2f}")
    print(f"  Reasoning: {candidate['match_reasoning']}")
    print(f"  Experience: {candidate['metadata']['experience_years']} years")
    print(f"  Location: {candidate['metadata']['location']}")
    print(f"  Skills: {', '.join(candidate['metadata']['skills'][:5])}")
    if candidate["filter_flags"]:
        print(f"  ⚠ Flags: {', '.join(candidate['filter_flags'])}")
```

#### JavaScript / Node.js (fetch)
```javascript
const SCREENER_URL = 'http://your-server:8000';
const API_KEY = process.env.SCREENER_API_KEY;

async function screenCandidates(jobDescription, topK = 10, filters = {}) {
  const response = await fetch(`${SCREENER_URL}/api/v1/screen`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({
      job_description: jobDescription,
      top_k: topK,
      filters,
    }),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Screener error ${response.status}: ${JSON.stringify(error)}`);
  }

  return response.json();
}

// Usage
const result = await screenCandidates(
  'Looking for a Data Analyst with SQL and Power BI...',
  10,
  { min_experience: 2, location: 'Mumbai', required_skills: ['SQL', 'Power BI'] }
);

console.log(`Screened at: ${result.screened_at}`);
console.log(`Filtered out: ${result.total_filtered_out} candidates`);
result.candidates.forEach((c, i) => {
  console.log(`${i + 1}. ${c.name} (${c.score.toFixed(2)}) — ${c.match_reasoning}`);
});
```

#### PHP
```php
<?php
function screenCandidates(string $jobDescription, int $topK = 10, array $filters = []): array {
    $url = 'http://your-server:8000/api/v1/screen';
    $apiKey = getenv('SCREENER_API_KEY');

    $payload = json_encode([
        'job_description' => $jobDescription,
        'top_k' => $topK,
        'filters' => $filters,
    ]);

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $payload,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'X-API-Key: ' . $apiKey,
        ],
    ]);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($httpCode !== 200) {
        throw new RuntimeException("Screener returned HTTP $httpCode: $response");
    }

    return json_decode($response, true);
}

// Usage
$result = screenCandidates(
    'Senior Accountant with Tally and GST experience...',
    10,
    ['min_experience' => 3, 'required_skills' => ['Tally', 'GST']]
);

foreach ($result['candidates'] as $i => $candidate) {
    echo ($i + 1) . ". {$candidate['name']} — Score: {$candidate['score']}\n";
    echo "   {$candidate['match_reasoning']}\n";
}
```

#### C# / .NET
```csharp
using System.Net.Http.Json;

public record ScreeningFilters(
    int? MinExperience = null,
    int? MaxExperience = null,
    string? Location = null,
    string[]? RequiredSkills = null,
    bool Strict = false
);

public record ScreeningRequest(
    string JobDescription,
    int TopK = 10,
    ScreeningFilters? Filters = null
);

public class ResumeScreenerClient
{
    private readonly HttpClient _http;
    private const string BaseUrl = "http://your-server:8000";

    public ResumeScreenerClient(string apiKey)
    {
        _http = new HttpClient { BaseAddress = new Uri(BaseUrl), Timeout = TimeSpan.FromSeconds(30) };
        _http.DefaultRequestHeaders.Add("X-API-Key", apiKey);
    }

    public async Task<JsonElement> ScreenAsync(ScreeningRequest request)
    {
        var response = await _http.PostAsJsonAsync("/api/v1/screen", request);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<JsonElement>();
    }
}

// Usage
var client = new ResumeScreenerClient(Environment.GetEnvironmentVariable("SCREENER_API_KEY")!);
var result = await client.ScreenAsync(new ScreeningRequest(
    JobDescription: "Senior Java Developer with Spring Boot and microservices...",
    TopK: 10,
    Filters: new ScreeningFilters(MinExperience: 5, RequiredSkills: ["Java", "Spring Boot"])
));

foreach (var candidate in result.GetProperty("candidates").EnumerateArray())
{
    Console.WriteLine($"{candidate.GetProperty("name")} — {candidate.GetProperty("score"):F2}");
    Console.WriteLine($"  {candidate.GetProperty("match_reasoning")}");
}
```

#### Java 11+
```java
import java.net.http.*;
import java.net.URI;
import com.fasterxml.jackson.databind.ObjectMapper;

public class ResumeScreenerClient {
    private static final String BASE_URL = "http://your-server:8000";
    private final HttpClient client = HttpClient.newHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String apiKey;

    public ResumeScreenerClient(String apiKey) { this.apiKey = apiKey; }

    public Map<String, Object> screen(String jobDescription, int topK, Map<String, Object> filters)
            throws Exception {
        var body = Map.of(
            "job_description", jobDescription,
            "top_k", topK,
            "filters", filters
        );

        var request = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + "/api/v1/screen"))
            .header("Content-Type", "application/json")
            .header("X-API-Key", apiKey)
            .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
            .timeout(Duration.ofSeconds(30))
            .build();

        var response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200)
            throw new RuntimeException("Screener error " + response.statusCode() + ": " + response.body());

        return mapper.readValue(response.body(), Map.class);
    }
}
```

### ERP Integration Patterns

#### Pattern 1 — Synchronous (Simple ERP, Recommended for most cases)

```
HR opens job form → fills JD → clicks "Screen Candidates"
→ ERP calls POST /api/v1/screen
→ ERP waits up to 30 seconds
→ ERP renders ranked candidate list inline
```

This works for typical response times of 3-8 seconds. Ensure the UI shows a loading spinner.

#### Pattern 2 — Asynchronous (For ERPs with strict timeout budgets)

```
HR submits JD
→ ERP calls POST /api/v1/screen in background job
→ HR sees "Screening in progress..." 
→ ERP polls or uses webhook notification when done
→ HR sees results
```

Use this if your ERP framework has a hard HTTP timeout < 10 seconds.

#### Pattern 3 — Health Gate (Check before exposing the screening form)

```javascript
async function isScreenerAvailable() {
  try {
    const res = await fetch('http://your-server:8000/health', { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}

// In your form rendering logic:
if (await isScreenerAvailable()) {
  showScreeningButton();
} else {
  showFallbackMessage("AI screening is temporarily unavailable. Please screen manually.");
}
```

#### Pattern 4 — Handling filter_flags in the UI

When a candidate has `filter_flags: ["experience_unknown"]`, it means the experience filter was active but experience could not be extracted from the CV. Suggested UI treatment:

```
┌─────────────────────────────────────────────────┐
│ 3. Rajesh Kumar          Score: 0.79            │
│    "Strong SQL and SAP background..."            │
│    📍 Mumbai  |  💼 Experience: Unknown          │
│    ⚠ Experience could not be verified from CV.  │
│      Please review CV manually.                  │
└─────────────────────────────────────────────────┘
```

---

## 10. Metadata Filtering System

### How Filtering Works

Filters are applied **after** vector search but **before** LLM reranking. This means:

1. Qdrant returns 90 candidates (3× the usual 30, buffered for filtering)
2. Python filter logic runs in-memory: negligible time
3. Surviving candidates (say 25) go to the LLM
4. LLM returns top 10

**Strict vs. Lenient mode:**

| Scenario | `strict: false` (default) | `strict: true` |
|---|---|---|
| CV has `experience_years: 6`, filter is `min_experience: 5` | ✅ Passes | ✅ Passes |
| CV has `experience_years: 3`, filter is `min_experience: 5` | ❌ Excluded | ❌ Excluded |
| CV has `experience_years: null` (couldn't extract), filter is `min_experience: 5` | ✅ Passes + `filter_flags: ["experience_unknown"]` | ❌ Excluded |

**When to use `strict: true`:** When you have high confidence that CVs in your database are well-structured (digital, not scanned) and that the metadata extractor will reliably find the relevant fields. In this case, candidates without extractable metadata are likely anomalies.

**When to use `strict: false` (default):** Always a safer choice. Candidates with unknown metadata are surfaced to HR with a flag, rather than silently discarded.

### Location Matching

The location filter uses **case-insensitive substring matching** on both the normalised city name and the raw extracted string:

| Filter value | Matches | Does not match |
|---|---|---|
| `"Delhi"` | Delhi, Delhi NCR, New Delhi, New Delhi, India | Mumbai, Bangalore |
| `"Bangalore"` | Bangalore, Bengaluru, Bangalore, Karnataka | Mysore |
| `"Mumbai"` | Mumbai, Mumbai, Maharashtra, Navi Mumbai | Pune |
| `"NCR"` | Delhi NCR | New Delhi (city only) |

**Note:** Location extraction covers 55+ Indian cities and major international cities. If a CV uses an unusual format or a smaller city name, `location` will be `null`. Set `strict: false` to include those candidates.

### Skills Filtering

`required_skills` uses **ALL-must-match** semantics:

```json
"required_skills": ["Python", "Docker", "PostgreSQL"]
```

A candidate must have **all three** skills in their extracted skill list to pass. If any one is missing, the candidate is excluded.

Skills are matched case-insensitively against the 130+ entry taxonomy in `indexer/metadata.py`. Valid skill names include: `Python`, `Java`, `JavaScript`, `TypeScript`, `React`, `Angular`, `Vue`, `FastAPI`, `Django`, `Flask`, `Spring Boot`, `Docker`, `Kubernetes`, `AWS`, `Azure`, `GCP`, `PostgreSQL`, `MySQL`, `MongoDB`, `Redis`, `Elasticsearch`, `Machine Learning`, `Deep Learning`, `NLP`, `SAP`, `ERP`, `Tally`, `Power BI`, `Tableau`, `Excel`, `Agile`, `Scrum`, and many more.

**View the full taxonomy:** Open `indexer/metadata.py` and look at the `_SKILL_TAXONOMY` dictionary.

---

## 11. OCR Support for Scanned PDFs

When a PDF has no readable text layer (scanned paper CV, image-embedded PDF), the indexer automatically falls back to Tesseract OCR.

### How OCR is Triggered

```python
# parser.py logic:
text = pdfplumber.extract()
if len(text.strip()) < 100 chars AND ENABLE_OCR=true:
    text = tesseract_ocr(pages)  # ~3 seconds per page
```

Regular digital PDFs are **never slowed down** by OCR — it only activates when text extraction fails.

### OCR Performance

| Scenario | Time per CV |
|---|---|
| Digital PDF (text layer) | ~0.1–0.5s |
| Scanned PDF (OCR) | ~3–10s depending on page count |
| DOCX | ~0.1–0.3s |

### OCR Quality

Tesseract works well on:
- Clean scans (300 DPI+)
- Single-column layouts
- Standard fonts

Tesseract struggles with:
- Low resolution scans (< 150 DPI)
- Complex multi-column layouts
- Handwritten text
- Non-English text (only English is installed)

### Disabling OCR

If all CVs are digital and you want faster indexing:
```env
ENABLE_OCR=false
```

---

## 12. Monitoring & Maintenance

### Daily Health Check

```bash
# Check all services are running
docker compose ps

# Quick health API check
curl http://localhost:8000/health

# View last 100 log lines from API
docker compose logs --tail=100 api

# View last 100 lines from Qdrant
docker compose logs --tail=100 qdrant
```

### Qdrant Dashboard

Qdrant ships with a web dashboard at `http://your-server:6333/dashboard`. It shows:
- Collection info (vector count, disk usage)
- Memory usage
- Query performance stats

No login required (it's local-only by default).

### Collection Stats

```bash
curl http://localhost:6333/collections/resumes
# Shows: vectors_count, indexed_vectors_count, disk_data_size
```

### Restarting Services

```bash
# Restart just the API (e.g. after .env change)
docker compose restart api

# Restart everything
docker compose down && docker compose up -d

# Rebuild after code changes
docker compose down
docker compose build
docker compose up -d
```

### Viewing Screening Logs

Every screening request is logged with:
- `job_id` (UUID)
- `top_k`
- JD character count
- Active filters
- Number of candidates filtered out
- Number of candidates returned

```bash
docker compose logs api | grep "Screening request"
docker compose logs api | grep "Returning"
```

---

## 13. Updating & Re-indexing

### Adding New CVs

1. Drop new PDF/DOCX files into the CV folder (`CV_FOLDER_PATH`)
2. Run the indexer:
   ```bash
   docker compose run --rm indexer
   ```
3. The indexer skips all existing files (hash match) and only processes new ones.
4. New candidates are immediately searchable — no API restart needed.

### Modifying Existing CVs

1. Replace the file in the CV folder
2. Run the indexer — it detects the hash change and re-indexes that file
3. Old vectors for that candidate are **overwritten** (same `candidate_id` because it's UUID5 from file path)

> **Note:** Old chunk vectors from a previous version of the file may remain in Qdrant if the new version produces fewer chunks. This is a known limitation. To clean up completely, reset the collection and re-index all CVs.

### Deleting CVs

Deleting a file from the CV folder does **not** automatically remove it from Qdrant. To fully remove a candidate:

```bash
# Find their candidate_id from index_state.json
cat data/index_state.json | python -c "
import json, sys
state = json.load(sys.stdin)
for path, info in state.items():
    if 'john_doe' in path:
        print(path, '->', info['candidate_id'])
"

# Delete their vectors from Qdrant
curl -X POST http://localhost:6333/collections/resumes/points/delete \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "candidate_id", "match": {"value": "UUID-HERE"}}]}}'

# Remove from index_state.json manually or re-run indexer which will clean missing files
```

### ⚠️ Re-indexing After a Code Update

**This is critical.** If you update the indexer (e.g. metadata.py or embedder.py) and want the new metadata fields to be stored for existing CVs, you must **reset Qdrant and re-index all CVs**:

```bash
# 1. Stop everything
docker compose down

# 2. Delete the Qdrant volume (destroys all vectors!)
docker volume rm resume_screener_qdrant_data

# 3. Delete the state file so indexer re-processes everything
rm data/index_state.json

# 4. Rebuild (if code changed)
docker compose build

# 5. Start Qdrant only
docker compose up -d qdrant

# 6. Re-index all CVs (this will take hours for 20k CVs)
docker compose run --rm indexer

# 7. Start the API
docker compose up -d api
```

**When is a full re-index required?**
- After updating `indexer/metadata.py` (new fields you want filterable)
- After updating `indexer/embedder.py` (changed payload structure)
- After changing `EMBEDDING_MODEL` (different vector dimensions)
- After upgrading Qdrant version (sometimes schema changes)

**When is a full re-index NOT required?**
- After updating `api/reranker.py` (no index changes)
- After updating `api/retriever.py` (no index changes)
- After updating `api/main.py` or `api/models.py`
- After changing `.env` values (except `EMBEDDING_MODEL`)

---

## 14. Troubleshooting

### Service won't start

```bash
docker compose logs api
```

**Common causes:**
- `.env` not created (copy from `.env.example`)
- `API_KEY` not set in `.env`
- `GROQ_API_KEY` not set in `.env`
- Port 8000 already in use: change `API_PORT` in `.env`

### Health check shows degraded

```bash
curl http://localhost:8000/health
# {"status":"degraded","qdrant_connected":false,"model_loaded":true}
```

Qdrant is not responding. Check:
```bash
docker compose ps qdrant     # should show "healthy"
docker compose logs qdrant   # look for error messages
```

### No candidates returned

**Cause 1:** CVs not indexed.
```bash
docker compose run --rm indexer
```

**Cause 2:** All candidates filtered out by hard filters.
Check `total_filtered_out` in the response. Try removing filters or using `strict: false`.

**Cause 3:** JD text is too short or too generic.
Use a specific, detailed job description (100+ words recommended).

### Screening returns 503

The API can't reach Qdrant. Usually means:
```bash
docker compose restart qdrant
docker compose restart api
```

### Indexer fails on some files

```bash
docker compose run --rm indexer 2>&1 | grep -i "failed\|error\|warning"
```

Each file failure is isolated — others continue indexing. Failed files are **not** marked in the state file, so re-running the indexer will retry them.

**PDF fails with no text extracted:**
- Digital PDF: check if it's password-protected
- Scanned PDF: ensure `ENABLE_OCR=true` in `.env`

**DOCX fails:**
- File may be corrupted. Try opening in LibreOffice.

### LLM reranking fails / slow

If Groq is down or rate-limited:
- The service auto-falls back to vector score ranking
- `match_reasoning` will say "Ranked by semantic similarity..."
- This is expected behaviour — not a bug

To check Groq status: [status.groq.com](https://status.groq.com)

To increase rate limit: upgrade Groq plan or switch to `GEMINI` provider.

### Scanned PDFs not being OCR'd

```bash
# Check OCR is enabled
grep ENABLE_OCR .env

# Check Tesseract is installed in the container
docker compose exec api tesseract --version
```

If Tesseract is missing, rebuild:
```bash
docker compose build --no-cache
```

---

## 15. Developer Handoff Notes

> **This section is written for the developer (and their agentic IDE) taking over this project.**

### Codebase Philosophy

- **No magic.** Every design decision is documented in the relevant module's docstring. Read `api/retriever.py` first — it has a particularly detailed explanation of why we use post-retrieval filtering over Qdrant native filters.
- **Fault isolation.** Every external call (Qdrant, LLM, file IO) has explicit error handling. The indexer never crashes on a single bad CV file. The API never returns 500 due to LLM issues.
- **Stateless API, stateful indexer.** The API holds no session state. The indexer maintains `data/index_state.json` for incremental runs.

### Module Responsibilities (Quick Map)

| Module | Responsibility | Key functions |
|---|---|---|
| `api/main.py` | App factory, lifespan, auth middleware, routes | `create_app()`, `screen_resumes()`, `health_check()` |
| `api/models.py` | API contract (Pydantic) | `ScreeningRequest`, `ScreeningResponse`, `Candidate`, `CandidateMetadata`, `ScreeningFilters` |
| `api/retriever.py` | Vector search + post-retrieval filtering | `retrieve_candidates()`, `build_candidate_response()`, `_apply_filters()` |
| `api/reranker.py` | LLM reranking with fallback | `rerank_candidates()`, `_fallback_ranking()`, `_build_prompt()` |
| `indexer/run.py` | Indexing orchestrator | `main()` — walk files, hash check, parse, embed, save state |
| `indexer/parser.py` | CV text extraction + OCR | `parse_file()`, `_parse_pdf()`, `_parse_pdf_ocr()`, `_chunk_text()` |
| `indexer/metadata.py` | Structured data extraction | `extract_metadata()`, `_extract_experience()`, `_extract_location()`, `_extract_skills()` |
| `indexer/embedder.py` | MiniLM encoding + Qdrant upsert | `embed_and_upsert()`, `ensure_collection()` |
| `indexer/utils.py` | Hashing, state file, logging | `compute_md5()`, `get_candidate_id()`, `load_state()`, `save_state()` |

### Key Data Flows to Understand

**1. Candidate ID stability**
`candidate_id` is a UUID5 derived from the absolute file path (`indexer/utils.py :: get_candidate_id()`). This means the same file always gets the same ID across re-index runs. This is intentional — it lets the ERP safely store a `candidate_id` as a foreign key.

**2. Metadata null propagation**
`CVMetadata` fields default to `None`. `None` flows through to the Qdrant payload, then to the `CandidateMetadata` response model. The filter logic in `retriever.py` explicitly handles `None` with the `strict` flag. **Never assume a metadata field is non-null.**

**3. Retrieval buffer when filtering**
`retrieve_candidates()` in `retriever.py` multiplies `top_n` by `_FILTER_RETRIEVAL_MULTIPLIER = 3` when any filter is active. This compensates for candidates that will be filtered out. If you're changing filter logic, be aware that tightening filters may need this multiplier increased.

**4. Single image, two roles**
The `Dockerfile` produces one image used for both `api` and `indexer`. The difference is the `CMD`: `api` runs `uvicorn`; `indexer` overrides with `python -m indexer.run`. Both share the same Python environment and pre-baked model.

### Where to Add Features

| Feature | Where to change |
|---|---|
| Add a new metadata field (e.g. `education_level`) | `indexer/metadata.py` (extraction) → `indexer/embedder.py` (payload) → `api/models.py` (response) → `api/retriever.py` (filter check) |
| Add a new filter type | `api/models.py` (field) → `api/retriever.py` (`_apply_filters`) |
| Add a new LLM provider | `api/reranker.py` (new branch in `rerank_candidates`) |
| Support `.doc` files | `indexer/parser.py` (add `_parse_doc` using `subprocess + libreoffice`) |
| Add multilingual OCR | `Dockerfile` (add `tesseract-ocr-hin` etc.) → `indexer/parser.py` (`lang` param to pytesseract) |
| Add webhook on index completion | `indexer/run.py` (call endpoint after summary) |
| Add authentication per ERP tenant | `api/main.py` (multi-key middleware) → `.env.example` (document key format) |

### Testing (Currently None — Add Before Production)

No tests exist. Priority order for first tests:
1. `tests/test_metadata.py` — unit test `extract_metadata()` with sample CV texts
2. `tests/test_retriever.py` — unit test `_apply_filters()` with mock candidate dicts
3. `tests/test_api.py` — integration test `/health` and `/api/v1/screen` with a real Qdrant instance
4. `tests/test_parser.py` — test `parse_file()` with sample PDFs and DOCX files

Recommended: pytest + httpx for async tests.

### Environment for Local Dev (Without Docker)

```bash
# Install system dependencies (required for OCR and PDF parsing)
# Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr poppler-utils
# Mac (Homebrew): brew install tesseract poppler

# Create venv
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install CPU torch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt

# Start Qdrant locally (Docker still needed for this)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.9.2

# Set env vars
cp .env.example .env
# Edit .env: set API_KEY, GROQ_API_KEY, CV_FOLDER_PATH, QDRANT_HOST=localhost

# Run API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run indexer
python -m indexer.run
```

### Known Limitations

1. **Deleted CV handling:** Removing a CV from the folder does not remove its vectors from Qdrant. Must be done manually (see Section 13).
2. **Metadata extraction is heuristic:** ~20-30% of CVs may have null experience/location. This is expected. The `strict` filter flag and `filter_flags` response field handle this gracefully.
3. **LLM reranker is English-only:** The Groq/Gemini prompt and reasoning will be in English regardless of the CV language. Non-English CVs still match semantically via MiniLM.
4. **No concurrent indexing protection:** Running two indexer instances simultaneously will cause duplicate Qdrant points. Only run one indexer at a time.
5. **OCR quality varies:** Tesseract at 200 DPI works well for clean scans but fails on very low quality or handwritten content.

---

## 16. FAQ

**Q: How accurate is the screening?**
A: Semantic accuracy for well-written JDs and CVs is high. The MiniLM embedding captures conceptual similarity (not just keywords), and the LLM reranker adds contextual reasoning. Expect to manually review borderline candidates (scores 0.5–0.7).

**Q: What happens if Groq goes down?**
A: The API automatically falls back to Qdrant vector score ranking. `match_reasoning` will say "Ranked by semantic similarity (AI reranker temporarily unavailable)." The service never goes down due to LLM issues.

**Q: Will filters miss candidates whose CVs don't mention their location or experience?**
A: With `strict: false` (default), those candidates are included and flagged with `filter_flags: ["experience_unknown"]` or `["location_unknown"]`. HR can review them manually. With `strict: true`, they're excluded.

**Q: How often should I re-index?**
A: Run the indexer whenever new CVs are added. It's incremental — only new/changed files are processed. For a rolling hiring pipeline, a weekly scheduled run is reasonable.

**Q: Is candidate data stored outside the server?**
A: CV content is sent to Groq/Gemini as part of the reranking prompt (CV excerpt up to 800 characters per candidate). If this is a concern, switch to `strict=false`, use longer JDs so semantic search is sufficient without LLM reranking, or run a locally-hosted LLM (would require code changes to `reranker.py`).

**Q: Can I run this without a Groq API key?**
A: Yes, but results will be lower quality. Without a valid Groq key, every request falls back to pure vector score ranking (no LLM reasoning). Set `LLM_PROVIDER=groq` and use an invalid key to trigger fallback mode intentionally.

**Q: The documentation mentions 20,000 resumes. Can the system scale to 1 Lakh (100k) or 2 Lakh (200k) resumes?**
A: Yes, without any architectural changes. Qdrant handles millions of vectors effortlessly using HNSW indexing, and the AI reranker only processes the top 90 matches regardless of database size, so API latency remains 3-8 seconds. The only bottlenecks are the *initial indexing time* and hardware. For 2 Lakh resumes, you will need:
- **RAM:** 8GB to 16GB (to hold the Qdrant index in memory for fast retrieval).
- **Disk Space:** ~200GB (to store the original PDFs/DOCXs) + 5GB for the Qdrant index.
- **Patience:** The *first* run of the indexer could take 30–60 hours on a single CPU core. Subsequent incremental runs for new daily CVs will remain extremely fast.

**Q: Can multiple users query the API simultaneously?**
A: Yes. The API is async (uvicorn with FastAPI). The embedding model and Qdrant client are loaded once and reused across all requests. Practical concurrent capacity: 3-5 simultaneous requests on a modern CPU server before latency degrades.

**Q: How do I backup the CV database?**
A: Three things to backup:
1. The CV files themselves (your `CV_FOLDER_PATH` directory)
2. `data/index_state.json` (the indexer state)
3. The Qdrant volume: `docker run --rm -v resume_screener_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz /data`

**Q: Can I change the embedding model?**
A: Technically yes, but it requires a full re-index (delete Qdrant volume + delete `index_state.json` + run indexer). You also need to update `EMBEDDING_MODEL` in `.env` and ensure the new model's output dimension matches the Qdrant collection dimension (delete collection if different). Not recommended unless you know what you're doing.

**Q: The indexer crashed halfway through. What now?**
A: Just re-run it. `data/index_state.json` tracks completed files. Only the files that successfully completed are in the state file — the interrupted one will be retried.
