# ══════════════════════════════════════════════════════════════
#  Resume Screener — Dockerfile
#
#  Single image used for both the API service and the indexer.
#  CMD is overridden in docker-compose.yml per service.
#
#  Key choices:
#    - python:3.10-slim  → small base, no unnecessary extras
#    - CPU-only torch    → avoids 3 GB+ CUDA download
#    - Model pre-baked   → no download on first container start
# ══════════════════════════════════════════════════════════════

FROM python:3.10.14-slim-bookworm

# ── System dependencies ─────────────────────────────────────────
# curl:          healthchecks and Qdrant connectivity tests
# tesseract-ocr: OCR engine for scanned PDFs (ENABLE_OCR=true)
# poppler-utils: PDF-to-image renderer used by pdf2image (for OCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────
# Install CPU-only PyTorch FIRST using the official CPU wheel index.
# This prevents pip from downloading the ~2 GB CUDA variant that
# sentence-transformers would otherwise pull in automatically.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model ────────────────────────────────
# Bakes the model into the image so first startup is instant.
# The model (~90 MB) is stored in the HuggingFace cache inside the image.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chown appuser /app/.cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2 ...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model ready.')"

# ── Application code ────────────────────────────────────────────
# mock_grpc.py is a top-level module and the FIRST import in both api/main.py and
# indexer/run.py. Leaving it out made every container exit immediately with
# "ModuleNotFoundError: No module named 'mock_grpc'".
COPY mock_grpc.py ./
COPY api/      ./api/
COPY indexer/  ./indexer/

# Persistent directories (mounted from host via docker-compose volumes).
# chown: the API writes here as appuser — SharePoint sync downloads into cvs/,
# and the embedded-Qdrant fallback opens data/qdrant_db for writing. Root-owned
# directories made both fail with EACCES.
RUN mkdir -p cvs data && chown -R appuser cvs data

USER appuser

# ── Default command (API service) ───────────────────────────────
# Overridden to "python -m indexer.run" for the indexer profile.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
