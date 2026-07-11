"""
Embedding and Qdrant upsert logic for the indexer.

Loads the SentenceTransformer model once per indexer run,
encodes CV chunks in batches, and upserts them into Qdrant
in batches to avoid memory spikes on large datasets.
"""
from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from indexer.parser import ParsedCV

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

VECTOR_DIM    = 384   # Output dimension of all-MiniLM-L6-v2
BATCH_ENCODE  = 64    # Chunks per SentenceTransformer encode call
BATCH_UPSERT  = 100   # Points per Qdrant upsert call


# ── Qdrant Collection Management ──────────────────────────────────────────────

def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Safe to call on every indexer run.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        logger.info("Creating Qdrant collection '%s' (dim=%d, cosine)", collection_name, VECTOR_DIM)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
    else:
        logger.debug("Collection '%s' already exists — skipping creation", collection_name)


# ── Embedding + Upsert ────────────────────────────────────────────────────────

def embed_and_upsert(
    client: QdrantClient,
    model: SentenceTransformer,
    parsed_cv: ParsedCV,
    collection_name: str,
) -> int:
    """
    Encode all chunks of a ParsedCV and upsert them into Qdrant.

    Each chunk becomes one Qdrant point. The payload stores all
    metadata needed for the API to reconstruct a candidate response
    without a secondary database lookup.

    Args:
        client:          Initialized Qdrant client.
        model:           Loaded SentenceTransformer model.
        parsed_cv:       Output of indexer.parser.parse_file().
        collection_name: Target Qdrant collection.

    Returns:
        Number of points successfully upserted.
    """
    chunks = parsed_cv.chunks
    if not chunks:
        logger.warning("No chunks for '%s' — skipping upsert", parsed_cv.name)
        return 0

    # ── Encode in batches ───────────────────────────────────────
    all_vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_ENCODE):
        batch_chunks = chunks[i : i + BATCH_ENCODE]
        batch_vectors = model.encode(
            batch_chunks,
            convert_to_list=True,
            show_progress_bar=False,
            batch_size=BATCH_ENCODE,
        )
        all_vectors.extend(batch_vectors)

    # ── Build Qdrant points ─────────────────────────────────────
    # All metadata fields are stored in every chunk's payload.
    # This means filtering and display work without a secondary DB lookup.
    # None values are stored as None — the retriever handles nulls.
    meta = parsed_cv.metadata
    base_payload: dict = {
        "candidate_id":   parsed_cv.candidate_id,
        "name":           parsed_cv.name,
        "cv_path":        parsed_cv.cv_path,
        # Metadata fields (used for post-retrieval filtering and display)
        "experience_years": meta.experience_years,          # int | None
        "location":         meta.location,                  # canonical city | None
        "location_raw":     meta.location_raw,              # raw extracted string | None
        "skills":           meta.skills,                    # list[str]
        "email":            meta.email,                     # str | None
        "ocr_used":         parsed_cv.ocr_used,             # bool (diagnostics)
    }

    points: list[PointStruct] = []
    for chunk_text, vector in zip(chunks, all_vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**base_payload, "chunk_text": chunk_text},
            )
        )

    # ── Upsert in batches ───────────────────────────────────────
    total_upserted = 0
    for i in range(0, len(points), BATCH_UPSERT):
        batch = points[i : i + BATCH_UPSERT]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        total_upserted += len(batch)

    logger.debug(
        "Upserted %d chunks for '%s' (candidate_id=%s)",
        total_upserted,
        parsed_cv.name,
        parsed_cv.candidate_id,
    )
    return total_upserted
