"""
Qdrant retrieval logic.

Embeds the job description and performs a vector similarity search
against the indexed CV chunks. Deduplicates results by candidate so
the LLM reranker receives one entry per person (not one per chunk).
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def retrieve_candidates(
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    jd_text: str,
    collection_name: str,
    top_n: int = 30,
) -> list[dict[str, Any]]:
    """
    Embed the job description and retrieve the top_n most similar
    CV chunks from Qdrant. Deduplicates by candidate_id, keeping
    the highest-scoring chunk per candidate.

    Args:
        qdrant_client:   Initialized Qdrant client.
        model:           Loaded SentenceTransformer model.
        jd_text:         Raw job description text.
        collection_name: Qdrant collection to search.
        top_n:           How many chunks to retrieve before dedup.

    Returns:
        List of candidate dicts, sorted by best_score descending.
        Each dict has: candidate_id, name, cv_path, best_score, best_chunk_text.
    """
    logger.info("Embedding job description (%d chars)", len(jd_text))
    jd_vector: list[float] = model.encode(jd_text, convert_to_list=True)

    logger.info("Querying Qdrant collection '%s' for top %d chunks", collection_name, top_n)
    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=jd_vector,
        limit=top_n,
        with_payload=True,
    )

    if not hits:
        logger.warning("Qdrant returned zero results. Is the collection indexed?")
        return []

    # Deduplicate: keep the best-scoring chunk per candidate
    best_per_candidate: dict[str, dict[str, Any]] = {}
    for hit in hits:
        payload = hit.payload or {}
        cid = payload.get("candidate_id", str(hit.id))
        score = float(hit.score)

        if cid not in best_per_candidate or score > best_per_candidate[cid]["best_score"]:
            best_per_candidate[cid] = {
                "candidate_id": cid,
                "name": payload.get("name", "Unknown"),
                "cv_path": payload.get("cv_path", ""),
                "best_score": score,
                "best_chunk_text": payload.get("chunk_text", ""),
            }

    candidates = sorted(
        best_per_candidate.values(),
        key=lambda x: x["best_score"],
        reverse=True,
    )

    logger.info(
        "Retrieved %d chunks → %d unique candidates after deduplication",
        len(hits),
        len(candidates),
    )
    return candidates
