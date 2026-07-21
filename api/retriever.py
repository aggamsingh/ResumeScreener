"""
Qdrant retrieval with post-retrieval metadata filtering.

Flow:
  1. Embed the job description with MiniLM
  2. Retrieve top_n chunks from Qdrant (increased when filters are active)
  3. Deduplicate by candidate_id (keep best-scoring chunk per candidate)
  4. Apply hard filters in Python: experience, location, required_skills
  5. Return filtered candidates sorted by vector score

Why post-retrieval filtering (not Qdrant-native filters):
  - Metadata extraction is best-effort. Many CVs will have None for
    experience_years or location. Qdrant native filters silently exclude
    those candidates. Post-filtering lets us handle None explicitly:
    either pass them through with a flag (strict=False, default) or
    exclude them (strict=True).
  - Qdrant text index setup would be required for substring location
    matching — adds schema complexity with no real benefit at 20k scale.
  - 30–90 candidate dicts filtered in Python is microseconds.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api.models import Candidate, CandidateMetadata, ScreeningFilters

logger = logging.getLogger(__name__)

# When filters are active, we retrieve more candidates to compensate
# for those that will be filtered out.
_FILTER_RETRIEVAL_MULTIPLIER = 3


# ── Filter Logic ───────────────────────────────────────────────────────────────

def _check_experience(
    candidate: dict[str, Any],
    filters: ScreeningFilters,
) -> tuple[bool, list[str]]:
    """
    Check experience_years against min/max filter.

    Returns (passes: bool, flags: list[str]).
    flags is non-empty when the check couldn't be done due to missing data.
    """
    flags: list[str] = []
    exp = candidate.get("experience_years")

    # No experience filter active → always pass
    if filters.min_experience is None and filters.max_experience is None:
        return True, flags

    if exp is None:
        flags.append("experience_unknown")
        # strict=True: exclude unknown; strict=False: include with flag
        return not filters.strict, flags

    if filters.min_experience is not None and exp < filters.min_experience:
        return False, flags

    if filters.max_experience is not None and exp > filters.max_experience:
        return False, flags

    return True, flags


def _check_location(
    candidate: dict[str, Any],
    filters: ScreeningFilters,
) -> tuple[bool, list[str]]:
    """
    Check location against filter string (case-insensitive substring match).

    We check both location (canonical) and location_raw to maximise recall.
    E.g. filter='Delhi' will match:
      location='Delhi NCR', location_raw='New Delhi, India'
    """
    flags: list[str] = []

    if not filters.location:
        return True, flags

    filter_loc = filters.location.lower().strip()
    loc_canonical = (candidate.get("location") or "").lower()
    loc_raw       = (candidate.get("location_raw") or "").lower()

    if not loc_canonical and not loc_raw:
        flags.append("location_unknown")
        return not filters.strict, flags

    if filter_loc in loc_canonical or filter_loc in loc_raw:
        return True, flags

    return False, flags


def _check_skills(
    candidate: dict[str, Any],
    filters: ScreeningFilters,
) -> tuple[bool, list[str]]:
    """
    Check that ALL required_skills appear in the candidate's skill list.
    Comparison is case-insensitive.
    """
    flags: list[str] = []

    if not filters.required_skills:
        return True, flags

    candidate_skills_lower = {s.lower() for s in (candidate.get("skills") or [])}

    missing = [
        s for s in filters.required_skills
        if s.lower() not in candidate_skills_lower
    ]

    if missing:
        return False, flags

    return True, flags


def _apply_filters(
    candidates: list[dict[str, Any]],
    filters: ScreeningFilters,
) -> tuple[list[dict[str, Any]], int]:
    """
    Apply all active hard filters to a deduplicated candidate list.

    Returns (passing_candidates, n_filtered_out).
    Passing candidates have their filter_flags list populated with any
    warnings about metadata fields that could not be verified.
    """
    if not filters.is_active():
        # No filters — attach empty flags and return everything
        for c in candidates:
            c["filter_flags"] = []
        return candidates, 0

    passing:     list[dict[str, Any]] = []
    filtered_out = 0

    for c in candidates:
        all_flags: list[str] = []
        passes_all = True

        exp_ok,  exp_flags  = _check_experience(c, filters)
        loc_ok,  loc_flags  = _check_location(c, filters)
        skill_ok, skill_flags = _check_skills(c, filters)

        all_flags.extend(exp_flags)
        all_flags.extend(loc_flags)
        all_flags.extend(skill_flags)

        if not (exp_ok and loc_ok and skill_ok):
            passes_all = False

        if passes_all:
            c["filter_flags"] = all_flags
            passing.append(c)
        else:
            filtered_out += 1

    logger.info(
        "Filters applied: %d passed, %d excluded",
        len(passing),
        filtered_out,
    )
    return passing, filtered_out


# ── Deduplication ──────────────────────────────────────────────────────────────

def _deduplicate(hits: list) -> list[dict[str, Any]]:
    """
    Deduplicate Qdrant search hits by candidate_id.
    Keeps the highest-scoring chunk per candidate and merges metadata.
    Returns list sorted by best_score descending.
    """
    best: dict[str, dict[str, Any]] = {}

    for hit in hits:
        payload = hit.payload or {}
        cid   = payload.get("candidate_id", str(hit.id))
        score = float(hit.score)

        if cid not in best or score > best[cid]["best_score"]:
            best[cid] = {
                "candidate_id":    cid,
                "name":            payload.get("name", "Unknown"),
                "cv_path":         payload.get("cv_path", ""),
                "best_score":      score,
                "best_chunk_text": payload.get("chunk_text", ""),
                # Metadata for filtering
                "experience_years": payload.get("experience_years"),
                "location":         payload.get("location"),
                "location_raw":     payload.get("location_raw"),
                "skills":           payload.get("skills", []),
                "email":            payload.get("email"),
                "ocr_used":         payload.get("ocr_used", False),
            }

    return sorted(best.values(), key=lambda x: x["best_score"], reverse=True)


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve_candidates(
    qdrant_client:   QdrantClient,
    model:           SentenceTransformer,
    jd_text:         str,
    collection_name: str,
    top_n:           int = 30,
    filters:         Optional[ScreeningFilters] = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Embed the JD, retrieve from Qdrant, deduplicate, and apply filters.

    Args:
        qdrant_client:    Initialized Qdrant client.
        model:            Loaded SentenceTransformer model.
        jd_text:          Raw job description text.
        collection_name:  Qdrant collection to search.
        top_n:            Base retrieval size (multiplied when filters are active).
        filters:          Optional ScreeningFilters to apply post-retrieval.

    Returns:
        Tuple of:
          - list of candidate dicts (filtered, sorted by score)
          - int: count of candidates excluded by hard filters
    """
    effective_filters = filters or ScreeningFilters()

    # Increase retrieval size when filters are active to compensate for exclusions
    actual_top_n = top_n
    if effective_filters.is_active():
        actual_top_n = top_n * _FILTER_RETRIEVAL_MULTIPLIER
        logger.info(
            "Filters active — increasing retrieval from %d to %d chunks",
            top_n,
            actual_top_n,
        )

    # ── Embed JD ─────────────────────────────────────────────────
    logger.info("Embedding job description (%d chars)", len(jd_text))
    jd_vector: list[float] = model.encode(jd_text).tolist()

    # ── Vector search ─────────────────────────────────────────────
    logger.info(
        "Querying Qdrant '%s' for top %d chunks",
        collection_name,
        actual_top_n,
    )
    try:
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=jd_vector,
            limit=actual_top_n,
            with_payload=True,
        )
    except Exception as e:
        logger.warning("Qdrant search error or collection empty: %s", e)
        return [], 0

    if not hits:
        logger.warning("Qdrant returned zero results — is the collection indexed?")
        return [], 0

    # ── Deduplicate ───────────────────────────────────────────────
    candidates = _deduplicate(hits)
    logger.info(
        "Retrieved %d chunks -> %d unique candidates after deduplication",
        len(hits),
        len(candidates),
    )

    # ── Apply hard filters ────────────────────────────────────────
    filtered_candidates, n_filtered_out = _apply_filters(candidates, effective_filters)

    return filtered_candidates, n_filtered_out


def build_candidate_response(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw retriever candidate dict into the fields expected
    by the reranker and response builder.

    Separates the metadata fields into a nested CandidateMetadata-shaped dict.
    """
    return {
        "candidate_id":    raw["candidate_id"],
        "name":            raw["name"],
        "cv_path":         raw["cv_path"],
        "best_score":      raw["best_score"],
        "best_chunk_text": raw["best_chunk_text"],
        "filter_flags":    raw.get("filter_flags", []),
        "metadata": {
            "experience_years": raw.get("experience_years"),
            "location":         raw.get("location"),
            "location_raw":     raw.get("location_raw"),
            "skills":           raw.get("skills", []),
            "email":            raw.get("email"),
        },
    }
