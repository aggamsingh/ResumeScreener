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
    flags is non-empty when data is missing or soft filter flags apply.
    """
    flags: list[str] = []
    exp = candidate.get("experience_years")

    # No experience filter active → always pass
    if filters.min_experience is None and filters.max_experience is None:
        return True, flags

    if exp is None:
        flags.append("experience_unknown")
        return not filters.strict, flags

    if filters.min_experience is not None and exp < filters.min_experience:
        if filters.strict:
            return False, flags
        flags.append(f"exp_below_min ({exp}y < {filters.min_experience}y)")

    if filters.max_experience is not None and exp > filters.max_experience:
        if filters.strict:
            return False, flags
        flags.append(f"exp_above_max ({exp}y > {filters.max_experience}y)")

    return True, flags


def _check_location(
    candidate: dict[str, Any],
    filters: ScreeningFilters,
) -> tuple[bool, list[str]]:
    """
    Check location against filter string (case-insensitive substring match).

    Allows 'remote', 'any', 'flexible', 'open', 'worldwide', 'hybrid' to match all candidates.
    If strict=False, non-matching location adds a soft flag instead of excluding candidates.
    """
    flags: list[str] = []

    if not filters.location:
        return True, flags

    filter_loc = filters.location.lower().strip()
    if filter_loc in ("remote", "any", "flexible", "open", "worldwide", "hybrid", "all"):
        return True, flags

    loc_canonical = (candidate.get("location") or "").lower()
    loc_raw       = (candidate.get("location_raw") or "").lower()
    chunk_text    = (candidate.get("best_chunk_text") or "").lower()

    if not loc_canonical and not loc_raw:
        flags.append("location_unknown")
        return not filters.strict, flags

    if filter_loc in loc_canonical or filter_loc in loc_raw or filter_loc in chunk_text:
        return True, flags

    if not filters.strict:
        flags.append(f"location_mismatch ({candidate.get('location') or 'unknown'})")
        return True, flags

    return False, flags


def _check_skills(
    candidate: dict[str, Any],
    filters: ScreeningFilters,
) -> tuple[bool, list[str]]:
    """
    Check that required_skills appear in candidate skills or text.
    If strict=True, requires ALL skills. If strict=False, treats missing skills as soft flags.
    """
    flags: list[str] = []

    if not filters.required_skills:
        return True, flags

    candidate_skills_lower = {s.lower() for s in (candidate.get("skills") or [])}
    chunk_text_lower = (candidate.get("best_chunk_text") or "").lower()

    missing = [
        s for s in filters.required_skills
        if s.lower() not in candidate_skills_lower and s.lower() not in chunk_text_lower
    ]

    if missing:
        if filters.strict:
            return False, flags
        else:
            flags.append(f"SKILLS_PARTIAL: missing [{', '.join(missing)}]")

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


def _calculate_qualification_score(
    candidate: dict[str, Any], 
    jd_text: str,
    filters: Optional[ScreeningFilters] = None
) -> float:
    """
    Compute a high-precision hybrid qualification match score (0.0 to 1.0)
    optimized for Job Title alignment, exact keyword matching, and experience fit.
    """
    if not jd_text:
        return float(candidate.get("best_score", 0.5))

    jd_lower = jd_text.lower()
    candidate_skills = [s.lower() for s in (candidate.get("skills") or [])]
    chunk_text_lower = (candidate.get("best_chunk_text") or "").lower()
    candidate_name = (candidate.get("name") or "").lower()
    
    # 1. Skill Matrix Overlap Score (Weight: 35%)
    req_skills = [s.lower() for s in (filters.required_skills if filters and filters.required_skills else [])]
    if req_skills:
        matched_req = sum(1 for req in req_skills if req in candidate_skills or req in chunk_text_lower)
        skill_score = matched_req / max(1, len(req_skills))
    elif candidate_skills:
        matched_in_jd = sum(1 for s in candidate_skills if s in jd_lower)
        skill_score = matched_in_jd / max(1, len(candidate_skills))
    else:
        skill_score = 0.3

    # 2. Job Title & Role Keyword Alignment (Weight: 30%)
    stop_words = {"a", "an", "and", "or", "the", "in", "of", "for", "with", "to", "at", "by", "on", "target", "job", "title"}
    jd_words = [w for w in jd_lower.split() if len(w) > 2 and w not in stop_words][:8]
    title_matches = sum(1 for w in jd_words if w in candidate_name or w in chunk_text_lower[:600])
    title_score = min(1.0, title_matches / max(1, min(4, len(jd_words))))

    # 3. Experience Range Fit (Weight: 15%)
    exp = candidate.get("experience_years")
    exp_score = 0.5
    if exp is not None and filters:
        min_e = filters.min_experience
        max_e = filters.max_experience
        if min_e is not None and max_e is not None:
            if min_e <= exp <= max_e:
                exp_score = 1.0
            elif exp >= min_e:
                exp_score = 0.8
            else:
                exp_score = 0.3
        elif min_e is not None and exp >= min_e:
            exp_score = 1.0
        elif max_e is not None and exp <= max_e:
            exp_score = 1.0

    # 4. Dense Vector Cosine Similarity (Weight: 20%)
    raw_vector_score = float(candidate.get("best_score", 0.5))

    # Hybrid Score Fusion: Skill Matrix (35%) + Title Alignment (30%) + Vector Similarity (20%) + Experience Fit (15%)
    final_score = (skill_score * 0.35) + (title_score * 0.30) + (raw_vector_score * 0.20) + (exp_score * 0.15)
    
    return min(1.0, max(0.0, round(final_score, 4)))


def _deduplicate(hits: list, jd_text: str = "", filters: Optional[ScreeningFilters] = None) -> list[dict[str, Any]]:
    """
    Deduplicate Qdrant search hits by candidate_id.
    Keeps the highest-scoring chunk per candidate and merges metadata.
    Calculates a high-precision hybrid qualification match score.
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

    # Enhance candidate scores using hybrid qualification matrix analysis
    for cand in best.values():
        cand["best_score"] = _calculate_qualification_score(cand, jd_text, filters)

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
        if hasattr(qdrant_client, "query_points"):
            res = qdrant_client.query_points(
                collection_name=collection_name,
                query=jd_vector,
                limit=actual_top_n,
                with_payload=True,
            )
            hits = getattr(res, "points", res)
        elif hasattr(qdrant_client, "search"):
            hits = qdrant_client.search(
                collection_name=collection_name,
                query_vector=jd_vector,
                limit=actual_top_n,
                with_payload=True,
            )
        else:
            hits = []
    except Exception as e:
        logger.warning("Qdrant search error or collection empty: %s", e)
        return [], 0

    if not hits:
        logger.warning("Qdrant returned zero results — is the collection indexed?")
        return [], 0

    # ── Deduplicate & Hybrid Score ────────────────────────────────
    candidates = _deduplicate(hits, jd_text=jd_text, filters=effective_filters)
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
    meta = raw.get("metadata")
    if not isinstance(meta, dict):
        meta = {
            "experience_years": raw.get("experience_years"),
            "location":         raw.get("location"),
            "location_raw":     raw.get("location_raw"),
            "skills":           raw.get("skills", []),
            "email":            raw.get("email"),
        }
    return {
        "candidate_id":    raw.get("candidate_id", ""),
        "name":            raw.get("name", "Candidate"),
        "cv_path":         raw.get("cv_path", ""),
        "best_score":      raw.get("best_score", 0.8),
        "best_chunk_text": raw.get("best_chunk_text", ""),
        "filter_flags":    raw.get("filter_flags", []),
        "metadata":        meta,
    }
