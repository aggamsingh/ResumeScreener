"""
LLM reranker: takes retrieved candidates and a job description,
calls Groq or Gemini to produce ranked results with reasoning,
and falls back gracefully to vector-score ordering if the LLM fails.

Design principles:
  - One API call per screening request
  - 15-second timeout, automatic fallback on any failure
  - Provider switched via LLM_PROVIDER env var ("groq" | "gemini")
  - Imports are lazy so the unused provider SDK never loads
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from api.models import Candidate, CandidateMetadata

logger = logging.getLogger(__name__)

# ── Configuration (read once at import time) ───────────────────────────────────

LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

_LLM_TIMEOUT_SECONDS = 20

# Maximum candidates sent to the LLM in a single prompt.
# With 800-char excerpts + metadata, 15 candidates fit comfortably in
# an 8k-token context window (llama-3.1-8b-instant, gemini-1.5-flash).
# Sending more risks silent truncation or a token-limit error.
_MAX_LLM_CANDIDATES = 15


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(jd_text: str, candidates: list[dict[str, Any]], top_k: int) -> str:
    def _candidate_block(i: int, c: dict) -> str:
        meta = c.get("metadata", {})
        lines = [
            f"--- Candidate {i + 1} ---",
            f"ID: {c['candidate_id']}",
            f"Name: {c['name']}",
        ]
        if meta.get("experience_years") is not None:
            lines.append(f"Experience: {meta['experience_years']} years")
        if meta.get("location"):
            lines.append(f"Location: {meta['location']}")
        if meta.get("skills"):
            lines.append(f"Key Skills: {', '.join(meta['skills'][:15])}")
        lines.append(f"CV Excerpt:\n{c['best_chunk_text'][:800]}")
        return "\n".join(lines)

    candidate_sections = "\n\n".join(
        _candidate_block(i, c) for i, c in enumerate(candidates)
    )

    return f"""You are an expert recruitment assistant helping an HR team shortlist candidates.

JOB DESCRIPTION:
{jd_text[:2500]}

CANDIDATES TO EVALUATE:
{candidate_sections}

TASK:
1. Select the top {top_k} most suitable candidates for this role.
2. Assign each a fit score from 0.0 (no fit) to 1.0 (perfect fit).
3. Write one concise sentence explaining why each candidate is a good fit.

RULES:
- Base your ranking on skills, experience, and role alignment.
- Consider any location or experience requirements mentioned in the job description.
- Return ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON.

REQUIRED OUTPUT FORMAT:
[
  {{
    "candidate_id": "<exact id from above>",
    "score": 0.95,
    "reasoning": "Concise one-sentence explanation of fit."
  }}
]"""


# ── LLM Callers ───────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    from groq import Groq  # type: ignore  # lazy import

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        timeout=_LLM_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content or ""


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai  # type: ignore  # lazy import

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.1),
        request_options={"timeout": _LLM_TIMEOUT_SECONDS},
    )
    return response.text or ""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping that some LLMs add."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1])
    return text.strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank_candidates(
    jd_text: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[Candidate]:
    """
    Rerank retrieved candidates using the configured LLM.

    Falls back to vector-score ordering with a note in match_reasoning
    if the LLM call fails for any reason (timeout, bad key, rate limit,
    malformed JSON, etc.). The service never goes down due to LLM issues.

    Args:
        jd_text:    Raw job description text.
        candidates: Deduplicated candidates from retriever.retrieve_candidates().
        top_k:      How many to return in the final response.

    Returns:
        List of Candidate objects, ranked best-first.
    """
    if not candidates:
        return []

    # Cap the number of candidates sent to the LLM.
    # Passing all retrieved candidates (up to top_k*3) can easily exceed
    # the model's context window, causing silent truncation or API errors.
    candidates_for_llm = candidates[:_MAX_LLM_CANDIDATES]
    if len(candidates) > _MAX_LLM_CANDIDATES:
        logger.debug(
            "Capping LLM input: %d candidates → %d (MAX_LLM_CANDIDATES=%d)",
            len(candidates), _MAX_LLM_CANDIDATES, _MAX_LLM_CANDIDATES,
        )

    prompt = _build_prompt(jd_text, candidates_for_llm, top_k)

    try:
        logger.info("Calling LLM reranker via provider='%s', model='%s'", LLM_PROVIDER, _get_model_name())

        if LLM_PROVIDER == "groq":
            raw = _call_groq(prompt)
        elif LLM_PROVIDER == "gemini":
            raw = _call_gemini(prompt)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Must be 'groq' or 'gemini'.")

        cleaned = _strip_markdown_fences(raw)
        ranked_items: list[dict] = json.loads(cleaned)

        if not isinstance(ranked_items, list):
            raise ValueError(f"LLM response was not a JSON array: {type(ranked_items)}")

        logger.info("LLM returned %d ranked candidates", len(ranked_items))

        # Map LLM output back to full candidate metadata
        lookup = {c["candidate_id"]: c for c in candidates}
        result: list[Candidate] = []

        for item in ranked_items[:top_k]:
            cid = item.get("candidate_id", "")
            if cid not in lookup:
                logger.warning("LLM returned unknown candidate_id='%s' — skipping", cid)
                continue

            meta = lookup[cid]
            try:
                score = min(1.0, max(0.0, float(item.get("score", meta["best_score"]))))
            except (TypeError, ValueError):
                score = min(1.0, max(0.0, round(meta["best_score"], 4)))
            result.append(
                Candidate(
                    candidate_id=cid,
                    name=meta["name"],
                    score=score,
                    match_reasoning=str(item.get("reasoning", "")).strip(),
                    cv_path=meta["cv_path"],
                    metadata=CandidateMetadata(**meta.get("metadata", {})),
                    filter_flags=meta.get("filter_flags", []),
                )
            )

        # Backfill: if the LLM hallucinated IDs and returned fewer than top_k,
        # fill remaining slots with the best vector-scored candidates not already included.
        if len(result) < top_k:
            used_ids = {c.candidate_id for c in result}
            remaining = sorted(
                [c for c in candidates if c["candidate_id"] not in used_ids],
                key=lambda x: x["best_score"],
                reverse=True,
            )
            for c in remaining[: top_k - len(result)]:
                result.append(
                    Candidate(
                        candidate_id=c["candidate_id"],
                        name=c["name"],
                        score=min(1.0, max(0.0, round(c["best_score"], 4))),
                        match_reasoning="Ranked by semantic similarity (LLM did not rank this candidate).",
                        cv_path=c["cv_path"],
                        metadata=CandidateMetadata(**c.get("metadata", {})),
                        filter_flags=c.get("filter_flags", []),
                    )
                )
            if len(result) > len(ranked_items[:top_k]):
                logger.info(
                    "Backfilled %d candidate(s) to reach top_k=%d",
                    len(result) - len(ranked_items[:top_k]),
                    top_k,
                )

        return result

    except Exception as exc:
        logger.error(
            "LLM reranking failed (%s: %s). Falling back to vector similarity scores.",
            type(exc).__name__,
            exc,
        )
        return _fallback_ranking(candidates, top_k)


def _get_model_name() -> str:
    return GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL


def _fallback_ranking(candidates: list[dict[str, Any]], top_k: int) -> list[Candidate]:
    """Return candidates sorted by Qdrant vector score when LLM is unavailable."""
    sorted_candidates = sorted(candidates, key=lambda x: x["best_score"], reverse=True)
    return [
        Candidate(
            candidate_id=c["candidate_id"],
            name=c["name"],
            # Clamp to [0.0, 1.0] — cosine scores can slightly exceed 1.0
            # due to floating-point precision. Without clamping, Pydantic's
            # le=1.0 constraint on Candidate.score raises a ValidationError,
            # crashing the fallback path that is meant to be the safety net.
            score=min(1.0, max(0.0, round(c["best_score"], 4))),
            match_reasoning="Ranked by semantic similarity (AI reranker temporarily unavailable).",
            cv_path=c["cv_path"],
            metadata=CandidateMetadata(**c.get("metadata", {})),
            filter_flags=c.get("filter_flags", []),
        )
        for c in sorted_candidates[:top_k]
    ]
