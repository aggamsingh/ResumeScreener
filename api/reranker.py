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


import json
import logging
import os
from typing import Any

import requests

from api.models import Candidate, CandidateMetadata

logger = logging.getLogger(__name__)

# ── Configuration (read once at import time) ───────────────────────────────────

LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "4"))

# Maximum candidates sent to the LLM in a single prompt.
# With 800-char excerpts + metadata, 15 candidates fit comfortably in
# an 8k-token context window (llama-3.1-8b-instant, gemini-1.5-flash).
# Sending more risks silent truncation or a token-limit error.
_MAX_LLM_CANDIDATES = 15


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(jd_text: str, candidates: list[dict[str, Any]], top_k: int) -> str:
    def _candidate_block(i: int, c: dict) -> str:
        meta = c.get("metadata") or {}
        lines = [
            f"Candidate ID: {c['candidate_id']}",
            f"Name: {c['name']}",
        ]
        if meta.get("experience_years") is not None:
            lines.append(f"Experience: {meta['experience_years']} years")
        if meta.get("location"):
            lines.append(f"Location: {meta['location']}")
        if meta.get("skills"):
            lines.append(f"Key Skills: {', '.join(meta['skills'][:15])}")
        lines.append(f"CV Excerpt:\n{c['best_chunk_text'][:1000]}")
        return "\n".join(lines)

    candidate_sections = "\n\n".join(
        _candidate_block(i, c) for i, c in enumerate(candidates)
    )

    return f"""You are an elite AI Talent Acquisition Lead evaluating candidate resumes for a target job opening.

TARGET JOB ROLE & REQUIREMENTS:
{jd_text[:10000]}

CANDIDATES TO EVALUATE:
{candidate_sections}

EVALUATION & RANKING INSTRUCTIONS:
1. Re-rank the candidates so that the MOST RELEVANT candidate appears FIRST (Rank #1 at the top of the list).
2. Rank #1 MUST be the candidate whose experience, exact job title alignment, and technical skills match the target job requirements most strongly.
3. Calculate a precise match score from 0.00 to 1.00 for each candidate based on:
   - Job Title & Domain Alignment (highest weight)
   - Mandatory Skill & Tech Stack Match
   - Experience Seniority Alignment
4. Provide a crisp 2-sentence match reasoning highlighting matching skills, years of experience, and domain fit.
5. Return the JSON array sorted in STRICT DESCENDING ORDER of score (highest match score first).

REQUIRED JSON FORMAT:
[
  {{
    "candidate_id": "<exact id from above>",
    "score": 0.96,
    "reasoning": "Top match: Possesses direct experience as Senior Accountant with 5+ years in Tally ERP, GST filing, and Bank Reconciliation."
  }}
]"""


# ── LLM Callers ───────────────────────────────────────────────────────────────

_groq_client = None

def _call_groq(prompt: str) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq  # type: ignore  # lazy import
        _groq_client = Groq(api_key=GROQ_API_KEY)

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        timeout=LLM_TIMEOUT,
    )
    return response.choices[0].message.content or ""


_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _call_gemini(prompt: str, api_key: str | None = None, temperature: float = 0.1) -> str:
    """
    Plain REST call, not the google-generativeai SDK.

    The SDK drags in google.api_core + grpc. mock_grpc.py has to stub grpc out at
    process start (the real cygrpc DLL is blocked in this environment), and the
    two do not coexist: api_core kept routing responses through its gRPC error
    path, so a simple HTTP 429 came back as "'HTTPStatus' object is not callable"
    and reranking silently fell back to vector ordering on every request.
    This is the same endpoint the frontend already calls directly.

    api_key overrides GEMINI_API_KEY, for callers that accept a per-request key.
    """
    res = requests.post(
        _GEMINI_URL.format(model=GEMINI_MODEL),
        params={"key": api_key or GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        },
        timeout=LLM_TIMEOUT,
    )
    if not res.ok:
        # Surface Google's own message — quota, bad key and retired model all
        # land here and each needs a different fix by the operator.
        raise RuntimeError(f"Gemini HTTP {res.status_code}: {res.text[:300]}")

    candidates = res.json().get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {res.text[:300]}")

    # Thinking models emit multiple parts and only some carry "text".
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p["text"] for p in parts if "text" in p)


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


def _apply_historical_feedback(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Query PostgreSQL candidate_feedback table and adjust scores based on past recruiter feedback."""
    try:
        import psycopg2
        pg_url = os.getenv("DATABASE_URL") or "postgresql://postgres:root@localhost:5432/resume_lens"
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()
        cur.execute("SELECT candidate_id, SUM(score_boost) FROM candidate_feedback GROUP BY candidate_id")
        feedback_map = {str(r[0]): float(r[1]) for r in cur.fetchall()}
        conn.close()

        for c in candidates:
            cid = str(c.get("candidate_id"))
            if cid in feedback_map:
                boost = feedback_map[cid]
                base_score = c.get("best_score", 0.7)
                c["best_score"] = min(1.0, max(0.0, base_score + boost))
    except Exception as e:
        logger.debug("Could not fetch candidate feedback boosts: %s", e)
    return candidates


def _fallback_ranking(candidates: list[dict[str, Any]], top_k: int) -> list[Candidate]:
    """Return candidates sorted by vector score + historical feedback boosts when LLM is unavailable."""
    candidates = _apply_historical_feedback(candidates)
    sorted_candidates = sorted(candidates, key=lambda x: x.get("best_score", 0.0), reverse=True)
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


def generate_llm_response(prompt: str, gemini_api_key_override: str | None = None) -> str:
    """
    Calls the configured LLM (Groq or Gemini) to generate text for a given prompt.
    Supports overriding the Gemini API key (useful if client sends X-Gemini-API-Key).
    """
    provider = LLM_PROVIDER
    
    # If a Gemini API key override is provided, force Gemini provider
    if gemini_api_key_override:
        provider = "gemini"
        
    logger.info("Calling LLM chat/generation via provider='%s'", provider)
    
    try:
        if provider == "groq":
            return _call_groq(prompt)
        elif provider == "gemini":
            # One call path for both the configured key and a per-request
            # override. The override branch used to build a google.generativeai
            # client, which cannot work here — mock_grpc.py stubs out grpc at
            # process start, so any SDK call fails and chat silently dropped to
            # the template reply. Chat is conversational, so it keeps the higher
            # temperature the SDK branch used.
            return _call_gemini(prompt, api_key=gemini_api_key_override, temperature=0.7)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER='{provider}'")
    except Exception as exc:
        logger.error("LLM text generation failed: %s", exc)
        raise exc

