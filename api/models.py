"""
Pydantic models for request validation and response serialization.
"""


import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ── Request Models ─────────────────────────────────────────────────────────────

class CandidateIndexRequest(BaseModel):
    id: str
    name: str
    resume_text: str
    skills: Optional[list[str]] = None
    years_experience: Optional[int] = 0
    location: Optional[str] = None
    cv_path: Optional[str] = None

class ScreeningFilters(BaseModel):
    """
    Hard filters applied before LLM reranking.

    Any candidate who fails a filter is excluded from results entirely —
    they will not be passed to the LLM and will not appear in the response.

    Important: filters only apply to candidates where the relevant metadata
    was successfully extracted from their CV. Candidates where a field could
    not be determined (e.g. experience_years is unknown) are included by
    default and flagged in the response so reviewers can manually verify.
    Set strict=true to exclude candidates with unknown metadata fields.
    """

    min_experience: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum years of total experience (inclusive)",
    )
    max_experience: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum years of total experience (inclusive)",
    )
    location: Optional[str] = Field(
        None,
        description=(
            "Filter by city or region. Case-insensitive substring match against "
            "the location extracted from each CV. E.g. 'Delhi' matches 'Delhi NCR', "
            "'New Delhi', etc."
        ),
    )
    required_skills: Optional[list[str]] = Field(
        None,
        description=(
            "Candidate must have ALL listed skills present in their CV. "
            "Skills are matched against a curated taxonomy (case-insensitive). "
            "Example: ['Python', 'Docker', 'PostgreSQL']"
        ),
    )
    strict: bool = Field(
        False,
        description=(
            "If true, candidates whose metadata could not be extracted are excluded "
            "when the corresponding filter is active. "
            "If false (default), candidates with unknown metadata pass through with a flag."
        ),
    )

    def is_active(self) -> bool:
        """Return True if any filter field has a meaningful value set."""
        return any([
            self.min_experience is not None,
            self.max_experience is not None,
            self.location is not None,
            # Treat empty list as inactive — [] provides no filtering criteria
            # but would otherwise trigger 3x retrieval multiplier for no benefit
            self.required_skills is not None and len(self.required_skills) > 0,
        ])

    @model_validator(mode="after")
    def validate_experience_range(self) -> "ScreeningFilters":
        """Ensure min_experience does not exceed max_experience.

        Without this, a request with min=10, max=5 silently returns zero
        results because no candidate can satisfy exp>=10 AND exp<=5.
        """
        if self.min_experience is not None and self.max_experience is not None:
            if self.min_experience > self.max_experience:
                raise ValueError(
                    f"min_experience ({self.min_experience}) cannot exceed "
                    f"max_experience ({self.max_experience})"
                )
        return self


class ScreeningRequest(BaseModel):
    """Body for POST /api/v1/screen"""

    job_description: str = Field(
        ...,
        min_length=1,
        max_length=50000,  # ~10 pages; prevents OOM on runaway inputs
        description="Full text of the job description to screen against",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Number of top candidates to return (max 50). "
            "Defaults to DEFAULT_TOP_K env var (default: 10)."
        ),
    )
    filters: ScreeningFilters = Field(
        default_factory=ScreeningFilters,
        description="Optional hard filters applied before AI reranking",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_description": (
                        "We are looking for a Senior Python Developer with 5+ years of "
                        "experience in FastAPI, PostgreSQL, and Docker. Experience with "
                        "machine learning pipelines is a plus. Location: Delhi NCR."
                    ),
                    "top_k": 10,
                    "filters": {
                        "min_experience": 5,
                        "max_experience": 15,
                        "location": "Delhi",
                        "required_skills": ["Python", "Docker"],
                        "strict": False,
                    },
                }
            ]
        }
    }


# ── Response Models ────────────────────────────────────────────────────────────

class CandidateMetadata(BaseModel):
    """
    Structured metadata extracted from the candidate's CV.
    All fields are optional — None means the field could not be extracted,
    not that the candidate lacks the qualification.
    """
    experience_years: Optional[int] = Field(
        None,
        description="Extracted years of total experience (None if unknown)",
    )
    location: Optional[str] = Field(
        None,
        description="Canonical city name extracted from CV (None if unknown)",
    )
    location_raw: Optional[str] = Field(
        None,
        description="Raw location string as found in CV before normalization",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Skill keywords matched from the CV",
    )
    email: Optional[str] = Field(
        None,
        description="Contact email extracted from CV (None if not found)",
    )


class Candidate(BaseModel):
    """A single ranked candidate in the screening response."""

    candidate_id:    str = Field(description="Unique identifier for this candidate")
    name:            str = Field(description="Candidate name extracted from their CV")
    score:           float = Field(ge=0.0, le=1.0, description="Fit score (0.0–1.0)")
    match_reasoning: str = Field(description="One-sentence AI explanation of fit")
    cv_path:         str = Field(description="Server-side path to the candidate's CV file")
    metadata:        CandidateMetadata = Field(
        default_factory=CandidateMetadata,
        description="Structured metadata extracted from the CV",
    )
    filter_flags:    list[str] = Field(
        default_factory=list,
        description=(
            "Warnings when a filter was active but the relevant metadata "
            "could not be extracted. Example: ['experience_unknown']. "
            "Empty list means all filter checks passed cleanly."
        ),
    )


class ScreeningResponse(BaseModel):
    """Response from POST /api/v1/screen"""

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this screening request",
    )
    candidates: list[Candidate] = Field(description="Ranked list of top candidates")
    total_filtered_out: int = Field(
        default=0,
        description="Number of candidates excluded by hard filters before reranking",
    )
    screened_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of when screening ran",
    )


class HealthResponse(BaseModel):
    """Response from GET /health"""

    status:           str  = Field(description="'healthy' or 'degraded'")
    qdrant_connected: bool = Field(description="Whether Qdrant is reachable")
    model_loaded:     bool = Field(description="Whether the embedding model is loaded")
    version:          str  = Field(default="1.0.0")


class ChatHistoryItem(BaseModel):
    role: str = Field(description="Role: 'user' or 'model'")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(description="User prompt / question")
    history: Optional[list[ChatHistoryItem]] = Field(default=[], description="Previous chat messages")


class ChatResponse(BaseModel):
    reply: str = Field(description="AI response text in markdown")
    candidates: Optional[list[dict]] = Field(default=[], description="Candidate profile objects matching the request")


class SyncRequest(BaseModel):
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    site_url: Optional[str] = None
    folder_path: Optional[str] = None


class SyncResponse(BaseModel):
    status: str = Field(description="Status: 'success' or 'error'")
    files_processed: int = Field(default=0, description="Total files processed")
    candidates_added: int = Field(default=0, description="New CVs added and indexed")
    message: str = Field(description="Summary message")


class JDRequest(BaseModel):
    job_title: str = Field(description="The job title to generate description for")
    keywords: Optional[list[str]] = Field(default_factory=list, description="User selected keywords/skills to incorporate into the JD")
    min_experience: Optional[int] = Field(None, description="Minimum years of experience")
    max_experience: Optional[int] = Field(None, description="Maximum years of experience")
    salary_lpa: Optional[str] = Field(None, description="Offering salary in LPA")
    location: Optional[str] = Field(None, description="Preferred location")
    education_level: Optional[str] = Field(None, description="Minimum education level")
    freshers_only: Optional[bool] = Field(False, description="Freshers only flag")


class JDResponse(BaseModel):
    job_description: str = Field(description="Synthesized job description")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords/skills")


class InterviewQuestion(BaseModel):
    id: int = Field(description="Question ID")
    question: str = Field(description="Technical or behavioral screening question")
    expectedAnswer: str = Field(description="Key concepts expected in a strong answer")


class InterviewGrade(BaseModel):
    questionId: int = Field(description="Question ID")
    score: int = Field(description="Score out of 100")
    feedback: str = Field(description="Detailed evaluation feedback")


class AssessmentReport(BaseModel):
    score: int = Field(description="Overall technical score out of 100")
    overallFeedback: str = Field(description="Summary verdict and hiring recommendation")
    grades: list[InterviewGrade] = Field(default_factory=list, description="Question grades")


class ScreeningSimulationRequest(BaseModel):
    candidate_id: str = Field(description="Candidate ID")
    job_role: str = Field(description="Target job title / role")
    candidate_name: Optional[str] = Field(None, description="Candidate full name")
    candidate_cv_text: Optional[str] = Field(None, description="Candidate CV text snippet")
    answers: Optional[dict[str, str]] = Field(default=None, description="Candidate answers map keyed by question ID")


class ScreeningSimulationResponse(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list, description="Generated technical screening questions")
    report: Optional[AssessmentReport] = Field(default=None, description="Grading report if answers were evaluated")


class JDMatchCandidate(BaseModel):
    candidate_id: str = Field(description="Candidate ID")
    name: str = Field(description="Candidate full name")
    score: float = Field(description="Match score 0.0 to 1.0")
    match_percentage: int = Field(description="Match percentage 0 to 100%")
    strengths: list[str] = Field(default_factory=list, description="Key candidate strengths for this position")
    gaps: list[str] = Field(default_factory=list, description="Identified skill or experience gaps")
    verdict: str = Field(description="Recruiter verdict and recommendation")
    cv_path: Optional[str] = Field(None, description="Path to candidate resume file")


class JDMatchRequest(BaseModel):
    job_description: str = Field(description="Job description or requirements text")
    top_k: Optional[int] = Field(10, description="Top N candidates to evaluate")


class CandidateFeedbackRequest(BaseModel):
    candidate_id: str = Field(description="Candidate ID")
    query_text: Optional[str] = Field("", description="Query or job role for which feedback was given")
    feedback_type: str = Field("shortlisted", description="Feedback type: shortlisted, hired, rejected, viewed")
    score_boost: Optional[float] = Field(0.1, description="Score adjustment multiplier")


class JDMatchResponse(BaseModel):
    candidates: list[JDMatchCandidate] = Field(default_factory=list, description="Matched candidate leaderboard")
