"""
CV Metadata Extraction — regex + heuristic based.

Extracts from raw CV text:
  - experience_years  (int | None)
  - location          (str | None)  — normalized city name
  - skills            (list[str])   — matched against a curated taxonomy
  - email             (str | None)

All extraction is best-effort. None is returned for any field that
cannot be determined reliably. Callers must handle null metadata gracefully.

Design note: This module uses only the Python standard library (re) plus
the text already extracted by the parser. No LLM calls, no network,
no additional model weight — pure string processing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  Data structure
# ══════════════════════════════════════════════════════════════

@dataclass
class CVMetadata:
    """Structured metadata extracted from a CV's raw text."""
    experience_years: Optional[int] = None
    location:         Optional[str] = None   # Canonical city name, e.g. "Bangalore"
    location_raw:     Optional[str] = None   # Original extracted string before normalization
    skills:           list[str] = field(default_factory=list)
    email:            Optional[str] = None


# ══════════════════════════════════════════════════════════════
#  Experience extraction
# ══════════════════════════════════════════════════════════════

# Ordered from most specific (explicit statement) to least specific (bare number near word)
_EXP_PATTERNS: list[tuple[str, int]] = [
    # "Total Experience: 6 years" / "Work Experience: 5+ yrs"
    (r'(?:total|overall|work|professional|industry)\s+experience\s*[:–\-]\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)', 0),
    # "6+ years of experience"  /  "6 years of professional experience"
    (r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+of\s+(?:relevant\s+|professional\s+|work\s+)?experience', 0),
    # "experience of 6 years"
    (r'experience\s+of\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)', 0),
    # "6 years experience" (no "of")
    (r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+experience', 0),
    # "over 5 years" / "more than 5 years" near experience
    (r'(?:over|more\s+than|above)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', 0),
]

# Cap extracted years at a sane maximum (avoids misparse of years like "2018")
_MAX_PLAUSIBLE_EXPERIENCE = 45
_MIN_PLAUSIBLE_EXPERIENCE = 0

def _extract_experience(text: str) -> Optional[int]:
    """
    Try each pattern in order of specificity.
    Returns the first plausible integer match, or None.
    """
    text_lower = text.lower()
    for pattern, _ in _EXP_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            try:
                years = float(match.group(1))
                if _MIN_PLAUSIBLE_EXPERIENCE <= years <= _MAX_PLAUSIBLE_EXPERIENCE:
                    return int(years)
            except (IndexError, ValueError):
                continue
    return None


# ══════════════════════════════════════════════════════════════
#  Location extraction
# ══════════════════════════════════════════════════════════════

# Mapping: lowercase alias → canonical display name
_CITY_ALIASES: dict[str, str] = {
    # Indian metros and major cities
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "ncr": "Delhi NCR",
    "delhi ncr": "Delhi NCR",
    "noida": "Noida",
    "gurgaon": "Gurgaon",
    "gurugram": "Gurgaon",
    "faridabad": "Faridabad",
    "ghaziabad": "Ghaziabad",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "navi mumbai": "Mumbai",
    "thane": "Mumbai",
    "pune": "Pune",
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "chennai": "Chennai",
    "madras": "Chennai",
    "hyderabad": "Hyderabad",
    "secunderabad": "Hyderabad",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "chandigarh": "Chandigarh",
    "lucknow": "Lucknow",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "coimbatore": "Coimbatore",
    "indore": "Indore",
    "bhopal": "Bhopal",
    "nagpur": "Nagpur",
    "surat": "Surat",
    "patna": "Patna",
    "bhubaneswar": "Bhubaneswar",
    "thiruvananthapuram": "Thiruvananthapuram",
    "trivandrum": "Thiruvananthapuram",
    "visakhapatnam": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "vadodara": "Vadodara",
    "baroda": "Vadodara",
    "mysore": "Mysore",
    "mysuru": "Mysore",
    "nashik": "Nashik",
    "rajkot": "Rajkot",
    "mangalore": "Mangalore",
    "mangaluru": "Mangalore",
    "agra": "Agra",
    "meerut": "Meerut",
    "varanasi": "Varanasi",
    "amritsar": "Amritsar",
    "ludhiana": "Ludhiana",
    "jodhpur": "Jodhpur",
    "udaipur": "Udaipur",
    "dehradun": "Dehradun",
    "shimla": "Shimla",
    "guwahati": "Guwahati",
    "bhopal": "Bhopal",
    "raipur": "Raipur",
    "ranchi": "Ranchi",
    # International
    "new york": "New York",
    "london": "London",
    "singapore": "Singapore",
    "dubai": "Dubai",
    "abu dhabi": "Abu Dhabi",
    "san francisco": "San Francisco",
    "seattle": "Seattle",
    "toronto": "Toronto",
    "sydney": "Sydney",
    "berlin": "Berlin",
    "paris": "Paris",
    "amsterdam": "Amsterdam",
    "austin": "Austin",
    "boston": "Boston",
    "chicago": "Chicago",
    "los angeles": "Los Angeles",
    "melbourne": "Melbourne",
    "kuala lumpur": "Kuala Lumpur",
}

# Lines starting with these markers strongly indicate a location follows
_LOCATION_MARKERS = re.compile(
    r'(?:^|\n)\s*(?:location|city|based\s+in|address|residing|current\s+city'
    r'|current\s+location|hometown|present\s+location)\s*[:\-–]\s*([^\n]{3,80})',
    re.IGNORECASE,
)

def _extract_location(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (canonical_city, raw_extracted_string).
    Tries marker-based extraction first, then scans for known city names.
    """
    text_lower = text.lower()

    # ── Strategy 1: labelled field ──────────────────────────────
    for match in _LOCATION_MARKERS.finditer(text):
        raw = match.group(1).strip().rstrip(".,;")
        raw_lower = raw.lower()
        # Check if a known city appears in this label value
        for alias, canonical in _CITY_ALIASES.items():
            if alias in raw_lower:
                return canonical, raw
        # Return raw even if not canonicalized (better than nothing)
        if 2 <= len(raw.split()) <= 5:
            return None, raw

    # ── Strategy 2: first 500 chars often have contact info ─────
    header = text[:500].lower()
    for alias, canonical in sorted(_CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        # Use word-boundary matching to avoid false positives
        if re.search(r'\b' + re.escape(alias) + r'\b', header):
            return canonical, alias.title()

    return None, None


# ══════════════════════════════════════════════════════════════
#  Skills extraction
# ══════════════════════════════════════════════════════════════

# Curated taxonomy. Stored lowercase here; displayed in their canonical form.
# Format: {lowercase_keyword: display_form}
_SKILL_TAXONOMY: dict[str, str] = {
    # Programming languages
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "c++": "C++", "c#": "C#",
    "golang": "Go", "go lang": "Go", "rust": "Rust", "php": "PHP",
    "ruby": "Ruby", "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala",
    "r language": "R", "matlab": "MATLAB", "perl": "Perl",
    "bash": "Bash", "shell scripting": "Shell Scripting",
    "powershell": "PowerShell",
    # Web / API
    "html": "HTML", "css": "CSS", "rest api": "REST API",
    "graphql": "GraphQL", "grpc": "gRPC", "websocket": "WebSocket",
    # Frameworks
    "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
    "spring boot": "Spring Boot", "spring": "Spring", ".net": ".NET",
    "asp.net": "ASP.NET", "laravel": "Laravel", "rails": "Ruby on Rails",
    "react": "React", "angular": "Angular", "vue": "Vue.js",
    "nextjs": "Next.js", "next.js": "Next.js",
    "nodejs": "Node.js", "node.js": "Node.js", "express": "Express.js",
    "fastify": "Fastify",
    # Databases
    "mysql": "MySQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mongodb": "MongoDB", "redis": "Redis", "elasticsearch": "Elasticsearch",
    "oracle": "Oracle DB", "sqlite": "SQLite", "dynamodb": "DynamoDB",
    "cassandra": "Cassandra", "neo4j": "Neo4j", "firebase": "Firebase",
    "mssql": "MSSQL", "sql server": "SQL Server", "mariadb": "MariaDB",
    # Cloud & DevOps
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "google cloud": "Google Cloud",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "terraform": "Terraform", "ansible": "Ansible", "jenkins": "Jenkins",
    "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
    "ci/cd": "CI/CD", "nginx": "Nginx", "apache": "Apache",
    "linux": "Linux", "unix": "Unix",
    # Data & Analytics
    "power bi": "Power BI", "tableau": "Tableau", "excel": "Excel",
    "apache spark": "Apache Spark", "spark": "Apache Spark",
    "hadoop": "Hadoop", "kafka": "Apache Kafka",
    "airflow": "Apache Airflow", "dbt": "dbt", "snowflake": "Snowflake",
    "bigquery": "BigQuery", "looker": "Looker", "grafana": "Grafana",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    # AI / ML
    "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "keras": "Keras", "hugging face": "Hugging Face",
    "langchain": "LangChain", "openai": "OpenAI API",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "computer vision": "Computer Vision", "llm": "LLM",
    "rag": "RAG", "generative ai": "Generative AI",
    "data science": "Data Science", "data engineering": "Data Engineering",
    "mlops": "MLOps",
    # Business / ERP
    "sap": "SAP", "erp": "ERP", "crm": "CRM",
    "salesforce": "Salesforce", "jira": "Jira", "confluence": "Confluence",
    "ms project": "MS Project", "asana": "Asana",
    # Methodologies
    "agile": "Agile", "scrum": "Scrum", "kanban": "Kanban",
    "devops": "DevOps", "microservices": "Microservices",
    "git": "Git", "project management": "Project Management",
    "system design": "System Design",
    # Accounting / Finance (common in ERP contexts)
    "tally": "Tally", "quickbooks": "QuickBooks", "zoho": "Zoho",
    "accounting": "Accounting", "gst": "GST", "payroll": "Payroll",
    "financial analysis": "Financial Analysis",
}

# Compile a single regex that matches any skill keyword (word boundaries)
_SKILL_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9])(' + '|'.join(re.escape(k) for k in sorted(_SKILL_TAXONOMY, key=len, reverse=True)) + r')(?![a-zA-Z0-9])',
    re.IGNORECASE,
)

def _extract_skills(text: str) -> list[str]:
    """
    Scan CV text for known skill keywords.
    Returns a deduplicated list of canonical skill display names.
    """
    found_canonical: dict[str, str] = {}  # canonical_lower → display
    for match in _SKILL_PATTERN.finditer(text):
        keyword = match.group(1).lower()
        if keyword in _SKILL_TAXONOMY:
            canonical = _SKILL_TAXONOMY[keyword]
            found_canonical[canonical.lower()] = canonical

    return sorted(found_canonical.values())


# ══════════════════════════════════════════════════════════════
#  Email extraction
# ══════════════════════════════════════════════════════════════

_EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
)

def _extract_email(text: str) -> Optional[str]:
    """Return the first email address found in the text, or None."""
    match = _EMAIL_PATTERN.search(text)
    return match.group(0).lower() if match else None


# ══════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════

def extract_metadata(text: str) -> CVMetadata:
    """
    Extract structured metadata from raw CV text.

    All fields are best-effort. None / empty list means the field
    could not be reliably determined — not that the candidate lacks it.
    Callers must not penalize candidates for missing metadata.

    Args:
        text: Raw plain-text content of a CV.

    Returns:
        CVMetadata instance with all available fields populated.
    """
    location_canonical, location_raw = _extract_location(text)
    return CVMetadata(
        experience_years=_extract_experience(text),
        location=location_canonical,
        location_raw=location_raw,
        skills=_extract_skills(text),
        email=_extract_email(text),
    )
