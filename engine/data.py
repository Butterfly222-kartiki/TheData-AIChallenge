"""
engine/data.py

Shared candidate data loading + text-construction helpers.

This matches the REAL Redrob hackathon schema (candidate_schema.json),
not a guessed one:

{
  "candidate_id": "CAND_0000001",
  "profile": {
      "anonymized_name", "headline", "summary", "location", "country",
      "years_of_experience", "current_title", "current_company",
      "current_company_size", "current_industry"
  },
  "career_history": [
      {"company","title","start_date","end_date","duration_months",
       "is_current","industry","company_size","description"}, ...
  ],
  "education": [ {"institution","degree","field_of_study","start_year",
                   "end_year","grade","tier"}, ... ],
  "skills": [ {"name","proficiency","endorsements","duration_months"}, ... ],
  "certifications": [...],
  "languages": [...],
  "redrob_signals": {
      "profile_completeness_score","signup_date","last_active_date",
      "open_to_work_flag","profile_views_received_30d",
      "applications_submitted_30d","recruiter_response_rate",
      "avg_response_time_hours","skill_assessment_scores",
      "connection_count","endorsements_received","notice_period_days",
      "expected_salary_range_inr_lpa":{"min","max"},"preferred_work_mode",
      "willing_to_relocate","github_activity_score","search_appearance_30d",
      "saved_by_recruiters_30d","interview_completion_rate",
      "offer_acceptance_rate","verified_email","verified_phone",
      "linkedin_connected"
  }
}
"""

from __future__ import annotations
import gzip
import json
from datetime import date, datetime
from typing import Iterator, Optional


def _open_any(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8" if "t" in mode else None)
    return open(path, mode, encoding="utf-8" if "t" in mode else None)


def load_candidates(path: str, limit: Optional[int] = None) -> list[dict]:
    """Loads candidates from a .jsonl, .jsonl.gz, or .json file."""
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with _open_any(path) as f:
            candidates = json.load(f)
        if limit:
            candidates = candidates[:limit]
        return candidates

    candidates = []
    with _open_any(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def iter_candidates(path: str) -> Iterator[dict]:
    """Memory-friendly streaming load for the full 100K-candidate pool."""
    if path.endswith(".json") and not path.endswith(".jsonl"):
        for c in load_candidates(path):
            yield c
        return
    with _open_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Field accessors — single source of truth for the real schema's field paths
# ---------------------------------------------------------------------------

def cid(c: dict) -> str:
    return c.get("candidate_id", "")


def yoe(c: dict) -> float:
    return float(c.get("profile", {}).get("years_of_experience", 0) or 0)


def current_title(c: dict) -> str:
    return c.get("profile", {}).get("current_title", "") or ""


def current_industry(c: dict) -> str:
    return c.get("profile", {}).get("current_industry", "") or ""


def current_company_size(c: dict) -> str:
    return c.get("profile", {}).get("current_company_size", "") or ""


def location_city(c: dict) -> str:
    # profile.location is "City, State" — take the city part.
    loc = c.get("profile", {}).get("location", "") or ""
    return loc.split(",")[0].strip() if loc else ""


def location_country(c: dict) -> str:
    return c.get("profile", {}).get("country", "") or ""


def career_history(c: dict) -> list[dict]:
    return c.get("career_history", []) or []


def skills(c: dict) -> list[dict]:
    return c.get("skills", []) or []


def signals(c: dict) -> dict:
    return c.get("redrob_signals", {}) or {}


def total_career_months(c: dict) -> int:
    return sum(int(r.get("duration_months", 0) or 0) for r in career_history(c))


def career_text(c: dict) -> str:
    """Concatenated career narrative used by Channel 1 (semantic)."""
    profile = c.get("profile", {})
    parts = [profile.get("headline", "") or "", profile.get("summary", "") or ""]
    for role in career_history(c):
        parts.append(role.get("title", "") or "")
        parts.append(role.get("company", "") or "")
        parts.append(role.get("description", "") or "")
    parts.append(profile.get("current_title", "") or "")
    return " . ".join(p for p in parts if p).strip()


def skill_text(c: dict) -> str:
    """Joined skill-name text used for skill embedding / stuffer detection."""
    return ", ".join(s.get("name", "") for s in skills(c) if s.get("name"))


def proficiency_weight(level: str) -> float:
    return {
        "beginner": 1.0,
        "intermediate": 2.0,
        "advanced": 3.0,
        "expert": 4.0,
    }.get((level or "").lower(), 1.0)


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_since(d: Optional[date], reference: date) -> int:
    if d is None:
        return 9999
    return (reference - d).days

