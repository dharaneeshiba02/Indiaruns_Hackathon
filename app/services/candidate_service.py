"""Candidate dataset loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.schemas import CandidateProfile
from app.utils.file_loader import extract_text_from_file
from app.utils.text import clean_text, extract_years, split_to_list


class CandidateService:
    """Loads CSV/JSON candidate datasets into normalized profiles."""

    def load_candidates(self, path: str | Path) -> list[CandidateProfile]:
        """Read CSV or JSON candidate data from disk."""

        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            rows = pd.read_csv(file_path).fillna("").to_dict(orient="records")
        elif suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("candidates", [])
        else:
            raise ValueError("Candidate data must be a CSV or JSON file.")
        return [self._normalize_row(row, index) for index, row in enumerate(rows, start=1)]

    def _normalize_row(self, row: dict[str, Any], index: int) -> CandidateProfile:
        normalized = {_normalize_key(key): value for key, value in row.items()}
        candidate_id = clean_text(
            _first(normalized, ["candidate_id", "id", "email", "profile_id"])
        ) or f"CAND-{index:04d}"
        name = clean_text(_first(normalized, ["candidate_name", "name", "full_name"])) or candidate_id
        resume_text = clean_text(_first(normalized, ["resume_text", "resume", "summary", "profile"]))
        resume_path = clean_text(_first(normalized, ["resume_path", "file_path"]))
        if resume_path:
            candidate_file = Path(resume_path)
            if candidate_file.exists():
                resume_text = f"{resume_text} {extract_text_from_file(candidate_file)}".strip()

        skills = split_to_list(_first(normalized, ["skills", "technical_skills", "skill_set"]))
        projects = split_to_list(_first(normalized, ["projects", "project"]))
        education = split_to_list(_first(normalized, ["education", "degree", "qualification"]))
        achievements = split_to_list(_first(normalized, ["achievements", "awards"]))
        traits = split_to_list(
            _first(normalized, ["behavioural_traits", "behavioral_traits", "soft_skills"])
        )
        recent_work = split_to_list(_first(normalized, ["recent_work", "recent_projects"]))
        platform_activity = _parse_activity(
            _first(normalized, ["platform_activity", "activity_score", "github_activity"])
        )

        return CandidateProfile(
            candidate_id=candidate_id,
            name=name,
            resume_text=resume_text,
            skills=skills,
            experience_years=extract_years(
                _first(normalized, ["experience_years", "years_experience", "experience"])
            ),
            projects=projects,
            education=education,
            achievements=achievements,
            behavioural_traits=traits,
            platform_activity=platform_activity,
            recent_work=recent_work,
            raw_profile=row,
        )

    @staticmethod
    def summarize(candidate: CandidateProfile) -> str:
        """Create a compact text summary for embedding and LLM prompts."""

        parts = [
            candidate.name,
            candidate.resume_text,
            "Skills: " + ", ".join(candidate.skills),
            "Projects: " + ", ".join(candidate.projects),
            "Education: " + ", ".join(candidate.education),
            "Achievements: " + ", ".join(candidate.achievements),
            "Traits: " + ", ".join(candidate.behavioural_traits),
            "Recent Work: " + ", ".join(candidate.recent_work),
        ]
        return clean_text(" ".join(part for part in parts if part))


def _normalize_key(key: str) -> str:
    return clean_text(key).lower().replace(" ", "_").replace("-", "_")


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_activity(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        score = float(value)
        return max(0.0, min(1.0, score / 100 if score > 1 else score))
    except ValueError:
        text = clean_text(value).lower()
        if any(word in text for word in ["high", "active", "strong"]):
            return 0.85
        if any(word in text for word in ["medium", "moderate"]):
            return 0.60
        if any(word in text for word in ["low", "inactive"]):
            return 0.25
    return None
