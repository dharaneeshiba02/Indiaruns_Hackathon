"""Candidate dataset loading and normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.schemas import CandidateProfile
from app.utils.file_loader import extract_text_from_file
from app.utils.text import clean_text, extract_years, split_to_list


class CandidateService:
    """Loads candidate datasets or single resume documents into profiles."""

    def load_candidates(self, path_or_paths: str | Path | list[str | Path]) -> list[CandidateProfile]:
        """Read CSV/JSON candidate data or resume document(s) from disk."""

        paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        
        profiles = []
        for path in paths:
            file_path = Path(path)
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                rows = pd.read_csv(file_path).fillna("").to_dict(orient="records")
                profiles.extend([self._normalize_row(row, len(profiles) + index) for index, row in enumerate(rows, start=1)])
            elif suffix == ".json":
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                rows = payload if isinstance(payload, list) else payload.get("candidates", [])
                profiles.extend([self._normalize_row(row, len(profiles) + index) for index, row in enumerate(rows, start=1)])
            elif suffix in {".pdf", ".docx", ".txt"}:
                profiles.append(self._profile_from_document(file_path))
            else:
                raise ValueError(f"Unsupported candidate data format for {file_path.name}")
        return profiles

    def _profile_from_document(self, path: Path) -> CandidateProfile:
        resume_text = extract_text_from_file(path)
        name = _extract_name_from_resume(resume_text) or path.stem.replace("_", " ").replace("-", " ").title()
        candidate_id = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") or path.stem.upper()
        return CandidateProfile(
            candidate_id=candidate_id,
            name=name,
            resume_text=resume_text,
            raw_profile={"resume_path": str(path)},
        )

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


def _extract_name_from_resume(text: str) -> str | None:
    """Extract a likely person name from the beginning of a resume."""

    header = clean_text(text)[:220]
    if not header:
        return None

    uppercase_match = re.match(r"^([A-Z][A-Z.'-]*(?:\s+[A-Z][A-Z.'-]*){1,4})(?=\s+[A-Z][a-z]|\s*[|,@]|\s+\+?\d)", header)
    if uppercase_match:
        return uppercase_match.group(1).title()

    contact_pattern = r"\s(?:email|phone|linkedin|github|mobile)\b|[|,@]|\+?\d{5,}"
    if not re.search(contact_pattern, header, flags=re.IGNORECASE):
        return None

    before_contact = re.split(contact_pattern, header, maxsplit=1, flags=re.IGNORECASE)[0]
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", before_contact)
    stop_words = {
        "resume",
        "curriculum",
        "vitae",
        "professional",
        "summary",
        "skills",
        "experience",
        "education",
    }
    name_words = [word for word in words[:5] if word.lower() not in stop_words]
    if 2 <= len(name_words) <= 5:
        return " ".join(name_words).title()
    return None


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
