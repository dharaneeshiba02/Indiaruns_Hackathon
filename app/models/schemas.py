"""Typed schemas shared by services and API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobRequirements(BaseModel):
    """Structured job requirements extracted from a job description."""

    role: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    education: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    industry: str = ""
    tools: list[str] = Field(default_factory=list)
    raw_text: str = ""


class CandidateProfile(BaseModel):
    """Normalized representation of one candidate."""

    candidate_id: str
    name: str
    resume_text: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_years: float | None = None
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    behavioural_traits: list[str] = Field(default_factory=list)
    platform_activity: float | None = None
    recent_work: list[str] = Field(default_factory=list)
    raw_profile: dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    """Explainable score breakdown for a candidate."""

    rank: int = 0
    candidate_id: str
    candidate_name: str
    overall_score: float
    semantic_score: float
    skill_score: float
    experience_score: float
    education_score: float
    behaviour_score: float
    activity_score: float
    confidence_score: float
    reason: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class RankingResponse(BaseModel):
    """API response for ranking results."""

    results: list[CandidateScore]
    output_file: str
