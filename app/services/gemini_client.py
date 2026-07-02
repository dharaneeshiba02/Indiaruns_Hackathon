"""Gemini LLM integration with deterministic local fallbacks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import Settings
from app.models.schemas import CandidateScore, JobRequirements
from app.utils.text import clean_text, split_to_list

logger = logging.getLogger(__name__)


class GeminiRecruiterClient:
    """LLM adapter with Gemini as the optional remote backend."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=settings.gemini_api_key)
                self._model = genai.GenerativeModel(settings.gemini_model)
                logger.info("Gemini model initialised: %s", settings.gemini_model)
            except ImportError:
                logger.warning("google-generativeai not installed; Gemini unavailable.")
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    @property
    def enabled(self) -> bool:
        """Whether Gemini is configured."""

        return self._model is not None

    def extract_job_requirements(self, job_text: str) -> JobRequirements:
        """Extract structured requirements from a job description."""

        if not self.enabled:
            return heuristic_job_extraction(job_text)

        prompt = (
            "Extract the following fields from the job description as strict JSON: "
            "role, responsibilities, required_skills, preferred_skills, "
            "years_experience, education, soft_skills, industry, tools. "
            "Use arrays for list fields and a number or null for years_experience.\n\n"
            f"Job Description:\n{job_text}"
        )
        try:
            content = self._generate_json(prompt)
            payload = _coerce_job_payload(_extract_json_object(content))
            payload["raw_text"] = job_text
            return JobRequirements.model_validate(payload)
        except Exception as exc:
            logger.warning("Gemini extraction failed; using heuristic fallback: %s", exc)
            return heuristic_job_extraction(job_text)

    def rerank_candidates(
        self, job: JobRequirements, scores: list[CandidateScore], summaries: dict[str, str]
    ) -> list[CandidateScore]:
        """Rerank top candidates with Gemini and attach recruiter-style notes."""

        if not self.enabled or not scores:
            return _fallback_rerank(scores)

        candidates_payload = [
            {
                "candidate_id": score.candidate_id,
                "name": score.candidate_name,
                "score": score.overall_score,
                "summary": summaries.get(score.candidate_id, ""),
            }
            for score in scores
        ]
        prompt = (
            "You are an expert recruiter. Given this structured job description and "
            "candidate summaries, rank candidates from best to worst. Return strict "
            "JSON with an array named rankings. Each item must include candidate_id, "
            "rank, reason, strengths, weaknesses.\n\n"
            f"Job:\n{job.model_dump_json()}\n\n"
            f"Candidates:\n{json.dumps(candidates_payload)}"
        )
        try:
            content = self._generate_json(prompt)
            payload = _extract_json_object(content)
            return _apply_llm_rankings(scores, payload.get("rankings", []))
        except Exception as exc:
            logger.warning("Gemini reranking failed; using score order: %s", exc)
            return _fallback_rerank(scores)

    def _generate_json(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        return response.text or "{}"


def heuristic_job_extraction(job_text: str) -> JobRequirements:
    """Extract a reasonable job schema without an external LLM."""

    text = clean_text(job_text)
    lower = text.lower()
    role = _extract_role(text)
    skills = _extract_terms(
        lower,
        [
            "python",
            "fastapi",
            "sql",
            "machine learning",
            "deep learning",
            "nlp",
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "pytorch",
            "docker",
            "aws",
            "azure",
            "gcp",
            "faiss",
            "langchain",
            "llm",
            "api",
        ],
    )
    soft_skills = _extract_terms(
        lower,
        ["communication", "leadership", "collaboration", "ownership", "problem solving"],
    )
    tools = _extract_terms(lower, ["git", "docker", "kubernetes", "jira", "mlflow", "faiss"])
    years = _extract_year_requirement(lower)
    education = _extract_terms(lower, ["bachelor", "master", "phd", "computer science"])
    return JobRequirements(
        role=role,
        responsibilities=_extract_section_items(text, ["responsibilities", "what you will do"]),
        required_skills=skills,
        preferred_skills=[],
        years_experience=years,
        education=education,
        soft_skills=soft_skills,
        industry="",
        tools=tools,
        raw_text=text,
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    list_fields = [
        "responsibilities",
        "required_skills",
        "preferred_skills",
        "education",
        "soft_skills",
        "tools",
    ]
    coerced = dict(payload)
    for field in list_fields:
        coerced[field] = split_to_list(coerced.get(field))
    if coerced.get("industry") is None:
        coerced["industry"] = ""
    if coerced.get("role") is None:
        coerced["role"] = ""
    try:
        coerced["years_experience"] = (
            None
            if coerced.get("years_experience") in (None, "")
            else float(coerced.get("years_experience"))
        )
    except (TypeError, ValueError):
        coerced["years_experience"] = None
    return coerced


def _extract_role(text: str) -> str:
    lines = [line.strip(" -:") for line in text.splitlines() if line.strip()]
    for line in lines[:5]:
        if any(token in line.lower() for token in ["engineer", "developer", "scientist", "analyst"]):
            return clean_text(line)
    return lines[0] if lines else "Unspecified Role"


def _extract_terms(text: str, vocabulary: list[str]) -> list[str]:
    return [term for term in vocabulary if term in text]


def _extract_year_requirement(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_section_items(text: str, headings: list[str]) -> list[str]:
    for heading in headings:
        pattern = rf"{heading}\s*[:\-]\s*(.*?)(?:\n[A-Z][A-Za-z ]{{2,}}[:\-]|\Z)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return split_to_list(match.group(1))[:8]
    return []


def _apply_llm_rankings(
    scores: list[CandidateScore], rankings: list[dict[str, Any]]
) -> list[CandidateScore]:
    by_id = {score.candidate_id: score for score in scores}
    ordered: list[CandidateScore] = []
    seen: set[str] = set()
    for item in rankings:
        candidate_id = str(item.get("candidate_id", ""))
        score = by_id.get(candidate_id)
        if not score:
            continue
        updated = score.model_copy(
            update={
                "rank": int(item.get("rank") or len(ordered) + 1),
                "reason": clean_text(item.get("reason")) or score.reason,
                "strengths": split_to_list(item.get("strengths")) or score.strengths,
                "weaknesses": split_to_list(item.get("weaknesses")) or score.weaknesses,
            }
        )
        ordered.append(updated)
        seen.add(candidate_id)

    for score in scores:
        if score.candidate_id not in seen:
            ordered.append(score)

    return _assign_ranks(ordered)


def _fallback_rerank(scores: list[CandidateScore]) -> list[CandidateScore]:
    return _assign_ranks(sorted(scores, key=lambda item: item.overall_score, reverse=True))


def _assign_ranks(scores: list[CandidateScore]) -> list[CandidateScore]:
    return [score.model_copy(update={"rank": index}) for index, score in enumerate(scores, start=1)]
