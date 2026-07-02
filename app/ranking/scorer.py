"""Hybrid candidate scoring."""

from __future__ import annotations

from app.models.schemas import CandidateProfile, CandidateScore, JobRequirements
from app.services.candidate_service import CandidateService
from app.utils.text import contains_phrase, lower_set


class CandidateScorer:
    """Computes weighted explainable candidate scores."""

    def __init__(self, weights: dict[str, float]):
        total = sum(weights.values()) or 1.0
        self.weights = {key: value / total for key, value in weights.items()}

    def score(
        self,
        job: JobRequirements,
        candidate: CandidateProfile,
        semantic_score: float,
    ) -> CandidateScore:
        """Return a complete score breakdown for one candidate."""

        skill_score = self._skill_score(job, candidate)
        experience_score = self._experience_score(job, candidate)
        education_score = self._education_score(job, candidate)
        behaviour_score = self._behaviour_score(job, candidate)
        activity_score = candidate.platform_activity if candidate.platform_activity is not None else 0.5

        overall = (
            semantic_score * self.weights.get("semantic", 0)
            + skill_score * self.weights.get("skills", 0)
            + experience_score * self.weights.get("experience", 0)
            + education_score * self.weights.get("education", 0)
            + behaviour_score * self.weights.get("behaviour", 0)
            + activity_score * self.weights.get("activity", 0)
        )
        strengths, weaknesses = self._explain(
            semantic_score,
            skill_score,
            experience_score,
            education_score,
            behaviour_score,
            activity_score,
        )
        return CandidateScore(
            candidate_id=candidate.candidate_id,
            candidate_name=candidate.name,
            overall_score=round(overall * 100, 2),
            semantic_score=round(semantic_score * 100, 2),
            skill_score=round(skill_score * 100, 2),
            experience_score=round(experience_score * 100, 2),
            education_score=round(education_score * 100, 2),
            behaviour_score=round(behaviour_score * 100, 2),
            activity_score=round(activity_score * 100, 2),
            confidence_score=round(self._confidence(job, candidate, overall) * 100, 2),
            reason=self._reason(
                candidate,
                strengths,
                weaknesses,
                semantic_score,
                skill_score,
                experience_score,
                education_score,
                behaviour_score,
            ),
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def _skill_score(self, job: JobRequirements, candidate: CandidateProfile) -> float:
        required = lower_set(job.required_skills + job.preferred_skills + job.tools)
        if not required:
            return 0.7
        candidate_terms = lower_set(candidate.skills)
        candidate_text = CandidateService.summarize(candidate).lower()
        matches = {
            skill
            for skill in required
            if skill in candidate_terms or contains_phrase(candidate_text, skill)
        }
        return len(matches) / len(required)

    def _experience_score(self, job: JobRequirements, candidate: CandidateProfile) -> float:
        if job.years_experience is None:
            return 0.7
        if candidate.experience_years is None:
            return 0.3
        return min(candidate.experience_years / max(job.years_experience, 1.0), 1.0)

    def _education_score(self, job: JobRequirements, candidate: CandidateProfile) -> float:
        required = lower_set(job.education)
        if not required:
            return 0.7
        candidate_text = " ".join(candidate.education + [candidate.resume_text]).lower()
        return 1.0 if any(item in candidate_text for item in required) else 0.3

    def _behaviour_score(self, job: JobRequirements, candidate: CandidateProfile) -> float:
        required = lower_set(job.soft_skills)
        if not required:
            return 0.7
        candidate_text = " ".join(candidate.behavioural_traits + [candidate.resume_text]).lower()
        matches = [trait for trait in required if trait in candidate_text]
        return len(matches) / len(required)

    def _confidence(
        self, job: JobRequirements, candidate: CandidateProfile, overall_score: float
    ) -> float:
        populated = [
            bool(job.raw_text),
            bool(job.required_skills),
            bool(candidate.resume_text),
            bool(candidate.skills),
            candidate.experience_years is not None,
            bool(candidate.education),
        ]
        data_quality = sum(populated) / len(populated)
        return min(1.0, 0.55 * overall_score + 0.45 * data_quality)

    def _explain(
        self,
        semantic: float,
        skills: float,
        experience: float,
        education: float,
        behaviour: float,
        activity: float,
    ) -> tuple[list[str], list[str]]:
        metrics = {
            "semantic fit": semantic,
            "skills": skills,
            "experience": experience,
            "education": education,
            "behaviour": behaviour,
            "platform activity": activity,
        }
        strengths = [name for name, score in metrics.items() if score >= 0.75]
        weaknesses = [name for name, score in metrics.items() if score < 0.45]
        return strengths[:4], weaknesses[:4]

    def _reason(
        self,
        candidate: CandidateProfile,
        strengths: list[str],
        weaknesses: list[str],
        semantic: float,
        skills: float,
        experience: float,
        education: float,
        behaviour: float,
    ) -> str:
        score_summary = (
            f"Score breakdown: semantic {semantic * 100:.1f}, skills {skills * 100:.1f}, "
            f"experience {experience * 100:.1f}, education {education * 100:.1f}, "
            f"behaviour {behaviour * 100:.1f}."
        )
        if strengths and weaknesses:
            return (
                f"{candidate.name} shows clear strength in {', '.join(strengths)}. "
                f"Recruiter review should focus on {', '.join(weaknesses)}. {score_summary}"
            )
        if strengths:
            return (
                f"{candidate.name} is a strong shortlist candidate because of "
                f"{', '.join(strengths)}. {score_summary}"
            )
        return (
            f"{candidate.name} has a partial match and should be reviewed carefully against "
            f"the job requirements. {score_summary}"
        )
