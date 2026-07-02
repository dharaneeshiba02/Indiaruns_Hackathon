"""End-to-end ranking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import Settings
from app.embeddings.encoder import EmbeddingEncoder
from app.embeddings.vector_store import VectorStore
from app.models.schemas import CandidateProfile, CandidateScore, JobRequirements
from app.ranking.scorer import CandidateScorer
from app.services.candidate_service import CandidateService
from app.services.gemini_client import GeminiRecruiterClient
from app.services.job_service import JobService
from app.utils.text import clean_text

logger = logging.getLogger(__name__)


class RankingPipeline:
    """Coordinates parsing, embeddings, hybrid scoring, reranking, and export."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.job_service = JobService(settings)
        self.candidate_service = CandidateService()
        self.encoder = EmbeddingEncoder(settings.embedding_model_name, settings.embedding_backend)
        self.scorer = CandidateScorer(settings.score_weights)
        self.llm = GeminiRecruiterClient(settings)

    def rank_from_files(
        self, job_path: str | Path, candidates_path: str | Path | list[str | Path]
    ) -> tuple[list[CandidateScore], Path, JobRequirements]:
        """Run the full pipeline from uploaded files."""

        job = self.job_service.load_from_file(job_path)
        candidates = self.candidate_service.load_candidates(candidates_path)
        results, output_path = self.rank(job, candidates)
        return results, output_path, job

    def rank(
        self, job: JobRequirements, candidates: list[CandidateProfile]
    ) -> tuple[list[CandidateScore], Path]:
        """Rank candidates for a structured job."""

        if not candidates:
            raise ValueError("No candidates were provided.")

        job_text = self._job_text(job)
        summaries = {
            candidate.candidate_id: CandidateService.summarize(candidate)
            for candidate in candidates
        }
        candidate_vectors = self.encoder.encode(summaries.values())
        job_vector = self.encoder.encode(job_text)
        store = VectorStore(candidate_vectors, list(summaries.keys()))
        semantic_scores = {candidate_id: _to_unit(score) for candidate_id, score in store.search(job_vector)}

        hybrid_scores = [
            self.scorer.score(job, candidate, semantic_scores.get(candidate.candidate_id, 0.0))
            for candidate in candidates
        ]
        hybrid_scores.sort(key=lambda item: item.overall_score, reverse=True)

        top_k = hybrid_scores[: self.settings.top_k_for_llm]
        reranked_top = self.llm.rerank_candidates(job, top_k, summaries)
        remaining = hybrid_scores[self.settings.top_k_for_llm :]
        results = _assign_ranks(reranked_top + remaining)
        output_path = self.write_results(results)
        return results, output_path

    def write_results(self, results: list[CandidateScore]) -> Path:
        """Write ranking results to ranked_candidates.csv."""

        output_path = self.settings.output_dir / "ranked_candidates.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "Rank": result.rank,
                "Candidate_ID": result.candidate_id,
                "Candidate_Name": result.candidate_name,
                "Overall_Score": result.overall_score,
                "Semantic_Score": result.semantic_score,
                "Skill_Score": result.skill_score,
                "Experience_Score": result.experience_score,
                "Education_Score": result.education_score,
                "Behaviour_Score": result.behaviour_score,
                "Activity_Score": result.activity_score,
                "Confidence_Score": result.confidence_score,
                "Reason": result.reason,
            }
            for result in results
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Wrote ranking output to %s", output_path)
        return output_path

    def _job_text(self, job: JobRequirements) -> str:
        parts = [
            job.role,
            "Responsibilities: " + ", ".join(job.responsibilities),
            "Required skills: " + ", ".join(job.required_skills),
            "Preferred skills: " + ", ".join(job.preferred_skills),
            "Education: " + ", ".join(job.education),
            "Soft skills: " + ", ".join(job.soft_skills),
            "Industry: " + job.industry,
            "Tools: " + ", ".join(job.tools),
            job.raw_text,
        ]
        return clean_text(" ".join(part for part in parts if part))


def _to_unit(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))


def _assign_ranks(scores: list[CandidateScore]) -> list[CandidateScore]:
    return [score.model_copy(update={"rank": index}) for index, score in enumerate(scores, start=1)]
