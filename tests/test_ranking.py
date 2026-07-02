"""Tests for the ranking pipeline."""

from config import Settings
from app.models.schemas import CandidateProfile, JobRequirements
from app.services.pipeline import RankingPipeline


def test_pipeline_ranks_semantic_candidate_first(tmp_path):
    settings = Settings(
        embedding_backend="hash",
        output_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        gemini_api_key=None,
    )
    job = JobRequirements(
        role="AI Engineer",
        required_skills=["python", "fastapi", "nlp"],
        years_experience=3,
        education=["computer science"],
        soft_skills=["communication"],
        raw_text="Build NLP APIs with Python and FastAPI.",
    )
    candidates = [
        CandidateProfile(
            candidate_id="C1",
            name="Strong Candidate",
            resume_text="Built NLP services and semantic search APIs.",
            skills=["Python", "FastAPI", "NLP"],
            experience_years=4,
            education=["B.Tech Computer Science"],
            behavioural_traits=["communication"],
            platform_activity=0.9,
        ),
        CandidateProfile(
            candidate_id="C2",
            name="Weak Candidate",
            resume_text="Created finance dashboards.",
            skills=["Excel", "SQL"],
            experience_years=1,
            education=["Commerce"],
            behavioural_traits=["teamwork"],
            platform_activity=0.4,
        ),
    ]

    results, output_path = RankingPipeline(settings).rank(job, candidates)

    assert results[0].candidate_id == "C1"
    assert results[0].overall_score > results[1].overall_score
    assert output_path.exists()
