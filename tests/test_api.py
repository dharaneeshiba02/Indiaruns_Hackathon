"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.models.schemas import CandidateScore


def test_health_endpoint():
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_rank_endpoint(monkeypatch, tmp_path):
    class FakePipeline:
        def __init__(self, settings):
            self.settings = settings

        def rank_from_files(self, job_path, candidates_path):
            output = tmp_path / "ranked_candidates.csv"
            output.write_text("Rank,Candidate_ID\n1,C1\n", encoding="utf-8")
            return [
                CandidateScore(
                    rank=1,
                    candidate_id="C1",
                    candidate_name="Aarav",
                    overall_score=91.0,
                    semantic_score=93.0,
                    skill_score=95.0,
                    experience_score=100.0,
                    education_score=100.0,
                    behaviour_score=80.0,
                    activity_score=90.0,
                    confidence_score=88.0,
                    reason="Strong match.",
                )
            ], output

    monkeypatch.setattr(api_main, "RankingPipeline", FakePipeline)
    client = TestClient(api_main.app)

    job_response = client.post(
        "/upload-job",
        files={"file": ("job.txt", b"AI Engineer with Python", "text/plain")},
    )
    candidate_response = client.post(
        "/upload-candidates",
        files={"file": ("candidates.csv", b"candidate_id,name\nC1,Aarav\n", "text/csv")},
    )
    rank_response = client.post("/rank")

    assert job_response.status_code == 200
    assert candidate_response.status_code == 200
    assert rank_response.status_code == 200
    assert rank_response.json()["results"][0]["candidate_id"] == "C1"


def test_candidate_pdf_upload_is_allowed():
    client = TestClient(api_main.app)

    response = client.post(
        "/upload-candidates",
        files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "candidates uploaded"
