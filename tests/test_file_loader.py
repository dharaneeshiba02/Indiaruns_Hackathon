"""Tests for document text extraction."""

from app.utils.file_loader import extract_text_from_file
from app.services.candidate_service import CandidateService


def test_extract_text_from_txt(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text("AI Engineer\n\nPython   FastAPI", encoding="utf-8")

    assert extract_text_from_file(path) == "AI Engineer Python FastAPI"


def test_candidate_service_loads_single_resume_txt(tmp_path):
    path = tmp_path / "dharanee_resume.txt"
    path.write_text("Python FastAPI NLP projects", encoding="utf-8")

    candidates = CandidateService().load_candidates(path)

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "DHARANEE_RESUME"
    assert "FastAPI" in candidates[0].resume_text
