"""Tests for document text extraction."""

from app.utils.file_loader import extract_text_from_file


def test_extract_text_from_txt(tmp_path):
    path = tmp_path / "job.txt"
    path.write_text("AI Engineer\n\nPython   FastAPI", encoding="utf-8")

    assert extract_text_from_file(path) == "AI Engineer Python FastAPI"
