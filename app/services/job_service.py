"""Job description ingestion and requirement extraction."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from app.models.schemas import JobRequirements
from app.services.groq_client import GroqRecruiterClient
from app.utils.file_loader import extract_text_from_file


class JobService:
    """Coordinates job description parsing and LLM extraction."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = GroqRecruiterClient(settings)

    def load_from_file(self, path: str | Path) -> JobRequirements:
        """Read a job description file and return structured requirements."""

        text = extract_text_from_file(path)
        return self.llm.extract_job_requirements(text)

    def load_from_text(self, text: str) -> JobRequirements:
        """Return structured requirements from raw job text."""

        return self.llm.extract_job_requirements(text)
