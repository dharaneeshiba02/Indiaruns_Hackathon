"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the candidate ranking system."""

    app_name: str = os.getenv("APP_NAME", "AI Candidate Ranking System")
    environment: str = os.getenv("ENVIRONMENT", "local")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")
    top_k_for_llm: int = int(os.getenv("TOP_K_FOR_LLM", "20"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": _float_env("WEIGHT_SEMANTIC", 0.40),
            "skills": _float_env("WEIGHT_SKILLS", 0.20),
            "experience": _float_env("WEIGHT_EXPERIENCE", 0.15),
            "education": _float_env("WEIGHT_EDUCATION", 0.10),
            "behaviour": _float_env("WEIGHT_BEHAVIOUR", 0.10),
            "activity": _float_env("WEIGHT_ACTIVITY", 0.05),
        }
    )

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not exist."""

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    settings = Settings()
    settings.ensure_directories()
    return settings
