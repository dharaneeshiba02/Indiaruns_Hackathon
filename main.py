"""Convenience entrypoint for running the FastAPI app.

Use the production command:
    uvicorn app.api.main:app --reload
"""

from app.api.main import app


__all__ = ["app"]
