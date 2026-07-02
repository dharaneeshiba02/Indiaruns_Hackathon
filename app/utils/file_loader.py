"""File readers for job descriptions and resume documents."""

from __future__ import annotations

import logging
from pathlib import Path

from app.utils.text import clean_text

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when a parser does not support a file extension."""


def extract_text_from_file(path: str | Path) -> str:
    """Extract clean text from txt, pdf, or docx files."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to parse PDF files.") from exc

    chunks: list[str] = []
    with fitz.open(path) as document:
        for page in document:
            chunks.append(page.get_text("text"))
    return clean_text(" ".join(chunks))


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required to parse DOCX files.") from exc

    document = Document(path)
    return clean_text(" ".join(paragraph.text for paragraph in document.paragraphs))
