"""Text cleaning and normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
_SPLIT_RE = re.compile(r"[,;/|]\s*|\n+")
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", re.IGNORECASE)


def clean_text(text: Any) -> str:
    """Convert arbitrary input to normalized plain text."""

    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text).replace("\x00", " ")).strip()


def split_to_list(value: Any) -> list[str]:
    """Normalize CSV/JSON list-like values into a list of clean strings."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[Any] = value
    else:
        items = _SPLIT_RE.split(str(value))
    return [clean_text(item) for item in items if clean_text(item)]


def lower_set(values: Iterable[str]) -> set[str]:
    """Return lower-cased, non-empty values."""

    return {clean_text(value).lower() for value in values if clean_text(value)}


def extract_years(value: Any) -> float | None:
    """Extract years of experience from numeric or textual values."""

    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    text = clean_text(value)
    try:
        return float(text)
    except ValueError:
        match = _YEARS_RE.search(text)
        return float(match.group(1)) if match else None


def contains_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive phrase check."""

    return clean_text(phrase).lower() in clean_text(text).lower()
