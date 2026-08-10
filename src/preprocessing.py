"""Text cleaning shared by training and inference."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\d[\s-]?){5,18}(?!\w)")
_CURRENCY_RE = re.compile(
    r"(?:[£$€¥]\s*\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s*(?:p|gbp|eur|usd))",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize structured spam clues while preserving ordinary words."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = _URL_RE.sub(" URL ", text)
    text = _CURRENCY_RE.sub(" CURRENCY ", text)

    def replace_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return " PHONE_NUMBER " if 5 <= len(digits) <= 18 else match.group(0)

    text = _PHONE_RE.sub(replace_phone, text)
    return _WHITESPACE_RE.sub(" ", text).strip().lower()
