"""Credential redaction for messages that may reach logs, tracebacks or issues."""
from __future__ import annotations

import os
import re

SECRET_NAMES = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FRED_API_KEY")
_QUERY_SECRET = re.compile(r"((?:api_key|apikey|secret_key|token)=)[^&\s'\"]+", re.IGNORECASE)


def redact_secrets(text: str, extra: tuple[str, ...] = ()) -> str:
    """Replace configured credential values and credential-like query values."""
    for value in [os.getenv(name, "") for name in SECRET_NAMES] + list(extra):
        if value and value.strip():
            text = text.replace(value.strip(), "[REDACTED]")
    return _QUERY_SECRET.sub(r"\1[REDACTED]", text)
