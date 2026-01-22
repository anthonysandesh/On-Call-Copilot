from __future__ import annotations

import re
from datetime import datetime, timedelta

from tools.tool_schemas import PromQLQuery

# Best-effort tokenizer: allow common PromQL characters.
PROMQL_ALLOWED_RE = re.compile(r"^[a-zA-Z0-9_:{}\[\](),=+\-*/\s\"'\.<>!~^%|]+$")


class PromQLValidationError(ValueError):
    pass


def validate_promql_syntax(query: str) -> bool:
    """Lightweight PromQL syntax guardrail."""
    if not query or len(query.strip()) < 3:
        raise PromQLValidationError("Query too short")
    if not PROMQL_ALLOWED_RE.match(query):
        raise PromQLValidationError("Query has illegal characters")
    if query.count("(") != query.count(")"):
        raise PromQLValidationError("Unbalanced parentheses")
    if query.count("[") != query.count("]"):
        raise PromQLValidationError("Unbalanced range selectors")
    return True


def validate_time_range(start: datetime, end: datetime, step_seconds: int, max_hours: int = 24) -> None:
    if end <= start:
        raise PromQLValidationError("End must be after start")
    if step_seconds <= 0:
        raise PromQLValidationError("step_seconds must be positive")
    if end - start > timedelta(hours=max_hours):
        raise PromQLValidationError("Time window is too large")


def ensure_valid_tool_call(call: PromQLQuery) -> PromQLQuery:
    validate_promql_syntax(call.query)
    validate_time_range(call.start, call.end, call.step_seconds)
    return call
