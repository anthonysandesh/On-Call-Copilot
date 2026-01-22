from __future__ import annotations

from typing import Dict, List, Set

from serving.schemas import TriageResponse
from tools.validators import ensure_valid_tool_call


def tool_call_valid_rate(responses: List[TriageResponse]) -> float:
    total = 0
    valid = 0
    for resp in responses:
        for call in resp.tool_calls:
            total += 1
            try:
                ensure_valid_tool_call(call.arguments)
                valid += 1
            except Exception:
                continue
    return valid / total if total else 1.0


def citation_coverage_rate(responses: List[TriageResponse]) -> float:
    total_steps = 0
    with_citations = 0
    for resp in responses:
        for step in resp.remediation_steps:
            total_steps += 1
            if step.citation_ids:
                with_citations += 1
    return with_citations / total_steps if total_steps else 1.0


def hallucination_flag_rate(responses: List[TriageResponse]) -> float:
    # Simple rule: if citations list is empty but remediation steps exist, flag hallucination.
    flags = 0
    total = 0
    for resp in responses:
        if resp.remediation_steps:
            total += 1
            if not resp.citations:
                flags += 1
    return flags / total if total else 0.0


def groundedness_rate(responses: List[TriageResponse]) -> float:
    total = 0
    grounded = 0
    for resp in responses:
        if not resp.citations:
            continue
        expected: Set[str] = set(resp.grounded_runbook_ids or resp.citations)
        total += 1
        if set(resp.citations).issubset(expected):
            grounded += 1
    return grounded / total if total else 1.0


def aggregate_metrics(
    responses: List[TriageResponse],
) -> Dict[str, float]:
    return {
        "tool_call_valid_rate": tool_call_valid_rate(responses),
        "citation_coverage_rate": citation_coverage_rate(responses),
        "hallucination_flag_rate": hallucination_flag_rate(responses),
        "groundedness_rate": groundedness_rate(responses),
    }
