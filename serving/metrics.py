from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter("triage_requests_total", "Total triage requests", ["outcome"])
REQUEST_LATENCY = Histogram("triage_request_latency_seconds", "Triage request latency seconds")
RETRIEVAL_LATENCY = Histogram("triage_retrieval_latency_seconds", "Retrieval latency seconds")
TOOL_CALL_COUNTER = Counter("triage_tool_calls_total", "Tool calls issued", ["tool_name"])


def record_request(outcome: str, duration_seconds: float) -> None:
    REQUEST_COUNTER.labels(outcome=outcome).inc()
    REQUEST_LATENCY.observe(duration_seconds)


def record_tool_call(tool_name: str) -> None:
    TOOL_CALL_COUNTER.labels(tool_name=tool_name).inc()
