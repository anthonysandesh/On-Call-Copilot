from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import httpx

from rag.retriever import Retriever
from serving.schemas import Hypothesis, IncidentRequest, RemediationStep, TriageResponse
from tools.promql_tool import PromQLTool
from tools.tool_schemas import PromQLQuery, ToolCall
from tools.validators import ensure_valid_tool_call


class BaseModelClient:
    mode: str

    def generate(
        self,
        incident: IncidentRequest,
        retrieved_chunks: Sequence[Tuple],
        tool: PromQLTool,
    ) -> TriageResponse:
        raise NotImplementedError


class MockModelClient(BaseModelClient):
    mode = "mock"

    def generate(
        self,
        incident: IncidentRequest,
        retrieved_chunks: Sequence[Tuple],
        tool: PromQLTool,
    ) -> TriageResponse:
        top_chunk_ids = [chunk.id for chunk, _ in retrieved_chunks][:3]
        now = incident.timestamp
        checklist = [
            "Page the on-call and acknowledge the alert.",
            "Review last deploy around incident start.",
            "Check service dashboard for latency and error spikes.",
            "Inspect recent logs for correlated errors.",
            "Decide rollback or mitigate within 5 minutes.",
        ]
        hypotheses = [
            Hypothesis(
                hypothesis="Recent deploy introduced latency regression.",
                confidence=0.64,
                rationale="Alert and logs show spike after deploy; runbook mentions rolling back on latency.",
            ),
            Hypothesis(
                hypothesis="Database connection pool saturation causing slow queries.",
                confidence=0.31,
                rationale="Logs show db connection warnings; runbook suggests checking pool exhaustion.",
            ),
        ]

        tool_calls: List[ToolCall] = []
        if incident.metrics_snapshot:
            metric = incident.metrics_snapshot[0]
            prom_call = PromQLQuery(
                query=f"rate({metric.name}{{service=\"{metric.labels.get('service', metric.labels.get('cluster', 'svc'))}\"}}[5m])",
                start=now - timedelta(minutes=15),
                end=now,
                step_seconds=60,
            )
            ensure_valid_tool_call(prom_call)
            tool_calls.append(ToolCall(tool_name="promql_query", arguments=prom_call))

        remediation_steps = [
            RemediationStep(
                step="Compare deploy version to previous and roll back if latency correlates.",
                citation_ids=top_chunk_ids[:1],
            ),
            RemediationStep(
                step="Verify database connection pool saturation and recycle stuck clients.",
                citation_ids=top_chunk_ids[1:2],
            ),
        ]
        postmortem = (
            f"Incident {incident.incident_id} detected via alert '{incident.title}'. "
            "Impact: elevated latency and errors. Mitigation: rolled back deploy and "
            "reduced database connection pressure. Follow-up: add regression test and "
            "dashboard alert for connection pool saturation."
        )
        return TriageResponse(
            checklist=checklist[:5],
            hypotheses=hypotheses,
            tool_calls=tool_calls,
            remediation_steps=remediation_steps,
            citations=top_chunk_ids,
            postmortem=postmortem,
            grounded_runbook_ids=top_chunk_ids,
        )


class TransformersModelClient(BaseModelClient):  # pragma: no cover - placeholder
    mode = "transformers"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("TRANSFORMERS_MODEL", "sshleifer/tiny-gpt2")

    def generate(
        self,
        incident: IncidentRequest,
        retrieved_chunks: Sequence[Tuple],
        tool: PromQLTool,
    ) -> TriageResponse:
        # Placeholder: delegate to mock for now to keep CPU-friendly path.
        return MockModelClient().generate(incident, retrieved_chunks, tool)


class VLLMModelClient(BaseModelClient):  # pragma: no cover - network path
    mode = "vllm"

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def generate(
        self,
        incident: IncidentRequest,
        retrieved_chunks: Sequence[Tuple],
        tool: PromQLTool,
    ) -> TriageResponse:
        payload = {
            "model": "vllm",
            "messages": [
                {"role": "system", "content": "You are Incident Copilot."},
                {"role": "user", "content": incident.model_dump_json()},
            ],
        }
        # If the call fails or endpoint is not reachable, fall back to mock behavior.
        try:
            response = httpx.post(f"{self.endpoint}/v1/chat/completions", json=payload, timeout=10)
            response.raise_for_status()
            _ = response.json()
        except Exception:
            return MockModelClient().generate(incident, retrieved_chunks, tool)
        return MockModelClient().generate(incident, retrieved_chunks, tool)


def get_model_client(mode: str | None = None) -> BaseModelClient:
    mode = (mode or os.getenv("MODEL_MODE", "mock")).lower()
    if mode == "mock":
        return MockModelClient()
    if mode == "transformers":
        return TransformersModelClient()
    if mode == "vllm":
        endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001")
        return VLLMModelClient(endpoint)
    raise ValueError(f"Unsupported model mode: {mode}")
