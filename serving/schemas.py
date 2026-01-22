from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from tools.tool_schemas import ToolCall


class MetricSnapshot(BaseModel):
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: datetime


class EnvironmentContext(BaseModel):
    service: str
    cluster: str
    region: str
    deploy_version: str


class IncidentRequest(BaseModel):
    incident_id: str
    title: str
    severity: str
    timestamp: datetime
    alert_text: str
    logs_text: str
    metrics_snapshot: List[MetricSnapshot]
    environment: EnvironmentContext

    @field_validator("severity")
    @classmethod
    def normalize_severity(cls, v: str) -> str:
        return v.lower()


class Hypothesis(BaseModel):
    hypothesis: str
    confidence: float = Field(..., ge=0, le=1)
    rationale: str


class RemediationStep(BaseModel):
    step: str
    citation_ids: List[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    checklist: List[str]
    hypotheses: List[Hypothesis]
    tool_calls: List[ToolCall]
    remediation_steps: List[RemediationStep]
    citations: List[str]
    postmortem: str
    grounded_runbook_ids: Optional[List[str]] = None
