from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class PromQLQuery(BaseModel):
    query: str = Field(..., description="PromQL query string")
    start: datetime = Field(..., description="Start time for the range query")
    end: datetime = Field(..., description="End time for the range query")
    step_seconds: int = Field(..., description="Step size in seconds")

    @field_validator("step_seconds")
    @classmethod
    def validate_step(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("step_seconds must be positive")
        return v

    @field_validator("end")
    @classmethod
    def validate_order(cls, v: datetime, info):
        start = info.data.get("start")
        if start and v <= start:
            raise ValueError("end must be greater than start")
        return v


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float
    labels: Dict[str, str]


class ToolCall(BaseModel):
    tool_name: str
    arguments: PromQLQuery


class ToolResult(BaseModel):
    tool_name: str
    query: PromQLQuery
    series: List[TimeSeriesPoint]
