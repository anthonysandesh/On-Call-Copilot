from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import List

from tools.tool_schemas import PromQLQuery, TimeSeriesPoint, ToolResult
from tools.validators import ensure_valid_tool_call


class PromQLTool:
    """PromQL tool interface with mock local execution."""

    def __init__(self, mode: str = "mock"):
        self.mode = mode.lower()

    def _mock_query(self, call: PromQLQuery) -> List[TimeSeriesPoint]:
        series: List[TimeSeriesPoint] = []
        total_seconds = int((call.end - call.start).total_seconds())
        steps = max(total_seconds // call.step_seconds, 1)
        seed = int(hashlib.sha1(call.query.encode("utf-8")).hexdigest()[:8], 16)
        for i in range(steps + 1):
            ts = call.start + timedelta(seconds=i * call.step_seconds)
            magnitude = (seed % 100) / 10.0
            value = magnitude + i * 0.1
            series.append(
                TimeSeriesPoint(
                    timestamp=ts,
                    value=float(f"{value:.3f}"),
                    labels={"source": "mock", "query_fingerprint": str(seed % 10000)},
                )
            )
        return series

    def run(self, call: PromQLQuery) -> ToolResult:
        ensure_valid_tool_call(call)
        if self.mode != "mock":
            # Placeholder for real Prometheus/vLLM sidecar integrations.
            # For now we mirror mock behavior to keep the interface stable.
            return ToolResult(tool_name="promql_query", query=call, series=self._mock_query(call))
        return ToolResult(tool_name="promql_query", query=call, series=self._mock_query(call))


def promql_query(query: str, start: datetime, end: datetime, step_seconds: int) -> ToolResult:
    call = PromQLQuery(query=query, start=start, end=end, step_seconds=step_seconds)
    tool = PromQLTool(mode="mock")
    return tool.run(call)
