#!/usr/bin/env bash
set -euo pipefail

export MODEL_MODE=${MODEL_MODE:-MOCK}

# Build index if missing
python rag/ingest_runbooks.py --embedding-model mock >/dev/null

python - <<'PY'
import json
from pathlib import Path

from incident_copilot import DEFAULT_ARTIFACT_DIR
from rag.retriever import Retriever
from serving.model_client import get_model_client
from serving.schemas import IncidentRequest
from tools.promql_tool import PromQLTool

retriever = Retriever.load(Path(DEFAULT_ARTIFACT_DIR))
sample = json.loads(Path("data/sample_incidents.jsonl").read_text().splitlines()[0])
incident = IncidentRequest(**sample)
retrieved = retriever.retrieve(f"{incident.alert_text}\n{incident.logs_text}", k=3)
client = get_model_client("mock")
response = client.generate(incident, retrieved, PromQLTool(mode="mock"))

print("=== Demo: incident-copilot (mock mode) ===")
print(f"Incident: {incident.incident_id} - {incident.title}")
print("\nChecklist:")
for item in response.checklist:
    print(f"- {item}")

print("\nHypotheses:")
for h in response.hypotheses:
    print(f"- ({h.confidence:.2f}) {h.hypothesis} :: {h.rationale}")

print("\nTool calls:")
for call in response.tool_calls:
    print(f"- {call.tool_name} -> {call.arguments.query} [{call.arguments.start} - {call.arguments.end}]")

print("\nRemediation:")
for step in response.remediation_steps:
    cites = ", ".join(step.citation_ids) if step.citation_ids else "none"
    print(f"- {step.step} (cites: {cites})")

print("\nCitations:", ", ".join(response.citations))
print("\nPostmortem draft:")
print(response.postmortem)
PY
