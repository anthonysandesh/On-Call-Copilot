from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import List, Tuple

from incident_copilot import DEFAULT_ARTIFACT_DIR
from rag.chunking import load_markdown_chunks
from rag.retriever import Retriever, get_embedder, persist_index
from serving.model_client import get_model_client
from serving.schemas import IncidentRequest, TriageResponse
from tools.promql_tool import PromQLTool
from eval.metrics import aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation harness")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/eval_set.jsonl"),
        help="Path to eval jsonl file",
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--model-mode", type=str, default="mock")
    return parser.parse_args()


def ensure_retriever(artifact_dir: Path) -> Retriever:
    try:
        return Retriever.load(artifact_dir)
    except FileNotFoundError:
        chunks = load_markdown_chunks(Path("data/sample_runbooks"))
        embedder = get_embedder("mock")
        persist_index(chunks, embedder, artifact_dir=artifact_dir)
        return Retriever.load(artifact_dir)


def run_sample(
    incident: IncidentRequest,
    retriever: Retriever,
    model_mode: str,
) -> Tuple[TriageResponse, float]:
    model_client = get_model_client(model_mode)
    tool = PromQLTool(mode="mock")
    start = time.perf_counter()
    retrieved = retriever.retrieve(f"{incident.alert_text}\n{incident.logs_text}", k=3)
    response = model_client.generate(incident, retrieved, tool)
    latency_ms = (time.perf_counter() - start) * 1000
    return response, latency_ms


def main() -> None:
    args = parse_args()
    retriever = ensure_retriever(args.artifact_dir)
    responses: List[TriageResponse] = []
    latencies: List[float] = []
    per_sample: List[dict] = []

    with args.data.open() as f:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            sample = json.loads(line)
            incident = IncidentRequest(**sample)
            resp, latency_ms = run_sample(incident, retriever, args.model_mode)
            responses.append(resp)
            latencies.append(latency_ms)
            per_sample.append(
                {
                    "incident_id": incident.incident_id,
                    "tool_calls": len(resp.tool_calls),
                    "citations": resp.citations,
                    "latency_ms": latency_ms,
                }
            )

    metrics = aggregate_metrics(responses)
    metrics["response_latency_ms_avg"] = mean(latencies) if latencies else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": per_sample,
        "metrics": metrics,
        "model_mode": args.model_mode,
    }

    out_dir = args.artifact_dir / "eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report-{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote eval report to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
