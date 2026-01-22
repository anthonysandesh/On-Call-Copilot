from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag.chunking import load_markdown_chunks
from rag.retriever import DEFAULT_ARTIFACT_DIR, Retriever, get_embedder, persist_index
from serving.metrics import RETRIEVAL_LATENCY, record_request, record_tool_call
from serving.model_client import get_model_client
from serving.schemas import IncidentRequest, TriageResponse
from tools.promql_tool import PromQLTool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Incident Copilot API", version="0.1.0")

_retriever: Optional[Retriever] = None


def _ensure_retriever() -> Optional[Retriever]:
    global _retriever
    if _retriever:
        return _retriever
    try:
        _retriever = Retriever.load(DEFAULT_ARTIFACT_DIR)
        return _retriever
    except FileNotFoundError:
        logger.warning("Artifacts missing, building mock index from sample runbooks")
        chunks = load_markdown_chunks(Path("data/sample_runbooks"))
        embedder = get_embedder("mock")
        persist_index(chunks, embedder, artifact_dir=DEFAULT_ARTIFACT_DIR)
        _retriever = Retriever.load(DEFAULT_ARTIFACT_DIR)
    except Exception as exc:
        logger.error("Unable to initialize retriever: %s", exc)
        _retriever = None
    return _retriever


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    # Using default registry
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/triage", response_model=TriageResponse)
def triage(request: IncidentRequest) -> JSONResponse:
    start_time = time.perf_counter()
    try:
        retriever = _ensure_retriever()
        retrieved = []
        if retriever:
            with RETRIEVAL_LATENCY.time():
                retrieved = retriever.retrieve(f"{request.alert_text}\n{request.logs_text}", k=3)

        model_client = get_model_client()
        tool = PromQLTool(mode="mock")
        response = model_client.generate(request, retrieved, tool)

        for call in response.tool_calls:
            record_tool_call(call.tool_name)

        record_request(outcome="success", duration_seconds=time.perf_counter() - start_time)
        return JSONResponse(content=jsonable_encoder(response))
    except Exception as exc:
        record_request(outcome="error", duration_seconds=time.perf_counter() - start_time)
        raise HTTPException(status_code=500, detail=str(exc))
