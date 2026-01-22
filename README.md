# Incident Copilot

SRE / On-Call copilot with RAG, PromQL tool-calling, Unsloth fine-tuning paths, and mock CPU mode. Ships with offline sample data and end-to-end evaluation so `make test` and `make demo` work without downloads.

## Features
- FastAPI API with `/v1/triage`, `/healthz`, `/metrics` (Prometheus).
- Mock/CPU model mode plus hooks for Transformers and vLLM OpenAI endpoints.
- RAG over markdown runbooks using sentence-transformers embeddings + FAISS (mock embedding fallback).
- PromQL tool schema + validator + mock executor.
- Offline evaluation harness for tool-call correctness, groundedness, hallucinations, and latency.
- Unsloth QLoRA SFT scripts, adapter merge helper, DPO scaffold, Docker + Helm + CI.

## Architecture
![Architecture/ Flowchart](oncall_copilot.png)

## Quickstart (mock mode, CPU)
```bash
make setup        # create .venv and install deps
make ingest       # builds FAISS index from sample runbooks (mock embeddings)
make run          # start FastAPI with MODEL_MODE=MOCK on :8000
```
Sample request:
```bash
curl -X POST http://localhost:8000/v1/triage \
  -H "Content-Type: application/json" \
  -d "$(head -n 1 data/sample_incidents.jsonl)"
```
Demo without server:
```bash
make demo
```

## Step-by-step setup & use
1) Clone repo and ensure Python 3.11 is available.  
2) Install deps and build mock RAG index: `make setup && make ingest` (offline-safe mock embeddings).  
3) Start API in mock mode: `make run` (FastAPI on http://localhost:8000).  
4) Health check: `curl http://localhost:8000/healthz`.  
5) Send a triage request:  
   ```bash
   curl -X POST http://localhost:8000/v1/triage \
     -H "Content-Type: application/json" \
     -d "$(head -n 1 data/sample_incidents.jsonl)"
   ```  
6) Explore metrics: `curl http://localhost:8000/metrics`.  
7) Run offline eval: `make eval`.  
8) Stop the server with Ctrl+C (or kill the uvicorn PID if backgrounded).

## RAG ingestion
- Run `make ingest` (uses mock embeddings by default for offline reproducibility).
- Artifacts land in `artifacts/{faiss.index,chunks.jsonl,index_meta.json}`.

## Evaluation
```bash
make eval         # runs eval/offline_eval.py in mock mode
```
Outputs JSON report under `artifacts/eval_reports/` with tool-call validity, citation coverage, groundedness, hallucination flag rate, and latency.

## Testing, lint, typecheck
```bash
make lint
make typecheck
make test
```

## Training (GPU required)
- Install training extras: `pip install .[training]`.
- SFT with Unsloth QLoRA (needs GPU, defaults to `BASE_MODEL_NAME` env):
  ```bash
  python training/sft_unsloth.py train data/training.jsonl --max-steps 200
  ```
- Merge adapters: `python training/merge_adapters.py <base_model> artifacts/adapters/<run_id> artifacts/merged-model`
- DPO scaffold: `python training/dpo_unsloth.py train <preference_data>`

## Serving with vLLM
- Build/start API + vLLM together:
  ```bash
  docker-compose up --build
  ```
- Point API at vLLM by setting `MODEL_MODE=vllm` and `VLLM_ENDPOINT=http://vllm:8001`.

## Helm (minimal)
`infra/helm` includes a minimal Deployment/Service. Adjust image and env vars, then `helm install incident-copilot infra/helm`.
