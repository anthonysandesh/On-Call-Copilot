SHELL := /bin/bash
PYTHON ?= python
VENV ?= .venv
ACTIVATE_CMD := if [ -d "$(VENV)/bin" ]; then source $(VENV)/bin/activate; fi;

export PYTHONPATH := .

.PHONY: setup lint typecheck test ingest run demo eval

setup:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(ACTIVATE_CMD) pip install --upgrade pip
	@$(ACTIVATE_CMD) pip install -e .[dev]

lint:
	@$(ACTIVATE_CMD) ruff check .

typecheck:
	@$(ACTIVATE_CMD) mypy .

test:
	@$(ACTIVATE_CMD) pytest

ingest:
	@$(ACTIVATE_CMD) $(PYTHON) rag/ingest_runbooks.py

run:
	@$(ACTIVATE_CMD) MODEL_MODE=MOCK uvicorn serving.api:app --host 0.0.0.0 --port 8000

demo:
	@bash scripts/demo.sh

eval:
	@$(ACTIVATE_CMD) MODEL_MODE=MOCK $(PYTHON) eval/offline_eval.py
