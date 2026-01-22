# Training Guide

This folder contains GPU-only paths for Unsloth QLoRA fine-tuning and adapter merging.

## Prerequisites
- GPU with >=24GB VRAM for meaningful runs (QLoRA loads models in 4bit).
- Python 3.11 with `pip install .[training]`.

## Supervised Fine-Tuning (SFT)
```bash
export BASE_MODEL_NAME="unsloth/llama-3-8b-bnb-4bit"  # override as needed
python training/sft_unsloth.py train data/training.jsonl --max-steps 200
```
Outputs adapters under `artifacts/adapters/<run_id>/`.

## Merge Adapters
```bash
python training/merge_adapters.py <base_model> artifacts/adapters/<run_id> artifacts/merged-model
```

## DPO (placeholder)
`training/dpo_unsloth.py` is scaffolded with TODOs for pairwise preference data.

## Notes
- Scripts default to `mock` embeddings for ingestion; training expects pre-tokenized ChatML text.
- See repository README for vLLM serving once adapters are merged.
