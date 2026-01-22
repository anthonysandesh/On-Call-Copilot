from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

DEFAULT_BASE_MODEL = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-bnb-4bit")


app = typer.Typer(help="Unsloth QLoRA SFT trainer for incident-copilot.")


def _load_dataset(path: Path):
    from datasets import load_dataset  # type: ignore

    return load_dataset("json", data_files=str(path), split="train")


@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Path to ChatML/ShareGPT-like jsonl training data."),
    output_dir: Path = typer.Option(Path("artifacts/adapters"), help="Where to store adapters."),
    base_model: str = typer.Option(DEFAULT_BASE_MODEL, help="Base model to fine-tune."),
    run_id: Optional[str] = typer.Option(None, help="Run identifier for output folder."),
    max_steps: int = typer.Option(100, help="Max training steps."),
    lr: float = typer.Option(2e-4, help="Learning rate."),
    micro_batch_size: int = typer.Option(1, help="Micro batch size."),
):
    """
    Fine-tune a base model with QLoRA adapters using Unsloth's FastLanguageModel helper.
    Requires GPU with sufficient VRAM; defaults to 4-bit loading for efficiency.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        typer.echo(f"Unsloth/TRL not installed or GPU unavailable: {exc}")
        raise typer.Exit(code=1)

    ds = _load_dataset(data_path)
    run_folder = output_dir / (run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    run_folder.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "v_proj"])

    training_args = TrainingArguments(
        output_dir=str(run_folder),
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=max_steps,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(run_folder)
    tokenizer.save_pretrained(run_folder)
    typer.echo(f"Saved adapters to {run_folder}")


@app.command()
def describe():
    typer.echo(
        json.dumps(
            {
                "base_model": DEFAULT_BASE_MODEL,
                "expected_hardware": "GPU with >=24GB VRAM for full training; CPU not supported.",
                "notes": "Uses QLoRA with 4-bit loading; see training/README.md for details.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
