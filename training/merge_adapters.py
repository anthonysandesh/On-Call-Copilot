from __future__ import annotations

import typer


app = typer.Typer(help="Merge LoRA adapters into a base model.")


@app.command()
def merge(
    base_model: str = typer.Argument(..., help="Base model name or path."),
    adapters_path: str = typer.Argument(..., help="Path to trained adapters."),
    output_dir: str = typer.Argument(..., help="Where to save merged model."),
):
    """
    Loads a base model and merges LoRA adapters for standalone serving.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Transformers/PEFT not available: {exc}")
        raise typer.Exit(code=1)

    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapters_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    app()
