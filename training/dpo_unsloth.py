"""
Scaffold for DPO training with Unsloth + TRL.
Fill in dataset loading and objective once preference data is available.
"""

import typer


app = typer.Typer(help="DPO training scaffold (placeholder).")


@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to preference data (jsonl with chosen/rejected)."),
    base_model: str = typer.Option("unsloth/llama-3-8b-bnb-4bit"),
    output_dir: str = typer.Option("artifacts/adapters/dpo-run"),
):
    typer.echo(
        "TODO: implement DPO training. Expected steps:\n"
        "- Load pairwise preference dataset.\n"
        "- Initialize Unsloth FastLanguageModel with QLoRA adapters.\n"
        "- Use TRL DPOTrainer to optimize on preferences.\n"
        "- Save adapters to output_dir."
    )


if __name__ == "__main__":
    app()
