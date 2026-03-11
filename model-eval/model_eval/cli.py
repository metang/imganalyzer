"""CLI entry point for the model evaluation tool."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="model-eval",
    help="Evaluate aesthetic and captioning models side-by-side.",
    no_args_is_help=True,
)
console = Console(stderr=True)

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "test_images"


@app.command()
def run(
    images: Path = typer.Option(
        DEFAULT_IMAGE_DIR,
        "--images", "-i",
        help="Directory containing test images.",
    ),
    models: str = typer.Option(
        "all",
        "--models", "-m",
        help="Comma-separated model names, or 'all', 'aesthetic', 'caption'.",
    ),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached results."),
) -> None:
    """Run model evaluation on test images."""
    from model_eval.config import list_model_names
    from model_eval.runner import run_evaluation

    if not images.is_dir():
        console.print(f"[red]Image directory not found: {images}[/red]")
        raise typer.Exit(1)

    # Resolve model list
    if models == "all":
        model_list = list_model_names()
    elif models == "aesthetic":
        model_list = list_model_names("aesthetic")
    elif models == "caption":
        model_list = list_model_names("caption")
    else:
        model_list = [m.strip() for m in models.split(",")]

    console.print(f"Models: [bold]{', '.join(model_list)}[/bold]")
    run_evaluation(model_list, images, device=device, skip_cached=not no_cache)


@app.command()
def report(
    images: Path = typer.Option(
        DEFAULT_IMAGE_DIR,
        "--images", "-i",
        help="Directory containing test images.",
    ),
    models: str = typer.Option(
        "all",
        "--models", "-m",
        help="Comma-separated model names, or 'all', 'aesthetic', 'caption'.",
    ),
    output: Path = typer.Option(
        Path("report.html"),
        "--output", "-o",
        help="Output HTML file path.",
    ),
) -> None:
    """Generate HTML comparison report from cached results."""
    from model_eval.config import list_model_names
    from model_eval.report.html_report import generate_report
    from model_eval.runner import load_all_results

    if models == "all":
        model_list = list_model_names()
    elif models == "aesthetic":
        model_list = list_model_names("aesthetic")
    elif models == "caption":
        model_list = list_model_names("caption")
    else:
        model_list = [m.strip() for m in models.split(",")]

    results = load_all_results(model_list, images)

    # Filter to models that actually have results
    results = {k: v for k, v in results.items() if v}
    if not results:
        console.print("[red]No cached results found. Run evaluation first.[/red]")
        raise typer.Exit(1)

    generate_report(results, images, output)
    console.print(f"[green]Report written to {output.resolve()}[/green]")


@app.command("list-models")
def list_models(
    category: Optional[str] = typer.Argument(
        None, help="Filter by 'aesthetic' or 'caption'."
    ),
) -> None:
    """List available models."""
    from model_eval.config import AESTHETIC_MODELS, ALL_MODELS, CAPTION_MODELS

    table = Table(title="Available Models")
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Module")

    if category == "aesthetic":
        source = AESTHETIC_MODELS
    elif category == "caption":
        source = CAPTION_MODELS
    else:
        source = ALL_MODELS

    for name in sorted(source.keys()):
        module_path, class_name = source[name]
        cat = "aesthetic" if name in AESTHETIC_MODELS else "caption"
        table.add_row(name, cat, f"{module_path}.{class_name}")

    console.print(table)


@app.command("clear-cache")
def clear_cache_cmd() -> None:
    """Clear all cached evaluation results."""
    from model_eval.cache import clear_cache

    count = clear_cache()
    console.print(f"[green]Cleared {count} cached results.[/green]")


if __name__ == "__main__":
    app()
