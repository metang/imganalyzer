"""Main CLI entry point using Typer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from imganalyzer.analyzer import Analyzer

app = typer.Typer(
    name="imganalyzer",
    help="Analyze images and camera RAW files. Outputs Lightroom-compatible XMP sidecars.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

AI_BACKENDS = ["openai", "anthropic", "google", "local", "none"]


@app.command()
def analyze(
    images: list[Path] = typer.Argument(..., help="Image file(s) to analyze. Supports glob patterns."),
    ai: Optional[str] = typer.Option(None, "--ai", help=f"AI backend: {', '.join(AI_BACKENDS)}"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output XMP path (single file only)"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI analysis"),
    no_technical: bool = typer.Option(False, "--no-technical", help="Skip technical analysis"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing XMP files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
) -> None:
    """Analyze one or more image files and generate XMP sidecar files."""
    from dotenv import load_dotenv
    load_dotenv()

    if ai and ai not in AI_BACKENDS:
        console.print(f"[red]Unknown AI backend '{ai}'. Choose from: {', '.join(AI_BACKENDS)}[/red]")
        raise typer.Exit(1)

    if no_ai:
        ai = "none"
    elif ai is None:
        import os
        ai = os.getenv("IMGANALYZER_DEFAULT_AI", "none")

    # Expand globs
    resolved: list[Path] = []
    for img in images:
        if img.exists():
            resolved.append(img)
        else:
            # Try glob expansion from cwd
            matches = list(Path(".").glob(str(img)))
            if matches:
                resolved.extend(matches)
            else:
                console.print(f"[yellow]Warning: '{img}' not found, skipping.[/yellow]")

    if not resolved:
        console.print("[red]No valid image files found.[/red]")
        raise typer.Exit(1)

    if output and len(resolved) > 1:
        console.print("[red]--output can only be used with a single input file.[/red]")
        raise typer.Exit(1)

    analyzer = Analyzer(
        ai_backend=ai,
        run_technical=not no_technical,
        verbose=verbose,
    )

    success = 0
    for img_path in resolved:
        xmp_path = output if output else img_path.with_suffix(".xmp")

        if xmp_path.exists() and not overwrite:
            if not quiet:
                console.print(f"[yellow]Skip:[/yellow] {xmp_path} already exists (use --overwrite)")
            continue

        if not quiet:
            console.print(f"[cyan]Analyzing:[/cyan] {img_path.name} ...")

        try:
            result = analyzer.analyze(img_path)
            result.write_xmp(xmp_path)
            success += 1
            if not quiet:
                _print_summary(result, xmp_path, verbose)
        except Exception as exc:
            console.print(f"[red]Error processing {img_path.name}: {exc}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()

    if not quiet:
        console.print(f"\n[green]Done.[/green] {success}/{len(resolved)} file(s) processed.")


@app.command()
def info(
    image: Path = typer.Argument(..., help="Image file to inspect"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, yaml"),
) -> None:
    """Display EXIF metadata and image info in the terminal (no XMP output)."""
    from dotenv import load_dotenv
    load_dotenv()

    if not image.exists():
        console.print(f"[red]File not found: {image}[/red]")
        raise typer.Exit(1)

    analyzer = Analyzer(ai_backend="none", run_technical=True, verbose=False)
    try:
        result = analyzer.analyze(image)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    if fmt == "json":
        import json
        rprint(json.dumps(result.to_dict(), indent=2, default=str))
    elif fmt == "yaml":
        try:
            import yaml
            rprint(yaml.dump(result.to_dict(), default_flow_style=False))
        except ImportError:
            console.print("[yellow]PyYAML not installed, falling back to JSON[/yellow]")
            import json
            rprint(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_info_table(result)


def _print_summary(result: "AnalysisResult", xmp_path: Path, verbose: bool) -> None:
    from imganalyzer.analyzer import AnalysisResult
    console.print(f"  [green]✓[/green] XMP written → {xmp_path}")
    if result.metadata:
        m = result.metadata
        parts = []
        if m.get("camera_make") or m.get("camera_model"):
            parts.append(f"{m.get('camera_make','')} {m.get('camera_model','')}".strip())
        if m.get("focal_length"):
            parts.append(f"{m['focal_length']}mm")
        if m.get("f_number"):
            parts.append(f"f/{m['f_number']}")
        if m.get("exposure_time"):
            parts.append(f"{m['exposure_time']}s")
        if m.get("iso"):
            parts.append(f"ISO {m['iso']}")
        if parts and verbose:
            console.print(f"  [dim]{' · '.join(parts)}[/dim]")
    if result.technical and verbose:
        t = result.technical
        console.print(f"  [dim]Sharpness: {t.get('sharpness_score', 0):.1f}  "
                       f"Exposure: {t.get('exposure_ev', 0):.2f} EV  "
                       f"Noise: {t.get('noise_level', 0):.3f}[/dim]")
    if result.ai_analysis and result.ai_analysis.get("description") and verbose:
        desc = result.ai_analysis["description"][:120]
        console.print(f"  [dim]AI: {desc}...[/dim]" if len(result.ai_analysis["description"]) > 120 else f"  [dim]AI: {desc}[/dim]")


def _print_info_table(result: "AnalysisResult") -> None:
    table = Table(title=f"Image Analysis: {result.source_path.name}", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold", width=24)
    table.add_column("Value")

    data = result.to_dict()
    sections = [
        ("📁 File", {
            "Path": str(result.source_path),
            "Format": data.get("format", ""),
            "Size": f"{data.get('width', '')} × {data.get('height', '')} px" if data.get("width") else "",
            "File size": f"{result.source_path.stat().st_size / 1024:.1f} KB",
        }),
        ("📷 Camera", {k: str(v) for k, v in (result.metadata or {}).items() if v}),
        ("📊 Technical", {k: str(round(v, 4)) if isinstance(v, float) else str(v)
                          for k, v in (result.technical or {}).items() if v is not None}),
        ("🤖 AI Analysis", {k: str(v) for k, v in (result.ai_analysis or {}).items() if v}),
    ]

    for section_name, fields in sections:
        if not any(v for v in fields.values()):
            continue
        table.add_row(f"[bold yellow]{section_name}[/bold yellow]", "", end_section=False)
        for k, v in fields.items():
            if v:
                table.add_row(f"  {k}", v)

    console.print(table)


if __name__ == "__main__":
    app()
