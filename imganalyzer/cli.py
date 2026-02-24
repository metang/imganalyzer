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
    detection_prompt: Optional[str] = typer.Option(
        None,
        "--detection-prompt",
        help="Custom object detection text prompt for GroundingDINO (local AI only). "
             "Categories separated by ' . '.  Overrides IMGANALYZER_DETECTION_PROMPT env var.",
    ),
    detection_threshold: Optional[float] = typer.Option(
        None,
        "--detection-threshold",
        min=0.0,
        max=1.0,
        help="Object detection confidence threshold 0–1 (default: 0.30). "
             "Overrides IMGANALYZER_DETECTION_THRESHOLD env var.",
    ),
    face_threshold: Optional[float] = typer.Option(
        None,
        "--face-threshold",
        min=0.0,
        max=1.0,
        help="Face recognition cosine similarity threshold 0–1 (default: 0.40). "
             "Overrides IMGANALYZER_FACE_DB_THRESHOLD env var.",
    ),
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

    # Expand globs and directories
    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif",
        ".heic", ".heif",
        ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
        ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
    }
    resolved: list[Path] = []
    for img in images:
        if img.is_dir():
            # Expand directory to all image files (non-recursive)
            found = sorted(
                p for p in img.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if found:
                resolved.extend(found)
            else:
                console.print(f"[yellow]Warning: No image files found in '{img}', skipping.[/yellow]")
        elif img.exists():
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
        detection_prompt=detection_prompt,
        detection_threshold=detection_threshold,
        face_match_threshold=face_threshold,
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


@app.command(name="register-face")
def register_face(
    name: str = typer.Argument(..., help="Identity name to register (e.g. 'Alice')."),
    images: list[Path] = typer.Argument(..., help="One or more face image files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Register a face identity for recognition in future analyses.

    Provide one or more clear face photos.  Multiple images improve matching
    accuracy across different angles and lighting conditions.

    Example::

        imganalyzer register-face Alice alice1.jpg alice2.jpg
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from imganalyzer.analysis.ai.faces import extract_embedding_from_image
        from imganalyzer.analysis.ai.face_db import FaceDatabase
    except ImportError as exc:
        console.print(f"[red]Face analysis requires local-ai extras: {exc}[/red]")
        raise typer.Exit(1)

    face_db = FaceDatabase()
    registered = 0

    for img_path in images:
        if not img_path.exists():
            console.print(f"[yellow]Warning: '{img_path}' not found, skipping.[/yellow]")
            continue

        console.print(f"[cyan]Processing:[/cyan] {img_path.name} ...")
        try:
            embedding = extract_embedding_from_image(img_path)
            if embedding is None:
                console.print(f"  [yellow]No face detected in {img_path.name}, skipping.[/yellow]")
                continue
            face_db.register(name, embedding)
            registered += 1
            if verbose:
                console.print(
                    f"  [green]Registered[/green] face from {img_path.name} "
                    f"(total embeddings for '{name}': {face_db.embedding_count(name)})"
                )
        except Exception as exc:
            console.print(f"[red]Error processing {img_path.name}: {exc}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()

    if registered:
        console.print(
            f"\n[green]Done.[/green] Registered {registered} face embedding(s) for "
            f"'[bold]{name}[/bold]' in {face_db.path}"
        )
        total = face_db.embedding_count(name)
        console.print(f"  Total embeddings for '{name}': {total}")
    else:
        console.print(f"[red]No faces were registered for '{name}'.[/red]")
        raise typer.Exit(1)


@app.command(name="list-faces")
def list_faces() -> None:
    """List all registered face identities."""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from imganalyzer.analysis.ai.face_db import FaceDatabase
    except ImportError as exc:
        console.print(f"[red]Face database requires local-ai extras: {exc}[/red]")
        raise typer.Exit(1)

    face_db = FaceDatabase()
    names = face_db.list_names()

    if not names:
        console.print("[yellow]No faces registered yet.[/yellow]")
        console.print("Use [bold]imganalyzer register-face NAME IMAGE[/bold] to register a face.")
        return

    table = Table(title="Registered Faces", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Embeddings", justify="right")

    for name in sorted(names):
        count = face_db.embedding_count(name)
        table.add_row(name, str(count))

    console.print(table)
    console.print(f"\nDatabase: {face_db.path}")


@app.command(name="remove-face")
def remove_face(
    name: str = typer.Argument(..., help="Identity name to remove."),
) -> None:
    """Remove a registered face identity from the database."""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from imganalyzer.analysis.ai.face_db import FaceDatabase
    except ImportError as exc:
        console.print(f"[red]Face database requires local-ai extras: {exc}[/red]")
        raise typer.Exit(1)

    face_db = FaceDatabase()
    if face_db.remove(name):
        console.print(f"[green]Removed[/green] '{name}' from face database.")
    else:
        console.print(f"[yellow]'{name}' not found in face database.[/yellow]")
        raise typer.Exit(1)


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
    if result.ai_analysis and verbose:
        ai = result.ai_analysis
        if ai.get("description"):
            desc = ai["description"][:120]
            suffix = "..." if len(ai["description"]) > 120 else ""
            console.print(f"  [dim]AI: {desc}{suffix}[/dim]")
        if ai.get("detected_objects"):
            objs = ", ".join(ai["detected_objects"][:5])
            console.print(f"  [dim]Objects: {objs}[/dim]")
        if ai.get("face_count"):
            ids = ", ".join(ai.get("face_identities") or [])
            console.print(f"  [dim]Faces: {ai['face_count']} detected ({ids})[/dim]")


def _print_info_table(result: "AnalysisResult") -> None:
    table = Table(title=f"Image Analysis: {result.source_path.name}", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold", width=24)
    table.add_column("Value")

    data = result.to_dict()
    sections = [
        ("File", {
            "Path": str(result.source_path),
            "Format": data.get("format", ""),
            "Size": f"{data.get('width', '')} x {data.get('height', '')} px" if data.get("width") else "",
            "File size": f"{result.source_path.stat().st_size / 1024:.1f} KB",
        }),
        ("Camera", {k: str(v) for k, v in (result.metadata or {}).items() if v}),
        ("Technical", {k: str(round(v, 4)) if isinstance(v, float) else str(v)
                       for k, v in (result.technical or {}).items() if v is not None}),
        ("AI Analysis", {k: str(v) for k, v in (result.ai_analysis or {}).items() if v}),
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
