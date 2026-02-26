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

AI_BACKENDS = ["openai", "anthropic", "google", "copilot", "local", "none"]


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
    skipped = 0
    for img_path in resolved:
        xmp_path = output if output else img_path.with_suffix(".xmp")

        if not img_path.exists():
            console.print(f"[yellow]Skip:[/yellow] {img_path.name} — file not found or inaccessible")
            skipped += 1
            continue

        if xmp_path.exists() and not overwrite:
            if not quiet:
                console.print(f"[yellow]Skip:[/yellow] {xmp_path} already exists (use --overwrite)")
            skipped += 1
            continue

        if not quiet:
            console.print(f"[cyan]Analyzing:[/cyan] {img_path.name} ...")

        try:
            result = analyzer.analyze(img_path)
            result.write_xmp(xmp_path)
            _persist_result_to_db(result, ai_backend=ai)
            success += 1
            if not quiet:
                _print_summary(result, xmp_path, verbose)
        except Exception as exc:
            console.print(f"[red]Error processing {img_path.name}: {exc}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()

    if not quiet:
        parts = [f"{success}/{len(resolved)} file(s) processed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        console.print(f"\n[green]Done.[/green] {', '.join(parts)}.")


def _persist_result_to_db(result: "AnalysisResult", ai_backend: str) -> None:
    """Store an AnalysisResult in the database for consistency with the batch pipeline.

    Best-effort: if the DB layer fails (e.g. missing deps), the XMP was already
    written so the user still gets output.  Errors are silently ignored.
    """
    try:
        from imganalyzer.db.connection import get_db
        from imganalyzer.db.repository import Repository

        conn = get_db()
        repo = Repository(conn)

        image_id = repo.register_image(
            file_path=str(result.source_path.resolve()),
            width=result.width,
            height=result.height,
            fmt=result.format,
        )

        # Wrap all upserts in a single transaction for atomicity.
        # In isolation_level=None mode, we need explicit BEGIN/COMMIT.
        conn.execute("BEGIN IMMEDIATE")
        try:
            if result.metadata:
                repo.upsert_metadata(image_id, dict(result.metadata))

            if result.technical:
                repo.upsert_technical(image_id, dict(result.technical))

            if result.ai_analysis:
                if ai_backend == "local":
                    data = dict(result.ai_analysis)
                    # Infer has_people from face_count if present
                    data.setdefault("has_people", bool(data.get("face_count", 0) > 0))
                    repo.upsert_local_ai(image_id, data)
                elif ai_backend in ("openai", "anthropic", "google", "copilot"):
                    cloud_data = dict(result.ai_analysis)
                    # Strip aesthetic fields — they go to analysis_aesthetic, not analysis_cloud_ai
                    aesthetic_score = cloud_data.pop("aesthetic_score", None)
                    aesthetic_label = cloud_data.pop("aesthetic_label", None)
                    aesthetic_reason = cloud_data.pop("aesthetic_reason", None)
                    repo.upsert_cloud_ai(image_id, ai_backend, cloud_data)
                    # Copilot also returns aesthetic fields — store them separately
                    if ai_backend == "copilot" and aesthetic_score is not None:
                        repo.upsert_aesthetic(image_id, {
                            "aesthetic_score": aesthetic_score,
                            "aesthetic_label": aesthetic_label or "",
                            "aesthetic_reason": aesthetic_reason or "",
                            "provider": "copilot/gpt-4.1",
                        })

            repo.update_search_index(image_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except Exception:
        pass  # best-effort — XMP already written


@app.command(name="register-face")
def register_face(
    name: str = typer.Argument(..., help="Identity name to register (e.g. 'Alice')."),
    images: list[Path] = typer.Argument(..., help="One or more face image files."),
    display: Optional[str] = typer.Option(None, "--display", help="Display name (defaults to identity name)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Register a face identity for recognition in future analyses.

    Provide one or more clear face photos.  Multiple images improve matching
    accuracy across different angles and lighting conditions.

    Face data is stored in the SQLite database (not the legacy JSON file).

    Example::

        imganalyzer register-face Alice alice1.jpg alice2.jpg
        imganalyzer register-face Alice alice3.jpg --display "Alice Chen"
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from imganalyzer.analysis.ai.faces import extract_embedding_from_image
    except ImportError as exc:
        console.print(f"[red]Face analysis requires local-ai extras: {exc}[/red]")
        raise typer.Exit(1)

    import numpy as np
    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)

    # Register (or get existing) face identity
    identity_id = repo.register_face_identity(name, display_name=display)
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
            # Store as raw bytes (float32)
            embedding_blob = embedding.astype(np.float32).tobytes()
            repo.add_face_embedding(
                identity_id, embedding_blob, source_image=str(img_path.resolve())
            )
            registered += 1
            if verbose:
                total = len(repo.get_face_embeddings(identity_id))
                console.print(
                    f"  [green]Registered[/green] face from {img_path.name} "
                    f"(total embeddings for '{name}': {total})"
                )
        except Exception as exc:
            console.print(f"[red]Error processing {img_path.name}: {exc}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()

    if registered:
        total = len(repo.get_face_embeddings(identity_id))
        console.print(
            f"\n[green]Done.[/green] Registered {registered} face embedding(s) for "
            f"'[bold]{name}[/bold]'"
        )
        console.print(f"  Total embeddings for '{name}': {total}")
    else:
        console.print(f"[red]No faces were registered for '{name}'.[/red]")
        raise typer.Exit(1)


@app.command(name="list-faces")
def list_faces() -> None:
    """List all registered face identities from the database."""
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db, get_db_path
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)
    identities = repo.list_face_identities()

    if not identities:
        console.print("[yellow]No faces registered yet.[/yellow]")
        console.print("Use [bold]imganalyzer register-face NAME IMAGE[/bold] to register a face.")
        return

    import json as _json
    table = Table(title="Registered Faces", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Display Name")
    table.add_column("Aliases")
    table.add_column("Embeddings", justify="right")

    for ident in identities:
        aliases = _json.loads(ident.get("aliases") or "[]")
        table.add_row(
            ident["canonical_name"],
            ident.get("display_name") or "",
            ", ".join(aliases) if aliases else "",
            str(ident.get("embedding_count", 0)),
        )

    console.print(table)
    console.print(f"\nDatabase: {get_db_path()}")


@app.command(name="remove-face")
def remove_face(
    name: str = typer.Argument(..., help="Identity name to remove."),
) -> None:
    """Remove a registered face identity and all its embeddings from the database."""
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)
    if repo.remove_face_identity(name):
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
    console.print(f"  [green]OK[/green] XMP written -> {xmp_path}")
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


# ── Batch processing commands ──────────────────────────────────────────────────

@app.command()
def ingest(
    folders: list[Path] = typer.Argument(..., help="Folder(s) to scan for images."),
    modules: Optional[str] = typer.Option(
        None, "--modules", "-m",
        help="Comma-separated modules to enqueue (default: all). "
             "Options: metadata,technical,local_ai,cloud_ai,aesthetic,embedding",
    ),
    force: bool = typer.Option(False, "--force", help="Re-enqueue even if already analyzed"),
    no_recursive: bool = typer.Option(False, "--no-recursive", help="Don't scan subfolders"),
    no_hash: bool = typer.Option(False, "--no-hash", help="Skip file hash computation (faster)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Scan folder(s) for images, register in database, and enqueue analysis jobs.

    After ingesting, run ``imganalyzer run`` to start processing.

    Example::

        imganalyzer ingest /photos/2024 /photos/2025
        imganalyzer ingest /photos --modules metadata,technical
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.pipeline.batch import BatchProcessor

    conn = get_db()
    processor = BatchProcessor(conn)

    mod_list = None
    if modules:
        mod_list = [m.strip() for m in modules.split(",")]

    processor.ingest(
        folders=folders,
        modules=mod_list,
        force=force,
        recursive=not no_recursive,
        compute_hash=not no_hash,
        verbose=verbose,
    )


@app.command(name="run")
def run_queue(
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers (IO-bound modules only)"),
    force: bool = typer.Option(False, "--force", help="Ignore cache, re-run everything"),
    cloud_provider: str = typer.Option("openai", "--cloud", help="Cloud AI provider for cloud_ai/aesthetic modules"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    batch_size: int = typer.Option(10, "--batch-size", help="Jobs to claim per batch"),
    no_xmp: bool = typer.Option(False, "--no-xmp", help="Skip XMP sidecar generation after processing"),
    detection_prompt: Optional[str] = typer.Option(None, "--detection-prompt"),
    detection_threshold: Optional[float] = typer.Option(None, "--detection-threshold", min=0.0, max=1.0),
    face_threshold: Optional[float] = typer.Option(None, "--face-threshold", min=0.0, max=1.0),
) -> None:
    """Start processing the job queue.

    Processes pending jobs from the database queue.  Press Ctrl+C to pause
    gracefully (current batch finishes, remaining jobs stay queued).

    Resume by running this command again.

    Example::

        imganalyzer ingest /photos
        imganalyzer run --workers 4
        # Ctrl+C to pause
        imganalyzer run  # resumes
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.pipeline.worker import Worker

    conn = get_db()
    worker = Worker(
        conn=conn,
        workers=workers,
        force=force,
        cloud_provider=cloud_provider,
        detection_prompt=detection_prompt,
        detection_threshold=detection_threshold,
        face_match_threshold=face_threshold,
        verbose=verbose,
        write_xmp=not no_xmp,
    )
    worker.run(batch_size=batch_size)


@app.command()
def status() -> None:
    """Show the current state of the processing queue."""
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    conn = get_db()
    queue = JobQueue(conn)
    repo = Repository(conn)

    total_images = repo.count_images()
    console.print(f"\n[bold]Database:[/bold] {total_images} images registered\n")

    module_stats = queue.stats()
    if not module_stats:
        console.print("[yellow]No jobs in queue.[/yellow]")
        return

    table = Table(title="Queue Status", show_header=True, header_style="bold cyan")
    table.add_column("Module", style="bold")
    table.add_column("Pending", justify="right")
    table.add_column("Running", justify="right")
    table.add_column("Done", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Skipped", justify="right", style="yellow")

    for module in ("metadata", "technical", "local_ai", "cloud_ai", "aesthetic", "embedding"):
        stats = module_stats.get(module, {})
        table.add_row(
            module,
            str(stats.get("pending", 0)),
            str(stats.get("running", 0)),
            str(stats.get("done", 0)),
            str(stats.get("failed", 0)),
            str(stats.get("skipped", 0)),
        )

    console.print(table)

    totals = queue.total_stats()
    console.print(
        f"\n[bold]Total:[/bold] "
        f"Pending: {totals.get('pending', 0)}  "
        f"Running: {totals.get('running', 0)}  "
        f"Done: {totals.get('done', 0)}  "
        f"Failed: {totals.get('failed', 0)}  "
        f"Skipped: {totals.get('skipped', 0)}"
    )


@app.command()
def rebuild(
    module: str = typer.Argument(
        ...,
        help="Module to rebuild: metadata, technical, local_ai, cloud_ai, aesthetic, embedding",
    ),
    image: Optional[str] = typer.Option(
        None, "--image", help="Rebuild for a single image (file path)"
    ),
    force: bool = typer.Option(True, "--force/--no-force", help="Force re-run (default: True)"),
) -> None:
    """Re-enqueue a specific analysis module for all (or one) image(s).

    Clears existing analysis data (respects manual overrides) and
    re-enqueues the jobs.  Then run ``imganalyzer run`` to process.

    Example::

        imganalyzer rebuild technical
        imganalyzer rebuild local_ai --image /photos/sunset.jpg
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.pipeline.batch import BatchProcessor

    conn = get_db()
    processor = BatchProcessor(conn)

    try:
        count = processor.rebuild_module(module=module, image_path=image, force=force)
        if count:
            console.print(f"\nRun [bold]imganalyzer run[/bold] to process the queued jobs.")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command(name="override")
def set_override(
    image_path: str = typer.Argument(..., help="Image file path"),
    field: str = typer.Argument(..., help="Field name to override (e.g. 'description', 'sharpness_score')"),
    value: str = typer.Argument(..., help="Override value"),
    table_name: Optional[str] = typer.Option(
        None, "--table",
        help="Target table (auto-detected if omitted): "
             "analysis_metadata, analysis_technical, analysis_local_ai, "
             "analysis_cloud_ai, analysis_aesthetic",
    ),
    note: Optional[str] = typer.Option(None, "--note", help="Note for the override"),
) -> None:
    """Manually override an analysis field for an image.

    Overridden fields are protected from being overwritten by re-analysis
    or rebuilds, unless the override is explicitly removed.

    Example::

        imganalyzer override /photos/sunset.jpg description "A stunning sunset over the lake"
        imganalyzer override /photos/portrait.jpg sharpness_score 95.0 --table analysis_technical
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)

    img = repo.get_image_by_path(image_path)
    if img is None:
        console.print(f"[red]Image not found in database: {image_path}[/red]")
        console.print("Run [bold]imganalyzer ingest[/bold] first to register images.")
        raise typer.Exit(1)

    # Auto-detect table from field name
    resolved_table = table_name or _detect_table_for_field(field)
    if resolved_table is None:
        console.print(
            f"[red]Cannot auto-detect table for field '{field}'. "
            f"Use --table to specify.[/red]"
        )
        raise typer.Exit(1)

    repo.set_override(img["id"], resolved_table, field, value, note=note)
    console.print(
        f"[green]Override set:[/green] {field} = {value} "
        f"(table: {resolved_table}, image: {image_path})"
    )


@app.command(name="alias-face")
def alias_face(
    name: str = typer.Argument(..., help="Canonical face identity name"),
    add: Optional[str] = typer.Option(None, "--add", help="Alias to add"),
    remove: Optional[str] = typer.Option(None, "--remove", help="Alias to remove"),
) -> None:
    """Add or remove an alias for a face identity.

    Example::

        imganalyzer alias-face Alice --add "Alicia"
        imganalyzer alias-face Alice --add "Al"
        imganalyzer alias-face Alice --remove "Al"
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)

    if not add and not remove:
        # Show current aliases
        identity = repo.get_face_identity(name)
        if identity is None:
            console.print(f"[red]Face identity '{name}' not found.[/red]")
            raise typer.Exit(1)
        import json
        aliases = json.loads(identity.get("aliases") or "[]")
        console.print(f"[bold]{name}[/bold] aliases: {', '.join(aliases) or '(none)'}")
        return

    try:
        if add:
            repo.add_face_alias(name, add)
            console.print(f"[green]Added alias[/green] '{add}' for '{name}'")
        if remove:
            if repo.remove_face_alias(name, remove):
                console.print(f"[green]Removed alias[/green] '{remove}' from '{name}'")
            else:
                console.print(f"[yellow]Alias '{remove}' not found for '{name}'[/yellow]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command(name="rename-face")
def rename_face_cmd(
    name: str = typer.Argument(..., help="Current canonical name"),
    display: str = typer.Option(..., "--display", help="New display name"),
) -> None:
    """Change the display name of a face identity.

    Example::

        imganalyzer rename-face Alice --display "Alice Chen"
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)
    repo.rename_face(name, display)
    console.print(f"[green]Display name updated:[/green] '{name}' -> '{display}'")


@app.command(name="merge-face")
def merge_face(
    keep: str = typer.Argument(..., help="Identity to keep"),
    merge: str = typer.Argument(..., help="Identity to merge into keep (will be deleted)"),
) -> None:
    """Merge two face identities into one.

    All embeddings from *merge* are moved to *keep*.  The *merge* identity
    is deleted and added as an alias of *keep*.

    Example::

        imganalyzer merge-face Alice Alicia
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.repository import Repository

    conn = get_db()
    repo = Repository(conn)

    try:
        repo.merge_faces(keep, merge)
        console.print(
            f"[green]Merged[/green] '{merge}' into '{keep}'. "
            f"'{merge}' is now an alias of '{keep}'."
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command(name="search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query text"),
    mode: str = typer.Option(
        "hybrid", "--mode",
        help="Search mode: text (FTS5), semantic (CLIP), hybrid (both)",
    ),
    face: Optional[str] = typer.Option(None, "--face", help="Search by face identity name/alias"),
    exif: Optional[str] = typer.Option(None, "--exif", help="Search by EXIF (camera/lens)"),
    location: Optional[str] = typer.Option(None, "--location", help="Search by location"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
    semantic_weight: float = typer.Option(0.5, "--semantic-weight", min=0.0, max=1.0),
) -> None:
    """Search the image database by text, face, EXIF metadata, or semantics.

    Example::

        imganalyzer search "sunset over mountain lake"
        imganalyzer search "dog" --mode semantic
        imganalyzer search "" --face Alice
        imganalyzer search "" --exif "Canon EOS R5"
        imganalyzer search "" --location "Paris"
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.search import SearchEngine

    conn = get_db()
    engine = SearchEngine(conn)

    results: list[dict] = []

    if face:
        results = engine.search_face(face, limit=limit)
    elif exif or location:
        results = engine.search_exif(
            camera=exif, location=location, limit=limit,
        )
    elif query:
        results = engine.search(
            query, limit=limit, semantic_weight=semantic_weight, mode=mode,
        )
    else:
        console.print("[yellow]Provide a query, --face, --exif, or --location.[/yellow]")
        raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", width=4)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Type", width=8)
    table.add_column("File", style="bold")
    table.add_column("Details")

    for i, r in enumerate(results, 1):
        file_path = r.get("file_path", "?")
        # Show only filename for brevity
        from pathlib import Path as _P
        short = _P(file_path).name if file_path != "?" else "?"
        table.add_row(
            str(i),
            f"{r.get('score', 0):.3f}",
            r.get("match_type", "?"),
            short,
            r.get("snippet", ""),
        )

    console.print(table)
    console.print(f"\n{len(results)} result(s)")


# ── Helper functions ──────────────────────────────────────────────────────────

_FIELD_TABLE_MAP: dict[str, str] = {
    # metadata fields
    "camera_make": "analysis_metadata",
    "camera_model": "analysis_metadata",
    "lens_model": "analysis_metadata",
    "focal_length": "analysis_metadata",
    "f_number": "analysis_metadata",
    "exposure_time": "analysis_metadata",
    "iso": "analysis_metadata",
    "date_time_original": "analysis_metadata",
    "gps_latitude": "analysis_metadata",
    "gps_longitude": "analysis_metadata",
    "location_city": "analysis_metadata",
    "location_country": "analysis_metadata",
    # technical fields
    "sharpness_score": "analysis_technical",
    "sharpness_label": "analysis_technical",
    "exposure_ev": "analysis_technical",
    "exposure_label": "analysis_technical",
    "noise_level": "analysis_technical",
    "noise_label": "analysis_technical",
    "snr_db": "analysis_technical",
    "dynamic_range_stops": "analysis_technical",
    # local AI fields
    "description": "analysis_local_ai",
    "scene_type": "analysis_local_ai",
    "main_subject": "analysis_local_ai",
    "lighting": "analysis_local_ai",
    "mood": "analysis_local_ai",
    "keywords": "analysis_local_ai",
    "detected_objects": "analysis_local_ai",
    "face_count": "analysis_local_ai",
    "face_identities": "analysis_local_ai",
    # aesthetic fields
    "aesthetic_score": "analysis_aesthetic",
    "aesthetic_label": "analysis_aesthetic",
    "aesthetic_reason": "analysis_aesthetic",
}


def _detect_table_for_field(field: str) -> str | None:
    """Auto-detect which table a field belongs to."""
    return _FIELD_TABLE_MAP.get(field)


if __name__ == "__main__":
    app()
