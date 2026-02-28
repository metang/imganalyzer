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
    detect_people: bool = typer.Option(
        False,
        "--detect-people",
        help="Run object detection only (no BLIP-2 / OCR / cloud AI) to populate "
             "has_people in the database.  Use before --ai copilot to ensure images "
             "with people are not sent to the cloud model.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive", "-r",
        help="Scan directories recursively (includes subdirectories).",
    ),
    no_xmp: bool = typer.Option(
        False,
        "--no-xmp",
        help="Do not write or update XMP sidecar files (DB-only mode).",
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
            glob_pattern = "**/*" if recursive else "*"
            found = sorted(
                p for p in img.glob(glob_pattern)
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
        detect_people=detect_people,
    )

    success = 0
    skipped = 0
    for img_path in resolved:
        xmp_path = output if output else img_path.with_suffix(".xmp")

        if not img_path.exists():
            console.print(f"[yellow]Skip:[/yellow] {img_path.name} — file not found or inaccessible")
            skipped += 1
            continue

        # When only detecting people or --no-xmp, skip the XMP-exists check — no XMP written.
        if not detect_people and not no_xmp and xmp_path.exists() and not overwrite:
            if not quiet:
                console.print(f"[yellow]Skip:[/yellow] {xmp_path} already exists (use --overwrite)")
            skipped += 1
            continue

        # For cloud AI with --no-xmp: skip images that already have an aesthetic row in the DB
        # (avoids redundant cloud calls on re-runs).
        if no_xmp and not detect_people and ai in ("openai", "anthropic", "google", "copilot") and not overwrite:
            try:
                from imganalyzer.db.connection import get_db as _get_db
                from imganalyzer.db.repository import Repository as _Repo
                _conn = _get_db()
                _repo = _Repo(_conn)
                _img = _repo.get_image_by_path(str(img_path.resolve()))
                if _img is not None:
                    _ae = _repo.get_analysis(_img["id"], "aesthetic")
                    if _ae is not None:
                        skipped += 1
                        continue
            except Exception:
                pass  # best-effort — proceed with analysis if DB check fails

        if not quiet:
            label = "Detecting people:" if detect_people else "Analyzing:"
            console.print(f"[cyan]{label}[/cyan] {img_path.name} ...")

        try:
            result = analyzer.analyze(img_path)
            if not detect_people and not no_xmp:
                result.write_xmp(xmp_path)
            _persist_result_to_db(result, ai_backend=ai, detect_people=detect_people)
            success += 1
            if not quiet and not detect_people and not no_xmp:
                _print_summary(result, xmp_path, verbose)
            elif not quiet and detect_people:
                has_p = result.ai_analysis.get("has_people", False)
                flag = "[red]people detected[/red]" if has_p else "[green]no people[/green]"
                console.print(f"  {flag}")
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


def _persist_result_to_db(
    result: "AnalysisResult",
    ai_backend: str,
    detect_people: bool = False,
) -> None:
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
                if detect_people or ai_backend == "local":
                    # People detection pre-pass or full local AI — both go to local_ai table.
                    data = dict(result.ai_analysis)
                    data.setdefault("has_people", bool(data.get("face_count", 0) > 0))
                    repo.upsert_local_ai(image_id, data)
                elif ai_backend in ("openai", "anthropic", "google", "copilot"):
                    cloud_data = dict(result.ai_analysis)
                    # Strip aesthetic fields — they go to analysis_aesthetic, not analysis_cloud_ai
                    aesthetic_score = cloud_data.pop("aesthetic_score", None)
                    aesthetic_label = cloud_data.pop("aesthetic_label", None)
                    aesthetic_reason = cloud_data.pop("aesthetic_reason", None)
                    repo.upsert_cloud_ai(image_id, ai_backend, cloud_data)
                    # Copilot also returns aesthetic fields — store them separately.
                    # People guard: do not store aesthetic scores for images with people.
                    if ai_backend == "copilot" and aesthetic_score is not None:
                        local_data = repo.get_analysis(image_id, "local_ai")
                        has_people = bool(local_data and local_data.get("has_people"))
                        if not has_people:
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
             "Options: metadata,technical,local_ai,blip2,objects,ocr,faces,cloud_ai,aesthetic,embedding",
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
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers (metadata/technical modules)"),
    cloud_workers: int = typer.Option(4, "--cloud-workers", help="Number of parallel workers for cloud AI modules (cloud_ai, aesthetic)"),
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
        cloud_workers=cloud_workers,
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
def status(
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON instead of a table"),
) -> None:
    """Show the current state of the processing queue."""
    import json as _json

    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository

    conn = get_db()
    queue = JobQueue(conn)
    repo = Repository(conn)

    total_images = repo.count_images()
    module_stats = queue.stats()
    totals = queue.total_stats()

    if json_output:
        # Emit a single JSON line — consumed by the Electron GUI poller
        all_modules = ("metadata", "technical", "local_ai", "cloud_ai", "aesthetic", "embedding")
        modules_out: dict[str, dict[str, int]] = {}
        for mod in all_modules:
            s = module_stats.get(mod, {})
            modules_out[mod] = {
                "pending": s.get("pending", 0),
                "running": s.get("running", 0),
                "done":    s.get("done", 0),
                "failed":  s.get("failed", 0),
                "skipped": s.get("skipped", 0),
            }
        payload = {
            "total_images": total_images,
            "modules": modules_out,
            "totals": {
                "pending": totals.get("pending", 0),
                "running": totals.get("running", 0),
                "done":    totals.get("done", 0),
                "failed":  totals.get("failed", 0),
                "skipped": totals.get("skipped", 0),
            },
        }
        print(_json.dumps(payload), flush=True)
        return

    console.print(f"\n[bold]Database:[/bold] {total_images} images registered\n")

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

    console.print(
        f"\n[bold]Total:[/bold] "
        f"Pending: {totals.get('pending', 0)}  "
        f"Running: {totals.get('running', 0)}  "
        f"Done: {totals.get('done', 0)}  "
        f"Failed: {totals.get('failed', 0)}  "
        f"Skipped: {totals.get('skipped', 0)}"
    )


@app.command(name="queue-clear")
def queue_clear(
    folder: Optional[str] = typer.Argument(
        None,
        help="Only clear jobs for images whose path starts with this folder. "
             "Omit to clear all jobs.",
    ),
    status_filter: str = typer.Option(
        "pending,running",
        "--status",
        help="Comma-separated statuses to delete (default: pending,running).",
    ),
) -> None:
    """Delete queued jobs — used by the GUI after a Stop action.

    By default removes only ``pending`` and ``running`` jobs so that
    already-completed results are preserved.  Pass ``--status all`` to
    wipe everything.

    Examples::

        imganalyzer queue-clear /photos/2024
        imganalyzer queue-clear /photos/2024 --status pending,running,failed
        imganalyzer queue-clear --status all
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.queue import JobQueue

    conn = get_db()
    queue = JobQueue(conn)

    if status_filter.strip().lower() == "all":
        statuses: list[str] | None = None
    else:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]

    if folder:
        deleted = queue.clear_by_folder(folder, statuses)
        scope = f"folder '{folder}'"
    else:
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            cur = conn.execute(
                f"DELETE FROM job_queue WHERE status IN ({placeholders})", statuses
            )
            conn.commit()
            deleted = cur.rowcount
        else:
            deleted = queue.clear_all()
        scope = "all images"

    console.print(f"[green]Cleared {deleted} job(s) for {scope}.[/green]")


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
    failed_only: bool = typer.Option(
        False, "--failed-only", help="Only re-enqueue images with a failed job (not all images)"
    ),
) -> None:
    """Re-enqueue a specific analysis module for all (or one) image(s).

    Clears existing analysis data (respects manual overrides) and
    re-enqueues the jobs.  Then run ``imganalyzer run`` to process.

    Example::

        imganalyzer rebuild technical
        imganalyzer rebuild local_ai --image /photos/sunset.jpg
        imganalyzer rebuild aesthetic --failed-only
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.pipeline.batch import BatchProcessor

    conn = get_db()
    processor = BatchProcessor(conn)

    try:
        count = processor.rebuild_module(module=module, image_path=image, force=force, failed_only=failed_only)
        if count:
            console.print(f"\nRun [bold]imganalyzer run[/bold] to process the queued jobs.")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command(name="purge-missing")
def purge_missing() -> None:
    """Remove DB entries and queue jobs for image files that no longer exist on disk.

    Scans all registered images and deletes any whose file_path does not exist,
    along with all their associated queue jobs and analysis data (via CASCADE).

    Example::

        imganalyzer purge-missing
    """
    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db

    conn = get_db()
    rows = conn.execute("SELECT id, file_path FROM images").fetchall()

    missing = [(r["id"], r["file_path"]) for r in rows if not Path(r["file_path"]).exists()]

    if not missing:
        console.print("[green]No missing files — database is clean.[/green]")
        return

    for image_id, file_path in missing:
        conn.execute("DELETE FROM images WHERE id = ?", [image_id])
        console.print(f"  [red]Removed[/red] {file_path}")

    conn.commit()
    console.print(f"\n[green]Purged {len(missing)} missing image(s) and their associated jobs/analysis data.[/green]")


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


@app.command(name="search-json")
def search_json_cmd(
    query: str = typer.Argument(default="", help="Semantic/keyword search query (empty = browse all)"),
    mode: str = typer.Option(
        "hybrid", "--mode",
        help="Search mode: text (FTS5), semantic (CLIP), hybrid (both), browse (no text filter)",
    ),
    # ── Text-based filters ──────────────────────────────────────────────
    face: Optional[str] = typer.Option(None, "--face", help="Filter by face identity name/alias"),
    camera: Optional[str] = typer.Option(None, "--camera", help="Filter by camera make/model"),
    lens: Optional[str] = typer.Option(None, "--lens", help="Filter by lens model"),
    location: Optional[str] = typer.Option(None, "--location", help="Filter by location"),
    # ── Numeric range filters ──────────────────────────────────────────
    aesthetic_min: Optional[float] = typer.Option(None, "--aesthetic-min", help="Min aesthetic score (0-10)"),
    aesthetic_max: Optional[float] = typer.Option(None, "--aesthetic-max", help="Max aesthetic score (0-10)"),
    sharpness_min: Optional[float] = typer.Option(None, "--sharpness-min", help="Min sharpness (0-100)"),
    sharpness_max: Optional[float] = typer.Option(None, "--sharpness-max", help="Max sharpness (0-100)"),
    noise_max: Optional[float] = typer.Option(None, "--noise-max", help="Max noise level"),
    iso_min: Optional[int] = typer.Option(None, "--iso-min", help="Min ISO"),
    iso_max: Optional[int] = typer.Option(None, "--iso-max", help="Max ISO"),
    faces_min: Optional[int] = typer.Option(None, "--faces-min", help="Min face count"),
    faces_max: Optional[int] = typer.Option(None, "--faces-max", help="Max face count"),
    date_from: Optional[str] = typer.Option(None, "--date-from", help="Date from (YYYY-MM-DD)"),
    date_to: Optional[str] = typer.Option(None, "--date-to", help="Date to (YYYY-MM-DD)"),
    has_people: Optional[bool] = typer.Option(None, "--has-people/--no-people", help="Filter by people presence"),
    # ── Pagination ─────────────────────────────────────────────────────
    limit: int = typer.Option(200, "--limit", "-n", help="Maximum results"),
    offset: int = typer.Option(0, "--offset", help="Result offset (for pagination)"),
    semantic_weight: float = typer.Option(0.5, "--semantic-weight", min=0.0, max=1.0),
) -> None:
    """Search the image database and output JSON for GUI consumption.

    Supports semantic/keyword query combined with metric-based filters.
    Returns a JSON array of image records with all analysis data merged.
    """
    import json as _json

    from dotenv import load_dotenv
    load_dotenv()

    from imganalyzer.db.connection import get_db
    from imganalyzer.db.search import SearchEngine

    conn = get_db()
    engine = SearchEngine(conn)

    # ── Step 1: Get candidate set from text/semantic/browse search ──────────
    has_text_query = bool(query and query.strip())
    has_text_filters = bool(face or camera or lens or location)  # noqa: F841

    candidate_ids: list[int] | None = None  # None = no restriction yet
    score_map: dict[int, float] = {}

    if face:
        text_results = engine.search_face(face, limit=limit * 4)
        candidate_ids = [r["image_id"] for r in text_results]
        score_map = {r["image_id"]: r["score"] for r in text_results}
    elif has_text_query and mode != "browse":
        text_results = engine.search(
            query, limit=limit * 4, semantic_weight=semantic_weight, mode=mode,
        )
        candidate_ids = [r["image_id"] for r in text_results]
        score_map = {r["image_id"]: r["score"] for r in text_results}
    else:
        score_map = {}

    # ── Step 2: Build rich SQL query joining all analysis tables ────────────
    # We join all analysis tables so metric filters and rich data can be fetched
    # in one pass.
    conditions: list[str] = []
    params: list = []

    if candidate_ids is not None:
        # Preserve ordering: use a CASE expression or filter by id set
        id_placeholders = ",".join("?" * len(candidate_ids))
        if id_placeholders:
            conditions.append(f"i.id IN ({id_placeholders})")
            params.extend(candidate_ids)
        else:
            # Empty result from text search — return nothing
            sys.stdout.write(_json.dumps({"results": [], "total": 0}) + "\n")
            sys.stdout.flush()
            return

    # Camera / lens / location filters
    if camera:
        conditions.append("(m.camera_make LIKE ? OR m.camera_model LIKE ?)")
        params.extend([f"%{camera}%", f"%{camera}%"])
    if lens:
        conditions.append("m.lens_model LIKE ?")
        params.append(f"%{lens}%")
    if location:
        conditions.append(
            "(m.location_city LIKE ? OR m.location_state LIKE ? OR m.location_country LIKE ?)"
        )
        params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])

    # Date range
    if date_from:
        conditions.append("m.date_time_original >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("m.date_time_original <= ?")
        params.append(date_to + "T23:59:59")

    # ISO range
    if iso_min is not None:
        conditions.append("CAST(m.iso AS REAL) >= ?")
        params.append(iso_min)
    if iso_max is not None:
        conditions.append("CAST(m.iso AS REAL) <= ?")
        params.append(iso_max)

    # Aesthetic score
    if aesthetic_min is not None:
        conditions.append("ae.aesthetic_score >= ?")
        params.append(aesthetic_min)
    if aesthetic_max is not None:
        conditions.append("ae.aesthetic_score <= ?")
        params.append(aesthetic_max)

    # Sharpness
    if sharpness_min is not None:
        conditions.append("t.sharpness_score >= ?")
        params.append(sharpness_min)
    if sharpness_max is not None:
        conditions.append("t.sharpness_score <= ?")
        params.append(sharpness_max)

    # Noise
    if noise_max is not None:
        conditions.append("t.noise_level <= ?")
        params.append(noise_max)

    # Face count
    if faces_min is not None:
        conditions.append("la.face_count >= ?")
        params.append(faces_min)
    if faces_max is not None:
        conditions.append("la.face_count <= ?")
        params.append(faces_max)

    # Has people
    if has_people is True:
        conditions.append("la.has_people = 1")
    elif has_people is False:
        conditions.append("(la.has_people = 0 OR la.has_people IS NULL)")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # When we have a score_map from semantic/text search, we must NOT apply
    # SQL LIMIT before Python re-sorting: SQLite returns rows in arbitrary
    # order from the IN(...) clause, so a LIMIT here would silently discard
    # high-scoring candidates that happen to fall outside the first N rows.
    # Instead, fetch ALL matching rows for the candidate set and slice in
    # Python after sorting.  For browse/filter-only queries (no score_map)
    # we apply LIMIT/OFFSET directly in SQL for efficiency.
    if score_map:
        sql = f"""
            SELECT
                i.id           AS image_id,
                i.file_path,
                i.width,
                i.height,
                i.file_size,
                -- metadata
                m.camera_make,
                m.camera_model,
                m.lens_model,
                m.focal_length,
                m.f_number,
                m.exposure_time,
                m.iso,
                m.date_time_original,
                m.gps_latitude,
                m.gps_longitude,
                m.location_city,
                m.location_state,
                m.location_country,
                -- technical
                t.sharpness_score,
                t.sharpness_label,
                t.exposure_ev,
                t.exposure_label,
                t.noise_level,
                t.noise_label,
                t.snr_db,
                t.dynamic_range_stops,
                t.highlight_clipping_pct,
                t.shadow_clipping_pct,
                t.avg_saturation,
                t.dominant_colors,
                -- local AI
                la.description,
                la.scene_type,
                la.main_subject,
                la.lighting,
                la.mood,
                la.keywords,
                la.detected_objects,
                la.face_count,
                la.face_identities,
                la.has_people,
                la.ocr_text,
                -- aesthetic
                ae.aesthetic_score,
                ae.aesthetic_label,
                ae.aesthetic_reason
            FROM images i
            LEFT JOIN analysis_metadata  m  ON m.image_id  = i.id
            LEFT JOIN analysis_technical t  ON t.image_id  = i.id
            LEFT JOIN analysis_local_ai  la ON la.image_id = i.id
            LEFT JOIN analysis_aesthetic ae ON ae.image_id = i.id
            {where_clause}
        """
        # No LIMIT/OFFSET — will slice after Python sort
    else:
        sql = f"""
            SELECT
                i.id           AS image_id,
                i.file_path,
                i.width,
                i.height,
                i.file_size,
                -- metadata
                m.camera_make,
                m.camera_model,
                m.lens_model,
                m.focal_length,
                m.f_number,
                m.exposure_time,
                m.iso,
                m.date_time_original,
                m.gps_latitude,
                m.gps_longitude,
                m.location_city,
                m.location_state,
                m.location_country,
                -- technical
                t.sharpness_score,
                t.sharpness_label,
                t.exposure_ev,
                t.exposure_label,
                t.noise_level,
                t.noise_label,
                t.snr_db,
                t.dynamic_range_stops,
                t.highlight_clipping_pct,
                t.shadow_clipping_pct,
                t.avg_saturation,
                t.dominant_colors,
                -- local AI
                la.description,
                la.scene_type,
                la.main_subject,
                la.lighting,
                la.mood,
                la.keywords,
                la.detected_objects,
                la.face_count,
                la.face_identities,
                la.has_people,
                la.ocr_text,
                -- aesthetic
                ae.aesthetic_score,
                ae.aesthetic_label,
                ae.aesthetic_reason
            FROM images i
            LEFT JOIN analysis_metadata  m  ON m.image_id  = i.id
            LEFT JOIN analysis_technical t  ON t.image_id  = i.id
            LEFT JOIN analysis_local_ai  la ON la.image_id = i.id
            LEFT JOIN analysis_aesthetic ae ON ae.image_id = i.id
            {where_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()

    # Build results, preserving text-search score ordering
    records = []
    for row in rows:
        iid = row["image_id"]
        score = score_map.get(iid, 0.0) if score_map else None

        def _json_field(val: str | None) -> list | None:
            if val is None:
                return None
            try:
                return _json.loads(val)
            except Exception:
                return val  # type: ignore[return-value]

        records.append({
            "image_id": iid,
            "file_path": row["file_path"],
            "score": score,
            "width": row["width"],
            "height": row["height"],
            "file_size": row["file_size"],
            # metadata
            "camera_make": row["camera_make"],
            "camera_model": row["camera_model"],
            "lens_model": row["lens_model"],
            "focal_length": row["focal_length"],
            "f_number": row["f_number"],
            "exposure_time": row["exposure_time"],
            "iso": row["iso"],
            "date_time_original": row["date_time_original"],
            "gps_latitude": row["gps_latitude"],
            "gps_longitude": row["gps_longitude"],
            "location_city": row["location_city"],
            "location_state": row["location_state"],
            "location_country": row["location_country"],
            # technical
            "sharpness_score": row["sharpness_score"],
            "sharpness_label": row["sharpness_label"],
            "exposure_ev": row["exposure_ev"],
            "exposure_label": row["exposure_label"],
            "noise_level": row["noise_level"],
            "noise_label": row["noise_label"],
            "snr_db": row["snr_db"],
            "dynamic_range_stops": row["dynamic_range_stops"],
            "highlight_clipping_pct": row["highlight_clipping_pct"],
            "shadow_clipping_pct": row["shadow_clipping_pct"],
            "avg_saturation": row["avg_saturation"],
            "dominant_colors": _json_field(row["dominant_colors"]),
            # local AI
            "description": row["description"],
            "scene_type": row["scene_type"],
            "main_subject": row["main_subject"],
            "lighting": row["lighting"],
            "mood": row["mood"],
            "keywords": _json_field(row["keywords"]),
            "detected_objects": _json_field(row["detected_objects"]),
            "face_count": row["face_count"],
            "face_identities": _json_field(row["face_identities"]),
            "has_people": bool(row["has_people"]) if row["has_people"] is not None else None,
            "ocr_text": row["ocr_text"],
            # aesthetic
            "aesthetic_score": row["aesthetic_score"],
            "aesthetic_label": row["aesthetic_label"],
            "aesthetic_reason": row["aesthetic_reason"],
        })

    # If we had a score_map, re-sort records by score descending then apply
    # offset/limit in Python (correct order; SQL LIMIT was not applied above).
    if score_map:
        records.sort(key=lambda r: -(r["score"] or 0.0))
        total = len(records)
        records = records[offset: offset + limit]
    else:
        total = len(records)  # SQL already applied LIMIT/OFFSET

    output = {"results": records, "total": total}
    sys.stdout.write(_json.dumps(output) + "\n")
    sys.stdout.flush()

    # Release GPU memory before the process exits.  search-json is spawned
    # fresh for every search, so the CLIP model is loaded once per call.
    # Explicitly deleting it and calling empty_cache ensures CUDA returns the
    # memory to the OS promptly instead of waiting for garbage collection.
    try:
        import torch
        from imganalyzer.embeddings.clip_embedder import CLIPEmbedder
        if CLIPEmbedder._model is not None:
            del CLIPEmbedder._model
            CLIPEmbedder._model = None
            CLIPEmbedder._preprocess = None
            CLIPEmbedder._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
