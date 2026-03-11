"""Orchestrator: run selected models against a set of images."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from model_eval.cache import get_cached, store_result
from model_eval.config import get_adapter

# Register HEIC support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Limit PyTorch CUDA memory to avoid virtual memory swap
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

console = Console(stderr=True)

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".tiff", ".tif", ".bmp", ".webp",
    ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf",
}


def collect_images(image_dir: Path) -> list[Path]:
    """Collect all image files from a directory (non-recursive)."""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    return images


def run_evaluation(
    model_names: list[str],
    image_dir: Path,
    device: str = "cuda",
    skip_cached: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run models against images sequentially, caching results.

    Returns nested dict: results[model_name][image_stem] = result_dict
    """
    images = collect_images(image_dir)
    if not images:
        console.print(f"[red]No images found in {image_dir}[/red]")
        return {}

    console.print(f"Found [bold]{len(images)}[/bold] images in {image_dir}")
    results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        console.rule(f"[bold blue]{model_name}[/bold blue]")
        results[model_name] = {}

        # Check how many are already cached
        uncached = [
            img for img in images
            if not skip_cached or get_cached(model_name, img) is None
        ]

        if skip_cached and len(uncached) < len(images):
            cached_count = len(images) - len(uncached)
            console.print(f"  [dim]{cached_count} results from cache[/dim]")
            # Load cached results
            for img in images:
                cached = get_cached(model_name, img)
                if cached is not None:
                    results[model_name][img.stem] = cached

        if not uncached:
            console.print("  [green]All cached, skipping model load[/green]")
            continue

        # Load model
        try:
            adapter = get_adapter(model_name)
            console.print(f"  Loading {adapter.model_id or model_name}...")
            adapter.load(device=device)
        except Exception as e:
            console.print(f"  [red]Failed to load: {e}[/red]")
            for img in uncached:
                error_result = {"error": str(e), "model": model_name}
                results[model_name][img.stem] = error_result
                store_result(model_name, img, error_result)
            continue

        # Run inference
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"  Running {model_name}...", total=len(uncached)
            )
            for img in uncached:
                progress.update(task, description=f"  {img.name}")
                try:
                    result = adapter.run_timed(img)
                except Exception as e:
                    result = {"error": str(e), "model": model_name}
                    console.print(f"  [red]Error on {img.name}: {e}[/red]")

                results[model_name][img.stem] = result
                store_result(model_name, img, result)
                progress.advance(task)

        # Unload model and free GPU
        try:
            adapter.unload()
        except Exception:
            pass
        _free_gpu_memory()
        if torch.cuda.is_available():
            used_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
            console.print(f"  [dim]GPU: {used_mb}MB allocated, {reserved_mb}MB reserved[/dim]")
        console.print(f"  [green]Done — {len(uncached)} images processed[/green]")

    return results


def _free_gpu_memory() -> None:
    """Aggressively free GPU memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def load_all_results(
    model_names: list[str],
    image_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load all cached results without running any inference."""
    images = collect_images(image_dir)
    results: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        results[model_name] = {}
        for img in images:
            cached = get_cached(model_name, img)
            if cached is not None:
                results[model_name][img.stem] = cached
    return results
