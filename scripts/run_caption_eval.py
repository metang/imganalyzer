"""Run caption evaluation for Qwen 3.5 and Qwen 2.5 VL models via Ollama."""

import sys
import json
from pathlib import Path

# Add model-eval to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model-eval"))

from model_eval.runner import run_evaluation, collect_images, console

IMAGE_DIR = Path(__file__).resolve().parent.parent / "test_images"

MODELS = [
    "qwen3.5-vl-2b",
    "qwen3.5-vl-4b",
    "qwen3.5-vl-9b",
    "qwen2.5-vl-3b",
    "qwen2.5-vl-7b",
]


def main():
    console.rule("[bold]Caption Evaluation — Qwen 3.5 + 2.5 VL via Ollama")
    console.print(f"Image dir: {IMAGE_DIR}")
    console.print(f"Models: {', '.join(MODELS)}")

    images = collect_images(IMAGE_DIR)
    console.print(f"Found {len(images)} images (including RAW/HEIC)")

    results = run_evaluation(MODELS, IMAGE_DIR, device="cuda", skip_cached=True)

    # Print comparison
    console.print()
    console.rule("[bold green]Results Summary")

    for img in images:
        console.print(f"\n[bold yellow]━━━ {img.name} ━━━[/bold yellow]")
        for model in MODELS:
            r = results.get(model, {}).get(img.stem, {})
            caption = r.get("caption", r.get("error", "N/A"))
            time_s = r.get("inference_time_s", "?")
            # Truncate for display
            if len(caption) > 300:
                caption = caption[:297] + "..."
            console.print(f"  [cyan]{model}[/cyan] ({time_s}s): {caption}")

    # Save full results
    out_path = Path(__file__).resolve().parent.parent / "model-eval" / "caption_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
