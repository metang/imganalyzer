"""LocalAIFull — orchestrates all local AI models with gating and parallelism.

Pipeline per image:
  1. BLIP-2 captioning
  2. GroundingDINO object detection  (sequential)
  3. InsightFace face analysis       (gated: only if has_person=True)
"""
from __future__ import annotations

from typing import Any


class LocalAIFull:
    """Full local AI pipeline for image analysis.

    Run with ``--ai local``.  All sub-models are lazily loaded singletons so
    the first image in a batch pays the model-loading cost; subsequent images
    in the same process are fast.
    """

    def analyze(
        self,
        image_data: dict[str, Any],
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
    ) -> dict[str, Any]:
        from rich.console import Console
        _con = Console()

        # ── Stage 1: BLIP-2 captioning ────────────────────────────────────
        _con.print("[dim]  [1/3] Captioning...[/dim]")

        blip_result: dict[str, Any] = {}

        def _run_blip() -> dict[str, Any]:
            from imganalyzer.analysis.ai.local import LocalAI
            return LocalAI().analyze(image_data)

        try:
            blip_result = _run_blip()
        except Exception as exc:
            _con.print(f"[yellow]  BLIP-2 warning: {exc}[/yellow]")

        # ── Stage 2: Object detection ──────────────────────────────────────
        _con.print("[dim]  [2/3] Object detection...[/dim]")
        object_result: dict[str, Any] = {}
        try:
            from imganalyzer.analysis.ai.objects import ObjectDetector
            object_result = ObjectDetector().analyze(
                image_data,
                prompt=detection_prompt,
                threshold=detection_threshold,
            )
        except Exception as exc:
            _con.print(f"[yellow]  Object detection warning: {exc}[/yellow]")

        has_person: bool = object_result.get("has_person", False)

        # ── Stage 3: Face analysis (gated on has_person) ───────────────────
        face_result: dict[str, Any] = {}
        if has_person:
            _con.print("[dim]  [3/3] Face detection & recognition...[/dim]")
            try:
                from imganalyzer.analysis.ai.faces import FaceAnalyzer
                from imganalyzer.analysis.ai.face_db import FaceDatabase
                face_db = FaceDatabase()
                face_result = FaceAnalyzer().analyze(
                    image_data,
                    face_db=face_db if len(face_db) > 0 else None,
                    match_threshold=face_match_threshold,
                )
            except Exception as exc:
                _con.print(f"[yellow]  Face analysis warning: {exc}[/yellow]")
        else:
            _con.print("[dim]  [3/3] No people detected — skipping face analysis.[/dim]")

        # ── Merge results ──────────────────────────────────────────────────
        merged: dict[str, Any] = {}
        merged.update(blip_result)
        merged.update(object_result)
        merged.update(face_result)

        # Merge detected object labels (stripped of confidence) into keywords
        keywords: list[str] = list(merged.get("keywords") or [])
        detected: list[str] = merged.get("detected_objects") or []
        obj_labels = list(dict.fromkeys(
            o.split(":")[0].strip() for o in detected if o
        ))
        for lbl in obj_labels:
            if lbl and lbl not in keywords:
                keywords.append(lbl)
        if keywords:
            merged["keywords"] = keywords

        # Remove internal flag — not needed in the output dict
        merged.pop("has_person", None)

        return merged
