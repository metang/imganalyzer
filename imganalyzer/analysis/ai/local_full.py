"""LocalAIFull — orchestrates all local AI models with gating and parallelism.

Pipeline per image:
  1. BLIP-2 captioning  ┐ run concurrently via ThreadPoolExecutor
  2. Aesthetic scoring  ┘
  3. GroundingDINO object detection  (sequential)
  4. InsightFace face analysis       (gated: only if has_person=True)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

import numpy as np


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

        # ── Stage 1: BLIP-2 + Aesthetic in parallel ───────────────────────
        _con.print("[dim]  [1/4] Caption + aesthetic scoring (parallel)...[/dim]")

        blip_result: dict[str, Any] = {}
        aesthetic_result: dict[str, Any] = {}
        blip_exc: BaseException | None = None
        aesthetic_exc: BaseException | None = None

        def _run_blip() -> dict[str, Any]:
            from imganalyzer.analysis.ai.local import LocalAI
            return LocalAI().analyze(image_data)

        def _run_aesthetic() -> dict[str, Any]:
            from imganalyzer.analysis.ai.aesthetic import AestheticScorer
            return AestheticScorer().analyze(image_data)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_blip: Future[dict[str, Any]] = executor.submit(_run_blip)
            future_aesthetic: Future[dict[str, Any]] = executor.submit(_run_aesthetic)

            try:
                blip_result = future_blip.result()
            except Exception as exc:
                blip_exc = exc
                _con.print(f"[yellow]  BLIP-2 warning: {exc}[/yellow]")

            try:
                aesthetic_result = future_aesthetic.result()
            except Exception as exc:
                aesthetic_exc = exc
                _con.print(f"[yellow]  Aesthetic scorer warning: {exc}[/yellow]")

        # ── Stage 2: Object detection ──────────────────────────────────────
        _con.print("[dim]  [2/4] Object detection...[/dim]")
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
            _con.print("[dim]  [3/4] Face detection & recognition...[/dim]")
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
            _con.print("[dim]  [3/4] No people detected — skipping face analysis.[/dim]")

        # ── Stage 4: Merge results ─────────────────────────────────────────
        _con.print("[dim]  [4/4] Merging results...[/dim]")
        merged: dict[str, Any] = {}
        merged.update(blip_result)
        merged.update(aesthetic_result)
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
