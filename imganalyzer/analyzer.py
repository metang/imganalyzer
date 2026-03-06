"""Main analysis orchestrator."""
from __future__ import annotations

import dataclasses
import logging
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()
_log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

# All RAW extensions supported by LibRaw/rawpy.
# NOTE: .tif / .tiff are standard TIFF files — Pillow handles them directly.
# They must NOT be listed here or they will be routed through rawpy (LibRaw),
# which does not support TIFF and raises LibRawFileUnsupportedError.
RAW_EXTENSIONS = {
    ".3fr", ".ari", ".arw", ".bay", ".braw", ".crw", ".cr2", ".cr3",
    ".cap", ".data", ".dcs", ".dcr", ".dng", ".drf", ".eip", ".erf",
    ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos",
    ".mrw", ".nef", ".nrw", ".obm", ".orf", ".pef", ".ptx", ".pxn",
    ".r3d", ".raf", ".raw", ".rwl", ".rw2", ".rwz", ".sr2", ".srf",
    ".srw", ".x3f",
}


@dataclasses.dataclass
class AnalysisResult:
    source_path: Path
    format: str = ""
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    technical: dict[str, Any] = dataclasses.field(default_factory=dict)
    ai_analysis: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
            "technical": self.technical,
            "ai_analysis": self.ai_analysis,
        }

    def write_xmp(self, path: Path) -> None:
        from imganalyzer.output.xmp import XMPWriter
        writer = XMPWriter(self)
        writer.write(path)


class AnalysisCancelled(Exception):
    """Raised when a single-image analysis is cancelled."""


class Analyzer:
    def __init__(
        self,
        ai_backend: str = "none",
        run_technical: bool = True,
        verbose: bool = False,
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
        detect_people: bool = False,
    ) -> None:
        self.ai_backend = ai_backend
        self.run_technical = run_technical
        self.verbose = verbose
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold
        self.detect_people = detect_people

    def _emit_progress(
        self,
        stage: str,
        *,
        progress_cb: ProgressCallback | None,
        console_stage: str | None = None,
    ) -> None:
        """Emit a stage update to the optional callback and verbose console."""
        if progress_cb is not None:
            progress_cb(stage)
        if self.verbose:
            console.print(console_stage or stage)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        """Raise when the caller has requested cancellation."""
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled("Analysis cancelled")

    def analyze(
        self,
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> AnalysisResult:
        suffix = path.suffix.lower()
        is_raw = suffix in RAW_EXTENSIONS

        self._check_cancel(cancel_event)
        self._emit_progress(
            f"Reading {'RAW' if is_raw else 'standard'} image...",
            progress_cb=progress_cb,
            console_stage=f"  [dim]Reading {'RAW' if is_raw else 'standard'} image...[/dim]",
        )

        # Read image
        reader: Any
        if is_raw:
            from imganalyzer.readers.raw import RawReader
            reader = RawReader(path)
        else:
            from imganalyzer.readers.standard import StandardReader
            reader = StandardReader(path)

        image_data = reader.read()
        self._check_cancel(cancel_event)

        result = AnalysisResult(
            source_path=path,
            format=image_data["format"],
            width=image_data["width"],
            height=image_data["height"],
        )

        # Metadata / EXIF
        self._emit_progress(
            "Extracting metadata...",
            progress_cb=progress_cb,
            console_stage="  [dim]Extracting metadata...[/dim]",
        )
        from imganalyzer.analysis.metadata import MetadataExtractor
        result.metadata = MetadataExtractor(path, image_data).extract()
        self._check_cancel(cancel_event)

        # Technical analysis
        if self.run_technical:
            self._emit_progress(
                "Running technical analysis...",
                progress_cb=progress_cb,
                console_stage="  [dim]Running technical analysis...[/dim]",
            )
            from imganalyzer.analysis.technical import TechnicalAnalyzer
            result.technical = TechnicalAnalyzer(image_data).analyze()
            self._check_cancel(cancel_event)

        # People detection only (fast pre-pass before cloud AI)
        if self.detect_people:
            self._emit_progress(
                "Running people detection...",
                progress_cb=progress_cb,
                console_stage="  [dim]Running people detection...[/dim]",
            )
            result.ai_analysis = self._run_people_detection(image_data)
            self._check_cancel(cancel_event)

        # AI analysis
        elif self.ai_backend and self.ai_backend != "none":
            self._emit_progress(
                f"Running AI analysis ({self.ai_backend})...",
                progress_cb=progress_cb,
                console_stage=f"  [dim]Running AI analysis ({self.ai_backend})...[/dim]",
            )
            self._check_cancel(cancel_event)
            if self.ai_backend == "local":
                from imganalyzer.analysis.ai.local_full import LocalAIFull
                result.ai_analysis = LocalAIFull().analyze(
                    image_data,
                    detection_prompt=self.detection_prompt,
                    detection_threshold=self.detection_threshold,
                    face_match_threshold=self.face_match_threshold,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                )
                self._check_cancel(cancel_event)
            elif self.ai_backend in ("openai", "anthropic", "google", "copilot"):
                # People guard: do not send images containing recognisable faces to
                # cloud AI.  Check the DB for a previously stored local_ai result; if
                # has_people is set, skip the cloud call entirely.
                has_people = self._db_has_people(path)
                self._check_cancel(cancel_event)
                if has_people:
                    _log.debug("Skip cloud AI: %s — people detected", path.name)
                else:
                    from imganalyzer.analysis.ai.cloud import CloudAI
                    result.ai_analysis = CloudAI(backend=self.ai_backend).analyze(path, image_data)
                self._check_cancel(cancel_event)

        return result

    def _run_people_detection(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """Run object detection only and return a minimal dict with has_people set.

        This is a lightweight pre-pass: only GroundingDINO runs (no BLIP-2,
        no OCR, no InsightFace).  The result is suitable for persisting to
        analysis_local_ai so the cloud AI guard can use has_people on the
        next run.
        """
        from imganalyzer.analysis.ai.objects import ObjectDetector
        obj = ObjectDetector().analyze(
            image_data,
            prompt=self.detection_prompt,
            threshold=self.detection_threshold,
        )
        has_people = bool(obj.get("has_person", False))
        return {
            "detected_objects": obj.get("detected_objects", []),
            "has_people": has_people,
        }

    def _db_has_people(self, path: Path) -> bool:
        """Return True if the DB records has_people=1 for this image (local_ai row).

        Best-effort: returns False on any error (missing DB, image not registered,
        local_ai not yet run) so the cloud call proceeds in ambiguous cases.
        """
        conn: sqlite3.Connection | None = None
        try:
            from imganalyzer.db.connection import get_db_path
            from imganalyzer.db.repository import Repository

            db_path = get_db_path()
            if not db_path.exists():
                return False

            conn = sqlite3.connect(
                str(db_path),
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")

            repo = Repository(conn)
            img = repo.get_image_by_path(str(path.resolve()))
            if img is None:
                return False
            local_data = repo.get_analysis(img["id"], "local_ai")
            if local_data is None:
                return False
            return bool(local_data.get("has_people"))
        except Exception as exc:
            _log.warning("Privacy guard lookup failed for %s: %s", path, exc)
            return False
        finally:
            if conn is not None:
                conn.close()
