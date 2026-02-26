"""Main analysis orchestrator."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# All RAW extensions supported by LibRaw/rawpy
RAW_EXTENSIONS = {
    ".3fr", ".ari", ".arw", ".bay", ".braw", ".crw", ".cr2", ".cr3",
    ".cap", ".data", ".dcs", ".dcr", ".dng", ".drf", ".eip", ".erf",
    ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos",
    ".mrw", ".nef", ".nrw", ".obm", ".orf", ".pef", ".ptx", ".pxn",
    ".r3d", ".raf", ".raw", ".rwl", ".rw2", ".rwz", ".sr2", ".srf",
    ".srw", ".tif", ".x3f",
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


class Analyzer:
    def __init__(
        self,
        ai_backend: str = "none",
        run_technical: bool = True,
        verbose: bool = False,
        detection_prompt: str | None = None,
        detection_threshold: float | None = None,
        face_match_threshold: float | None = None,
    ) -> None:
        self.ai_backend = ai_backend
        self.run_technical = run_technical
        self.verbose = verbose
        self.detection_prompt = detection_prompt
        self.detection_threshold = detection_threshold
        self.face_match_threshold = face_match_threshold

    def analyze(self, path: Path) -> AnalysisResult:
        suffix = path.suffix.lower()
        is_raw = suffix in RAW_EXTENSIONS

        if self.verbose:
            console.print(f"  [dim]Reading {'RAW' if is_raw else 'standard'} image...[/dim]")

        # Read image
        if is_raw:
            from imganalyzer.readers.raw import RawReader
            reader = RawReader(path)
        else:
            from imganalyzer.readers.standard import StandardReader
            reader = StandardReader(path)

        image_data = reader.read()

        result = AnalysisResult(
            source_path=path,
            format=image_data["format"],
            width=image_data["width"],
            height=image_data["height"],
        )

        # Metadata / EXIF
        if self.verbose:
            console.print("  [dim]Extracting metadata...[/dim]")
        from imganalyzer.analysis.metadata import MetadataExtractor
        result.metadata = MetadataExtractor(path, image_data).extract()

        # Technical analysis
        if self.run_technical:
            if self.verbose:
                console.print("  [dim]Running technical analysis...[/dim]")
            from imganalyzer.analysis.technical import TechnicalAnalyzer
            result.technical = TechnicalAnalyzer(image_data).analyze()

        # AI analysis
        if self.ai_backend and self.ai_backend != "none":
            if self.verbose:
                console.print(f"  [dim]Running AI analysis ({self.ai_backend})...[/dim]")
            if self.ai_backend == "local":
                from imganalyzer.analysis.ai.local_full import LocalAIFull
                result.ai_analysis = LocalAIFull().analyze(
                    image_data,
                    detection_prompt=self.detection_prompt,
                    detection_threshold=self.detection_threshold,
                    face_match_threshold=self.face_match_threshold,
                )
            elif self.ai_backend in ("openai", "anthropic", "google", "copilot"):
                # People guard: do not send images containing recognisable faces to
                # cloud AI.  Check the DB for a previously stored local_ai result; if
                # has_people is set, skip the cloud call entirely.
                has_people = self._db_has_people(path)
                if has_people:
                    console.print(
                        f"  [yellow]Skip cloud AI:[/yellow] {path.name} — "
                        "people detected, image will not be sent to cloud model"
                    )
                else:
                    from imganalyzer.analysis.ai.cloud import CloudAI
                    result.ai_analysis = CloudAI(backend=self.ai_backend).analyze(path, image_data)

        return result

    def _db_has_people(self, path: Path) -> bool:
        """Return True if the DB records has_people=1 for this image (local_ai row).

        Best-effort: returns False on any error (missing DB, image not registered,
        local_ai not yet run) so the cloud call proceeds in ambiguous cases.
        """
        try:
            from imganalyzer.db.connection import get_db
            from imganalyzer.db.repository import Repository
            conn = get_db()
            repo = Repository(conn)
            img = repo.get_image_by_path(str(path.resolve()))
            if img is None:
                return False
            local_data = repo.get_analysis(img["id"], "local_ai")
            if local_data is None:
                return False
            return bool(local_data.get("has_people"))
        except Exception:
            return False

