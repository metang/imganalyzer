"""Tests for imganalyzer."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
import types
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image_data():
    """Synthetic 100×100 RGB image data dict."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(80, 200, (100, 100, 3), dtype=np.uint8)
    return {
        "format": "JPEG",
        "width": 100,
        "height": 100,
        "rgb_array": rgb,
        "is_raw": False,
    }


@pytest.fixture
def dark_image_data():
    rgb = np.full((100, 100, 3), 30, dtype=np.uint8)
    return {"format": "JPEG", "width": 100, "height": 100, "rgb_array": rgb, "is_raw": False}


@pytest.fixture
def bright_image_data():
    rgb = np.full((100, 100, 3), 240, dtype=np.uint8)
    return {"format": "JPEG", "width": 100, "height": 100, "rgb_array": rgb, "is_raw": False}


@pytest.fixture
def face_embedding():
    """A synthetic 512-d unit-norm face embedding."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ── TechnicalAnalyzer ─────────────────────────────────────────────────────────

class TestTechnicalAnalyzer:
    def test_basic_output_keys(self, synthetic_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(synthetic_image_data).analyze()
        assert "sharpness_score" in result
        assert "exposure_ev" in result
        assert "noise_level" in result
        assert "mean_luminance" in result
        assert "dominant_colors" in result

    def test_sharpness_score_range(self, synthetic_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(synthetic_image_data).analyze()
        assert 0.0 <= result["sharpness_score"] <= 100.0

    def test_dark_image_exposure(self, dark_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(dark_image_data).analyze()
        assert result["exposure_ev"] < 0  # underexposed

    def test_bright_image_exposure(self, bright_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(bright_image_data).analyze()
        assert result["exposure_ev"] > 0  # overexposed

    def test_dominant_colors_hex_format(self, synthetic_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(synthetic_image_data).analyze()
        colors = result.get("dominant_colors", [])
        for c in colors:
            assert c.startswith("#"), f"Expected hex color, got: {c}"
            assert len(c) == 7

    def test_zone_distribution_sums_to_one(self, synthetic_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(synthetic_image_data).analyze()
        zones = result.get("zone_distribution", [])
        assert len(zones) == 10
        assert abs(sum(zones) - 1.0) < 0.01

    def test_noise_level_non_negative(self, synthetic_image_data):
        from imganalyzer.analysis.technical import TechnicalAnalyzer
        result = TechnicalAnalyzer(synthetic_image_data).analyze()
        assert result["noise_level"] >= 0


# ── MetadataExtractor ─────────────────────────────────────────────────────────

class TestMetadataExtractor:
    def test_returns_dict(self, tmp_path, synthetic_image_data):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.fromarray(synthetic_image_data["rgb_array"]).save(str(img_path))

        from imganalyzer.analysis.metadata import MetadataExtractor
        result = MetadataExtractor(img_path, synthetic_image_data).extract()
        assert isinstance(result, dict)

    def test_extract_logs_debug_on_exifread_failure(self, tmp_path, synthetic_image_data, caplog):
        from PIL import Image
        import imganalyzer.analysis.metadata as metadata_module

        img_path = tmp_path / "bad-exif.jpg"
        Image.fromarray(synthetic_image_data["rgb_array"]).save(str(img_path))

        def _fail_process_file(*_args, **_kwargs):
            raise ValueError("bad EXIF payload")

        fake_exifread = types.SimpleNamespace(process_file=_fail_process_file)

        with patch.dict(sys.modules, {"exifread": fake_exifread}):
            with caplog.at_level(logging.DEBUG, logger=metadata_module.__name__):
                result = metadata_module.MetadataExtractor(img_path, synthetic_image_data).extract()

        assert isinstance(result, dict)
        assert "stage=exifread" in caplog.text

    def test_no_crash_on_missing_exif(self, tmp_path, synthetic_image_data):
        """Images without EXIF should return empty dict without crashing."""
        from PIL import Image
        img_path = tmp_path / "noexif.png"
        Image.fromarray(synthetic_image_data["rgb_array"]).save(str(img_path))

        from imganalyzer.analysis.metadata import MetadataExtractor
        result = MetadataExtractor(img_path, synthetic_image_data).extract()
        assert isinstance(result, dict)


# ── StandardReader ──────────────────────────────────────────────────────────────

class TestStandardReader:
    def test_read_wraps_decode_errors_and_suppresses_tiff_logger(self, tmp_path, caplog):
        import imganalyzer.readers.standard as standard

        path = tmp_path / "bad.tiff"

        def _fail_open(*_args, **_kwargs):
            logging.getLogger("PIL.TiffImagePlugin").error(
                "More samples per pixel than can be decoded: %s", 10
            )
            raise SyntaxError("Invalid value for samples per pixel")

        with caplog.at_level(logging.ERROR, logger="PIL.TiffImagePlugin"), \
             patch("PIL.Image.open", side_effect=_fail_open):
            with pytest.raises(
                ValueError,
                match=r"Pillow cannot decode bad\.tiff: Invalid value for samples per pixel",
            ):
                standard.read(path)

        assert "More samples per pixel than can be decoded" not in caplog.text

    def test_read_headers_wraps_decode_errors_and_suppresses_tiff_logger(
        self,
        tmp_path,
        caplog,
    ):
        import imganalyzer.readers.standard as standard

        path = tmp_path / "headers-bad.tiff"

        def _fail_open(*_args, **_kwargs):
            logging.getLogger("PIL.TiffImagePlugin").error(
                "More samples per pixel than can be decoded: %s", 10
            )
            raise SyntaxError("Invalid value for samples per pixel")

        with caplog.at_level(logging.ERROR, logger="PIL.TiffImagePlugin"), \
             patch("PIL.Image.open", side_effect=_fail_open):
            with pytest.raises(
                ValueError,
                match=r"Pillow cannot decode headers-bad\.tiff: Invalid value for samples per pixel",
            ):
                standard.read_headers(path)

        assert "More samples per pixel than can be decoded" not in caplog.text


# ── Analyzer control path ─────────────────────────────────────────────────────

class TestAnalyzer:
    def test_cancel_event_stops_before_caption(self, tmp_path):
        from imganalyzer.analyzer import AnalysisCancelled, Analyzer

        image_path = tmp_path / "cancel.jpg"
        cancel_event = threading.Event()
        local_ai_calls = {"n": 0}
        image_data = {
            "format": "JPEG",
            "width": 64,
            "height": 64,
            "rgb_array": np.zeros((64, 64, 3), dtype=np.uint8),
            "is_raw": False,
        }

        class FakeReader:
            def __init__(self, path):
                self.path = path

            def read(self):
                return image_data

        class FakeMetadataExtractor:
            def __init__(self, path, data):
                self.path = path
                self.data = data

            def extract(self):
                return {}

        class FakeTechnicalAnalyzer:
            def __init__(self, data):
                self.data = data

            def analyze(self):
                cancel_event.set()
                return {}

        class FakeLocalAIFull:
            def analyze(self, *args, **kwargs):
                local_ai_calls["n"] += 1
                return {"description": "should not run"}

        import imganalyzer.readers.standard as standard_module
        import imganalyzer.analysis.metadata as metadata_module
        import imganalyzer.analysis.technical as technical_module
        import imganalyzer.analysis.ai.local_full as local_full_module

        with patch.object(standard_module, "StandardReader", FakeReader), \
             patch.object(metadata_module, "MetadataExtractor", FakeMetadataExtractor), \
             patch.object(technical_module, "TechnicalAnalyzer", FakeTechnicalAnalyzer), \
             patch.object(local_full_module, "LocalAIFull", FakeLocalAIFull):

            with pytest.raises(AnalysisCancelled):
                Analyzer(ai_backend="local", run_technical=True).analyze(
                    image_path,
                    cancel_event=cancel_event,
                )

        assert local_ai_calls["n"] == 0

    def test_db_has_people_uses_fresh_connection_from_background_thread(
        self,
        tmp_path,
        monkeypatch,
    ):
        from imganalyzer.analyzer import Analyzer
        from imganalyzer.db.connection import close_db, get_db
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.schema import ensure_schema

        db_path = tmp_path / "imganalyzer.db"
        image_path = tmp_path / "person.jpg"
        image_path.write_bytes(b"stub")

        close_db()
        monkeypatch.setenv("IMGANALYZER_DB_PATH", str(db_path))

        conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        ensure_schema(conn)

        try:
            repo = Repository(conn)
            image_id = repo.register_image(str(image_path.resolve()))
            repo.upsert_caption(image_id, {"has_people": True})
            conn.commit()

            # Prime the global singleton on the main thread so the background
            # lookup would fail if it tried to reuse this same connection.
            get_db()

            result_box: dict[str, bool] = {}
            error_box: list[Exception] = []

            def _lookup() -> None:
                try:
                    result_box["value"] = Analyzer(ai_backend="openai")._db_has_people(image_path)
                except Exception as exc:  # pragma: no cover - diagnostic only
                    error_box.append(exc)

            thread = threading.Thread(target=_lookup)
            thread.start()
            thread.join()

            assert not error_box
            assert result_box["value"] is True
        finally:
            close_db()
            conn.close()


# ── CloudAI cleanup ─────────────────────────────────────────────────────────────

class TestCloudAI:
    def test_copilot_backend_deletes_session_and_stops_client(self, tmp_path, monkeypatch):
        from imganalyzer.analysis.ai.cloud import CloudAI

        image_path = tmp_path / "copilot.jpg"
        image_path.write_bytes(b"stub")
        created_clients: list = []
        deleted_session_ids: list[str] = []

        class FakeSession:
            session_id = "test-session-1"

            async def send_and_wait(self, payload, timeout):
                class _Data:
                    content = json.dumps({"description": "ok", "keywords": ["tag"]})

                class _Event:
                    data = _Data()

                return _Event()

        class FakeCopilotClient:
            def __init__(self):
                self.stop_calls = 0
                created_clients.append(self)

            async def create_session(self, config):
                handler = config.get("on_permission_request")
                assert callable(handler)
                result = handler(None, {})
                assert result["kind"] == "approved"
                return FakeSession()

            async def delete_session(self, session_id):
                deleted_session_ids.append(session_id)

            async def stop(self):
                self.stop_calls += 1
                return []

        monkeypatch.setitem(
            sys.modules,
            "copilot",
            types.SimpleNamespace(
                CopilotClient=FakeCopilotClient,
            ),
        )

        result = CloudAI("copilot")._copilot(image_path, {})

        assert result["description"] == "ok"
        assert deleted_session_ids == ["test-session-1"]
        assert created_clients[0].stop_calls == 1

    def test_copilot_backend_deletes_session_after_failure(self, tmp_path, monkeypatch):
        from imganalyzer.analysis.ai.cloud import CloudAI

        image_path = tmp_path / "copilot.jpg"
        image_path.write_bytes(b"stub")
        created_clients: list = []
        deleted_session_ids: list[str] = []

        class FakeSession:
            session_id = "test-session-fail"

            async def send_and_wait(self, payload, timeout):
                raise RuntimeError("boom")

        class FakeCopilotClient:
            def __init__(self):
                self.stop_calls = 0
                created_clients.append(self)

            async def create_session(self, config):
                handler = config.get("on_permission_request")
                assert callable(handler)
                return FakeSession()

            async def delete_session(self, session_id):
                deleted_session_ids.append(session_id)

            async def stop(self):
                self.stop_calls += 1
                return []

        monkeypatch.setitem(
            sys.modules,
            "copilot",
            types.SimpleNamespace(
                CopilotClient=FakeCopilotClient,
            ),
        )

        with pytest.raises(RuntimeError, match="boom"):
            CloudAI("copilot")._copilot(image_path, {})

        # Session should still be deleted even on failure.
        assert deleted_session_ids == ["test-session-fail"]
        assert created_clients[0].stop_calls == 1


# ── FaceDatabase ──────────────────────────────────────────────────────────────

class TestFaceDatabase:
    def test_register_and_list(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)

        assert len(db) == 0
        db.register("Alice", face_embedding)
        assert len(db) == 1
        assert "Alice" in db.list_names()

    def test_register_multiple_embeddings(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)

        db.register("Bob", face_embedding)
        db.register("Bob", face_embedding)
        assert db.embedding_count("Bob") == 2

    def test_match_exact(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)
        db.register("Alice", face_embedding)

        name, sim = db.match(face_embedding, threshold=0.4)
        assert name == "Alice"
        assert sim > 0.99  # same vector → cosine ≈ 1.0

    def test_match_unknown_below_threshold(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)
        db.register("Alice", face_embedding)

        # Random orthogonal vector should not match
        rng = np.random.default_rng(99)
        other = rng.standard_normal(512).astype(np.float32)
        other = other / np.linalg.norm(other)
        name, sim = db.match(other, threshold=0.4)
        assert name == "Unknown"

    def test_match_empty_db(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)

        name, sim = db.match(face_embedding)
        assert name == "Unknown"
        assert sim == 0.0

    def test_remove(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)
        db.register("Alice", face_embedding)

        removed = db.remove("Alice")
        assert removed is True
        assert len(db) == 0

    def test_remove_nonexistent(self, tmp_path):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)

        removed = db.remove("Nobody")
        assert removed is False

    def test_persistence(self, tmp_path, face_embedding):
        """Data written by one FaceDatabase instance should be readable by another."""
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"

        db1 = FaceDatabase(path=db_path)
        db1.register("Carol", face_embedding)

        db2 = FaceDatabase(path=db_path)
        assert "Carol" in db2.list_names()
        name, sim = db2.match(face_embedding, threshold=0.4)
        assert name == "Carol"

    def test_db_file_is_valid_json(self, tmp_path, face_embedding):
        from imganalyzer.analysis.ai.face_db import FaceDatabase
        db_path = tmp_path / "faces.json"
        db = FaceDatabase(path=db_path)
        db.register("Dave", face_embedding)

        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "Dave" in data
        assert "embeddings" in data["Dave"]
        assert isinstance(data["Dave"]["embeddings"][0], list)


# ── XMP Writer ────────────────────────────────────────────────────────────────

class TestXMPWriter:
    def _make_result(self, tmp_path, ai_extras: dict | None = None):
        from imganalyzer.analyzer import AnalysisResult
        ai = {
            "description": "A golden hour landscape with mountains reflected in a calm lake.",
            "scene_type": "landscape",
            "mood": "serene",
            "keywords": ["mountain", "lake", "golden hour", "reflection", "nature"],
            "lighting": "Golden hour",
            "main_subject": "Mountain lake",
        }
        if ai_extras:
            ai.update(ai_extras)
        return AnalysisResult(
            source_path=tmp_path / "test.jpg",
            format="JPEG",
            width=1920,
            height=1280,
            metadata={
                "camera_make": "Canon",
                "camera_model": "EOS R5",
                "lens_model": "RF 24-70mm F2.8 L IS USM",
                "iso": 400,
                "f_number": 2.8,
                "exposure_time": "1/500",
                "focal_length": 35.0,
                "date_time_original": "2024:06:15 14:30:00",
                "gps_latitude": 48.8566,
                "gps_longitude": 2.3522,
            },
            technical={
                "sharpness_score": 72.5,
                "sharpness_label": "Sharp",
                "exposure_ev": 0.3,
                "exposure_label": "Good",
                "noise_level": 0.02,
                "noise_label": "Good",
                "snr_db": 34.5,
                "dynamic_range_stops": 7.2,
                "dominant_colors": ["#3a6b8c", "#f0c040", "#1a1a1a"],
            },
            ai_analysis=ai,
        )

    def test_xmp_is_valid_xml(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        from xml.etree import ElementTree as ET
        result = self._make_result(tmp_path)
        writer = XMPWriter(result)
        xmp_path = tmp_path / "test.xmp"
        writer.write(xmp_path)

        content = xmp_path.read_text()
        assert "xmpmeta" in content
        ET.fromstring(content.encode())

    def test_xmp_contains_camera_info(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path)
        writer = XMPWriter(result)
        xmp_path = tmp_path / "test.xmp"
        writer.write(xmp_path)

        content = xmp_path.read_text()
        assert "Canon" in content
        assert "EOS R5" in content

    def test_xmp_contains_ai_keywords(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path)
        writer = XMPWriter(result)
        xmp_path = tmp_path / "test.xmp"
        writer.write(xmp_path)

        content = xmp_path.read_text()
        assert "mountain" in content.lower()
        assert "landscape" in content.lower()

    def test_xmp_contains_technical_scores(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path)
        writer = XMPWriter(result)
        xmp_path = tmp_path / "test.xmp"
        writer.write(xmp_path)

        content = xmp_path.read_text()
        assert "SharpnessScore" in content
        assert "ExposureEV" in content
        assert "NoiseLevel" in content

    def test_xmp_gps_present(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path)
        writer = XMPWriter(result)
        xmp_path = tmp_path / "test.xmp"
        writer.write(xmp_path)

        content = xmp_path.read_text()
        assert "GPS" in content

    def test_xmp_imganalyzer_namespace(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path)
        xmp_path = tmp_path / "test.xmp"
        XMPWriter(result).write(xmp_path)
        content = xmp_path.read_text()
        assert "imganalyzer" in content

    def test_xmp_aesthetic_score(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path, ai_extras={
            "aesthetic_score": 7.3,
            "aesthetic_label": "High",
        })
        xmp_path = tmp_path / "test.xmp"
        XMPWriter(result).write(xmp_path)
        content = xmp_path.read_text()
        assert "AestheticScore" in content
        assert "7.3" in content
        assert "AestheticLabel" in content
        assert "High" in content

    def test_xmp_face_fields(self, tmp_path):
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path, ai_extras={
            "face_count": 2,
            "face_identities": ["Alice", "Unknown"],
            "face_details": ["Alice:28:Female", "Unknown:34:Male"],
        })
        xmp_path = tmp_path / "test.xmp"
        XMPWriter(result).write(xmp_path)
        content = xmp_path.read_text()
        assert "FaceCount" in content
        assert "2" in content
        assert "FaceIdentities" in content
        assert "Alice" in content
        assert "FaceDetails" in content
        assert "Alice:28:Female" in content

    def test_xmp_detected_objects_in_keywords(self, tmp_path):
        """Detected object labels should appear in dc:subject keywords."""
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path, ai_extras={
            "detected_objects": ["person:87%", "bicycle:62%"],
        })
        xmp_path = tmp_path / "test.xmp"
        XMPWriter(result).write(xmp_path)
        content = xmp_path.read_text()
        assert "AIDetectedObjects" in content
        # Labels should also be in dc:subject
        assert "person" in content
        assert "bicycle" in content

    def test_xmp_face_count_zero(self, tmp_path):
        """face_count=0 should be written to XMP (absence of faces is informative)."""
        from imganalyzer.output.xmp import XMPWriter
        result = self._make_result(tmp_path, ai_extras={"face_count": 0})
        xmp_path = tmp_path / "test.xmp"
        XMPWriter(result).write(xmp_path)
        content = xmp_path.read_text()
        assert "FaceCount" in content


# ── AnalysisResult ────────────────────────────────────────────────────────────

class TestAnalysisResult:
    def test_to_dict(self, tmp_path):
        from imganalyzer.analyzer import AnalysisResult
        r = AnalysisResult(source_path=tmp_path / "x.jpg", format="JPEG", width=800, height=600)
        d = r.to_dict()
        assert d["format"] == "JPEG"
        assert d["width"] == 800

    def test_write_xmp_creates_file(self, tmp_path):
        from imganalyzer.analyzer import AnalysisResult
        r = AnalysisResult(source_path=tmp_path / "x.jpg", format="JPEG", width=800, height=600)
        xmp_path = tmp_path / "x.xmp"
        r.write_xmp(xmp_path)
        assert xmp_path.exists()
        assert xmp_path.stat().st_size > 0


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_rational_str(self):
        from imganalyzer.output.xmp import _rational_str
        assert _rational_str(2.8) == "28/10"
        assert _rational_str(4.0) == "4/1"

    def test_decimal_to_dms(self):
        from imganalyzer.output.xmp import _decimal_to_dms_str
        result = _decimal_to_dms_str(48.8566, "lat")
        assert result.endswith("N")
        result_s = _decimal_to_dms_str(-33.8688, "lat")
        assert result_s.endswith("S")

    def test_exposure_str(self):
        from imganalyzer.output.xmp import _exposure_str
        assert _exposure_str("1/500") == "1/500"
        assert _exposure_str(0.002) == "1/500"
        assert _exposure_str(2.0) == "2/1"


# ── LocalAIFull orchestration (mocked) ───────────────────────────────────────

class TestLocalAIFull:
    """Tests that verify orchestration logic without loading real models."""

    def _make_image_data(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        return {"format": "JPEG", "width": 64, "height": 64, "rgb_array": rgb, "is_raw": False}

    def test_face_analysis_skipped_when_no_person(self):
        """FaceAnalyzer should NOT be called when ObjectDetector says has_person=False."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull

        blip_out = {"description": "A landscape.", "keywords": ["mountain"], "scene_type": "landscape",
                    "main_subject": "mountain", "lighting": "daylight", "mood": "calm"}
        object_out = {"detected_objects": ["tree:70%", "mountain:80%"], "has_person": False}

        face_call_count = {"n": 0}

        class FakeFaceAnalyzer:
            def analyze(self, *args, **kwargs):
                face_call_count["n"] += 1
                return {"face_count": 0, "face_identities": [], "face_details": []}

        class FakeBlip:
            def analyze(self, *args, **kwargs):
                return blip_out

        class FakeObjects:
            def analyze(self, *args, **kwargs):
                return object_out

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.objects as objects_module
        import imganalyzer.analysis.ai.faces as faces_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(objects_module, "ObjectDetector", FakeObjects), \
             patch.object(faces_module, "FaceAnalyzer", FakeFaceAnalyzer):

            result = LocalAIFull().analyze(self._make_image_data())

        assert face_call_count["n"] == 0, "FaceAnalyzer should not be called when no person detected"
        assert "detected_objects" in result
        # has_person flag must not leak into output
        assert "has_person" not in result

    def test_face_analysis_called_when_person_present(self):
        """FaceAnalyzer MUST be called when ObjectDetector reports has_person=True."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull

        blip_out = {"description": "A portrait.", "keywords": ["person"], "scene_type": "portrait",
                    "main_subject": "person", "lighting": "studio", "mood": "confident"}
        object_out = {"detected_objects": ["person:91%"], "has_person": True}
        face_out = {"face_count": 1, "face_identities": ["Alice"], "face_details": ["Alice:30:Female"]}

        face_call_count = {"n": 0}

        class FakeFaceAnalyzer:
            def analyze(self, *args, **kwargs):
                face_call_count["n"] += 1
                return face_out

        class FakeBlip:
            def analyze(self, *args, **kwargs):
                return blip_out

        class FakeObjects:
            def analyze(self, *args, **kwargs):
                return object_out

        class FakeFaceDB:
            def __init__(self, *args, **kwargs):
                pass
            def __len__(self):
                return 1
            def match(self, *args, **kwargs):
                return ("Alice", 0.92)

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.objects as objects_module
        import imganalyzer.analysis.ai.faces as faces_module
        import imganalyzer.analysis.ai.face_db as face_db_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(objects_module, "ObjectDetector", FakeObjects), \
             patch.object(faces_module, "FaceAnalyzer", FakeFaceAnalyzer), \
             patch.object(face_db_module, "FaceDatabase", FakeFaceDB):

            result = LocalAIFull().analyze(self._make_image_data())

        assert face_call_count["n"] == 1, "FaceAnalyzer should be called when person detected"
        assert result["face_count"] == 1
        assert result["face_identities"] == ["Alice"]
        assert "has_person" not in result

    def test_keywords_merged_with_object_labels(self):
        """Object detection labels should be merged into keywords."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull

        blip_out = {"description": "Outdoors.", "keywords": ["nature"], "scene_type": "outdoor",
                    "main_subject": "tree", "lighting": "daylight", "mood": "peaceful"}
        object_out = {"detected_objects": ["tree:75%", "sky:80%"], "has_person": False}

        class FakeBlip:
            def analyze(self, *a, **kw): return blip_out

        class FakeObjects:
            def analyze(self, *a, **kw): return object_out

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.objects as objects_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(objects_module, "ObjectDetector", FakeObjects):

            result = LocalAIFull().analyze(self._make_image_data())

        keywords = result.get("keywords", [])
        assert "tree" in keywords
        assert "sky" in keywords
        assert "nature" in keywords  # original BLIP keyword preserved

    def test_progress_callback_emits_stage_messages(self):
        """Progress callback should receive plain stage messages in order."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull

        progress: list[str] = []
        blip_out = {
            "description": "Outdoors.",
            "keywords": ["nature"],
            "scene_type": "outdoor",
            "main_subject": "tree",
            "lighting": "daylight",
            "mood": "peaceful",
        }
        object_out = {
            "detected_objects": ["tree:75%", "sky:80%"],
            "has_person": False,
            "has_text": False,
        }

        class FakeBlip:
            def analyze(self, *a, **kw):
                return blip_out

        class FakeObjects:
            def analyze(self, *a, **kw):
                return object_out

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.objects as objects_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(objects_module, "ObjectDetector", FakeObjects):

            LocalAIFull().analyze(self._make_image_data(), progress_cb=progress.append)

        assert progress[0] == "[1/4] Captioning..."
        assert progress[1] == "[2/4] Object detection..."
        assert progress[2].startswith("[3/4] No text detected")
        assert progress[3].startswith("[4/4] No people detected")

    def test_cancel_event_raises_between_stages(self):
        """Cancellation should stop the pipeline before the next stage begins."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull
        from imganalyzer.analyzer import AnalysisCancelled

        cancel_event = threading.Event()
        progress: list[str] = []

        class FakeBlip:
            def analyze(self, *a, **kw):
                return {"description": "Outdoors.", "keywords": ["nature"]}

        class FakeObjects:
            def analyze(self, *a, **kw):
                cancel_event.set()
                return {"detected_objects": ["tree:75%"], "has_person": False, "has_text": False}

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.objects as objects_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(objects_module, "ObjectDetector", FakeObjects):

            with pytest.raises(AnalysisCancelled):
                LocalAIFull().analyze(
                    self._make_image_data(),
                    cancel_event=cancel_event,
                    progress_cb=progress.append,
                )

        assert progress == [
            "[1/4] Captioning...",
            "[2/4] Object detection...",
        ]


# ── ObjectDetector (mocked) ───────────────────────────────────────────────────

class TestObjectDetector:
    def test_has_person_detection(self, synthetic_image_data):
        """Verifies that person labels set has_person=True."""
        from imganalyzer.analysis.ai.objects import ObjectDetector
        import torch

        mock_results = [{
            "scores": torch.tensor([0.91]),
            "labels": ["person"],
            "boxes": torch.zeros(1, 4),
        }]

        with patch.object(ObjectDetector, "_load_models"):
            # The processor call returns an object with .to(device) that yields
            # a dict-like object supporting item access for "input_ids".
            fake_inputs = MagicMock()
            fake_inputs.__getitem__ = lambda self, key: MagicMock()
            fake_inputs.to.return_value = fake_inputs

            fake_proc = MagicMock()
            fake_proc.post_process_grounded_object_detection.return_value = mock_results
            fake_proc.return_value = fake_inputs

            fake_model = MagicMock()
            fake_model.parameters.return_value = iter([torch.zeros(1)])
            fake_model.return_value = MagicMock()

            ObjectDetector._processor = fake_proc
            ObjectDetector._model = fake_model

            detector = ObjectDetector()
            result = detector.analyze(synthetic_image_data)

            ObjectDetector._processor = None
            ObjectDetector._model = None

        assert result.get("has_person") is True
        assert any("person" in o for o in result.get("detected_objects", []))

    def test_no_person_when_only_objects(self, synthetic_image_data):
        """Non-person labels should not set has_person."""
        from imganalyzer.analysis.ai.objects import ObjectDetector
        import torch

        mock_results = [{
            "scores": torch.tensor([0.85, 0.70]),
            "labels": ["tree", "mountain"],
            "boxes": torch.zeros(2, 4),
        }]

        with patch.object(ObjectDetector, "_load_models"):
            fake_inputs = MagicMock()
            fake_inputs.__getitem__ = lambda self, key: MagicMock()
            fake_inputs.to.return_value = fake_inputs

            fake_proc = MagicMock()
            fake_proc.post_process_grounded_object_detection.return_value = mock_results
            fake_proc.return_value = fake_inputs

            fake_model = MagicMock()
            fake_model.parameters.return_value = iter([torch.zeros(1)])
            fake_model.return_value = MagicMock()

            ObjectDetector._processor = fake_proc
            ObjectDetector._model = fake_model

            result = ObjectDetector().analyze(synthetic_image_data)

            ObjectDetector._processor = None
            ObjectDetector._model = None

        assert result.get("has_person") is False

    def test_confidence_format(self, synthetic_image_data):
        """Detected object labels should be formatted as 'label:XX%'."""
        from imganalyzer.analysis.ai.objects import ObjectDetector
        import torch

        mock_results = [{
            "scores": torch.tensor([0.75]),
            "labels": ["car"],
            "boxes": torch.zeros(1, 4),
        }]

        with patch.object(ObjectDetector, "_load_models"):
            fake_inputs = MagicMock()
            fake_inputs.__getitem__ = lambda self, key: MagicMock()
            fake_inputs.to.return_value = fake_inputs

            fake_proc = MagicMock()
            fake_proc.post_process_grounded_object_detection.return_value = mock_results
            fake_proc.return_value = fake_inputs

            fake_model = MagicMock()
            fake_model.parameters.return_value = iter([torch.zeros(1)])
            fake_model.return_value = MagicMock()

            ObjectDetector._processor = fake_proc
            ObjectDetector._model = fake_model

            result = ObjectDetector().analyze(synthetic_image_data)

            ObjectDetector._processor = None
            ObjectDetector._model = None

        objs = result.get("detected_objects", [])
        assert len(objs) == 1
        assert objs[0] == "car:75%"

    def test_load_models_falls_back_to_slow_processor_on_import_error(self):
        import types
        from imganalyzer.analysis.ai.objects import ObjectDetector

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda(), float32="float32")
        fast_processor = object()
        slow_processor = object()
        processor_calls: list[dict[str, object]] = []

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                processor_calls.append({"model_id": model_id, **kwargs})
                if kwargs.get("use_fast") is not False:
                    raise ImportError("GroundingDinoImageProcessorFast requires PyTorch")
                return slow_processor

        class FakeModel:
            def to(self, _device: str):
                return self

            def eval(self):
                return None

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs):
                return FakeModel()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeAutoProcessor,
            AutoModelForZeroShotObjectDetection=FakeAutoModel,
        )

        ObjectDetector._processor = None
        ObjectDetector._model = None
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            ObjectDetector._load_models()

        try:
            assert ObjectDetector._processor is slow_processor
            assert len(processor_calls) == 2
            assert "use_fast" not in processor_calls[0]
            assert processor_calls[1]["use_fast"] is False
        finally:
            ObjectDetector._processor = None
            ObjectDetector._model = None

    def test_load_models_uses_fast_processor_when_available(self):
        import types
        from imganalyzer.analysis.ai.objects import ObjectDetector

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch = types.SimpleNamespace(cuda=FakeCuda(), float32="float32")
        fast_processor = object()
        processor_calls: list[dict[str, object]] = []

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                processor_calls.append({"model_id": model_id, **kwargs})
                return fast_processor

        class FakeModel:
            def to(self, _device: str):
                return self

            def eval(self):
                return None

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs):
                return FakeModel()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeAutoProcessor,
            AutoModelForZeroShotObjectDetection=FakeAutoModel,
        )

        ObjectDetector._processor = None
        ObjectDetector._model = None
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            ObjectDetector._load_models()

        try:
            assert ObjectDetector._processor is fast_processor
            assert len(processor_calls) == 1
            assert "use_fast" not in processor_calls[0]
        finally:
            ObjectDetector._processor = None
            ObjectDetector._model = None


class TestHuggingFaceCache:
    def test_load_pretrained_prefers_local_cache(self):
        from imganalyzer.analysis.ai.hf_cache import load_pretrained

        calls: list[dict[str, object]] = []

        def fake_loader(model_id: str, **kwargs):
            calls.append({"model_id": model_id, **kwargs})
            return "loaded"

        with patch("imganalyzer.analysis.ai.hf_cache.has_local_snapshot", return_value=True):
            result = load_pretrained(fake_loader, "repo/model", cache_dir="cache-dir")

        assert result == "loaded"
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is True

    def test_load_pretrained_falls_back_to_network_when_cache_is_incomplete(self):
        from imganalyzer.analysis.ai.hf_cache import load_pretrained

        calls: list[dict[str, object]] = []

        def fake_loader(model_id: str, **kwargs):
            calls.append({"model_id": model_id, **kwargs})
            if kwargs.get("local_files_only"):
                raise OSError("missing local shard")
            return "loaded"

        with patch("imganalyzer.analysis.ai.hf_cache.has_local_snapshot", return_value=True):
            result = load_pretrained(fake_loader, "repo/model", cache_dir="cache-dir")

        assert result == "loaded"
        assert len(calls) == 2
        assert calls[0]["local_files_only"] is True
        assert "local_files_only" not in calls[1]


# ── Batch Processing DB Integration ──────────────────────────────────────────

def _make_test_db(tmp_path):
    """Create a fresh SQLite DB with schema for testing (bypasses singleton)."""
    import sqlite3
    from imganalyzer.db.schema import ensure_schema

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


class TestDatabaseLayer:
    """Tests for the DB layer: repository, queue, overrides, search index."""

    def test_register_image_and_retrieve(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(
            file_path="/photos/test.jpg", width=1920, height=1080, fmt="JPEG"
        )
        assert img_id > 0

        img = repo.get_image(img_id)
        assert img is not None
        assert img["file_path"] == "/photos/test.jpg"
        assert img["width"] == 1920

    def test_register_image_idempotent(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        id1 = repo.register_image(file_path="/photos/a.jpg")
        id2 = repo.register_image(file_path="/photos/a.jpg")
        assert id1 == id2
        assert repo.count_images() == 1

    def test_upsert_metadata(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        repo.upsert_metadata(img_id, {
            "camera_make": "Canon",
            "camera_model": "EOS R5",
            "iso": 400,
        })
        conn.commit()

        data = repo.get_analysis(img_id, "metadata")
        assert data is not None
        assert data["camera_make"] == "Canon"
        assert data["iso"] == 400
        assert data["analyzed_at"] is not None

    def test_upsert_technical(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        repo.upsert_technical(img_id, {
            "sharpness_score": 72.5,
            "exposure_ev": 0.3,
            "noise_level": 0.02,
            "dominant_colors": ["#3a6b8c", "#f0c040"],
        })
        conn.commit()

        data = repo.get_analysis(img_id, "technical")
        assert data is not None
        assert data["sharpness_score"] == 72.5
        # dominant_colors should be JSON-encoded
        import json as _json
        colors = _json.loads(data["dominant_colors"])
        assert len(colors) == 2

    def test_upsert_caption_with_people(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/portrait.jpg")
        repo.upsert_caption(img_id, {
            "description": "A portrait of a woman",
            "scene_type": "portrait",
            "keywords": ["person", "portrait"],
            "face_count": 1,
            "face_identities": ["Alice"],
            "has_people": True,
        })
        conn.commit()

        data = repo.get_analysis(img_id, "caption")
        assert data is not None
        assert data["has_people"] == 1  # stored as int
        assert data["face_count"] == 1

    def test_is_analyzed_returns_true_after_upsert(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        assert repo.is_analyzed(img_id, "metadata") is False

        repo.upsert_metadata(img_id, {"camera_make": "Nikon"})
        conn.commit()

        assert repo.is_analyzed(img_id, "metadata") is True

    def test_override_protection(self, tmp_path):
        """Overrides must NOT be overwritten by subsequent upserts.

        The override mask removes the field from the write data so it stays NULL
        in the analysis table.  The override value is stored in the `overrides`
        table and applied on read via `get_full_result`.
        """
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")

        # First upsert
        repo.upsert_metadata(img_id, {"camera_make": "Canon", "camera_model": "R5"})
        conn.commit()

        # Set an override on camera_make
        repo.set_override(img_id, "analysis_metadata", "camera_make", "Nikon")

        # Re-upsert with different data — camera_make should be masked out
        repo.upsert_metadata(img_id, {"camera_make": "Sony", "camera_model": "A7IV"})
        conn.commit()

        # Raw analysis row has NULL for camera_make (masked out)
        raw = repo.get_analysis(img_id, "metadata")
        assert raw["camera_make"] is None  # masked out by override
        assert raw["camera_model"] == "A7IV"  # non-overridden field updated

        # get_full_result applies overrides on top
        full = repo.get_full_result(img_id)
        assert full["metadata"]["camera_make"] == "Nikon"  # override value
        assert full["metadata"]["camera_model"] == "A7IV"

    def test_upsert_cloud_ai(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/landscape.jpg")
        repo.upsert_cloud_ai(img_id, "openai", {
            "description": "A sunset over mountains",
            "keywords": ["sunset", "mountain"],
        })
        conn.commit()

        data = repo.get_analysis(img_id, "cloud_ai")
        assert data is not None
        assert "providers" in data
        assert len(data["providers"]) == 1
        assert data["providers"][0]["provider"] == "openai"


class TestJobQueue:
    """Tests for the job queue system."""

    def test_enqueue_and_claim(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        job_id = queue.enqueue(img_id, "metadata")
        assert job_id is not None

        jobs = queue.claim(batch_size=5)
        assert len(jobs) == 1
        assert jobs[0]["image_id"] == img_id
        assert jobs[0]["module"] == "metadata"

    def test_enqueue_skip_duplicate(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        j1 = queue.enqueue(img_id, "metadata")
        j2 = queue.enqueue(img_id, "metadata")
        assert j1 is not None
        assert j2 is None  # duplicate, skipped

    def test_enqueue_batch(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        id1 = repo.register_image(file_path="/photos/a.jpg")
        id2 = repo.register_image(file_path="/photos/b.jpg")
        count = queue.enqueue_batch([id1, id2], ["metadata", "technical"])
        assert count == 4  # 2 images × 2 modules

    def test_queue_stats(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        queue.enqueue(img_id, "metadata")
        queue.enqueue(img_id, "technical")

        stats = queue.stats()
        assert "metadata" in stats
        assert stats["metadata"]["pending"] == 1
        assert "technical" in stats
        assert stats["technical"]["pending"] == 1

    def test_module_avg_processing_ms_uses_last_done_jobs_only(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        image_ids = [
            repo.register_image(file_path=f"/photos/caption-{idx}.jpg")
            for idx in range(4)
        ]
        job_ids: list[int] = []
        for image_id in image_ids:
            job_id = queue.enqueue(image_id, "caption")
            assert job_id is not None
            job_ids.append(job_id)

        conn.execute(
            """UPDATE job_queue
               SET status = 'done',
                   started_at = '2026-01-01 00:00:00.000',
                   completed_at = '2026-01-01 00:00:01.000'
               WHERE id = ?""",
            [job_ids[0]],
        )
        conn.execute(
            """UPDATE job_queue
               SET status = 'done',
                   started_at = '2026-01-01 00:00:02.000',
                   completed_at = '2026-01-01 00:00:04.000'
               WHERE id = ?""",
            [job_ids[1]],
        )
        conn.execute(
            """UPDATE job_queue
               SET status = 'done',
                   started_at = '2026-01-01 00:00:05.000',
                   completed_at = '2026-01-01 00:00:08.000'
               WHERE id = ?""",
            [job_ids[2]],
        )
        conn.execute(
            """UPDATE job_queue
               SET status = 'failed',
                   started_at = '2026-01-01 00:00:09.000',
                   completed_at = '2026-01-01 00:00:18.000'
               WHERE id = ?""",
            [job_ids[3]],
        )
        conn.commit()

        avg = queue.module_avg_processing_ms(last_n=2)
        assert avg["caption"] == 2500

    def test_recover_stale_zero_reclaims_all_running(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/recover.jpg")
        job_id = queue.enqueue(img_id, "metadata")
        assert job_id is not None
        claimed = queue.claim(batch_size=1, module="metadata")
        assert len(claimed) == 1

        recovered = queue.recover_stale(timeout_minutes=0)
        assert recovered == 1

        row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [job_id]).fetchone()
        assert row is not None
        assert row["status"] == "pending"

    def test_recover_stale_fails_jobs_over_max_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/stale_max.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        # Simulate a job that has been retried past max_attempts
        conn.execute(
            "UPDATE job_queue SET status = 'running', attempts = 5 WHERE id = ?",
            [job_id],
        )
        conn.commit()

        recovered = queue.recover_stale(timeout_minutes=0)
        # Should NOT be re-queued (over max)
        assert recovered == 0

        row = conn.execute(
            "SELECT status, error_message FROM job_queue WHERE id = ?", [job_id]
        ).fetchone()
        assert row is not None
        assert row["status"] == "failed"
        assert "max attempts" in row["error_message"].lower()
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/leased.jpg")
        job_id = queue.enqueue(img_id, "objects")
        assert job_id is not None

        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=120, batch_size=1)
        assert len(claimed) == 1
        assert claimed[0]["id"] == job_id
        assert claimed[0]["module"] == "objects"
        assert claimed[0]["lease_token"]

        lease_row = conn.execute(
            "SELECT worker_id, lease_token FROM job_leases WHERE job_id = ?",
            [job_id],
        ).fetchone()
        assert lease_row is not None
        assert lease_row["worker_id"] == "macbook-pro"
        assert lease_row["lease_token"] == claimed[0]["lease_token"]

    def test_release_expired_leases_requeues_jobs(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/expired.jpg")
        job_id = queue.enqueue(img_id, "objects")
        assert job_id is not None

        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=1, batch_size=1)
        assert len(claimed) == 1

        conn.execute(
            "UPDATE job_leases SET lease_expires_at = datetime('now', '-1 minutes') WHERE job_id = ?",
            [job_id],
        )
        conn.commit()

        released = queue.release_expired_leases()
        assert released == 1

        row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [job_id]).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        lease_row = conn.execute("SELECT 1 FROM job_leases WHERE job_id = ?", [job_id]).fetchone()
        assert lease_row is None

    def test_claim_skips_jobs_over_max_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/overclaimed.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        # Set attempts > max_attempts but status = pending (simulates
        # the state after recover_stale resets without the fix)
        conn.execute(
            "UPDATE job_queue SET attempts = 10 WHERE id = ?", [job_id]
        )
        conn.commit()

        # Local claim should skip it
        claimed = queue.claim(batch_size=10, module="caption")
        assert len(claimed) == 0

    def test_claim_leased_skips_jobs_over_max_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["test-worker", "Test Worker", "linux", "online"],
        )
        img_id = repo.register_image(file_path="/photos/overclaimed2.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        conn.execute(
            "UPDATE job_queue SET attempts = 10 WHERE id = ?", [job_id]
        )
        conn.commit()

        # Leased claim should also skip it
        claimed = queue.claim_leased(
            worker_id="test-worker", lease_ttl_seconds=120, batch_size=10,
        )
        assert len(claimed) == 0

    def test_release_expired_leases_fails_jobs_over_max_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["test-worker", "Test Worker", "linux", "online"],
        )
        img_id = repo.register_image(file_path="/photos/expired_max.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        # Claim, then set attempts high and expire the lease
        claimed = queue.claim_leased(
            worker_id="test-worker", lease_ttl_seconds=1, batch_size=1,
        )
        assert len(claimed) == 1
        conn.execute(
            "UPDATE job_queue SET attempts = 5 WHERE id = ?", [job_id]
        )
        conn.execute(
            "UPDATE job_leases SET lease_expires_at = datetime('now', '-1 minutes') WHERE job_id = ?",
            [job_id],
        )
        conn.commit()

        released = queue.release_expired_leases()
        assert released == 1

        row = conn.execute(
            "SELECT status, error_message FROM job_queue WHERE id = ?", [job_id]
        ).fetchone()
        assert row is not None
        assert row["status"] == "failed"
        assert "max attempts" in row["error_message"].lower()

        # Lease should be cleaned up
        lease_row = conn.execute(
            "SELECT 1 FROM job_leases WHERE job_id = ?", [job_id]
        ).fetchone()
        assert lease_row is None

    def test_mark_done_leased_requires_matching_token(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/complete.jpg")
        job_id = queue.enqueue(img_id, "objects")
        assert job_id is not None
        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=120, batch_size=1)
        token = claimed[0]["lease_token"]

        assert queue.mark_done_leased(job_id, "wrong-token") is False
        assert queue.mark_done_leased(job_id, token) is True

        row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [job_id]).fetchone()
        assert row is not None
        assert row["status"] == "done"
        lease_row = conn.execute("SELECT 1 FROM job_leases WHERE job_id = ?", [job_id]).fetchone()
        assert lease_row is None

    def test_heartbeat_leased_extends_expiry_and_rejects_bad_token(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/heartbeat.jpg")
        job_id = queue.enqueue(img_id, "objects")
        assert job_id is not None
        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=60, batch_size=1)
        token = claimed[0]["lease_token"]
        before = conn.execute(
            "SELECT heartbeat_at, lease_expires_at FROM job_leases WHERE job_id = ?",
            [job_id],
        ).fetchone()
        assert before is not None

        assert queue.heartbeat_lease(job_id, "wrong-token", extend_ttl_seconds=300) is False
        assert queue.heartbeat_lease(job_id, token, extend_ttl_seconds=300) is True

        after = conn.execute(
            "SELECT heartbeat_at, lease_expires_at FROM job_leases WHERE job_id = ?",
            [job_id],
        ).fetchone()
        assert after is not None
        assert after["heartbeat_at"] >= before["heartbeat_at"]
        assert after["lease_expires_at"] > before["lease_expires_at"]

    def test_release_and_skip_leased_require_matching_token(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/release-and-skip.jpg")

        release_job = queue.enqueue(img_id, "objects")
        assert release_job is not None
        release_claim = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=60, batch_size=1)
        release_token = release_claim[0]["lease_token"]
        assert queue.release_leased(release_job, "wrong-token") is False
        assert queue.release_leased(release_job, release_token) is True
        release_row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [release_job]).fetchone()
        assert release_row is not None
        assert release_row["status"] == "pending"

        skip_image_id = repo.register_image(file_path="/photos/skip-only.jpg")
        skip_job = queue.enqueue(skip_image_id, "faces")
        assert skip_job is not None
        skip_claim = queue.claim_leased(
            worker_id="macbook-pro",
            lease_ttl_seconds=60,
            batch_size=1,
            module="faces",
        )
        skip_token = skip_claim[0]["lease_token"]
        assert queue.mark_skipped_leased(skip_job, "wrong-token", "has_people") is False
        assert queue.mark_skipped_leased(skip_job, skip_token, "has_people") is True
        skip_row = conn.execute(
            "SELECT status, skip_reason FROM job_queue WHERE id = ?",
            [skip_job],
        ).fetchone()
        assert skip_row is not None
        assert skip_row["status"] == "skipped"
        assert skip_row["skip_reason"] == "has_people"

    def test_release_leased_does_not_consume_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/soft-release.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=60, batch_size=1)
        token = claimed[0]["lease_token"]

        assert queue.release_leased(job_id, token) is True

        row = conn.execute(
            "SELECT status, attempts FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["attempts"] == 0

    def test_release_leased_with_delay_requeues_into_the_future(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["worker-1", "Worker 1", "linux", "online"],
        )
        img_id = repo.register_image(file_path="/photos/delayed-release.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        before = conn.execute("SELECT queued_at FROM job_queue WHERE id = ?", [job_id]).fetchone()
        assert before is not None

        claimed = queue.claim_leased(worker_id="worker-1", lease_ttl_seconds=60, batch_size=1)
        token = claimed[0]["lease_token"]

        assert queue.release_leased(job_id, token, delay_seconds=15) is True

        row = conn.execute(
            "SELECT status, attempts, queued_at FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["attempts"] == 0
        assert row["queued_at"] > before["queued_at"]

    def test_retry_failed_recovers_exhausted_pending_jobs(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/exhausted-pending.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        conn.execute(
            """UPDATE job_queue
               SET status = 'failed',
                   attempts = 9,
                   error_message = 'Exceeded max attempts (exhausted pending)',
                   completed_at = datetime('now')
               WHERE id = ?""",
            [job_id],
        )
        conn.commit()

        retried = queue.retry_failed()
        assert retried == 1

        row = conn.execute(
            "SELECT status, attempts, error_message, completed_at FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["attempts"] == 0
        assert row["error_message"] is None
        assert row["completed_at"] is None

    def test_batch_skip_release_leased_does_not_consume_attempts(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["worker-1", "Worker 1", "linux", "online"],
        )
        img_id = repo.register_image(file_path="/photos/batch-soft-release.jpg")
        job_id = queue.enqueue(img_id, "caption")
        assert job_id is not None

        claimed = queue.claim_leased(worker_id="worker-1", lease_ttl_seconds=60, batch_size=1)
        token = claimed[0]["lease_token"]

        queue.batch_skip_release_leased([], [(job_id, token)])

        row = conn.execute(
            "SELECT status, attempts FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["attempts"] == 0

    def test_release_worker_leases_requeues_claimed_jobs(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["macbook-pro", "MacBook Pro", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/release.jpg")
        job_id = queue.enqueue(img_id, "objects")
        assert job_id is not None
        claimed = queue.claim_leased(worker_id="macbook-pro", lease_ttl_seconds=120, batch_size=1)
        assert len(claimed) == 1

        released = queue.release_worker_leases("macbook-pro")
        assert released == 1

        row = conn.execute(
            "SELECT status, attempts FROM job_queue WHERE id = ?",
            [job_id],
        ).fetchone()
        assert row is not None
        assert row["status"] == "pending"
        assert row["attempts"] == 0

    def test_reconcile_runtime_state_recovers_orphans_and_dangling_leases(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["worker-1", "Worker 1", "linux", "online"],
        )

        worker_image = repo.register_image(file_path="/photos/worker-orphan.jpg")
        worker_job = queue.enqueue(worker_image, "objects")
        assert worker_job is not None
        worker_claim = queue.claim_leased(worker_id="worker-1", batch_size=1, module="objects")
        assert len(worker_claim) == 1
        conn.execute("DELETE FROM job_leases WHERE job_id = ?", [worker_job])

        master_image = repo.register_image(file_path="/photos/master-orphan.jpg")
        master_job = queue.enqueue(master_image, "metadata")
        assert master_job is not None
        claimed_master = queue.claim(batch_size=1, module="metadata")
        assert len(claimed_master) == 1

        dangling_image = repo.register_image(file_path="/photos/dangling-lease.jpg")
        dangling_job = queue.enqueue(dangling_image, "faces")
        assert dangling_job is not None
        dangling_claim = queue.claim_leased(worker_id="worker-1", batch_size=1, module="faces")
        assert len(dangling_claim) == 1
        conn.execute(
            "UPDATE job_queue SET status = 'done', completed_at = datetime('now') WHERE id = ?",
            [dangling_job],
        )
        conn.commit()

        summary = queue.reconcile_runtime_state(recover_master_jobs=True)
        assert summary == {
            "dangling_leases": 1,
            "worker_orphans": 1,
            "master_orphans": 1,
        }

        worker_row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [worker_job]).fetchone()
        master_row = conn.execute("SELECT status FROM job_queue WHERE id = ?", [master_job]).fetchone()
        dangling_lease = conn.execute("SELECT 1 FROM job_leases WHERE job_id = ?", [dangling_job]).fetchone()
        assert worker_row is not None and worker_row["status"] == "pending"
        assert master_row is not None and master_row["status"] == "pending"
        assert dangling_lease is None

    def test_get_pending_image_ids(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        id1 = repo.register_image(file_path="/photos/a.jpg")
        id2 = repo.register_image(file_path="/photos/b.jpg")
        id3 = repo.register_image(file_path="/photos/c.jpg")
        queue.enqueue(id1, "metadata")
        queue.enqueue(id2, "objects")
        queue.enqueue(id3, "metadata")
        queue.enqueue(id3, "objects")

        ids = queue.get_pending_image_ids()
        assert sorted(ids) == sorted([id1, id2, id3])

        ids_meta = queue.get_pending_image_ids(modules=["metadata"])
        assert sorted(ids_meta) == sorted([id1, id3])

    def test_claim_with_image_ids_filter(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        id1 = repo.register_image(file_path="/photos/a.jpg")
        id2 = repo.register_image(file_path="/photos/b.jpg")
        queue.enqueue(id1, "metadata")
        queue.enqueue(id2, "metadata")

        # Only claim jobs for id1
        jobs = queue.claim(batch_size=10, image_ids={id1})
        assert len(jobs) == 1
        assert jobs[0]["image_id"] == id1

        # id2 is still pending
        jobs2 = queue.claim(batch_size=10)
        assert len(jobs2) == 1
        assert jobs2[0]["image_id"] == id2

    def test_claim_with_large_image_ids_filter(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        image_ids = [
            repo.register_image(file_path=f"/photos/large-claim-{idx}.jpg")
            for idx in range(1100)
        ]
        image_id_set = set(image_ids)
        for image_id in image_ids:
            queue.enqueue(image_id, "metadata")

        jobs = queue.claim(batch_size=5, image_ids=image_id_set)

        assert len(jobs) == 5
        assert {int(job["image_id"]) for job in jobs}.issubset(image_id_set)

    def test_pending_count_with_image_ids_filter(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        id1 = repo.register_image(file_path="/photos/a.jpg")
        id2 = repo.register_image(file_path="/photos/b.jpg")
        queue.enqueue(id1, "metadata")
        queue.enqueue(id1, "objects")
        queue.enqueue(id2, "metadata")

        assert queue.pending_count(image_ids={id1}) == 2
        assert queue.pending_count(module="metadata", image_ids={id1}) == 1
        assert queue.pending_count(image_ids={id2}) == 1
        assert queue.pending_count() == 3  # backward compat

    def test_remap_pending_modules_converts_legacy_blip2_jobs(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        id1 = repo.register_image(file_path="/photos/legacy-1.jpg")
        id2 = repo.register_image(file_path="/photos/legacy-2.jpg")
        id3 = repo.register_image(file_path="/photos/legacy-3.jpg")
        id4 = repo.register_image(file_path="/photos/legacy-4.jpg")

        queue.enqueue(id1, "blip2")
        queue.enqueue(id2, "blip2")
        queue.enqueue(id3, "blip2")
        queue.enqueue(id2, "caption")  # target row already exists -> source should be deleted
        done_legacy = queue.enqueue(id4, "blip2")
        assert done_legacy is not None
        queue.mark_done(done_legacy)

        queue.claim(batch_size=1, module="blip2")  # ensure running rows also remap

        remapped = queue.remap_pending_modules({"blip2": "caption"})
        assert remapped == {"updated": 2, "deleted": 1}

        active_legacy = conn.execute(
            """SELECT COUNT(*) AS cnt FROM job_queue
               WHERE module = 'blip2' AND status IN ('pending', 'running')"""
        ).fetchone()
        assert active_legacy is not None
        assert active_legacy["cnt"] == 0

        done_legacy_rows = conn.execute(
            "SELECT COUNT(*) AS cnt FROM job_queue WHERE module = 'blip2' AND status = 'done'"
        ).fetchone()
        assert done_legacy_rows is not None
        assert done_legacy_rows["cnt"] == 1

        target_rows = conn.execute(
            """SELECT COUNT(*) AS cnt FROM job_queue
               WHERE module = 'caption' AND status IN ('pending', 'running')"""
        ).fetchone()
        assert target_rows is not None
        assert target_rows["cnt"] == 3

    def test_claim_leased_prefer_module(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["worker-1", "Worker 1", "linux", "online"],
        )
        img_id = repo.register_image(file_path="/photos/affinity.jpg")
        queue.enqueue(img_id, "metadata", priority=50)
        queue.enqueue(img_id, "objects", priority=100)  # higher priority

        # Without prefer_module, objects (priority 100) comes first
        claimed = queue.claim_leased(worker_id="worker-1", batch_size=2)
        assert claimed[0]["module"] == "objects"
        # Release them
        for c in claimed:
            queue.release_leased(c["id"], c["lease_token"])

        # With prefer_module="metadata", metadata comes first despite lower priority
        claimed = queue.claim_leased(worker_id="worker-1", batch_size=2, prefer_module="metadata")
        assert claimed[0]["module"] == "metadata"


class TestWorkerQueueLiveness:
    def test_pending_wait_keeps_master_alive_while_jobs_running(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1)

        class FakeQueue:
            def __init__(self) -> None:
                self.pending_calls = 0
                self.running_calls = 0
                self.release_calls = 0

            def get_pending_image_ids(self) -> list[int]:
                self.pending_calls += 1
                if self.pending_calls >= 3:
                    return [123]
                return []

            def running_count(self) -> int:
                self.running_calls += 1
                return 1

            def release_expired_leases(self) -> int:
                self.release_calls += 1
                return 0

        fake_queue = FakeQueue()
        worker.queue = fake_queue  # type: ignore[assignment]

        image_ids = worker._pending_image_ids_with_running_wait(poll_interval_s=0.0)

        assert image_ids == [123]
        assert fake_queue.running_calls >= 2
        assert fake_queue.release_calls == 0

    def test_pending_wait_exits_when_running_queue_is_empty(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1)

        class FakeQueue:
            def get_pending_image_ids(self) -> list[int]:
                return []

            def running_count(self) -> int:
                return 0

            def release_expired_leases(self) -> int:
                raise AssertionError("Should not reclaim leases when nothing is running")

        worker.queue = FakeQueue()  # type: ignore[assignment]
        image_ids = worker._pending_image_ids_with_running_wait(poll_interval_s=0.0)
        assert image_ids == []

    def test_pending_wait_tolerates_busy_db_during_lease_reclaim(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1, verbose=True)

        class FakeQueue:
            def __init__(self) -> None:
                self.pending_calls = 0
                self.reclaim_calls = 0

            def get_pending_image_ids(self) -> list[int]:
                self.pending_calls += 1
                if self.pending_calls >= 3:
                    return [456]
                return []

            def running_count(self) -> int:
                return 1

            def release_expired_leases(self) -> int:
                self.reclaim_calls += 1
                if self.reclaim_calls == 1:
                    raise sqlite3.OperationalError("database is locked")
                return 0

        fake_queue = FakeQueue()
        worker.queue = fake_queue  # type: ignore[assignment]

        image_ids = worker._pending_image_ids_with_running_wait(
            poll_interval_s=0.0,
            reclaim_interval_s=0.0,
        )

        assert image_ids == [456]
        assert fake_queue.reclaim_calls >= 1

    def test_perception_vram_timeout_is_deferred_not_fatal(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1, verbose=False)

        timeout = RuntimeError("Timed out waiting for VRAM for perception: need 13.80 GB")
        assert worker._is_perception_vram_timeout(["perception"], timeout)
        assert not worker._is_perception_vram_timeout(["caption"], timeout)

        worker._defer_perception_phase(cooldown_s=5.0)
        assert worker._perception_on_cooldown(["perception"])
        assert not worker._perception_on_cooldown(["caption"])


class TestWorkerFlushRecovery:
    def test_flush_fts_requeues_failed_ids(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1)
        worker._fts_dirty = {101, 102}

        def _update_search_artifacts(image_id: int) -> None:
            if image_id == 102:
                raise RuntimeError("fts rebuild failed")

        worker.repo.update_search_artifacts = _update_search_artifacts  # type: ignore[method-assign]

        rebuilt = worker._flush_fts_dirty()
        assert rebuilt == 1
        assert worker._fts_dirty == {102}

    def test_write_pending_xmps_requeues_failed_ids(self, tmp_path):
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        worker = Worker(conn, workers=1, write_xmp=True)
        worker._xmp_candidates = {201, 202}

        def _write_xmp(_repo, image_id: int):
            if image_id == 202:
                raise RuntimeError("xmp failed")
            return f"/tmp/{image_id}.xmp"

        with patch("imganalyzer.pipeline.worker.write_xmp_from_db", side_effect=_write_xmp):
            written = worker._write_pending_xmps()

        assert written == 1
        assert worker._xmp_candidates == {202}

    def test_process_job_skips_pillow_decode_errors_and_pending_siblings(self, tmp_path):
        from imganalyzer.db.queue import JobQueue
        from imganalyzer.db.repository import Repository
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        image_path = tmp_path / "broken.tiff"
        image_id = repo.register_image(file_path=str(image_path))
        first_job = queue.enqueue(image_id, "metadata")
        second_job = queue.enqueue(image_id, "technical")
        assert first_job is not None
        assert second_job is not None

        worker = Worker(conn, workers=1)

        class FakeRunner:
            def should_run(self, _image_id: int, _module: str) -> bool:
                return True

            def run(self, _image_id: int, _module: str) -> dict[str, object]:
                raise ValueError(
                    "Pillow cannot decode broken.tiff: Invalid value for samples per pixel"
                )

        worker._get_thread_db = lambda: (conn, repo, queue, FakeRunner())  # type: ignore[method-assign]

        status = worker._process_job({"id": first_job, "image_id": image_id, "module": "metadata"})

        assert status == "skipped"

        rows = conn.execute(
            "SELECT id, status, skip_reason FROM job_queue WHERE image_id = ? ORDER BY id",
            [image_id],
        ).fetchall()
        assert [(row["id"], row["status"], row["skip_reason"]) for row in rows] == [
            (first_job, "skipped", "corrupt_file"),
            (second_job, "skipped", "corrupt_file"),
        ]

        corrupt = conn.execute(
            "SELECT file_path, error_msg FROM corrupt_files WHERE image_id = ?",
            [image_id],
        ).fetchone()
        assert corrupt is not None
        assert corrupt["file_path"] == str(image_path)
        assert "Pillow cannot decode broken.tiff" in corrupt["error_msg"]

    def test_process_job_marks_missing_dependency_skipped(self, tmp_path):
        from imganalyzer.db.queue import JobQueue
        from imganalyzer.db.repository import Repository
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        image_path = tmp_path / "missing-torch.jpg"
        image_id = repo.register_image(file_path=str(image_path))
        objects_job = queue.enqueue(image_id, "objects")
        faces_job = queue.enqueue(image_id, "faces")
        assert objects_job is not None
        assert faces_job is not None

        worker = Worker(conn, workers=1)

        class FakeRunner:
            def should_run(self, _image_id: int, _module: str) -> bool:
                return True

            def run(self, _image_id: int, _module: str) -> dict[str, object]:
                raise ImportError("PyTorch library was not found")

        worker._get_thread_db = lambda: (conn, repo, queue, FakeRunner())  # type: ignore[method-assign]

        status = worker._process_job({"id": objects_job, "image_id": image_id, "module": "objects"})
        assert status == "skipped"

        rows = conn.execute(
            "SELECT id, status, skip_reason FROM job_queue WHERE image_id = ? ORDER BY id",
            [image_id],
        ).fetchall()
        assert [(row["id"], row["status"], row["skip_reason"]) for row in rows] == [
            (objects_job, "skipped", "missing_dependency"),
            (faces_job, "skipped", "prerequisite_objects_missing_dependency"),
        ]

    def test_process_job_skips_when_prerequisite_failed(self, tmp_path):
        from imganalyzer.db.queue import JobQueue
        from imganalyzer.db.repository import Repository
        from imganalyzer.pipeline.worker import Worker

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        image_id = repo.register_image(file_path=str(tmp_path / "prereq.jpg"))
        objects_job = queue.enqueue(image_id, "objects")
        faces_job = queue.enqueue(image_id, "faces")
        assert objects_job is not None
        assert faces_job is not None
        queue.mark_failed(objects_job, "ImportError: PyTorch not installed")

        worker = Worker(conn, workers=1)

        class FakeRunner:
            def should_run(self, _image_id: int, _module: str) -> bool:
                return True

            def run(self, _image_id: int, _module: str) -> dict[str, object]:
                raise AssertionError("faces should not run when objects already failed")

        worker._get_thread_db = lambda: (conn, repo, queue, FakeRunner())  # type: ignore[method-assign]

        status = worker._process_job({"id": faces_job, "image_id": image_id, "module": "faces"})
        assert status == "skipped"

        row = conn.execute(
            "SELECT status, skip_reason FROM job_queue WHERE id = ?",
            [faces_job],
        ).fetchone()
        assert row is not None
        assert row["status"] == "skipped"
        assert row["skip_reason"] == "prerequisite_objects_failed"


class TestSearchIndex:
    """Tests for the FTS5 search index."""

    def test_search_index_populated_after_upsert(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/sunset.jpg")
        repo.upsert_caption(img_id, {
            "description": "A beautiful sunset over the ocean",
            "scene_type": "landscape",
            "main_subject": "sunset",
            "keywords": ["sunset", "ocean", "golden hour"],
            "mood": "serene",
            "lighting": "golden hour",
            "face_count": 0,
            "has_people": False,
        })
        conn.commit()
        repo.update_search_index(img_id)
        conn.commit()

        # Search using FTS5 directly
        row = conn.execute(
            "SELECT * FROM search_index WHERE search_index MATCH ?",
            ["sunset"],
        ).fetchone()
        assert row is not None

    def test_search_index_includes_metadata(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/canon.jpg")
        repo.upsert_metadata(img_id, {
            "camera_make": "Canon",
            "camera_model": "EOS R5",
        })
        conn.commit()
        repo.update_search_index(img_id)
        conn.commit()

        row = conn.execute(
            "SELECT * FROM search_index WHERE search_index MATCH ?",
            ["Canon"],
        ).fetchone()
        assert row is not None

    def test_search_index_update_replaces_old_tokens(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/token-update.jpg")

        repo.upsert_caption(img_id, {
            "description": "pink cloud sky",
            "scene_type": "sky",
            "main_subject": "cloud",
            "keywords": ["pink", "cloud"],
            "face_count": 0,
            "has_people": False,
        })
        repo.update_search_artifacts(img_id)
        conn.commit()

        row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["pink", img_id],
        ).fetchone()
        assert row is not None

        repo.upsert_caption(img_id, {
            "description": "child with red boots on street",
            "scene_type": "street",
            "main_subject": "child",
            "keywords": ["child", "boots"],
            "face_count": 0,
            "has_people": False,
        })
        repo.update_search_artifacts(img_id)
        conn.commit()

        old_row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["pink", img_id],
        ).fetchone()
        assert old_row is None

        new_row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["child", img_id],
        ).fetchone()
        assert new_row is not None

    def test_search_index_update_replaces_tokens_on_legacy_contentless_table(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        # Simulate legacy DBs created before contentless_delete support.
        conn.execute("DROP TABLE IF EXISTS search_index")
        conn.execute(
            """
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                tokenize='porter unicode61'
            )
            """
        )

        img_id = repo.register_image(file_path="/photos/legacy-contentless.jpg")
        repo.upsert_caption(img_id, {
            "description": "pink cloud sky",
            "scene_type": "sky",
            "main_subject": "cloud",
            "keywords": ["pink", "cloud"],
            "face_count": 0,
            "has_people": False,
        })
        repo.update_search_artifacts(img_id)
        conn.commit()

        row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["pink", img_id],
        ).fetchone()
        assert row is not None

        # Existing-row refresh should remove old tokens even on legacy contentless FTS.
        repo.upsert_caption(img_id, {
            "description": "child with red boots on street",
            "scene_type": "street",
            "main_subject": "child",
            "keywords": ["child", "boots"],
            "face_count": 0,
            "has_people": False,
        })
        repo.update_search_artifacts(img_id)
        conn.commit()

        old_row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["pink", img_id],
        ).fetchone()
        assert old_row is None

        new_row = conn.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ? AND rowid = ?",
            ["child", img_id],
        ).fetchone()
        assert new_row is not None


class TestFaceIdentityDB:
    """Tests for the DB-backed face identity system."""

    def test_register_and_list_faces(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        face_id = repo.register_face_identity("alice", display_name="Alice Smith")
        assert face_id > 0

        faces = repo.list_face_identities()
        assert len(faces) == 1
        assert faces[0]["canonical_name"] == "alice"
        assert faces[0]["display_name"] == "Alice Smith"

    def test_add_face_embedding(self, tmp_path, face_embedding):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        face_id = repo.register_face_identity("bob")
        emb_bytes = face_embedding.tobytes()
        repo.add_face_embedding(face_id, emb_bytes, source_image="/photos/bob1.jpg")

        embeddings = repo.get_face_embeddings(face_id)
        assert len(embeddings) == 1

    def test_remove_face_cascades(self, tmp_path, face_embedding):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        face_id = repo.register_face_identity("carol")
        repo.add_face_embedding(face_id, face_embedding.tobytes())
        repo.remove_face_identity("carol")

        faces = repo.list_face_identities()
        assert len(faces) == 0
        embeddings = repo.get_face_embeddings(face_id)
        assert len(embeddings) == 0


class TestFacePersons:
    """Tests for the cross-age person identity linking feature."""

    def _seed_clusters(self, repo, conn):
        """Insert test images + face occurrences in 3 clusters."""
        for i in range(1, 4):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/{i}.jpg", f"hash{i}", 100],
            )
        emb = np.zeros(512, dtype=np.float32).tobytes()
        rows = [
            (1, 1, 0, emb, 1, 0.0, 0.0, 1.0, 1.0),
            (2, 1, 1, emb, 1, 0.0, 0.0, 1.0, 1.0),
            (3, 2, 0, emb, 2, 0.0, 0.0, 1.0, 1.0),
            (4, 2, 1, emb, 2, 0.0, 0.0, 1.0, 1.0),
            (5, 3, 0, emb, 3, 0.0, 0.0, 1.0, 1.0),
        ]
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    def test_create_and_list_persons(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        pid = repo.create_person("Dad")
        assert pid > 0

        persons = repo.list_persons()
        assert len(persons) == 1
        assert persons[0]["name"] == "Dad"
        assert persons[0]["face_count"] == 0

    def test_rename_person(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        pid = repo.create_person("Dad")
        repo.rename_person(pid, "Father")
        persons = repo.list_persons()
        assert persons[0]["name"] == "Father"

    def test_link_cluster_to_person(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        updated = repo.link_cluster_to_person(1, pid)
        assert updated == 2  # cluster 1 has 2 occurrences

        persons = repo.list_persons()
        assert persons[0]["face_count"] == 2
        assert persons[0]["cluster_count"] == 1

    def test_link_multiple_clusters(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.link_cluster_to_person(1, pid)
        repo.link_cluster_to_person(2, pid)

        persons = repo.list_persons()
        assert persons[0]["face_count"] == 4
        assert persons[0]["cluster_count"] == 2

        pc = repo.get_person_clusters(pid)
        assert len(pc) == 2

    def test_unlink_cluster(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.link_cluster_to_person(1, pid)
        repo.unlink_cluster_from_person(1)

        persons = repo.list_persons()
        assert persons[0]["face_count"] == 0

    def test_delete_person_clears_links(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.link_cluster_to_person(1, pid)
        repo.delete_person(pid)

        persons = repo.list_persons()
        assert len(persons) == 0
        # person_id should be cleared
        row = conn.execute(
            "SELECT person_id FROM face_occurrences WHERE id = 1"
        ).fetchone()
        assert row["person_id"] is None

    def test_auto_assign_after_recluster(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.link_cluster_to_person(1, pid)

        # Simulate re-clustering: shift cluster_ids (1→10, 2→20, 3→30)
        # but keep person_id on the occurrence
        conn.execute("UPDATE face_occurrences SET cluster_id = cluster_id * 10")
        conn.commit()

        # Now cluster 10 has 2 occurrences with person_id = pid
        updated = repo.auto_assign_persons_after_recluster()
        assert updated == 0  # already assigned, no new ones to update

        # Add an untagged occurrence to cluster 10
        conn.execute(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
            "VALUES (100, 1, 5, ?, 10, NULL, 0.0, 0.0, 1.0, 1.0)",
            [np.zeros(512, dtype=np.float32).tobytes()],
        )
        conn.commit()

        updated = repo.auto_assign_persons_after_recluster()
        assert updated == 1  # the new occurrence should be assigned to Dad
        row = conn.execute(
            "SELECT person_id FROM face_occurrences WHERE id = 100"
        ).fetchone()
        assert row["person_id"] == pid

    def test_list_clusters_includes_person_id(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.link_cluster_to_person(1, pid)

        clusters, total_count = repo.list_face_clusters()
        cluster_map = {c["cluster_id"]: c for c in clusters}
        assert cluster_map[1]["person_id"] == pid
        assert cluster_map[2]["person_id"] is None
        assert cluster_map[3]["person_id"] is None

    def test_list_clusters_includes_unclustered_rows_when_mixed(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [10, "/img/10.jpg", "hash10", 100],
        )
        emb = np.zeros(512, dtype=np.float32).tobytes()
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (100, 10, 0, emb, None, "new_person", 0.0, 0.0, 1.0, 1.0),
                (101, 10, 1, emb, None, "new_person", 0.0, 0.0, 2.0, 2.0),
            ],
        )
        conn.commit()

        clusters, total_count = repo.list_face_clusters()
        assert total_count == 4

        unclustered = next(
            c for c in clusters
            if c["cluster_id"] is None and c["identity_name"] == "new_person"
        )
        assert unclustered["face_count"] == 2
        assert unclustered["image_count"] == 1

    def test_get_cluster_occurrences_identity_returns_only_unclustered(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        conn.execute(
            "UPDATE face_occurrences SET identity_name = ? WHERE id = ?",
            ["alice", 1],
        )
        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [11, "/img/11.jpg", "hash11", 100],
        )
        emb = np.zeros(512, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [110, 11, 0, emb, None, "alice", 0.0, 0.0, 1.5, 1.5],
        )
        conn.commit()

        occurrences = repo.get_cluster_occurrences(identity_name="alice", limit=10)
        assert [row["id"] for row in occurrences] == [110]

    def test_suggest_cluster_link_targets_scores_persons_and_aliases(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        for i in range(1, 7):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/suggest_{i}.jpg", f"hash_suggest_{i}", 100],
            )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        likely_person_id = repo.create_person("Likely Person")
        wrong_person_id = repo.create_person("Wrong Person")

        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (2, 1, 1, emb([0.98, 0.02, 0.0]), 1, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (3, 2, 0, emb([0.97, 0.03, 0.0]), 2, likely_person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (4, 3, 0, emb([-1.0, 0.0, 0.0]), 3, wrong_person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (5, 4, 0, emb([0.96, 0.04, 0.0]), 4, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (6, 5, 0, emb([-0.95, 0.05, 0.0]), 5, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
            ],
        )
        conn.executemany(
            "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)",
            [
                (4, "Likely Alias"),
                (5, "Wrong Alias"),
            ],
        )
        conn.commit()

        suggestions = repo.suggest_cluster_link_targets(1, limit=10)
        assert suggestions

        likely_person = next(
            item
            for item in suggestions
            if item["target_type"] == "person" and item["person_id"] == likely_person_id
        )
        wrong_person = next(
            item
            for item in suggestions
            if item["target_type"] == "person" and item["person_id"] == wrong_person_id
        )
        likely_alias = next(
            item
            for item in suggestions
            if item["target_type"] == "alias" and item["cluster_id"] == 4
        )
        wrong_alias = next(
            item
            for item in suggestions
            if item["target_type"] == "alias" and item["cluster_id"] == 5
        )

        assert likely_person["score"] > wrong_person["score"]
        assert likely_alias["score"] > wrong_alias["score"]
        assert suggestions[0]["label"] in {"Likely Person", "Likely Alias"}
        assert likely_person["representative_id"] is not None
        assert likely_alias["representative_id"] is not None
        assert likely_person["reason"]

    def test_suggest_cluster_link_targets_returns_empty_without_source_embedding(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/no_embedding.jpg", "hash_no_embedding", 100],
        )
        conn.execute(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [1, 1, 0, None, 1, None, "Unknown", 0.0, 0.0, 1.0, 1.0],
        )
        conn.commit()

        assert repo.suggest_cluster_link_targets(1, limit=5) == []

    def test_suggest_person_link_clusters_ranks_unlinked_clusters(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        for i in range(1, 6):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/person_suggest_{i}.jpg", f"hash_person_suggest_{i}", 100],
            )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        target_person_id = repo.create_person("Target Person")
        other_person_id = repo.create_person("Other Person")

        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, target_person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (2, 1, 1, emb([0.98, 0.02, 0.0]), 1, target_person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (3, 2, 0, emb([0.97, 0.03, 0.0]), 2, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (4, 3, 0, emb([-1.0, 0.0, 0.0]), 3, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (5, 4, 0, emb([0.95, 0.05, 0.0]), 4, other_person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
            ],
        )
        conn.execute(
            "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)",
            [2, "Likely Candidate"],
        )
        conn.commit()

        suggestions = repo.suggest_person_link_clusters(target_person_id, limit=10)
        assert suggestions

        likely = next(item for item in suggestions if item["cluster_id"] == 2)
        unlikely = next(item for item in suggestions if item["cluster_id"] == 3)

        assert likely["score"] > unlikely["score"]
        assert likely["label"] == "Likely Candidate"
        assert likely["representative_id"] is not None
        assert all(item["cluster_id"] != 4 for item in suggestions)

    def test_suggest_person_link_clusters_empty_without_person_embeddings(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        person_id = repo.create_person("No Embedding Person")
        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/no_person_embedding.jpg", "hash_no_person_embedding", 100],
        )
        conn.execute(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [1, 1, 0, None, 1, person_id, "Unknown", 0.0, 0.0, 1.0, 1.0],
        )
        conn.commit()

        assert repo.suggest_person_link_clusters(person_id, limit=5) == []

    def test_suggest_person_link_clusters_respects_limit(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        for i in range(1, 6):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/person_limit_{i}.jpg", f"hash_person_limit_{i}", 100],
            )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        person_id = repo.create_person("Limit Person")
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, person_id, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (2, 2, 0, emb([0.99, 0.01, 0.0]), 2, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (3, 3, 0, emb([0.95, 0.05, 0.0]), 3, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
                (4, 4, 0, emb([-1.0, 0.0, 0.0]), 4, None, "Unknown", 0.0, 0.0, 2.0, 2.0),
            ],
        )
        conn.commit()

        suggestions = repo.suggest_person_link_clusters(person_id, limit=1)
        assert len(suggestions) == 1
        assert suggestions[0]["cluster_id"] == 2

    def test_suggest_person_link_clusters_multi_cluster_diversity(self, tmp_path):
        """A person with clusters at two orthogonal extremes should find
        candidates near EITHER extreme, not just the centroid average."""
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        for i in range(1, 8):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/diversity_{i}.jpg", f"hash_diversity_{i}", 100],
            )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        person_id = repo.create_person("Diverse Person")

        # Cluster 1 (linked): points along axis 0  — "young" appearance
        # Cluster 2 (linked): points along axis 1  — "old" appearance
        # Cluster 3 (unlinked): close to axis 0 → should match "young"
        # Cluster 4 (unlinked): close to axis 1 → should match "old"
        # Cluster 5 (unlinked): opposite direction → should NOT match well
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, "
            "cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Person cluster 1 — "young" (axis 0)
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
                (2, 1, 1, emb([0.98, 0.02, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
                # Person cluster 2 — "old" (axis 1)
                (3, 2, 0, emb([0.0, 1.0, 0.0]), 2, person_id, "Unknown", 0, 0, 2, 2),
                (4, 2, 1, emb([0.02, 0.98, 0.0]), 2, person_id, "Unknown", 0, 0, 2, 2),
                # Unlinked cluster 3 — close to "young"
                (5, 3, 0, emb([0.95, 0.05, 0.0]), 3, None, "Unknown", 0, 0, 2, 2),
                # Unlinked cluster 4 — close to "old"
                (6, 4, 0, emb([0.05, 0.95, 0.0]), 4, None, "Unknown", 0, 0, 2, 2),
                # Unlinked cluster 5 — opposite direction
                (7, 5, 0, emb([-1.0, 0.0, 0.0]), 5, None, "Unknown", 0, 0, 2, 2),
            ],
        )
        conn.commit()

        suggestions = repo.suggest_person_link_clusters(person_id, limit=10)

        # Both cluster 3 ("young"-like) and cluster 4 ("old"-like) should
        # score highly — the algorithm must not average the person into a
        # midpoint that matches neither.
        scores = {s["cluster_id"]: s["score"] for s in suggestions}
        assert 3 in scores, "Candidate near 'young' extreme should be found"
        assert 4 in scores, "Candidate near 'old' extreme should be found"
        assert scores[3] > 0.9, f"'Young' candidate score {scores[3]:.3f} should be > 0.9"
        assert scores[4] > 0.9, f"'Old' candidate score {scores[4]:.3f} should be > 0.9"
        # Opposite-direction cluster should score poorly
        if 5 in scores:
            assert scores[5] < 0.0, f"Opposite cluster score {scores[5]:.3f} should be negative"

    def test_split_cluster_separates_mixed_embeddings(self, tmp_path):
        """Cluster with two distinct embedding groups should split into 2 sub-clusters."""
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/split_1.jpg", "hash_split_1", 100],
        )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        # All in cluster 1 — but group A (ids 1-3) and group B (ids 4-5) are orthogonal
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "identity_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (2, 1, 1, emb([0.98, 0.02, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (3, 1, 2, emb([0.97, 0.03, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (4, 1, 3, emb([0.0, 1.0, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (5, 1, 4, emb([0.02, 0.98, 0.0]), 1, "Unknown", 0, 0, 2, 2),
            ],
        )
        conn.commit()

        result = repo.split_cluster(1)
        assert result["split_count"] == 2
        assert len(result["new_cluster_ids"]) == 1

        # Largest sub-cluster (3 faces) should keep cluster_id 1
        kept = conn.execute(
            "SELECT COUNT(*) FROM face_occurrences WHERE cluster_id = 1"
        ).fetchone()[0]
        assert kept == 3

        new_cid = result["new_cluster_ids"][0]
        moved = conn.execute(
            "SELECT COUNT(*) FROM face_occurrences WHERE cluster_id = ?", [new_cid]
        ).fetchone()[0]
        assert moved == 2

    def test_split_cluster_preserves_person_on_largest(self, tmp_path):
        """When splitting a linked cluster, person_id stays on largest sub-cluster."""
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/split_person.jpg", "hash_sp", 100],
        )

        person_id = repo.create_person("Split Test Person")

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "person_id, identity_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
                (2, 1, 1, emb([0.98, 0.02, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
                (3, 1, 2, emb([0.97, 0.03, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
                (4, 1, 3, emb([0.0, 1.0, 0.0]), 1, person_id, "Unknown", 0, 0, 2, 2),
            ],
        )
        conn.commit()

        result = repo.split_cluster(1)
        assert result["split_count"] == 2

        # Largest (3 faces) keeps person_id
        kept_person = conn.execute(
            "SELECT person_id FROM face_occurrences WHERE cluster_id = 1 LIMIT 1"
        ).fetchone()[0]
        assert kept_person == person_id

        # Smaller sub-cluster has person_id cleared
        new_cid = result["new_cluster_ids"][0]
        split_person = conn.execute(
            "SELECT person_id FROM face_occurrences WHERE cluster_id = ?", [new_cid]
        ).fetchone()[0]
        assert split_person is None

    def test_split_cluster_no_op_on_pure(self, tmp_path):
        """A pure cluster (all similar embeddings) should not be split."""
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/pure.jpg", "hash_pure", 100],
        )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "identity_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (2, 1, 1, emb([0.99, 0.01, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (3, 1, 2, emb([0.98, 0.02, 0.0]), 1, "Unknown", 0, 0, 2, 2),
            ],
        )
        conn.commit()

        result = repo.split_cluster(1)
        assert result["split_count"] == 1
        assert result["new_cluster_ids"] == []

    def test_compute_cluster_purity(self, tmp_path):
        """Purity score should be low for mixed clusters, high for pure clusters."""
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        conn.execute(
            "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
            [1, "/img/purity.jpg", "hash_purity", 100],
        )

        def emb(values: list[float]) -> bytes:
            return np.array(values, dtype=np.float32).tobytes()

        # Pure cluster (cluster 1)
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "identity_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (2, 1, 1, emb([0.99, 0.01, 0.0]), 1, "Unknown", 0, 0, 2, 2),
                (3, 1, 2, emb([0.98, 0.02, 0.0]), 1, "Unknown", 0, 0, 2, 2),
            ],
        )
        # Mixed cluster (cluster 2) — orthogonal embeddings
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, "
            "identity_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (4, 1, 3, emb([1.0, 0.0, 0.0]), 2, "Unknown", 0, 0, 2, 2),
                (5, 1, 4, emb([0.0, 1.0, 0.0]), 2, "Unknown", 0, 0, 2, 2),
            ],
        )
        conn.commit()

        pure = repo.compute_cluster_purity(1)
        mixed = repo.compute_cluster_purity(2)

        assert pure["purity_score"] > 0.95, f"Pure cluster purity {pure['purity_score']} should be > 0.95"
        assert mixed["purity_score"] < 0.8, f"Mixed cluster purity {mixed['purity_score']} should be < 0.8"
        assert pure["member_count"] == 3
        assert mixed["member_count"] == 2

    def test_relink_cluster_updates_label_and_person_together(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        updated = repo.relink_cluster(1, "Dad Alias", pid, update_person=True)

        assert updated == 2
        label_row = conn.execute(
            "SELECT display_name FROM face_cluster_labels WHERE cluster_id = ?",
            [1],
        ).fetchone()
        assert label_row["display_name"] == "Dad Alias"
        person_ids = conn.execute(
            "SELECT DISTINCT person_id FROM face_occurrences WHERE cluster_id = ?",
            [1],
        ).fetchall()
        assert {row["person_id"] for row in person_ids} == {pid}

    def test_relink_cluster_can_clear_label_and_unlink_person(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        self._seed_clusters(repo, conn)

        pid = repo.create_person("Dad")
        repo.relink_cluster(1, "Dad Alias", pid, update_person=True)

        updated = repo.relink_cluster(1, None, None, update_person=True)

        assert updated == 2
        label_row = conn.execute(
            "SELECT display_name FROM face_cluster_labels WHERE cluster_id = ?",
            [1],
        ).fetchone()
        assert label_row is None
        person_ids = conn.execute(
            "SELECT DISTINCT person_id FROM face_occurrences WHERE cluster_id = ?",
            [1],
        ).fetchall()
        assert {row["person_id"] for row in person_ids} == {None}

    def test_get_person_direct_links_excludes_only_fully_linked_clusters(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        # Images
        for i in range(1, 5):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/direct_{i}.jpg", f"hash_direct_{i}", 100],
            )

        pid = repo.create_person("Direct Link Person")
        other_pid = repo.create_person("Other Person")

        emb = np.zeros(512, dtype=np.float32).tobytes()
        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Cluster 10 is fully linked to pid -> should be excluded from direct links
                (1, 1, 0, emb, 10, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                (2, 1, 1, emb, 10, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                # Cluster 20 is partially linked -> should be included as direct link
                (3, 2, 0, emb, 20, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                (4, 2, 1, emb, 20, None, "Unknown", 0.0, 0.0, 1.0, 1.0),
                # Cluster 30 has only one linked occurrence for pid but others for another person
                (5, 3, 0, emb, 30, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                (6, 3, 1, emb, 30, other_pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                # Unclustered linked occurrence -> should be included
                (7, 4, 0, emb, None, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
            ],
        )
        conn.commit()

        links = repo.get_person_direct_links(pid)
        image_ids = {row["image_id"] for row in links}
        assert 1 not in image_ids
        assert 2 in image_ids
        assert 3 in image_ids
        assert 4 in image_ids

    def test_find_similar_images_for_person_respects_min_similarity(self, tmp_path):
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        for i in range(1, 4):
            conn.execute(
                "INSERT INTO images (id, file_path, file_hash, file_size) VALUES (?, ?, ?, ?)",
                [i, f"/img/sim_{i}.jpg", f"hash_sim_{i}", 100],
            )

        pid = repo.create_person("Similarity Person")

        def emb(values: list[float]) -> bytes:
            vec = np.array(values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tobytes()

        conn.executemany(
            "INSERT INTO face_occurrences (id, image_id, face_idx, embedding, cluster_id, person_id, identity_name, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Person reference embedding (image 1 is excluded from results)
                (1, 1, 0, emb([1.0, 0.0, 0.0]), 10, pid, "Unknown", 0.0, 0.0, 1.0, 1.0),
                # High-similarity candidate (~0.80)
                (2, 2, 0, emb([0.8, 0.6, 0.0]), 20, None, "Unknown", 0.0, 0.0, 1.0, 1.0),
                # Borderline candidate (~0.34)
                (3, 3, 0, emb([0.34, 0.94, 0.0]), 30, None, "Unknown", 0.0, 0.0, 1.0, 1.0),
            ],
        )
        conn.commit()

        relaxed = repo.find_similar_images_for_person(pid, limit=10, min_similarity=0.3)
        strict = repo.find_similar_images_for_person(pid, limit=10, min_similarity=0.5)

        relaxed_ids = {row["image_id"] for row in relaxed}
        strict_ids = {row["image_id"] for row in strict}

        assert relaxed_ids == {2, 3}
        assert strict_ids == {2}


class TestPersistResultToDB:
    """Test the _persist_result_to_db helper used by the analyze command."""

    def test_persist_stores_all_data(self, tmp_path):
        """Verify that _persist_result_to_db correctly stores analysis data."""
        import sqlite3
        from imganalyzer.db.schema import ensure_schema
        from imganalyzer.db.repository import Repository
        from imganalyzer.analyzer import AnalysisResult

        # Create a fresh DB
        db_path = tmp_path / "persist_test.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        ensure_schema(conn)

        repo = Repository(conn)

        result = AnalysisResult(
            source_path=tmp_path / "test.jpg",
            format="JPEG",
            width=1920,
            height=1080,
            metadata={"camera_make": "Canon", "camera_model": "R5", "iso": 200},
            technical={"sharpness_score": 80.0, "exposure_ev": 0.1},
            ai_analysis={
                "description": "A landscape",
                "scene_type": "landscape",
                "keywords": ["nature"],
                "face_count": 0,
                "has_people": False,
            },
        )

        # Simulate what _persist_result_to_db does
        image_id = repo.register_image(
            file_path=str(result.source_path.resolve()),
            width=result.width,
            height=result.height,
            fmt=result.format,
        )

        conn.execute("BEGIN IMMEDIATE")
        repo.upsert_metadata(image_id, dict(result.metadata))
        repo.upsert_technical(image_id, dict(result.technical))
        data = dict(result.ai_analysis)
        data.setdefault("has_people", bool(data.get("face_count", 0) > 0))
        repo.upsert_caption(image_id, data)
        repo.update_search_artifacts(image_id)
        conn.execute("COMMIT")

        # Verify everything was stored
        img = repo.get_image(image_id)
        assert img["width"] == 1920

        meta = repo.get_analysis(image_id, "metadata")
        assert meta["camera_make"] == "Canon"

        tech = repo.get_analysis(image_id, "technical")
        assert tech["sharpness_score"] == 80.0

        local = repo.get_analysis(image_id, "caption")
        assert local["description"] == "A landscape"
        assert local["has_people"] == 0  # False stored as 0

        # Search index should be populated
        row = conn.execute(
            "SELECT * FROM search_index WHERE search_index MATCH ?",
            ["landscape"],
        ).fetchone()
        assert row is not None

        conn.close()


class TestProfiler:
    """Tests for the batch processing profiler."""

    def _make_db(self, tmp_path):
        import sqlite3
        from imganalyzer.db.schema import ensure_schema
        db_path = tmp_path / "profiler_test.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        ensure_schema(conn)
        return conn

    def test_null_profiler_noop(self):
        """NullProfiler should do nothing with zero overhead."""
        from imganalyzer.pipeline.profiler import NullProfiler
        p = NullProfiler()
        assert not p.enabled
        assert p.run_id is None
        with p.span("test_event"):
            pass
        p.record_event("test", 100.0)
        p.flush()

    def test_collector_start_end_run(self, tmp_path):
        """ProfileCollector should create and finalize a run."""
        from imganalyzer.pipeline.profiler import ProfileCollector
        conn = self._make_db(tmp_path)
        p = ProfileCollector(conn)

        run_id = p.start_run(total_images=42)
        assert run_id is not None
        assert p.enabled

        p.end_run()
        assert not p.enabled

        row = conn.execute(
            "SELECT * FROM profiler_runs WHERE id = ?", [run_id]
        ).fetchone()
        assert row is not None
        assert row["total_images"] == 42
        assert row["ended_at"] is not None
        conn.close()

    def test_span_records_event(self, tmp_path):
        """Span context manager should record a timed event."""
        import time
        from imganalyzer.pipeline.profiler import ProfileCollector
        conn = self._make_db(tmp_path)
        p = ProfileCollector(conn)

        run_id = p.start_run(total_images=1)
        with p.span("io_read", image_id=1, module="objects",
                     image_file_size=5000, image_format=".jpg"):
            time.sleep(0.01)  # ~10ms

        p.flush()
        rows = conn.execute(
            "SELECT * FROM profiler_events WHERE run_id = ?", [run_id]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "io_read"
        assert rows[0]["module"] == "objects"
        assert rows[0]["image_id"] == 1
        assert rows[0]["image_file_size"] == 5000
        assert rows[0]["duration_ms"] >= 5  # at least 5ms

        p.end_run()
        conn.close()

    def test_record_event_manual(self, tmp_path):
        """record_event should store a pre-timed event."""
        from imganalyzer.pipeline.profiler import ProfileCollector
        conn = self._make_db(tmp_path)
        p = ProfileCollector(conn)

        run_id = p.start_run()
        p.record_event("cache_hit", 0.0, image_id=5, module="faces")
        p.flush()

        rows = conn.execute(
            "SELECT * FROM profiler_events WHERE run_id = ?", [run_id]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "cache_hit"
        assert rows[0]["duration_ms"] == 0.0

        p.end_run()
        conn.close()

    def test_multiple_events_flushed(self, tmp_path):
        """Multiple events should all be flushed to DB."""
        from imganalyzer.pipeline.profiler import ProfileCollector
        conn = self._make_db(tmp_path)
        p = ProfileCollector(conn)

        run_id = p.start_run()
        for i in range(10):
            p.record_event("gpu_infer", float(i * 10), image_id=i, module="objects")
        p.flush()

        count = conn.execute(
            "SELECT COUNT(*) as c FROM profiler_events WHERE run_id = ?", [run_id]
        ).fetchone()["c"]
        assert count == 10

        p.end_run()
        conn.close()

    def test_disabled_profiler_no_events(self, tmp_path):
        """Events recorded before start_run should be ignored."""
        from imganalyzer.pipeline.profiler import ProfileCollector
        conn = self._make_db(tmp_path)
        p = ProfileCollector(conn)

        # Not started — should be no-op
        with p.span("test"):
            pass
        p.record_event("test", 100.0)
        p.flush()

        count = conn.execute(
            "SELECT COUNT(*) as c FROM profiler_events"
        ).fetchone()["c"]
        assert count == 0
        conn.close()


class TestProbeAvailableModules:
    """Tests for _probe_available_modules capability detection."""

    def test_sets_mps_fallback_env_when_unset(self, monkeypatch):
        from imganalyzer.pipeline.distributed_worker import _probe_available_modules

        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        _probe_available_modules()
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    def test_master_only_modules_excluded(self):
        from imganalyzer.pipeline.distributed_worker import _probe_available_modules

        modules = _probe_available_modules()
        # metadata, technical, and faces are master-only
        assert "metadata" not in modules
        assert "technical" not in modules
        assert "faces" not in modules

    def test_copilot_provider_includes_cloud_modules(self):
        from imganalyzer.pipeline.distributed_worker import _probe_available_modules

        modules = _probe_available_modules()
        # cloud_ai and aesthetic are legacy — should NOT be probed
        assert "cloud_ai" not in modules
        assert "aesthetic" not in modules
        # master-only modules excluded
        assert "metadata" not in modules
        assert "technical" not in modules

    def test_returns_sorted_unique_list(self):
        from imganalyzer.pipeline.distributed_worker import _probe_available_modules

        modules = _probe_available_modules()
        assert modules == sorted(set(modules))

    def test_accepts_legacy_cloud_provider_arg(self):
        from imganalyzer.pipeline.distributed_worker import _probe_available_modules

        modules = _probe_available_modules("copilot")
        # master-only modules excluded
        assert "metadata" not in modules
        assert "technical" not in modules


class TestClaimLeasedModulesFilter:
    """Tests for claim_leased with the new modules list filter."""

    def test_claim_leased_with_modules_list_filter(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        conn.execute(
            """INSERT INTO worker_nodes (id, display_name, platform, status)
               VALUES (?, ?, ?, ?)""",
            ["w1", "Worker 1", "darwin", "online"],
        )
        img_id = repo.register_image(file_path="/photos/test.jpg")
        queue.enqueue(img_id, "objects")
        queue.enqueue(img_id, "metadata")
        queue.enqueue(img_id, "caption")

        # Only claim metadata and caption (not objects)
        claimed = queue.claim_leased(
            worker_id="w1", lease_ttl_seconds=120, batch_size=10,
            modules=["metadata", "caption"],
        )
        claimed_modules = {c["module"] for c in claimed}
        assert "metadata" in claimed_modules
        assert "caption" in claimed_modules
        assert "objects" not in claimed_modules

    def test_pending_count_with_modules_filter(self, tmp_path):
        from imganalyzer.db.repository import Repository
        from imganalyzer.db.queue import JobQueue

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)
        queue = JobQueue(conn)

        img_id = repo.register_image(file_path="/photos/test.jpg")
        queue.enqueue(img_id, "objects")
        queue.enqueue(img_id, "metadata")
        queue.enqueue(img_id, "caption")

        assert queue.pending_count(modules=["metadata", "caption"]) == 2
        assert queue.pending_count(modules=["objects"]) == 1
        assert queue.pending_count() == 3


class TestFaceCropBatchCaching:
    def test_crop_batch_opens_shared_image_once_and_persists_backfill(self, tmp_path, monkeypatch):
        from PIL import Image

        from imganalyzer import server
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        image_path = tmp_path / "source.jpg"
        Image.new("RGB", (120, 120), color=(180, 120, 90)).save(image_path, format="JPEG")
        image_id = repo.register_image(file_path=str(image_path), width=120, height=120, fmt="JPEG")

        occ1 = conn.execute(
            """
            INSERT INTO face_occurrences
                (image_id, face_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2, identity_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [image_id, 0, 10.0, 10.0, 50.0, 50.0, "Unknown"],
        ).lastrowid
        occ2 = conn.execute(
            """
            INSERT INTO face_occurrences
                (image_id, face_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2, identity_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [image_id, 1, 30.0, 25.0, 80.0, 85.0, "Unknown"],
        ).lastrowid

        original_open = Image.open
        open_calls: list[str] = []

        def counting_open(*args, **kwargs):
            open_calls.append(str(args[0]))
            return original_open(*args, **kwargs)

        monkeypatch.setattr("PIL.Image.open", counting_open)
        monkeypatch.setattr(server, "_get_db", lambda: conn)

        result = server._handle_faces_crop_batch({"ids": [occ1, occ2]})

        assert set(result["thumbnails"]) == {str(occ1), str(occ2)}
        assert len(open_calls) == 1

        persisted = conn.execute(
            "SELECT id, thumbnail FROM face_occurrences WHERE id IN (?, ?)",
            [occ1, occ2],
        ).fetchall()
        assert len(persisted) == 2
        assert all(row["thumbnail"] is not None for row in persisted)

    def test_crop_batch_returns_thumbnail_even_when_thumbnail_update_is_temporarily_locked(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image

        from imganalyzer import server
        from imganalyzer.db.repository import Repository

        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        image_path = tmp_path / "source-locked.jpg"
        Image.new("RGB", (120, 120), color=(120, 180, 90)).save(image_path, format="JPEG")
        image_id = repo.register_image(file_path=str(image_path), width=120, height=120, fmt="JPEG")

        occ = conn.execute(
            """
            INSERT INTO face_occurrences
                (image_id, face_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2, identity_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [image_id, 0, 12.0, 14.0, 64.0, 66.0, "Unknown"],
        ).lastrowid
        assert occ is not None

        original_set = Repository.set_face_occurrence_thumbnail
        call_count = 0

        def flaky_set(self, occurrence_id, thumbnail):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            return original_set(self, occurrence_id, thumbnail)

        sleep_calls: list[float] = []
        monkeypatch.setattr(Repository, "set_face_occurrence_thumbnail", flaky_set)
        monkeypatch.setattr(server, "_get_db", lambda: conn)
        monkeypatch.setattr(server.time, "sleep", lambda seconds: sleep_calls.append(seconds))

        result = server._handle_faces_crop_batch({"ids": [occ]})

        assert str(occ) in result["thumbnails"]
        assert call_count >= 2
        assert sleep_calls == [server._LOCK_RETRY_INITIAL_DELAY_S]
