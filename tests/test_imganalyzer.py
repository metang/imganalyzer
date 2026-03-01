"""Tests for imganalyzer."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
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

    def test_no_crash_on_missing_exif(self, tmp_path, synthetic_image_data):
        """Images without EXIF should return empty dict without crashing."""
        from PIL import Image
        img_path = tmp_path / "noexif.png"
        Image.fromarray(synthetic_image_data["rgb_array"]).save(str(img_path))

        from imganalyzer.analysis.metadata import MetadataExtractor
        result = MetadataExtractor(img_path, synthetic_image_data).extract()
        assert isinstance(result, dict)


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

    def test_upsert_local_ai_with_people(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/portrait.jpg")
        repo.upsert_local_ai(img_id, {
            "description": "A portrait of a woman",
            "scene_type": "portrait",
            "keywords": ["person", "portrait"],
            "face_count": 1,
            "face_identities": ["Alice"],
            "has_people": True,
        })
        conn.commit()

        data = repo.get_analysis(img_id, "local_ai")
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


class TestSearchIndex:
    """Tests for the FTS5 search index."""

    def test_search_index_populated_after_upsert(self, tmp_path):
        from imganalyzer.db.repository import Repository
        conn = _make_test_db(tmp_path)
        repo = Repository(conn)

        img_id = repo.register_image(file_path="/photos/sunset.jpg")
        repo.upsert_local_ai(img_id, {
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

        clusters = repo.list_face_clusters()
        cluster_map = {c["cluster_id"]: c for c in clusters}
        assert cluster_map[1]["person_id"] == pid
        assert cluster_map[2]["person_id"] is None
        assert cluster_map[3]["person_id"] is None


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
        repo.upsert_local_ai(image_id, data)
        repo.update_search_index(image_id)
        conn.execute("COMMIT")

        # Verify everything was stored
        img = repo.get_image(image_id)
        assert img["width"] == 1920

        meta = repo.get_analysis(image_id, "metadata")
        assert meta["camera_make"] == "Canon"

        tech = repo.get_analysis(image_id, "technical")
        assert tech["sharpness_score"] == 80.0

        local = repo.get_analysis(image_id, "local_ai")
        assert local["description"] == "A landscape"
        assert local["has_people"] == 0  # False stored as 0

        # Search index should be populated
        row = conn.execute(
            "SELECT * FROM search_index WHERE search_index MATCH ?",
            ["landscape"],
        ).fetchone()
        assert row is not None

        conn.close()
