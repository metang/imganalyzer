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
        aesthetic_out = {"aesthetic_score": 6.5, "aesthetic_label": "Medium"}
        object_out = {"detected_objects": ["tree:70%", "mountain:80%"], "has_person": False}

        with patch("imganalyzer.analysis.ai.local_full.LocalAIFull") as MockFull:
            # Directly test the logic by patching sub-module classes
            pass

        # Patch sub-modules directly
        with patch("imganalyzer.analysis.ai.local.LocalAI") as MockBlip, \
             patch("imganalyzer.analysis.ai.aesthetic.AestheticScorer") as MockAesthetic, \
             patch("imganalyzer.analysis.ai.objects.ObjectDetector") as MockObjects, \
             patch("imganalyzer.analysis.ai.faces.FaceAnalyzer") as MockFaces:

            MockBlip.return_value.analyze.return_value = blip_out
            MockAesthetic.return_value.analyze.return_value = aesthetic_out
            MockObjects.return_value.analyze.return_value = object_out

            orchestrator = LocalAIFull()

            # Patch the imports inside local_full at the module level
            import imganalyzer.analysis.ai.local_full as lf_module
            with patch.object(lf_module, "_import_local_ai", create=True):
                pass  # Not using internal helpers — test via integration

        # Integration-style: verify has_person=False prevents face analysis call
        # We verify this through the FaceDatabase / FaceAnalyzer not being called
        face_call_count = {"n": 0}

        real_fa_analyze = None

        class FakeFaceAnalyzer:
            def analyze(self, *args, **kwargs):
                face_call_count["n"] += 1
                return {"face_count": 0, "face_identities": [], "face_details": []}

        class FakeBlip:
            def analyze(self, *args, **kwargs):
                return blip_out

        class FakeAesthetic:
            def analyze(self, *args, **kwargs):
                return aesthetic_out

        class FakeObjects:
            def analyze(self, *args, **kwargs):
                return object_out

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.aesthetic as aesthetic_module
        import imganalyzer.analysis.ai.objects as objects_module
        import imganalyzer.analysis.ai.faces as faces_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(aesthetic_module, "AestheticScorer", FakeAesthetic), \
             patch.object(objects_module, "ObjectDetector", FakeObjects), \
             patch.object(faces_module, "FaceAnalyzer", FakeFaceAnalyzer):

            result = LocalAIFull().analyze(self._make_image_data())

        assert face_call_count["n"] == 0, "FaceAnalyzer should not be called when no person detected"
        assert "aesthetic_score" in result
        assert result["aesthetic_score"] == 6.5
        assert "detected_objects" in result
        # has_person flag must not leak into output
        assert "has_person" not in result

    def test_face_analysis_called_when_person_present(self):
        """FaceAnalyzer MUST be called when ObjectDetector reports has_person=True."""
        from imganalyzer.analysis.ai.local_full import LocalAIFull

        blip_out = {"description": "A portrait.", "keywords": ["person"], "scene_type": "portrait",
                    "main_subject": "person", "lighting": "studio", "mood": "confident"}
        aesthetic_out = {"aesthetic_score": 7.8, "aesthetic_label": "High"}
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

        class FakeAesthetic:
            def analyze(self, *args, **kwargs):
                return aesthetic_out

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
        import imganalyzer.analysis.ai.aesthetic as aesthetic_module
        import imganalyzer.analysis.ai.objects as objects_module
        import imganalyzer.analysis.ai.faces as faces_module
        import imganalyzer.analysis.ai.face_db as face_db_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(aesthetic_module, "AestheticScorer", FakeAesthetic), \
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
        aesthetic_out = {"aesthetic_score": 5.5, "aesthetic_label": "Medium"}
        object_out = {"detected_objects": ["tree:75%", "sky:80%"], "has_person": False}

        class FakeBlip:
            def analyze(self, *a, **kw): return blip_out

        class FakeAesthetic:
            def analyze(self, *a, **kw): return aesthetic_out

        class FakeObjects:
            def analyze(self, *a, **kw): return object_out

        import imganalyzer.analysis.ai.local as local_module
        import imganalyzer.analysis.ai.aesthetic as aesthetic_module
        import imganalyzer.analysis.ai.objects as objects_module

        with patch.object(local_module, "LocalAI", FakeBlip), \
             patch.object(aesthetic_module, "AestheticScorer", FakeAesthetic), \
             patch.object(objects_module, "ObjectDetector", FakeObjects):

            result = LocalAIFull().analyze(self._make_image_data())

        keywords = result.get("keywords", [])
        assert "tree" in keywords
        assert "sky" in keywords
        assert "nature" in keywords  # original BLIP keyword preserved


# ── AestheticScorer (mocked) ──────────────────────────────────────────────────

class TestAestheticScorer:
    def test_output_keys_and_range(self, synthetic_image_data):
        """Smoke test with mocked model — verifies interface contract."""
        from imganalyzer.analysis.ai.aesthetic import AestheticScorer, _aesthetic_label

        mock_score = 6.8
        with patch.object(AestheticScorer, "_load_models"):
            # Inject fake models
            import types, torch
            fake_clip = MagicMock()
            fake_clip.encode_image.return_value = torch.ones(1, 768)
            fake_linear = MagicMock()
            fake_linear.parameters.return_value = iter([torch.zeros(1)])
            fake_linear.return_value = torch.tensor([[mock_score]])
            AestheticScorer._clip_model = fake_clip
            AestheticScorer._preprocess = lambda img: torch.zeros(3, 224, 224)
            AestheticScorer._model = fake_linear

            scorer = AestheticScorer()
            result = scorer.analyze(synthetic_image_data)

            # Reset singletons so other tests are unaffected
            AestheticScorer._model = None
            AestheticScorer._clip_model = None
            AestheticScorer._preprocess = None

        assert "aesthetic_score" in result
        assert "aesthetic_label" in result
        assert 0.0 <= result["aesthetic_score"] <= 10.0

    def test_aesthetic_label_thresholds(self):
        from imganalyzer.analysis.ai.aesthetic import _aesthetic_label
        assert _aesthetic_label(1.0) == "Very Low"
        assert _aesthetic_label(4.0) == "Low"
        assert _aesthetic_label(5.5) == "Medium"
        assert _aesthetic_label(7.0) == "High"
        assert _aesthetic_label(9.0) == "Exceptional"
        assert _aesthetic_label(10.0) == "Exceptional"


# ── ObjectDetector (mocked) ───────────────────────────────────────────────────

class TestObjectDetector:
    def test_has_person_detection(self, synthetic_image_data):
        """Verifies that person labels set has_person=True."""
        from imganalyzer.analysis.ai.objects import ObjectDetector
        import torch

        mock_results = [{
            "scores": torch.tensor([0.91]),
            "labels": ["person"],
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
