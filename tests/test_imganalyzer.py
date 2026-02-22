"""Tests for imganalyzer."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


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


# ── XMP Writer ────────────────────────────────────────────────────────────────

class TestXMPWriter:
    def _make_result(self, tmp_path):
        from imganalyzer.analyzer import AnalysisResult
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
            ai_analysis={
                "description": "A golden hour landscape with mountains reflected in a calm lake.",
                "scene_type": "landscape",
                "mood": "serene",
                "keywords": ["mountain", "lake", "golden hour", "reflection", "nature"],
                "lighting": "Golden hour",
                "main_subject": "Mountain lake",
            },
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
        # Should parse without error
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
