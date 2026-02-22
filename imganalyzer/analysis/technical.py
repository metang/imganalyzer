"""Technical image analysis: sharpness, exposure, histogram, noise, color."""
from __future__ import annotations

from typing import Any

import numpy as np


class TechnicalAnalyzer:
    def __init__(self, image_data: dict[str, Any]) -> None:
        self.image_data = image_data

    def analyze(self) -> dict[str, Any]:
        rgb: np.ndarray = self.image_data["rgb_array"]  # uint8 H×W×3
        gray = self._to_gray(rgb)
        result: dict[str, Any] = {}

        result.update(self._sharpness(gray))
        result.update(self._exposure(gray, rgb))
        result.update(self._noise(gray))
        result.update(self._histogram(gray, rgb))
        result.update(self._color_analysis(rgb))

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        # ITU-R BT.601 luminance
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)

    # ── Sharpness ─────────────────────────────────────────────────────────────

    def _sharpness(self, gray: np.ndarray) -> dict[str, Any]:
        """Laplacian variance — higher = sharper."""
        try:
            from scipy.ndimage import laplace
            lap = laplace(gray)
            score = float(np.var(lap))
        except ImportError:
            # Manual Laplacian kernel without scipy
            from skimage.filters import laplace as sk_laplace
            lap = sk_laplace(gray / 255.0)
            score = float(np.var(lap) * 1e6)

        # Normalise to 0-100 range (empirical scale)
        score_norm = min(100.0, score / 200.0)
        return {
            "sharpness_score": round(score_norm, 2),
            "sharpness_raw": round(score, 2),
            "sharpness_label": _label(score_norm, [(20, "Blurry"), (50, "Soft"), (75, "Sharp"), (100, "Very Sharp")]),
        }

    # ── Exposure ──────────────────────────────────────────────────────────────

    def _exposure(self, gray: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
        mean_lum = float(np.mean(gray))  # 0-255
        std_lum = float(np.std(gray))

        # Estimate EV relative to 18% grey (117.75 / 255)
        exposure_ev = float(np.log2(max(mean_lum, 1) / 117.75))

        # Clipping
        total_px = gray.size
        highlight_clip = float(np.sum(gray > 250) / total_px * 100)
        shadow_clip = float(np.sum(gray < 5) / total_px * 100)

        # Dynamic range estimate (stops)
        p1 = np.percentile(gray, 1)
        p99 = np.percentile(gray, 99)
        dr_stops = float(np.log2(max(p99 - p1, 1) + 1))

        exposure_label = _label(mean_lum, [
            (60, "Underexposed"), (95, "Slightly Dark"), (165, "Good"), (210, "Slightly Bright"), (255, "Overexposed")
        ])

        return {
            "mean_luminance": round(mean_lum, 2),
            "luminance_std": round(std_lum, 2),
            "exposure_ev": round(exposure_ev, 2),
            "exposure_label": exposure_label,
            "highlight_clipping_pct": round(highlight_clip, 3),
            "shadow_clipping_pct": round(shadow_clip, 3),
            "dynamic_range_stops": round(dr_stops, 2),
        }

    # ── Noise ─────────────────────────────────────────────────────────────────

    def _noise(self, gray: np.ndarray) -> dict[str, Any]:
        """Estimate noise via high-frequency residuals."""
        try:
            from skimage.restoration import estimate_sigma
            sigma = float(estimate_sigma(gray / 255.0, channel_axis=None))
        except Exception:
            # Fallback: std of Laplacian residuals in smooth areas
            from skimage.filters import gaussian
            smoothed = gaussian(gray / 255.0, sigma=2)
            residual = gray / 255.0 - smoothed
            sigma = float(np.std(residual))

        snr_db = float(20 * np.log10(max(np.mean(gray) / 255.0, 1e-6) / max(sigma, 1e-6)))

        return {
            "noise_level": round(sigma, 5),
            "noise_label": _label(sigma, [(0.01, "Excellent"), (0.03, "Good"), (0.06, "Moderate"), (1.0, "High")]),
            "snr_db": round(snr_db, 2),
        }

    # ── Histogram ────────────────────────────────────────────────────────────

    def _histogram(self, gray: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()

        # Zone distribution (Ansel Adams zones 0-10)
        zones = np.split(hist_norm, 10)
        zone_dist = [round(float(z.sum()), 4) for z in zones]

        return {
            "histogram_mean": round(float(np.mean(gray)), 2),
            "histogram_median": round(float(np.median(gray)), 2),
            "histogram_p5": round(float(np.percentile(gray, 5)), 2),
            "histogram_p95": round(float(np.percentile(gray, 95)), 2),
            "zone_distribution": zone_dist,
        }

    # ── Color analysis ────────────────────────────────────────────────────────

    def _color_analysis(self, rgb: np.ndarray) -> dict[str, Any]:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))

        # Color temperature estimate (warm vs cool)
        warm_cool_ratio = round((mean_r - mean_b) / max(mean_g, 1), 4)

        # Dominant colors (K-means, 5 clusters)
        dominant_colors: list[str] = []
        try:
            from skimage.color import rgb2lab
            # Sample pixels for performance
            flat = rgb.reshape(-1, 3)
            if len(flat) > 10000:
                idx = np.random.choice(len(flat), 10000, replace=False)
                flat = flat[idx]

            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=3)
            km.fit(flat)
            centers = km.cluster_centers_.astype(int)
            # Sort by frequency
            labels = km.labels_
            counts = np.bincount(labels)
            order = np.argsort(-counts)
            dominant_colors = [f"#{int(centers[i][0]):02x}{int(centers[i][1]):02x}{int(centers[i][2]):02x}" for i in order]
        except Exception:
            # Simple average quantization fallback
            dominant_colors = [f"#{int(mean_r):02x}{int(mean_g):02x}{int(mean_b):02x}"]

        # Saturation estimate
        max_rgb = np.maximum(np.maximum(r, g), b).astype(float)
        min_rgb = np.minimum(np.minimum(r, g), b).astype(float)
        saturation = float(np.mean((max_rgb - min_rgb) / np.maximum(max_rgb, 1)))

        return {
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2),
            "warm_cool_ratio": warm_cool_ratio,
            "avg_saturation": round(saturation, 4),
            "dominant_colors": dominant_colors,
        }


def _label(value: float, thresholds: list[tuple[float, str]]) -> str:
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    return thresholds[-1][1]
