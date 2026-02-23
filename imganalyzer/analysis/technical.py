"""Technical image analysis: sharpness, exposure, histogram, noise, color."""
from __future__ import annotations

from typing import Any

import numpy as np


class TechnicalAnalyzer:
    def __init__(self, image_data: dict[str, Any]) -> None:
        self.image_data = image_data

    def analyze(self) -> dict[str, Any]:
        rgb: np.ndarray = self.image_data["rgb_array"]  # uint8 H×W×3

        # Downsample to max 3000px on longest side for analysis (sufficient for accuracy)
        h, w = rgb.shape[:2]
        max_dim = 3000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from PIL import Image as _PILImage
            pil_img = _PILImage.fromarray(rgb)
            pil_img = pil_img.resize((new_w, new_h), _PILImage.LANCZOS)
            rgb = np.array(pil_img)

        gray = self._to_gray(rgb)
        result: dict[str, Any] = {}

        result.update(self._sharpness(gray, rgb))
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

    def _sharpness(self, gray: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
        """Patch-based, saliency-weighted sharpness using Laplacian variance.

        Divides the image into 64×64 patches, applies a Gaussian center weight
        combined with spectral saliency weighting (Achanta 2009) to focus on
        the visually prominent subject.  Takes the top 15% of patches by the
        combined rank score and returns their weighted mean.  The raw score is
        log-normalised to a 0–100 scale.

        Analysis is performed on a 1200 px cap to match the calibration of the
        log-normalisation reference (ref=2000).
        """
        # Downsample to 1200 px for consistent calibration
        from PIL import Image as _PILImage
        h, w = rgb.shape[:2]
        sharp_max = 1200
        if max(h, w) > sharp_max:
            sc = sharp_max / max(h, w)
            pil_s = _PILImage.fromarray(rgb).resize((int(w * sc), int(h * sc)), _PILImage.LANCZOS)
            rgb_s = np.array(pil_s)
            gray_s = self._to_gray(rgb_s)
        else:
            rgb_s = rgb
            gray_s = gray

        sal_map = _spectral_saliency(rgb_s)
        score = _patch_sharpness(gray_s, sal_map=sal_map)
        # Log-scale normalisation: ref=2000 calibrated against a test corpus
        # (well-focused full-res RAW ≈ 1000–4500, smooth content-limited ≈ 5–20)
        score_norm = min(100.0, float(np.log1p(score) / np.log1p(2000.0) * 100.0))
        return {
            "sharpness_score": round(score_norm, 2),
            "sharpness_raw": round(score, 2),
            "sharpness_label": _label(score_norm, [(40, "Blurry"), (70, "Soft"), (85, "Sharp"), (100, "Very Sharp")]),
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

        # Zone distribution (Ansel Adams zones 0-9, split 256 bins into 10 equal parts)
        zone_size = 256 // 10  # 25 bins each, skip last 6 bins
        zone_dist = [round(float(hist_norm[i * zone_size:(i + 1) * zone_size].sum()), 4) for i in range(10)]

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


def _spectral_saliency(rgb: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Frequency-tuned saliency map (Achanta et al. 2009).

    Computes per-pixel saliency as the squared distance in LAB space between
    the image mean and a Gaussian-blurred version of the image.  The result is
    a float32 map normalised to [0, 1] with the same spatial dimensions as
    *rgb*.  Salient (visually distinct) regions score near 1; uniform regions
    near 0.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.color import rgb2lab

    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    mu = lab.mean(axis=(0, 1), keepdims=True)
    blurred = np.stack(
        [gaussian_filter(lab[:, :, i], sigma=sigma) for i in range(3)], axis=2
    )
    sal = ((blurred - mu) ** 2).sum(axis=2)
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min)
    return sal.astype(np.float32)


def _patch_sharpness(
    gray: np.ndarray,
    patch_size: int = 64,
    top_pct: float = 0.15,
    sal_map: np.ndarray | None = None,
    sal_power: float = 2.0,
) -> float:
    """Compute saliency- and center-weighted patch-based Laplacian sharpness.

    Splits *gray* into non-overlapping *patch_size*×*patch_size* tiles and
    scores each tile by its Laplacian variance.  Patches are ranked by
    ``score × center_weight × saliency_weight`` and the top *top_pct* fraction
    is averaged (weighted by ``center_weight × saliency_weight``).

    If *sal_map* is provided (a float32 saliency map in [0, 1] from
    :func:`_spectral_saliency`), the mean saliency of each patch is raised to
    *sal_power* and used as an additional weight factor.  This suppresses
    background patches that are sharp but not the primary subject.

    Falls back to a plain global Laplacian variance for very small images.
    """
    from scipy.ndimage import laplace

    h, w = gray.shape
    rows, cols = h // patch_size, w // patch_size
    if rows == 0 or cols == 0:
        return float(np.var(laplace(gray)))

    scores: list[float] = []
    cweights: list[float] = []
    sweights: list[float] = []
    cy, cx = rows / 2.0, cols / 2.0
    sigma = max(rows, cols) * 0.5

    for r in range(rows):
        for c in range(cols):
            patch = gray[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            patch_score = float(np.var(laplace(patch)))
            dist2 = ((r - cy) / sigma) ** 2 + ((c - cx) / sigma) ** 2
            cw = float(np.exp(-0.5 * dist2))
            if sal_map is not None:
                sal_patch = sal_map[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
                sw = float(sal_patch.mean()) ** sal_power
            else:
                sw = 1.0
            scores.append(patch_score)
            cweights.append(cw)
            sweights.append(sw)

    scores_arr = np.array(scores)
    cweights_arr = np.array(cweights)
    sweights_arr = np.array(sweights)
    combined = scores_arr * cweights_arr * sweights_arr
    k = max(1, int(len(scores_arr) * top_pct))
    top_idx = np.argpartition(combined, -k)[-k:]
    top_weights = cweights_arr[top_idx] * sweights_arr[top_idx]
    if top_weights.sum() == 0:
        top_weights = np.ones(len(top_weights))
    top_weights = top_weights / top_weights.sum()
    return float(np.dot(scores_arr[top_idx], top_weights))


def _label(value: float, thresholds: list[tuple[float, str]]) -> str:
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    return thresholds[-1][1]
