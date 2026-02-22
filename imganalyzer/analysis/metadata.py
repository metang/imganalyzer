"""EXIF and metadata extraction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _parse_rational(tag_value: Any) -> float | None:
    """Parse exifread IfdTag rational to float."""
    try:
        v = tag_value.values
        if isinstance(v, list) and len(v) > 0:
            r = v[0]
            if hasattr(r, "num") and hasattr(r, "den") and r.den:
                return r.num / r.den
            return float(r)
        return float(v)
    except Exception:
        return None


def _dms_to_decimal(dms: list, ref: str) -> float | None:
    """Convert degrees/minutes/seconds to decimal degrees."""
    try:
        d = float(dms[0].num) / float(dms[0].den)
        m = float(dms[1].num) / float(dms[1].den)
        s = float(dms[2].num) / float(dms[2].den)
        decimal = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def _reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    """Best-effort reverse geocoding via nominatim (no API key needed)."""
    try:
        import httpx
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "imganalyzer/0.1.0"},
            timeout=5.0,
        )
        data = resp.json()
        addr = data.get("address", {})
        return {
            "location_city": addr.get("city") or addr.get("town") or addr.get("village", ""),
            "location_state": addr.get("state", ""),
            "location_country": addr.get("country", ""),
            "location_country_code": addr.get("country_code", "").upper(),
        }
    except Exception:
        return {}


class MetadataExtractor:
    def __init__(self, path: Path, image_data: dict[str, Any]) -> None:
        self.path = path
        self.image_data = image_data

    def extract(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}

        # Try exifread (works for JPEG, TIFF, and RAW files)
        try:
            import exifread
            with open(self.path, "rb") as f:
                tags = exifread.process_file(f, details=False, strict=False)
            meta.update(self._parse_exifread_tags(tags))
        except Exception:
            pass

        # Fallback: piexif for JPEG
        if not meta and not self.image_data.get("is_raw"):
            try:
                import piexif
                exif_bytes = self.image_data.get("exif_bytes")
                if exif_bytes:
                    exif_dict = piexif.load(exif_bytes)
                    meta.update(self._parse_piexif(exif_dict))
            except Exception:
                pass

        # DPI from Pillow
        if self.image_data.get("dpi"):
            dpi = self.image_data["dpi"]
            if isinstance(dpi, tuple):
                meta["resolution_x_dpi"] = dpi[0]
                meta["resolution_y_dpi"] = dpi[1]

        # RAW-specific
        if self.image_data.get("is_raw"):
            if self.image_data.get("camera_wb"):
                wb = self.image_data["camera_wb"]
                meta["raw_camera_wb"] = [round(float(x), 4) for x in wb if x]
            if self.image_data.get("white_level"):
                meta["raw_white_level"] = int(self.image_data["white_level"])

        # GPS reverse geocode
        if meta.get("gps_latitude") and meta.get("gps_longitude"):
            try:
                geo = _reverse_geocode(meta["gps_latitude"], meta["gps_longitude"])
                meta.update(geo)
            except Exception:
                pass

        return meta

    def _parse_exifread_tags(self, tags: dict) -> dict[str, Any]:
        m: dict[str, Any] = {}

        def get(key: str) -> Any:
            return tags.get(key) or tags.get(f"EXIF {key}") or tags.get(f"Image {key}")

        # Camera
        make = tags.get("Image Make")
        model = tags.get("Image Model")
        if make:
            m["camera_make"] = _safe_str(make)
        if model:
            m["camera_model"] = _safe_str(model)

        lens = tags.get("EXIF LensModel") or tags.get("MakerNote LensModel")
        if lens:
            m["lens_model"] = _safe_str(lens)

        lens_make = tags.get("EXIF LensMake")
        if lens_make:
            m["lens_make"] = _safe_str(lens_make)

        serial = tags.get("EXIF BodySerialNumber") or tags.get("MakerNote SerialNumber")
        if serial:
            m["camera_serial"] = _safe_str(serial)

        sw = tags.get("Image Software")
        if sw:
            m["software"] = _safe_str(sw)

        # Date/time
        dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if dt:
            m["date_time_original"] = _safe_str(dt)

        # Exposure
        fn = tags.get("EXIF FNumber")
        if fn:
            v = _parse_rational(fn)
            if v:
                m["f_number"] = round(v, 1)

        iso = tags.get("EXIF ISOSpeedRatings")
        if iso:
            try:
                m["iso"] = int(_safe_str(iso))
            except ValueError:
                pass

        exp = tags.get("EXIF ExposureTime")
        if exp:
            v = _parse_rational(exp)
            if v:
                m["exposure_time"] = f"1/{int(1/v)}" if v < 1 else str(round(v, 4))

        fl = tags.get("EXIF FocalLength")
        if fl:
            v = _parse_rational(fl)
            if v:
                m["focal_length"] = round(v, 1)

        fl35 = tags.get("EXIF FocalLengthIn35mmFilm")
        if fl35:
            try:
                m["focal_length_35mm"] = int(_safe_str(fl35))
            except ValueError:
                pass

        ev = tags.get("EXIF ExposureBiasValue")
        if ev:
            v = _parse_rational(ev)
            if v is not None:
                m["exposure_bias"] = round(v, 2)

        meter = tags.get("EXIF MeteringMode")
        if meter:
            m["metering_mode"] = _safe_str(meter)

        flash = tags.get("EXIF Flash")
        if flash:
            m["flash"] = _safe_str(flash)

        wb = tags.get("EXIF WhiteBalance")
        if wb:
            m["white_balance"] = _safe_str(wb)

        # Image properties
        orient = tags.get("Image Orientation")
        if orient:
            m["orientation"] = _safe_str(orient)

        color_space = tags.get("EXIF ColorSpace")
        if color_space:
            m["color_space"] = _safe_str(color_space)

        # GPS
        gps_lat = tags.get("GPS GPSLatitude")
        gps_lat_ref = tags.get("GPS GPSLatitudeRef")
        gps_lon = tags.get("GPS GPSLongitude")
        gps_lon_ref = tags.get("GPS GPSLongitudeRef")
        gps_alt = tags.get("GPS GPSAltitude")

        if gps_lat and gps_lon:
            lat = _dms_to_decimal(gps_lat.values, _safe_str(gps_lat_ref) or "N")
            lon = _dms_to_decimal(gps_lon.values, _safe_str(gps_lon_ref) or "E")
            if lat is not None:
                m["gps_latitude"] = lat
            if lon is not None:
                m["gps_longitude"] = lon
            if gps_alt:
                alt = _parse_rational(gps_alt)
                if alt is not None:
                    m["gps_altitude"] = round(alt, 1)

        return m

    def _parse_piexif(self, exif_dict: dict) -> dict[str, Any]:
        """Parse piexif dict as fallback."""
        m: dict[str, Any] = {}
        import piexif

        def get_ifd(ifd: str, tag_id: int) -> Any:
            return exif_dict.get(ifd, {}).get(tag_id)

        def decode(v: Any) -> str:
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace").rstrip("\x00").strip()
            return str(v)

        make = get_ifd("0th", piexif.ImageIFD.Make)
        model = get_ifd("0th", piexif.ImageIFD.Model)
        if make:
            m["camera_make"] = decode(make)
        if model:
            m["camera_model"] = decode(model)

        return m
