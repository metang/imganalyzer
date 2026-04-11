"""EXIF and metadata extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from imganalyzer import __version__

log = logging.getLogger(__name__)


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
    except Exception as exc:
        log.debug(
            "metadata rational parse failed tag_value=%r error_type=%s error=%s",
            tag_value,
            type(exc).__name__,
            exc,
        )
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
    except Exception as exc:
        log.debug(
            "metadata GPS DMS parse failed ref=%s dms=%r error_type=%s error=%s",
            ref,
            dms,
            type(exc).__name__,
            exc,
        )
        return None


def _reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    """Best-effort reverse geocoding via nominatim (no API key needed).

    Results are cached by GPS coordinates rounded to 4 decimal places
    (~11 m resolution) so that images taken at the same location reuse
    the result.  At 500K images this avoids ~400K redundant HTTP requests
    (Nominatim rate-limits to 1 req/s → ~111 hours without the cache).
    """
    # Round to 4 decimal places (~11m) — images from the same spot
    # will share the cached result without a visible location difference.
    cache_key = (round(lat, 4), round(lon, 4))
    cached = _geocode_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import httpx
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": f"imganalyzer/{__version__}"},
            timeout=5.0,
        )
        data = resp.json()
        addr = data.get("address", {})
        result = {
            "location_city": addr.get("city") or addr.get("town") or addr.get("village", ""),
            "location_state": addr.get("state", ""),
            "location_country": addr.get("country", ""),
            "location_country_code": addr.get("country_code", "").upper(),
        }
    except Exception as exc:
        error_key = f"{type(exc).__name__}:{exc}"
        if error_key not in _geocode_warning_keys:
            log.warning(
                "metadata reverse geocode failed lat=%.6f lon=%.6f "
                "error_type=%s error=%s",
                lat,
                lon,
                type(exc).__name__,
                exc,
            )
            _geocode_warning_keys.add(error_key)
        else:
            log.debug(
                "metadata reverse geocode failed (repeat) lat=%.6f lon=%.6f "
                "error_type=%s error=%s",
                lat,
                lon,
                type(exc).__name__,
                exc,
            )
        result = {}

    _geocode_cache[cache_key] = result
    return result


# GPS reverse-geocoding cache.  Keyed by (lat, lon) rounded to 4 decimal
# places (~11 m resolution).  Survives the full ingest run so that images
# from the same location share a single HTTP request.
_geocode_cache: dict[tuple[float, float], dict[str, str]] = {}
_geocode_warning_keys: set[str] = set()


class MetadataExtractor:
    def __init__(self, path: Path, image_data: dict[str, Any]) -> None:
        self.path = path
        self.image_data = image_data

    def extract(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}

        # 1. Try pre-parsed EXIF from sidecar (remote worker path — no
        #    original file on disk, but coordinator extracted EXIF during
        #    pre-decode and stored the result in the sidecar).
        parsed_exif = self.image_data.get("parsed_exif")
        if parsed_exif and isinstance(parsed_exif, dict):
            meta.update(parsed_exif)

        # 2. Try piexif from raw EXIF bytes (works without file access)
        if not meta.get("date_time_original"):
            exif_bytes = self.image_data.get("exif_bytes")
            if exif_bytes:
                try:
                    import piexif
                    exif_dict = piexif.load(exif_bytes)
                    piexif_meta = self._parse_piexif(exif_dict)
                    # Merge: only fill in keys not already present
                    for k, v in piexif_meta.items():
                        if k not in meta:
                            meta[k] = v
                except Exception as exc:
                    log.debug(
                        "metadata extraction failed stage=piexif path=%s "
                        "error_type=%s error=%s",
                        self.path,
                        type(exc).__name__,
                        exc,
                    )

        # 3. Try exifread from the original file (master device path)
        if not meta.get("date_time_original"):
            try:
                import exifread
                with open(self.path, "rb") as f:
                    tags = exifread.process_file(f, details=False, strict=False)
                exifread_meta = self._parse_exifread_tags(tags)
                for k, v in exifread_meta.items():
                    if k not in meta:
                        meta[k] = v
            except Exception as exc:
                log.debug(
                    "metadata extraction failed stage=exifread path=%s "
                    "error_type=%s error=%s",
                    self.path,
                    type(exc).__name__,
                    exc,
                )

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
            # Compute geohash for map clustering
            try:
                from imganalyzer.db.geohash import encode as geohash_encode

                meta["geohash"] = geohash_encode(
                    meta["gps_latitude"], meta["gps_longitude"], precision=8
                )
            except Exception:
                pass

            try:
                geo = _reverse_geocode(meta["gps_latitude"], meta["gps_longitude"])
                meta.update(geo)
            except Exception as exc:
                log.warning(
                    "metadata geocode update failed path=%s lat=%.6f lon=%.6f "
                    "error_type=%s error=%s",
                    self.path,
                    meta["gps_latitude"],
                    meta["gps_longitude"],
                    type(exc).__name__,
                    exc,
                )

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
            except ValueError as exc:
                log.debug(
                    "metadata ISO parse failed value=%r error_type=%s error=%s",
                    iso,
                    type(exc).__name__,
                    exc,
                )

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
            except ValueError as exc:
                log.debug(
                    "metadata focal_length_35mm parse failed value=%r "
                    "error_type=%s error=%s",
                    fl35,
                    type(exc).__name__,
                    exc,
                )

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

        def rational_to_float(v: Any) -> float | None:
            if isinstance(v, tuple) and len(v) == 2 and v[1]:
                return v[0] / v[1]
            return None

        # Camera
        make = get_ifd("0th", piexif.ImageIFD.Make)
        model = get_ifd("0th", piexif.ImageIFD.Model)
        if make:
            m["camera_make"] = decode(make)
        if model:
            m["camera_model"] = decode(model)

        sw = get_ifd("0th", piexif.ImageIFD.Software)
        if sw:
            m["software"] = decode(sw)

        # Date/time
        dt = get_ifd("Exif", piexif.ExifIFD.DateTimeOriginal)
        if not dt:
            dt = get_ifd("0th", piexif.ImageIFD.DateTime)
        if dt:
            m["date_time_original"] = decode(dt)

        # Lens
        lens = get_ifd("Exif", piexif.ExifIFD.LensModel)
        if lens:
            m["lens_model"] = decode(lens)
        lens_make = get_ifd("Exif", piexif.ExifIFD.LensMake)
        if lens_make:
            m["lens_make"] = decode(lens_make)

        # Exposure
        fn = get_ifd("Exif", piexif.ExifIFD.FNumber)
        if fn:
            v = rational_to_float(fn)
            if v:
                m["f_number"] = round(v, 1)

        iso = get_ifd("Exif", piexif.ExifIFD.ISOSpeedRatings)
        if iso:
            m["iso"] = int(iso)

        exp = get_ifd("Exif", piexif.ExifIFD.ExposureTime)
        if exp:
            v = rational_to_float(exp)
            if v:
                m["exposure_time"] = f"1/{int(1/v)}" if v < 1 else str(round(v, 4))

        fl = get_ifd("Exif", piexif.ExifIFD.FocalLength)
        if fl:
            v = rational_to_float(fl)
            if v:
                m["focal_length"] = round(v, 1)

        fl35 = get_ifd("Exif", piexif.ExifIFD.FocalLengthIn35mmFilm)
        if fl35:
            m["focal_length_35mm"] = int(fl35)

        # GPS
        gps = exif_dict.get("GPS", {})
        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
        alt = gps.get(piexif.GPSIFD.GPSAltitude)

        if lat and lon:
            try:
                def _piexif_dms(dms: list, ref: bytes) -> float | None:
                    d = dms[0][0] / dms[0][1]
                    mi = dms[1][0] / dms[1][1]
                    s = dms[2][0] / dms[2][1]
                    dec = d + mi / 60 + s / 3600
                    if ref in (b"S", b"W"):
                        dec = -dec
                    return round(dec, 6)

                lat_dec = _piexif_dms(lat, lat_ref or b"N")
                lon_dec = _piexif_dms(lon, lon_ref or b"E")
                if lat_dec is not None:
                    m["gps_latitude"] = lat_dec
                if lon_dec is not None:
                    m["gps_longitude"] = lon_dec
                if alt:
                    alt_v = rational_to_float(alt)
                    if alt_v is not None:
                        m["gps_altitude"] = round(alt_v, 1)
            except Exception:
                pass

        return m
