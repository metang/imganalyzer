"""XMP sidecar writer — Adobe Lightroom-compatible format."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.dom import minidom

# XMP Namespaces
NS = {
    "x":             "adobe:ns:meta/",
    "rdf":           "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc":            "http://purl.org/dc/elements/1.1/",
    "xmp":           "http://ns.adobe.com/xap/1.0/",
    "xmpMM":         "http://ns.adobe.com/xap/1.0/mm/",
    "exif":          "http://ns.adobe.com/exif/1.0/",
    "tiff":          "http://ns.adobe.com/tiff/1.0/",
    "Iptc4xmpCore":  "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "photoshop":     "http://ns.adobe.com/photoshop/1.0/",
    "crs":           "http://ns.adobe.com/camera-raw-settings/1.0/",
    "imganalyzer":   "http://ns.imganalyzer.io/1.0/",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def _ns(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def _rdf_bag(parent: ET.Element, items: list[str]) -> None:
    bag = ET.SubElement(parent, _ns("rdf", "Bag"))
    for item in items:
        li = ET.SubElement(bag, _ns("rdf", "li"))
        li.text = item


def _rdf_seq(parent: ET.Element, items: list[str]) -> None:
    seq = ET.SubElement(parent, _ns("rdf", "Seq"))
    for item in items:
        li = ET.SubElement(seq, _ns("rdf", "li"))
        li.text = item


def _alt_lang(parent: ET.Element, text: str, lang: str = "x-default") -> None:
    alt = ET.SubElement(parent, _ns("rdf", "Alt"))
    li = ET.SubElement(alt, _ns("rdf", "li"))
    li.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
    li.text = text


class XMPWriter:
    def __init__(self, result: Any) -> None:
        self.result = result

    def write(self, path: Path) -> None:
        xml_str = self._build_xml()
        path.write_text(xml_str, encoding="utf-8")

    def _build_xml(self) -> str:
        result = self.result
        meta = result.metadata or {}
        tech = result.technical or {}
        ai = result.ai_analysis or {}

        # Root
        xmpmeta = ET.Element(_ns("x", "xmpmeta"))
        xmpmeta.set(_ns("x", "xmptk"), "imganalyzer 0.1.0")

        rdf_root = ET.SubElement(xmpmeta, _ns("rdf", "RDF"))
        desc = ET.SubElement(rdf_root, _ns("rdf", "Description"))
        desc.set(_ns("rdf", "about"), "")

        # ── dc: Dublin Core ────────────────────────────────────────────────
        # Caption / description (AI-generated or empty)
        ai_desc = ai.get("description") or ""
        if ai_desc:
            dc_desc = ET.SubElement(desc, _ns("dc", "description"))
            _alt_lang(dc_desc, ai_desc)

        # Subject / keywords — merge AI keywords, scene/mood, and detected object labels
        keywords: list[str] = []
        if ai.get("keywords"):
            if isinstance(ai["keywords"], list):
                keywords.extend(ai["keywords"])
            elif isinstance(ai["keywords"], str):
                keywords.extend(ai["keywords"].split(","))
        if ai.get("scene_type"):
            keywords.append(ai["scene_type"])
        if ai.get("mood"):
            keywords.append(ai["mood"])
        # Add detected object labels (strip confidence percentages) as keywords
        for obj in (ai.get("detected_objects") or []):
            label = obj.split(":")[0].strip()
            if label and label not in keywords:
                keywords.append(label)
        if keywords:
            dc_subject = ET.SubElement(desc, _ns("dc", "subject"))
            _rdf_bag(dc_subject, [k.strip() for k in keywords if k.strip()])

        # ── xmp: Basic XMP ─────────────────────────────────────────────────
        if meta.get("date_time_original"):
            desc.set(_ns("xmp", "CreateDate"), meta["date_time_original"])

        desc.set(_ns("xmp", "CreatorTool"), "imganalyzer 0.1.0")

        # ── xmpMM: Media Management ────────────────────────────────────────
        import uuid
        desc.set(_ns("xmpMM", "DocumentID"), f"xmp.did:{uuid.uuid4()}")
        desc.set(_ns("xmpMM", "OriginalDocumentID"), f"xmp.did:{uuid.uuid4()}")

        # ── tiff: TIFF/Image properties ────────────────────────────────────
        if result.width:
            desc.set(_ns("tiff", "ImageWidth"), str(result.width))
        if result.height:
            desc.set(_ns("tiff", "ImageHeight"), str(result.height))
        if meta.get("camera_make"):
            desc.set(_ns("tiff", "Make"), meta["camera_make"])
        if meta.get("camera_model"):
            desc.set(_ns("tiff", "Model"), meta["camera_model"])
        if meta.get("orientation"):
            # Map text orientation to TIFF numeric value
            orient_map = {"Horizontal": "1", "Rotated 90 CW": "6", "Rotated 90 CCW": "8", "Rotated 180": "3"}
            orient_str = str(meta["orientation"])
            for k, v in orient_map.items():
                if k.lower() in orient_str.lower():
                    desc.set(_ns("tiff", "Orientation"), v)
                    break

        # ── exif: EXIF properties ──────────────────────────────────────────
        if meta.get("iso"):
            iso_tag = ET.SubElement(desc, _ns("exif", "ISOSpeedRatings"))
            _rdf_seq(iso_tag, [str(meta["iso"])])
        if meta.get("f_number"):
            desc.set(_ns("exif", "FNumber"), _rational_str(meta["f_number"]))
        if meta.get("exposure_time"):
            desc.set(_ns("exif", "ExposureTime"), _exposure_str(meta["exposure_time"]))
        if meta.get("focal_length"):
            desc.set(_ns("exif", "FocalLength"), _rational_str(meta["focal_length"]))
        if meta.get("focal_length_35mm"):
            desc.set(_ns("exif", "FocalLengthIn35mmFilm"), str(meta["focal_length_35mm"]))
        if meta.get("exposure_bias") is not None:
            desc.set(_ns("exif", "ExposureBiasValue"), str(meta["exposure_bias"]))
        if meta.get("white_balance"):
            wb_map = {"auto": "0", "manual": "1"}
            wb_str = str(meta["white_balance"]).lower()
            desc.set(_ns("exif", "WhiteBalance"), wb_map.get(wb_str, "0"))
        if meta.get("metering_mode"):
            desc.set(_ns("exif", "MeteringMode"), _metering_mode(meta["metering_mode"]))
        if meta.get("flash"):
            desc.set(_ns("exif", "Flash"), _flash_value(meta["flash"]))
        if meta.get("color_space"):
            cs_map = {"sRGB": "1", "srgb": "1", "Adobe RGB": "65535", "Uncalibrated": "65535"}
            desc.set(_ns("exif", "ColorSpace"), cs_map.get(str(meta["color_space"]), "1"))

        # GPS
        if meta.get("gps_latitude") is not None and meta.get("gps_longitude") is not None:
            lat = meta["gps_latitude"]
            lon = meta["gps_longitude"]
            gps_desc = ET.SubElement(desc, _ns("exif", "GPSCoordinates"))
            # Lightroom uses deg,min.secN/S format
            gps_desc.text = f"{_decimal_to_dms_str(lat, 'lat')},{_decimal_to_dms_str(lon, 'lon')}"
            # Also as individual attributes
            desc.set(_ns("exif", "GPSLatitude"), _decimal_to_dms_str(lat, "lat"))
            desc.set(_ns("exif", "GPSLongitude"), _decimal_to_dms_str(lon, "lon"))
        if meta.get("gps_altitude") is not None:
            desc.set(_ns("exif", "GPSAltitude"), _rational_str(meta["gps_altitude"]))
            desc.set(_ns("exif", "GPSAltitudeRef"), "0")

        # ── Iptc4xmpCore: IPTC location ────────────────────────────────────
        if meta.get("location_city"):
            desc.set(_ns("Iptc4xmpCore", "Location"), meta.get("location_city", ""))
        if meta.get("location_state"):
            desc.set(_ns("Iptc4xmpCore", "ProvinceState"), meta["location_state"])
        if meta.get("location_country"):
            desc.set(_ns("Iptc4xmpCore", "CountryName"), meta["location_country"])
        if meta.get("location_country_code"):
            desc.set(_ns("Iptc4xmpCore", "CountryCode"), meta["location_country_code"])

        # ── photoshop: Additional metadata ────────────────────────────────
        if meta.get("lens_model"):
            desc.set(_ns("photoshop", "Lens"), meta["lens_model"])

        # ── crs: Camera Raw Settings (Lightroom-specific) ──────────────────
        if meta.get("lens_model"):
            desc.set(_ns("crs", "LensProfileName"), meta["lens_model"])
        if meta.get("camera_make") or meta.get("camera_model"):
            make = meta.get("camera_make", "")
            model = meta.get("camera_model", "")
            desc.set(_ns("crs", "CameraProfile"), f"{make} {model}".strip() or "Adobe Standard")
        desc.set(_ns("crs", "Version"), "15.4")

        # ── imganalyzer: Custom namespace for analysis results ─────────────
        # Technical scores
        if tech.get("sharpness_score") is not None:
            desc.set(_ns("imganalyzer", "SharpnessScore"), str(tech["sharpness_score"]))
        if tech.get("sharpness_label"):
            desc.set(_ns("imganalyzer", "SharpnessLabel"), tech["sharpness_label"])
        if tech.get("exposure_ev") is not None:
            desc.set(_ns("imganalyzer", "ExposureEV"), str(tech["exposure_ev"]))
        if tech.get("exposure_label"):
            desc.set(_ns("imganalyzer", "ExposureLabel"), tech["exposure_label"])
        if tech.get("noise_level") is not None:
            desc.set(_ns("imganalyzer", "NoiseLevel"), str(tech["noise_level"]))
        if tech.get("noise_label"):
            desc.set(_ns("imganalyzer", "NoiseLabel"), tech["noise_label"])
        if tech.get("snr_db") is not None:
            desc.set(_ns("imganalyzer", "SNR_dB"), str(tech["snr_db"]))
        if tech.get("dynamic_range_stops") is not None:
            desc.set(_ns("imganalyzer", "DynamicRangeStops"), str(tech["dynamic_range_stops"]))
        if tech.get("highlight_clipping_pct") is not None:
            desc.set(_ns("imganalyzer", "HighlightClippingPct"), str(tech["highlight_clipping_pct"]))
        if tech.get("shadow_clipping_pct") is not None:
            desc.set(_ns("imganalyzer", "ShadowClippingPct"), str(tech["shadow_clipping_pct"]))
        if tech.get("avg_saturation") is not None:
            desc.set(_ns("imganalyzer", "AvgSaturation"), str(tech["avg_saturation"]))
        if tech.get("warm_cool_ratio") is not None:
            desc.set(_ns("imganalyzer", "WarmCoolRatio"), str(tech["warm_cool_ratio"]))
        if tech.get("dominant_colors"):
            dc_colors = ET.SubElement(desc, _ns("imganalyzer", "DominantColors"))
            _rdf_seq(dc_colors, tech["dominant_colors"])

        # AI results
        if ai.get("scene_type"):
            desc.set(_ns("imganalyzer", "AISceneType"), ai["scene_type"])
        if ai.get("main_subject"):
            desc.set(_ns("imganalyzer", "AIMainSubject"), ai["main_subject"])
        if ai.get("lighting"):
            desc.set(_ns("imganalyzer", "AILighting"), ai["lighting"])
        if ai.get("mood"):
            desc.set(_ns("imganalyzer", "AIMood"), ai["mood"])
        if ai.get("technical_notes"):
            desc.set(_ns("imganalyzer", "AITechnicalNotes"), ai["technical_notes"])
        if ai.get("dominant_colors_ai"):
            ai_colors = ET.SubElement(desc, _ns("imganalyzer", "AIDominantColors"))
            _rdf_seq(ai_colors, ai["dominant_colors_ai"])
        if ai.get("detected_objects"):
            objs = ET.SubElement(desc, _ns("imganalyzer", "AIDetectedObjects"))
            _rdf_bag(objs, ai["detected_objects"])
        if ai.get("landmark"):
            desc.set(_ns("imganalyzer", "AILandmark"), ai["landmark"])
        if ai.get("ocr_text"):
            # Store multi-line OCR text as an element (not an attribute) so
            # raw newlines are valid XML text content rather than illegal
            # attribute characters.
            ocr_elem = ET.SubElement(desc, _ns("imganalyzer", "AIOCRText"))
            ocr_elem.text = ai["ocr_text"]

        # Face analysis
        if ai.get("face_count") is not None:
            desc.set(_ns("imganalyzer", "FaceCount"), str(ai["face_count"]))
        if ai.get("face_identities"):
            fi_elem = ET.SubElement(desc, _ns("imganalyzer", "FaceIdentities"))
            _rdf_bag(fi_elem, ai["face_identities"])
        if ai.get("face_details"):
            fd_elem = ET.SubElement(desc, _ns("imganalyzer", "FaceDetails"))
            _rdf_bag(fd_elem, ai["face_details"])

        # RAW-specific
        if meta.get("raw_white_level"):
            desc.set(_ns("imganalyzer", "RAWWhiteLevel"), str(meta["raw_white_level"]))

        # ── Serialize to pretty XML ────────────────────────────────────────
        tree_str = ET.tostring(xmpmeta, encoding="unicode", xml_declaration=False)
        dom = minidom.parseString(f'<?xml version="1.0" encoding="UTF-8"?>\n{tree_str}')
        return dom.toprettyxml(indent="  ", encoding=None).replace('<?xml version="1.0" ?>', '')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rational_str(value: float) -> str:
    """Convert float to EXIF rational string e.g. 2.8 → '28/10'."""
    if value == int(value):
        return f"{int(value)}/1"
    s = str(round(value, 4))
    dec_places = len(s.split(".")[-1]) if "." in s else 0
    denom = 10 ** dec_places
    num = int(round(value * denom))
    return f"{num}/{denom}"


def _exposure_str(exp: str | float) -> str:
    """Normalise exposure time to rational string."""
    s = str(exp)
    if "/" in s:
        return s
    try:
        v = float(s)
        if v >= 1:
            return f"{int(v)}/1"
        return f"1/{int(round(1/v))}"
    except ValueError:
        return s


def _decimal_to_dms_str(decimal: float, axis: str) -> str:
    """Convert decimal degrees to XMP DMS string e.g. '48,51.4N'."""
    ref = "N" if axis == "lat" and decimal >= 0 else "S" if axis == "lat" else "E" if decimal >= 0 else "W"
    decimal = abs(decimal)
    deg = int(decimal)
    minutes = (decimal - deg) * 60
    return f"{deg},{minutes:.4f}{ref}"


def _metering_mode(mode_str: str) -> str:
    mode_map = {
        "average": "1", "center": "2", "centerweighted": "2",
        "spot": "3", "multispot": "4", "pattern": "5", "evaluative": "5",
        "multi": "5", "partial": "6",
    }
    key = str(mode_str).lower().replace(" ", "").replace("-", "")
    for k, v in mode_map.items():
        if k in key:
            return v
    return "0"


def _flash_value(flash_str: str) -> str:
    s = str(flash_str).lower()
    if "did not fire" in s or "no flash" in s:
        return "0"
    if "fired" in s:
        return "1"
    return "0"
