"""Pure-Python geohash encoding for spatial clustering.

Encodes (latitude, longitude) into a geohash string.  Used by the map
feature to enable fast server-side clustering via ``GROUP BY substr(geohash, 1, N)``.

No external dependencies — the algorithm is a simple interleaved binary
encoding of quantised lat/lng into a base-32 string.
"""
from __future__ import annotations

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode(latitude: float, longitude: float, precision: int = 8) -> str:
    """Encode *latitude* / *longitude* to a geohash string.

    Parameters
    ----------
    latitude:
        Decimal latitude (-90 to 90).
    longitude:
        Decimal longitude (-180 to 180).
    precision:
        Length of the returned geohash (1–12).  Default 8 gives ~19 m
        resolution which is sufficient for photo clustering.

    Returns
    -------
    str
        Geohash string of the requested *precision*.
    """
    lat_range = (-90.0, 90.0)
    lng_range = (-180.0, 180.0)
    bits = 0
    count = 0
    is_lng = True
    chars: list[str] = []

    while len(chars) < precision:
        if is_lng:
            mid = (lng_range[0] + lng_range[1]) / 2
            if longitude >= mid:
                bits = (bits << 1) | 1
                lng_range = (mid, lng_range[1])
            else:
                bits = bits << 1
                lng_range = (lng_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:
                bits = (bits << 1) | 1
                lat_range = (mid, lat_range[1])
            else:
                bits = bits << 1
                lat_range = (lat_range[0], mid)
        is_lng = not is_lng
        count += 1
        if count == 5:
            chars.append(_BASE32[bits])
            bits = 0
            count = 0

    return "".join(chars)
