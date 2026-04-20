from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from imganalyzer.readers import is_decode_error, open_as_pil


def test_is_decode_error_recognizes_libraw_unsupported_format() -> None:
    class LibRawDataError(RuntimeError):
        pass

    exc = LibRawDataError("Data error or unsupported file format")
    assert is_decode_error(exc) is True


def test_open_as_pil_raw_uses_shared_raw_reader(tmp_path: Path) -> None:
    raw_path = tmp_path / "sample.dng"
    raw_path.write_bytes(b"raw")
    rgb = np.zeros((6, 4, 3), dtype=np.uint8)

    with patch("imganalyzer.readers.raw.read", return_value={"rgb_array": rgb}) as mock_read:
        img = open_as_pil(raw_path)

    mock_read.assert_called_once_with(raw_path, half_size=True)
    assert img.mode == "RGB"
    assert img.size == (4, 6)
