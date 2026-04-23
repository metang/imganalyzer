"""Verify register_heif_opener is called at most once across many reads."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_heif_flag():
    import imganalyzer.readers.standard as standard
    original = standard._HEIF_REGISTERED
    standard._HEIF_REGISTERED = False
    yield
    standard._HEIF_REGISTERED = original


def test_heif_register_called_once_over_many_reads():
    import imganalyzer.readers.standard as standard

    call_count = {"n": 0}

    def fake_register() -> None:
        call_count["n"] += 1

    fake_module = types.SimpleNamespace(register_heif_opener=fake_register)

    paths = [Path(f"img_{i}.heic") for i in range(25)]

    with patch.dict(sys.modules, {"pillow_heif": fake_module}):
        for p in paths:
            standard.register_optional_pillow_opener(p)

    assert call_count["n"] == 1
    assert standard._HEIF_REGISTERED is True


def test_heif_register_skipped_for_non_heif():
    import imganalyzer.readers.standard as standard

    call_count = {"n": 0}

    def fake_register() -> None:
        call_count["n"] += 1

    fake_module = types.SimpleNamespace(register_heif_opener=fake_register)

    with patch.dict(sys.modules, {"pillow_heif": fake_module}):
        for _ in range(10):
            standard.register_optional_pillow_opener(Path("img.jpg"))

    assert call_count["n"] == 0
    assert standard._HEIF_REGISTERED is False
