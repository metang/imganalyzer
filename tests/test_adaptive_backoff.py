"""Tests for the adaptive idle-poll backoff helper used by worker.py
and distributed_worker.py to throttle empty-claim polling (B6).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from imganalyzer.pipeline.worker import _AdaptiveBackoff, _adaptive_wait


def _make_mock_event() -> MagicMock:
    """An Event-like mock whose ``wait`` records the requested interval
    and returns False (i.e., shutdown is NOT set)."""
    ev = MagicMock(spec=threading.Event)
    ev.wait.return_value = False
    ev.is_set.return_value = False
    return ev


def test_first_wait_uses_min() -> None:
    b = _AdaptiveBackoff(min_s=0.5, max_s=15.0, factor=2.0)
    assert b.current == pytest.approx(0.5)
    ev = _make_mock_event()
    b.wait(ev)
    ev.wait.assert_called_once_with(pytest.approx(0.5))


def test_doubles_on_consecutive_empty_ticks() -> None:
    b = _AdaptiveBackoff(min_s=0.5, max_s=15.0, factor=2.0)
    ev = _make_mock_event()
    intervals = []
    for _ in range(5):
        intervals.append(b.current)
        b.wait(ev)
    # Starts at 0.5 and doubles.
    assert intervals == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_caps_at_max() -> None:
    b = _AdaptiveBackoff(min_s=0.5, max_s=15.0, factor=2.0)
    ev = _make_mock_event()
    # Push far past the cap.
    for _ in range(30):
        b.wait(ev)
    assert b.current == pytest.approx(15.0)
    # And stays there.
    b.wait(ev)
    assert b.current == pytest.approx(15.0)
    # The wait() call at the cap should also request the cap interval.
    ev.wait.assert_called_with(pytest.approx(15.0))


def test_reset_returns_to_min_on_successful_work_signal() -> None:
    b = _AdaptiveBackoff(min_s=0.5, max_s=15.0, factor=2.0)
    ev = _make_mock_event()
    for _ in range(4):  # advance well past min
        b.wait(ev)
    assert b.current > 0.5
    b.reset()
    assert b.current == pytest.approx(0.5)
    # Next wait uses the reset (min) interval — preserves latency for
    # the "jobs arrived, pick up quickly" case.
    ev.reset_mock()
    b.wait(ev)
    ev.wait.assert_called_once_with(pytest.approx(0.5))


def test_shutdown_set_mid_wait_is_returned() -> None:
    b = _AdaptiveBackoff(min_s=0.1, max_s=1.0, factor=2.0)
    ev = MagicMock(spec=threading.Event)
    ev.wait.return_value = True  # simulate shutdown signalled
    ev.is_set.return_value = True
    assert b.wait(ev) is True


def test_zero_min_is_tolerated() -> None:
    """Used by tests that pass ``poll_interval_s=0.0`` to avoid real sleeps."""
    b = _AdaptiveBackoff(min_s=0.0, max_s=0.0, factor=2.0)
    ev = _make_mock_event()
    # Should not call event.wait with a positive timeout; just peek is_set().
    b.wait(ev)
    ev.wait.assert_not_called()
    ev.is_set.assert_called()


def test_adaptive_wait_helper_delegates_to_state() -> None:
    b = _AdaptiveBackoff(min_s=0.5, max_s=15.0, factor=2.0)
    ev = _make_mock_event()
    _adaptive_wait(ev, b)
    ev.wait.assert_called_once_with(pytest.approx(0.5))
    assert b.current == pytest.approx(1.0)
