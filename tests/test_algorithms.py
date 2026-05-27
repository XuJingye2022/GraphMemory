"""Tests for algorithmic helpers — strength decay and load scoring (RED phase).

All functions under test are pure: no side-effects, no database dependency.
"""

from __future__ import annotations

import math

import pytest

from graph_memory.algorithms import (
    IN_HOURS,
    compute_load_score,
    current_strength,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hours_ago(hours: float) -> str:
    """Return ISO-format timestamp *hours* before 2026-05-27T00:00:00."""
    from datetime import datetime, timedelta

    base = datetime(2026, 5, 27, 0, 0, 0)
    return (base - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# TestCurrentStrength
# ---------------------------------------------------------------------------


class TestCurrentStrength:
    """current_strength() — exponential decay with review-dependent half-life."""

    def test_no_decay_at_zero_hours(self):
        """dt=0 -> strength == base_stability."""
        t = "2026-05-27T00:00:00"
        assert current_strength(0.5, t, 0, t) == pytest.approx(0.5, rel=1e-4)

    def test_decay_after_one_half_life(self):
        """dt == half_life -> strength == base_stability * e^{-1}."""
        t0 = _hours_ago(24)  # 24 hours ago
        t1 = "2026-05-27T00:00:00"
        # review_count=0 -> half_life = 24.0 * (1 + 0.2*0) = 24.0
        # dt = 24 h -> strength = 0.5 * exp(-24/24) = 0.5 * exp(-1)
        expected = 0.5 * math.exp(-1)
        assert current_strength(0.5, t0, 0, t1) == pytest.approx(expected, rel=1e-4)

    def test_decay_after_two_half_lives(self):
        """dt == 2 * half_life -> strength == base_stability * e^{-2}."""
        t0 = _hours_ago(48)
        t1 = "2026-05-27T00:00:00"
        expected = 0.5 * math.exp(-2)
        assert current_strength(0.5, t0, 0, t1) == pytest.approx(expected, rel=1e-4)

    def test_review_count_slows_decay(self):
        """Higher review_count -> longer half_life -> slower decay."""
        t0 = _hours_ago(24)
        t1 = "2026-05-27T00:00:00"
        s0 = current_strength(0.5, t0, 0, t1)  # half_life = 24
        s3 = current_strength(0.5, t0, 3, t1)  # half_life = 24 * (1 + 0.2*3) = 38.4
        assert s3 > s0  # slower decay -> higher remaining strength

    def test_strength_bounded_0_to_1(self):
        """Return value always in [0, 1] regardless of input."""
        t = "2026-05-27T00:00:00"
        # Negative base_stability clamped to 0
        assert current_strength(-0.5, t, 0, t) == pytest.approx(0.0, abs=1e-4)
        # Overly large base_stability clamped to 1
        assert current_strength(1.5, t, 0, t) == pytest.approx(1.0, abs=1e-4)
        # Normal in-range value
        val = current_strength(0.7, t, 2, t)
        assert 0.0 <= val <= 1.0

    def test_current_strength_clamps_to_zero(self):
        """Very large dt -> strength ~0, never negative."""
        t0 = "2024-01-01T00:00:00"
        t1 = "2026-05-27T00:00:00"
        val = current_strength(0.5, t0, 0, t1)
        assert val == pytest.approx(0.0, abs=1e-4)
        assert val >= 0


# ---------------------------------------------------------------------------
# TestComputeLoadScore
# ---------------------------------------------------------------------------


class TestComputeLoadScore:
    """compute_load_score() — weighted linear combination."""

    def test_basic_weighting(self):
        """Verify weighted sum: c*beta + r*gamma + s*delta."""
        score = compute_load_score(
            coreness=0.7,
            relevance=0.5,
            avg_outgoing_strength=0.3,
            beta=0.4,
            gamma=0.4,
            delta=0.2,
        )
        expected = 0.7 * 0.4 + 0.5 * 0.4 + 0.3 * 0.2
        assert score == pytest.approx(expected, rel=1e-4)

    def test_weights_sum_independent(self):
        """Changing one weight does not affect the other terms."""
        s1 = compute_load_score(0.7, 0.5, 0.3, beta=0.4, gamma=0.4, delta=0.2)
        s2 = compute_load_score(0.7, 0.5, 0.3, beta=0.8, gamma=0.4, delta=0.2)
        # Only beta term changed: delta = 0.7 * (0.8 - 0.4) = 0.28
        assert s2 - s1 == pytest.approx(0.7 * 0.4, rel=1e-4)

    def test_uses_default_weights(self):
        """Calling with only the three scores uses default weights."""
        score = compute_load_score(1.0, 0.0, 0.5)
        expected = 1.0 * 0.4 + 0.0 * 0.4 + 0.5 * 0.2
        assert score == pytest.approx(expected, rel=1e-4)
