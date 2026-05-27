"""Algorithmic helpers — strength decay and load scoring (GREEN phase).

All functions are pure: no side-effects, no database dependency.
"""

from __future__ import annotations

import math
from datetime import datetime

IN_HOURS: float = 1.0
"""Semantic constant: one hour expressed in hours (identity factor)."""


def current_strength(
    base_stability: float,
    last_review: str,
    review_count: int,
    now: str,
    initial_half_life: float = 24.0,
    alpha: float = 0.2,
) -> float:
    """Compute edge strength after exponential decay since last review.

    Parameters
    ----------
    base_stability:
        Stability value at the time of the most recent review [0, 1].
    last_review:
        ISO-format timestamp of the most recent review.
    review_count:
        Total number of reviews the edge has received.
    now:
        ISO-format timestamp of the current time.
    initial_half_life:
        Half-life in hours when *review_count* is 0.
    alpha:
        How much each additional review extends the half-life.

    Returns
    -------
    float
        Decayed strength, clamped to [0, 1].
    """
    # ------------------------------------------------------------------
    # Half-life grows with each review (spacing effect).
    # ------------------------------------------------------------------
    half_life = initial_half_life * (1.0 + alpha * review_count)

    # ------------------------------------------------------------------
    # Elapsed time (dt) in hours.
    # ------------------------------------------------------------------
    dt = (
        datetime.fromisoformat(now) - datetime.fromisoformat(last_review)
    ).total_seconds() / 3600.0

    # ------------------------------------------------------------------
    # Exponential decay.
    # ------------------------------------------------------------------
    if dt < 0.0:
        dt = 0.0

    strength = base_stability * math.exp(-dt / half_life)

    # ------------------------------------------------------------------
    # Clamp to legal range [0, 1].
    # ------------------------------------------------------------------
    if strength < 0.0:
        strength = 0.0
    elif strength > 1.0:
        strength = 1.0

    return strength


def compute_load_score(
    coreness: float,
    relevance: float,
    avg_outgoing_strength: float,
    beta: float = 0.4,
    gamma: float = 0.4,
    delta: float = 0.2,
) -> float:
    """Compute a load/priority score for a node.

    The score is a weighted linear combination of three factors:

        score = coreness * beta + relevance * gamma + avg_outgoing_strength * delta

    Parameters
    ----------
    coreness:
        Intrinsic importance of the node [0, 1].
    relevance:
        Query-level relevance score [0, 1].
    avg_outgoing_strength:
        Mean current_strength of outgoing edges [0, 1].
    beta, gamma, delta:
        Weights for each component.  The caller is responsible for
        ensuring they sum to 1.0 (not enforced).

    Returns
    -------
    float
    """
    return (
        coreness * beta + relevance * gamma + avg_outgoing_strength * delta
    )
