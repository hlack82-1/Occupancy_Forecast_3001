"""Tests for occupancy.performance."""

import numpy as np
import pandas as pd
import pytest

from occupancy.data import to_regular_grid
from occupancy.performance import (
    linear_regression_flops,
    matmul_intensity,
    mlp_training_flops,
    mlp_weight_count,
    to_regular_grid_loop,
)
from tests.test_data import make_events


def random_events(n=500, seed=0):
    """Irregular events over about two days, crossing a daylight-saving change."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(300, n).clip(1)  # seconds, some long gaps
    gaps[rng.integers(0, n, 5)] = 4 * 3600   # a few dropouts
    times = pd.Timestamp("2024-04-06 12:00", tz="Australia/Melbourne") + pd.to_timedelta(gaps.cumsum(), unit="s")
    counts = rng.choice([0, 0, 1, 2, 3, 5], n)
    return pd.DataFrame({"timestamp": times, "headcount": counts})


def test_loop_matches_vectorised_version():
    events = random_events()
    fast = to_regular_grid(events)
    slow = to_regular_grid_loop(events)
    # Same blocks, and values agree to within a millionth of a person
    # (tiny floating-point differences only)
    assert fast.index.equals(slow.index)
    assert np.allclose(fast.to_numpy(), slow.to_numpy(), rtol=0, atol=1e-6)


def test_loop_matches_on_simple_case():
    events = make_events([
        ("2024-05-01 09:00", 0),
        ("2024-05-01 09:05", 3),
        ("2024-05-01 09:15", 3),
    ])
    assert to_regular_grid_loop(events).iloc[0] == pytest.approx(2.0)


def test_flop_helpers():
    assert linear_regression_flops(100, 10) == 2 * 100 * 100
    # 3 inputs -> 4 hidden -> 1 output: (3*4 + 4) + (4*1 + 1) = 21
    assert mlp_weight_count(3, (4,)) == 21
    assert mlp_training_flops(10, 3, (4,), epochs=2) == 6 * 21 * 10 * 2
    # square matmul intensity grows with size
    assert matmul_intensity(1000, 1000, 1000) > matmul_intensity(10, 10, 10)
