"""Tests for occupancy.evaluate."""

import numpy as np
import pandas as pd
import pytest

from occupancy.evaluate import score, score_table


def make(values, start="2024-05-06 09:00"):
    index = pd.date_range(start, periods=len(values), freq="15min", tz="Australia/Melbourne")
    return pd.Series(values, index=index, dtype=float)


def test_score_known_values():
    actual = make([2, 2, 0, 0])
    forecast = make([3, 1, 0, 2])  # errors +1, -1, 0, +2
    s = score(actual, forecast)
    assert s["mae"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 1 + 0 + 4) / 4))
    assert s["bias"] == pytest.approx(0.5)
    assert s["peak_error"] == pytest.approx(1.0)        # max 3 vs max 2
    assert s["under_supply_ph_per_day"] == pytest.approx(1 * 0.25)
    assert s["over_supply_ph_per_day"] == pytest.approx(3 * 0.25)
    assert s["blocks_scored"] == 4


def test_perfect_forecast_scores_zero():
    actual = make([0, 1, 5, 2])
    s = score(actual, actual.copy())
    assert s[["mae", "rmse", "bias", "peak_error",
              "under_supply_ph_per_day", "over_supply_ph_per_day"]].eq(0).all()


def test_missing_values_are_skipped():
    actual = make([1, np.nan, 3])
    forecast = make([1, 2, np.nan])
    assert score(actual, forecast)["blocks_scored"] == 1


def test_score_rejects_bad_input():
    with pytest.raises(ValueError):
        score(make([1, 2]), make([1, 2], start="2025-01-01"))
    with pytest.raises(ValueError):
        score(make([np.nan]), make([1]))


def test_score_table_uses_common_blocks():
    actual = make([1, 1, 1])
    table = score_table(actual, {"a": make([1, 1, np.nan]), "b": make([2, 2, 2])})
    assert (table["blocks_scored"] == 2).all()
    assert table.loc["a", "mae"] == 0 and table.loc["b", "mae"] == 1


def test_supply_uses_block_length_from_data():
    index = pd.date_range("2024-05-06 09:00", periods=3, freq="1h", tz="Australia/Melbourne")
    actual = pd.Series([2.0, 2.0, 2.0], index=index)
    forecast = pd.Series([1.0, 2.0, 2.0], index=index)
    # 1 person short for one 1-hour block = 1 person-hour
    assert score(actual, forecast)["under_supply_ph_per_day"] == pytest.approx(1.0)
