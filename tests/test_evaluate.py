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


# ---------------------------------------------------------------------------
# Airflow and CO2 cross-check
# ---------------------------------------------------------------------------
from occupancy.evaluate import airflow_summary, co2_crosscheck, headcount_to_airflow  # noqa: E402


def test_headcount_to_airflow():
    assert headcount_to_airflow(3, 10.0) == 30.0
    with pytest.raises(ValueError):
        headcount_to_airflow(3, 0)
    with pytest.raises(ValueError):
        headcount_to_airflow(3, None)


def test_airflow_summary_known_values():
    # One day of 15-minute blocks: 2 people for 4 blocks, forecast 1 person.
    actual = make([2, 2, 2, 2])
    forecast = make([1, 1, 1, 1])
    s = airflow_summary(actual, forecast, litres_per_person=10.0)
    assert s["mean_required_ls"] == pytest.approx(20.0)
    assert s["peak_required_ls"] == pytest.approx(20.0)
    # short 10 L/s for 4 x 900 s = 36,000 L = 36 m3 in one day
    assert s["under_supply_m3_per_day"] == pytest.approx(36.0)
    assert s["over_supply_m3_per_day"] == 0
    assert s["blocks_under_by_1_person"] == 1.0


def test_co2_crosscheck_finds_matching_pattern():
    index = pd.date_range("2024-05-06", periods=14 * 96, freq="15min", tz="Australia/Melbourne")
    busy = (index.dayofweek < 5) & (index.hour >= 9) & (index.hour < 17)
    headcount = pd.Series(np.where(busy, 3.0, 0.0), index=index)
    co2 = pd.Series(420 + 30 * headcount.to_numpy(), index=index)  # CO2 follows people
    s = co2_crosscheck(headcount, co2)
    assert s["corr_weekday_hourly_profile"] == pytest.approx(1.0)
    assert s["corr_hourly_lag0h"] == pytest.approx(1.0)
    assert s["overlap_days"] == 14


def test_co2_crosscheck_needs_overlap():
    a = make([1, 2, 3], start="2024-05-06 09:00")
    b = make([400, 410, 420], start="2025-05-06 09:00")
    with pytest.raises(ValueError):
        co2_crosscheck(a, b)
