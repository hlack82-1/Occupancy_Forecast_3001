"""Tests for occupancy.features."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from occupancy.data import LOCAL_TZ
from occupancy.features import (
    PERIODS,
    academic_period,
    make_features,
    split_by_date,
)


def daily_pattern_series(start, days, value_for_day=lambda d: 1.0):
    """15-minute series where every block of day d has value_for_day(d)."""
    index = pd.date_range(start, periods=days * 96, freq="15min", tz=LOCAL_TZ)
    day_number = (index - index[0]).days
    values = [value_for_day(d) for d in day_number]
    return pd.Series(values, index=index, name="headcount")


def test_lags_use_only_previous_days():
    # Day d has headcount d, so lags are easy to check.
    s = daily_pattern_series("2024-05-06", 40, value_for_day=float)
    t = make_features(s)
    day35 = t[t.index.date == date(2024, 6, 10)]  # day 35
    assert (day35["lag_1d"] == 34).all()
    assert (day35["lag_7d"] == 28).all()
    assert (day35["mean_4wk"] == (28 + 21 + 14 + 7) / 4).all()
    assert (day35["mean_prev_day"] == 34).all()
    assert (day35["mean_prev_7d"] == np.mean(range(28, 35))).all()


def test_lags_missing_at_start():
    s = daily_pattern_series("2024-05-06", 10)
    t = make_features(s)
    assert t["lag_1d"].iloc[:96].isna().all()
    assert t["mean_4wk"].isna().all()  # needs 28 days of history


def test_calendar_columns():
    s = daily_pattern_series("2024-04-24", 3)  # Wed 24, Thu 25 (Anzac Day), Fri 26
    t = make_features(s)
    anzac = t[t.index.date == date(2024, 4, 25)]
    assert (anzac["is_holiday"] == 1).all()
    assert (anzac["weekday"] == 3).all()
    assert (t["block"].between(0, 95)).all()
    period_cols = [f"period_{p}" for p in PERIODS]
    assert (t[period_cols].sum(axis=1) == 1).all()  # exactly one period each


def test_lag_follows_local_clock_across_dst():
    # Clocks go back 7 Apr 2024. Mark 9:00am on 3 Apr; a week later the
    # 9:00am lag on 10 Apr must point to it, not to 8:00am or 10:00am.
    s = daily_pattern_series("2024-04-01", 14, value_for_day=lambda d: 0.0)
    s[pd.Timestamp("2024-04-03 09:00", tz=LOCAL_TZ)] = 5.0
    t = make_features(s)
    assert t.loc[pd.Timestamp("2024-04-10 09:00", tz=LOCAL_TZ), "lag_7d"] == 5.0
    assert t.loc[pd.Timestamp("2024-04-10 10:00", tz=LOCAL_TZ), "lag_7d"] == 0.0


def test_academic_period_examples():
    assert academic_period(date(2025, 4, 22)) == "midsem_break"
    assert academic_period(date(2025, 3, 3)) == "teaching"
    assert academic_period(date(2025, 6, 10)) == "swot_exams"
    assert academic_period(date(2025, 1, 15)) == "break"


def test_split_by_date():
    t = make_features(daily_pattern_series("2024-05-06", 10))
    train, test = split_by_date(t, "2024-05-11")
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(t)
    assert str(test.index.min().date()) == "2024-05-11"
    with pytest.raises(ValueError):
        split_by_date(t, "2030-01-01")
