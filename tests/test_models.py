"""Tests for occupancy.models baselines."""

import pandas as pd

from occupancy.features import make_features, split_by_date
from occupancy.models import (
    fit_weekly_profile,
    forecast_empty,
    forecast_fixed_profile,
    forecast_last_week,
    forecast_rolling_profile,
)
from tests.test_features import daily_pattern_series


def test_fixed_profile_is_training_average_per_weekday_and_block():
    # Weekdays 2 people, weekends 0, so the profile must reproduce that.
    s = daily_pattern_series("2024-05-06", 21)  # starts on a Monday
    s[s.index.dayofweek >= 5] = 0.0
    s[s.index.dayofweek < 5] = 2.0
    t = make_features(s)
    train, test = split_by_date(t, "2024-05-20")
    forecast = forecast_fixed_profile(fit_weekly_profile(train), test)
    assert (forecast[test["weekday"] < 5] == 2.0).all()
    assert (forecast[test["weekday"] >= 5] == 0.0).all()


def test_fixed_profile_does_not_adapt_but_rolling_does():
    # Use drops from 4 people to 1 person half-way through.
    s = daily_pattern_series("2024-05-06", 70, value_for_day=lambda d: 4.0 if d < 35 else 1.0)
    t = make_features(s)
    train, test = split_by_date(t, "2024-06-10")  # day 35
    fixed = forecast_fixed_profile(fit_weekly_profile(train), test)
    rolling = forecast_rolling_profile(test)
    late = test.index >= pd.Timestamp("2024-07-10", tz=test.index.tz)
    assert (fixed[late] == 4.0).all()      # still forecasting the old level
    assert (rolling[late] == 1.0).all()    # caught up after 4 weeks


def test_last_week_equals_lag_7d():
    t = make_features(daily_pattern_series("2024-05-06", 14, value_for_day=float))
    assert forecast_last_week(t).equals(t["lag_7d"].rename("last_week"))


def test_empty_forecast_is_all_zero():
    t = make_features(daily_pattern_series("2024-05-06", 3))
    assert (forecast_empty(t) == 0).all()
