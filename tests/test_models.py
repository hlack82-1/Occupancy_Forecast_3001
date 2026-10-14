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


# ---------------------------------------------------------------------------
# Machine-learning models
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from occupancy.models import (  # noqa: E402
    MODEL_KINDS,
    SeedAverage,
    build_model,
    forecast_model,
    train_model,
)


def learnable_table():
    """Weekdays 9am to 5pm busy (3 people), everything else empty, 8 weeks."""
    s = daily_pattern_series("2024-05-06", 56, value_for_day=lambda d: 0.0)
    busy = (s.index.dayofweek < 5) & (s.index.hour >= 9) & (s.index.hour < 17)
    s[busy] = 3.0
    return make_features(s)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_models_learn_a_simple_weekly_pattern(kind):
    t = learnable_table()
    train, test = split_by_date(t, "2024-06-17")
    settings = {"seeds": (0,)} if kind == "mlp" else {}
    model = train_model(kind, train, **settings)
    forecast = forecast_model(kind, model, test)
    error = (forecast - test["headcount"]).abs().mean()
    assert error < 0.5          # always-empty would score 3 * 8 / 24 = 1.0
    assert (forecast >= 0).all()
    assert forecast.index.equals(test.index)


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        build_model("random_forest")


def test_seed_average_is_mean_of_members():
    class Constant:
        def __init__(self, seed):
            self.seed = seed

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), float(self.seed))

    avg = SeedAverage(Constant, seeds=[1, 2, 3]).fit([[0], [0]], [0, 0])
    assert list(avg.predict([[0], [0]])) == [2.0, 2.0]
