"""Models stage: baseline forecasts and the machine-learning model.

Baselines set the bar the ML model has to beat. Each one is a simple rule
someone could apply without any machine learning:

- **Same time last week:** tomorrow at 10:00 will look like last week's
  same weekday at 10:00.
- **Fixed weekly profile:** the average for that weekday and time over the
  whole training period. It never updates, so it shows what happens when
  room use changes after training.
- **Rolling 4-week profile:** the average for that weekday and time over
  the previous 4 weeks. Like the fixed profile, but it keeps up with change.

There is also an "always empty" forecast (0 people everywhere). It is not a
real method, but it is a sanity check: because the room is often empty, a
method has to beat it clearly on error to be worth anything.

Then three machine-learning models (linear regression, decision tree,
neural network), described further down.

All forecasts are day-ahead: they only use data up to the end of the day
before the day being forecast (see ``occupancy.features``).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def forecast_empty(table):
    """Forecast 0 people for every block (sanity-check reference).

    Parameters
    ----------
    table : pandas.DataFrame
        Rows to forecast.

    Returns
    -------
    pandas.Series
        Zeros, one per row.
    """
    return pd.Series(0.0, index=table.index, name="always_empty")


def forecast_last_week(table):
    """Same-time-last-week forecast.

    Parameters
    ----------
    table : pandas.DataFrame
        Feature table from ``occupancy.features.make_features`` (any rows).

    Returns
    -------
    pandas.Series
        Forecast headcount per block (NaN where last week is missing).
    """
    return table["lag_7d"].rename("last_week")


def fit_weekly_profile(train):
    """Average headcount for each (weekday, block) over the training rows.

    Parameters
    ----------
    train : pandas.DataFrame
        Training part of the feature table.

    Returns
    -------
    pandas.Series
        Average headcount indexed by (weekday, block): 7 x 96 values.
    """
    return train.groupby(["weekday", "block"])["headcount"].mean()


def forecast_fixed_profile(profile, table):
    """Forecast every block with the training-period weekly profile.

    Parameters
    ----------
    profile : pandas.Series
        Output of :func:`fit_weekly_profile`.
    table : pandas.DataFrame
        Rows to forecast.

    Returns
    -------
    pandas.Series
        Forecast headcount per block.
    """
    keys = pd.MultiIndex.from_arrays([table["weekday"], table["block"]])
    values = profile.reindex(keys).to_numpy()
    return pd.Series(values, index=table.index, name="fixed_profile")


def forecast_rolling_profile(table):
    """Average of the same weekday and block over the previous 4 weeks.

    Parameters
    ----------
    table : pandas.DataFrame
        Feature table (needs the ``mean_4wk`` column).

    Returns
    -------
    pandas.Series
        Forecast headcount per block (NaN in the first 4 weeks of data).
    """
    return table["mean_4wk"].rename("rolling_profile")


# ---------------------------------------------------------------------------
# Machine-learning models
# ---------------------------------------------------------------------------
#
# Three models from the unit, in increasing complexity:
#
# - Linear regression: adds up a weight for each input. Fast and easy to
#   explain, but can't capture effects that combine (e.g. "10am on a
#   teaching Tuesday").
# - Decision tree: learns yes/no rules ("is it a weekday? is last week's
#   value above 2?"), so it can capture combined effects. Readable, but a
#   single tree is unstable.
# - Neural network (MLP): layers of weighted sums with a non-linear step in
#   between. The most flexible, but needs scaled inputs, has random starting
#   weights, and can be unstable on small data. Trained with a Poisson loss,
#   which suits counts that can't go below zero, and averaged over several
#   random seeds to reduce run-to-run variation.

#: Inputs shared by all three models (lags, recent level and calendar flags).
_COMMON = [
    "is_weekend", "is_holiday",
    "period_teaching", "period_midsem_break", "period_swot_exams", "period_break",
    "lag_1d", "lag_7d", "mean_4wk", "mean_prev_day", "mean_prev_7d",
]
#: The linear model and MLP see time of day as sin/cos and weekday one-hot,
#: because they treat every input as a number on a straight line.
SMOOTH_FEATURES = ["block_sin", "block_cos"] + _COMMON
#: The tree can split on block and weekday numbers directly.
TREE_FEATURES = ["block", "weekday"] + _COMMON

MODEL_KINDS = ("linear", "tree", "mlp")


def _scaled_inputs():
    """Fill missing lags with the training median, scale, one-hot weekday."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer([
        ("numeric", numeric, SMOOTH_FEATURES),
        ("weekday", OneHotEncoder(handle_unknown="ignore"), ["weekday"]),
    ])


class SeedAverage:
    """Average the predictions of the same model trained with several seeds.

    A neural network starts from random weights, so two training runs give
    different answers. Averaging a few runs makes the result repeatable and
    usually a little more accurate.

    Parameters
    ----------
    make_model : callable
        Function taking a seed (int) and returning an unfitted model.
    seeds : sequence of int
        Random seeds to train with.
    """

    def __init__(self, make_model, seeds):
        self.make_model = make_model
        self.seeds = list(seeds)
        self.models_ = []

    def fit(self, X, y):
        """Train one model per seed."""
        self.models_ = [self.make_model(seed).fit(X, y) for seed in self.seeds]
        return self

    def predict(self, X):
        """Average prediction over all seeds."""
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def build_model(kind, tree_depth=6, hidden_layers=(64, 32), alpha=1e-4,
                seeds=(0, 1, 2, 3, 4)):
    """Create an untrained model.

    Parameters
    ----------
    kind : {"linear", "tree", "mlp"}
        Which model.
    tree_depth : int, optional
        Maximum number of yes/no questions in a row (tree only).
    hidden_layers : tuple of int, optional
        Neurons per hidden layer (MLP only).
    alpha : float, optional
        Weight penalty that discourages over-fitting (MLP only).
    seeds : sequence of int, optional
        Random seeds to average over (MLP only).

    Returns
    -------
    object
        A model with ``fit(X, y)`` and ``predict(X)``.

    Raises
    ------
    ValueError
        If ``kind`` is not recognised.
    """
    if kind == "linear":
        return make_pipeline(_scaled_inputs(), LinearRegression())
    if kind == "tree":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=tree_depth, random_state=0),
        )
    if kind == "mlp":
        def one_network(seed):
            return make_pipeline(_scaled_inputs(), MLPRegressor(
                hidden_layer_sizes=hidden_layers, loss="poisson", alpha=alpha,
                early_stopping=True, max_iter=500, random_state=seed))
        return SeedAverage(one_network, seeds)
    raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")


def model_inputs(kind, table):
    """Select the columns a model of this kind uses."""
    return table[TREE_FEATURES] if kind == "tree" else table


def train_model(kind, train, **settings):
    """Build and train a model on the training rows.

    Parameters
    ----------
    kind : {"linear", "tree", "mlp"}
    train : pandas.DataFrame
        Training part of the feature table (must include ``headcount``).
    **settings
        Passed to :func:`build_model`.

    Returns
    -------
    object
        The trained model.
    """
    model = build_model(kind, **settings)
    return model.fit(model_inputs(kind, train), train["headcount"])


def forecast_model(kind, model, table):
    """Forecast headcount with a trained model.

    Negative predictions (possible for the linear model) are raised to 0,
    because a room can't hold fewer than zero people.

    Parameters
    ----------
    kind : {"linear", "tree", "mlp"}
    model : object
        Output of :func:`train_model`.
    table : pandas.DataFrame
        Rows to forecast.

    Returns
    -------
    pandas.Series
        Forecast headcount per block, never below 0.
    """
    predicted = model.predict(model_inputs(kind, table))
    return pd.Series(np.clip(predicted, 0, None), index=table.index, name=kind)


#: Settings tried for each model on the validation window. One setting per
#: kind is picked there, before the test period is ever looked at.
CANDIDATES = {
    "linear": {"linear": {}},
    "tree": {
        "tree_depth4": {"tree_depth": 4},
        "tree_depth6": {"tree_depth": 6},
        "tree_depth8": {"tree_depth": 8},
        "tree_depth10": {"tree_depth": 10},
    },
    "mlp": {
        "mlp_32": {"hidden_layers": (32,)},
        "mlp_64_32": {"hidden_layers": (64, 32)},
        "mlp_64_32_alpha0.01": {"hidden_layers": (64, 32), "alpha": 1e-2},
    },
}
