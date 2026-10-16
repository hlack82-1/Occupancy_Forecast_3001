"""Evaluation stage: error metrics for headcount forecasts.

Two kinds of measure:

- **Accuracy** in people per 15-minute block: MAE, RMSE, bias, daily peak
  error.
- **Engineering consequence**, in person-hours per day. A forecast that is
  too low means fresh air is under-supplied (stuffy room). A forecast that
  is too high means air is conditioned for people who aren't there (wasted
  energy). Splitting the error this way shows which side a method errs on.

One 15-minute block is 0.25 hours, so 1 person missed for one block is
0.25 person-hours. The block length is read from the data, so hourly
blocks work too.
"""

import numpy as np
import pandas as pd

def block_hours(index):
    """Length of one block in hours, from the most common gap in the index."""
    if len(index) < 2:
        return 0.25
    return pd.Series(index).diff().mode().iloc[0].total_seconds() / 3600


def score(actual, forecast):
    """Compare a forecast with the measured headcount.

    Blocks where either value is missing are left out, and the number of
    blocks actually scored is reported so methods can be compared fairly.

    Parameters
    ----------
    actual : pandas.Series
        Measured headcount per block (people).
    forecast : pandas.Series
        Forecast headcount per block, same index as ``actual``.

    Returns
    -------
    pandas.Series
        - ``mae``: mean absolute error (people)
        - ``rmse``: root mean squared error (people); punishes big misses more
        - ``bias``: mean of forecast minus actual (people); positive means
          the method over-forecasts on average
        - ``peak_error``: mean over days of |forecast daily max - actual
          daily max| (people); how well the busiest moment is predicted
        - ``under_supply_ph_per_day``: person-hours per day the forecast
          was below the actual headcount
        - ``over_supply_ph_per_day``: person-hours per day the forecast was
          above the actual headcount
        - ``blocks_scored``: number of blocks compared

    Raises
    ------
    ValueError
        If the indexes differ or no block has both values.
    """
    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must share the same index")
    both = actual.notna() & forecast.notna()
    if not both.any():
        raise ValueError("no blocks have both an actual and a forecast value")
    a = actual[both].astype(float)
    f = forecast[both].astype(float)
    error = f - a
    hours = block_hours(actual.index)

    days = a.index.date
    n_days = len(set(days))
    daily_peak_gap = (f.groupby(days).max() - a.groupby(days).max()).abs()

    return pd.Series({
        "mae": error.abs().mean(),
        "rmse": np.sqrt((error ** 2).mean()),
        "bias": error.mean(),
        "peak_error": daily_peak_gap.mean(),
        "under_supply_ph_per_day": (-error).clip(lower=0).sum() * hours / n_days,
        "over_supply_ph_per_day": error.clip(lower=0).sum() * hours / n_days,
        "blocks_scored": int(both.sum()),
    })


def score_table(actual, forecasts):
    """Score several forecasts on exactly the same blocks.

    Only blocks where every forecast has a value are used, so no method
    gets an easier set of blocks than another.

    Parameters
    ----------
    actual : pandas.Series
        Measured headcount.
    forecasts : dict of str to pandas.Series
        Method name to forecast.

    Returns
    -------
    pandas.DataFrame
        One row per method, columns as in :func:`score`.
    """
    common = actual.notna()
    for f in forecasts.values():
        common &= f.notna()
    rows = {name: score(actual[common], f[common]) for name, f in forecasts.items()}
    return pd.DataFrame(rows).T
