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

All forecasts are day-ahead: they only use data up to the end of the day
before the day being forecast (see ``occupancy.features``).
"""

import pandas as pd


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
