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


# ---------------------------------------------------------------------------
# Airflow: turning headcount into required fresh air
# ---------------------------------------------------------------------------

def headcount_to_airflow(headcount, litres_per_person):
    """Required outdoor (fresh) air for a headcount.

    Ventilation standards set a minimum outdoor-air rate per person, so the
    required airflow is simply headcount x rate.

    Parameters
    ----------
    headcount : pandas.Series or float
        People (actual or forecast).
    litres_per_person : float
        Outdoor-air rate in L/s per person (from AS 1668.2 for this type of
        space).

    Returns
    -------
    pandas.Series or float
        Required outdoor air in L/s.

    Raises
    ------
    ValueError
        If the rate is not positive.
    """
    if not litres_per_person or litres_per_person <= 0:
        raise ValueError("litres_per_person must be a positive number of L/s")
    return headcount * litres_per_person


def airflow_summary(actual, forecast, litres_per_person):
    """What a forecast would mean for fresh-air supply if the system followed it.

    Assumes the ventilation delivers exactly the forecast airflow in each
    block. Under-supply is fresh air the room needed but didn't get (comfort
    and air-quality risk); over-supply is air delivered for people who
    weren't there (wasted fan and conditioning energy).

    Parameters
    ----------
    actual, forecast : pandas.Series
        Measured and forecast headcount per block, same index.
    litres_per_person : float
        Outdoor-air rate in L/s per person.

    Returns
    -------
    pandas.Series
        - ``mean_required_ls``: average fresh air the room actually needed (L/s)
        - ``peak_required_ls``: highest block-average need (L/s)
        - ``under_supply_m3_per_day``: fresh air short per day (cubic metres)
        - ``over_supply_m3_per_day``: excess fresh air per day (cubic metres)
        - ``blocks_under_by_1_person``: share of blocks where supply fell
          short by at least one person's worth of air
    """
    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must share the same index")
    both = actual.notna() & forecast.notna()
    a, f = actual[both], forecast[both]
    need = headcount_to_airflow(a, litres_per_person)
    supply = headcount_to_airflow(f, litres_per_person)
    seconds_per_block = block_hours(a.index) * 3600
    n_days = len(set(a.index.date))
    to_m3_per_day = seconds_per_block / 1000 / n_days  # L/s per block -> m3/day
    return pd.Series({
        "mean_required_ls": need.mean(),
        "peak_required_ls": need.max(),
        "under_supply_m3_per_day": (need - supply).clip(lower=0).sum() * to_m3_per_day,
        "over_supply_m3_per_day": (supply - need).clip(lower=0).sum() * to_m3_per_day,
        "blocks_under_by_1_person": float((a - f >= 1).mean()),
    })


# ---------------------------------------------------------------------------
# Independent check: does CO2 in a nearby room follow the occupancy?
# ---------------------------------------------------------------------------

def co2_crosscheck(headcount, co2, lags_hours=(0, 1, 2)):
    """Compare G.25 headcount with CO2 measured by another sensor.

    People breathe out CO2, so if the occupancy data reflects real use of
    the building, CO2 nearby should rise and fall at the same times. Both
    series are averaged to hours over the period they overlap.

    Parameters
    ----------
    headcount : pandas.Series
        Regular headcount series (tz-aware local index).
    co2 : pandas.Series
        CO2 readings in ppm (tz-aware local index), e.g. from
        ``occupancy.data.load_env_sensor``.
    lags_hours : sequence of int, optional
        Delays to test, with CO2 shifted behind occupancy (CO2 builds up
        after people arrive).

    Returns
    -------
    pandas.Series
        - ``overlap_start``, ``overlap_end``, ``overlap_hours``, ``overlap_days``
        - ``corr_weekday_hourly_profile``: correlation between the average
          weekday pattern (24 hourly values) of each
        - ``corr_daily_mean``: correlation between daily averages
        - ``corr_hourly_lag<k>h``: hour-by-hour correlation with CO2 k hours
          behind occupancy

    Raises
    ------
    ValueError
        If the two series don't overlap.
    """
    hourly = pd.concat({
        "headcount": headcount.resample("1h").mean(),
        "co2": co2.resample("1h").mean(),
    }, axis=1, sort=True).dropna()
    if hourly.empty:
        raise ValueError("headcount and co2 do not overlap in time")

    weekdays = hourly[hourly.index.dayofweek < 5]
    profile = weekdays.groupby(weekdays.index.hour).mean()
    daily = hourly.resample("D").mean().dropna()

    result = {
        "overlap_start": hourly.index.min(),
        "overlap_end": hourly.index.max(),
        "overlap_hours": len(hourly),
        "overlap_days": len(daily),
        "corr_weekday_hourly_profile": profile["headcount"].corr(profile["co2"]),
        "corr_daily_mean": daily["headcount"].corr(daily["co2"]),
    }
    for lag in lags_hours:
        result[f"corr_hourly_lag{lag}h"] = hourly["headcount"].shift(lag).corr(hourly["co2"])
    return pd.Series(result)
