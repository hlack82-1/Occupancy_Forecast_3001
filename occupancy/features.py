"""Features stage: calendar and lag features, and the train/test split.

Turns the regular headcount series from ``occupancy.data`` into one table
with a row per 15-minute block and the inputs the forecasting models use.

Forecasting set-up (day-ahead): every block of day D is forecast at the end
of day D-1. So a feature may only use headcounts from day D-1 or earlier.
All lag features below respect that rule.

Lags are taken by **local clock time**, not by a fixed number of hours.
"Same time last week" for 9:00am means 9:00am seven calendar days earlier,
even if daylight saving changed in between.
"""

from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Calendar data
# ---------------------------------------------------------------------------

#: Victorian public holidays, from Business Victoria
#: (business.vic.gov.au, "Public holidays in Victoria for 2024/2025/2026").
PUBLIC_HOLIDAYS = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 26), date(2024, 3, 11), date(2024, 3, 29),
    date(2024, 3, 30), date(2024, 3, 31), date(2024, 4, 1), date(2024, 4, 25),
    date(2024, 6, 10), date(2024, 9, 27), date(2024, 11, 5), date(2024, 12, 25),
    date(2024, 12, 26),
    # 2025
    date(2025, 1, 1), date(2025, 1, 27), date(2025, 3, 10), date(2025, 4, 18),
    date(2025, 4, 19), date(2025, 4, 20), date(2025, 4, 21), date(2025, 4, 25),
    date(2025, 6, 9), date(2025, 9, 26), date(2025, 11, 4), date(2025, 12, 25),
    date(2025, 12, 26),
    # 2026
    date(2026, 1, 1), date(2026, 1, 26), date(2026, 3, 9), date(2026, 4, 3),
    date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 6), date(2026, 4, 25),
    date(2026, 6, 8), date(2026, 9, 25), date(2026, 11, 3), date(2026, 12, 25),
    date(2026, 12, 26), date(2026, 12, 28),
}

#: Monash Clayton standard semesters (S1-01 and S2-01) as
#: (first day, last day, period name). Any date not covered is a "break"
#: (summer, winter, between semesters). Weekends next to a period take the
#: label of the period they sit between.
#: Sources:
#:   2024 and 2025: Monash University "Important dates" academic calendars
#:     for 2024 and 2025 (official; teaching, mid-semester break, swot vac
#:     and final assessment dates).
#:   2026 S1: Monash study abroad "Key dates" page (official).
ACADEMIC_PERIODS = [
    # 2024 semester 1 (official)
    (date(2024, 2, 26), date(2024, 3, 31), "teaching"),
    (date(2024, 4, 1), date(2024, 4, 5), "midsem_break"),
    (date(2024, 4, 6), date(2024, 5, 24), "teaching"),
    (date(2024, 5, 25), date(2024, 6, 21), "swot_exams"),
    # 2024 semester 2 (official)
    (date(2024, 7, 22), date(2024, 9, 22), "teaching"),
    (date(2024, 9, 23), date(2024, 9, 27), "midsem_break"),
    (date(2024, 9, 28), date(2024, 10, 18), "teaching"),
    (date(2024, 10, 19), date(2024, 11, 15), "swot_exams"),
    # 2025 semester 1 (official)
    (date(2025, 3, 3), date(2025, 4, 20), "teaching"),
    (date(2025, 4, 21), date(2025, 4, 25), "midsem_break"),
    (date(2025, 4, 26), date(2025, 5, 30), "teaching"),
    (date(2025, 5, 31), date(2025, 6, 27), "swot_exams"),
    # 2025 semester 2 (official)
    (date(2025, 7, 28), date(2025, 9, 28), "teaching"),
    (date(2025, 9, 29), date(2025, 10, 3), "midsem_break"),
    (date(2025, 10, 4), date(2025, 10, 24), "teaching"),
    (date(2025, 10, 25), date(2025, 11, 19), "swot_exams"),
    # 2026 semester 1 (official)
    (date(2026, 3, 2), date(2026, 4, 5), "teaching"),
    (date(2026, 4, 6), date(2026, 4, 10), "midsem_break"),
    (date(2026, 4, 11), date(2026, 5, 29), "teaching"),
    (date(2026, 5, 30), date(2026, 6, 26), "swot_exams"),
]

#: All period names, in a fixed order so one-hot columns are always the same.
PERIODS = ["teaching", "midsem_break", "swot_exams", "break"]

BLOCKS_PER_DAY = 96  # 15-minute blocks


def academic_period(day):
    """Return the academic period name for one calendar date.

    Parameters
    ----------
    day : datetime.date

    Returns
    -------
    str
        One of ``PERIODS``. Dates outside every listed semester are "break".
    """
    for first, last, name in ACADEMIC_PERIODS:
        if first <= day <= last:
            return name
    return "break"


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------

def _daily_matrix(series):
    """Reshape the series into one row per local date and one column per block.

    Block number = position in the local day (0 = 00:00, 95 = 23:45).
    On the day clocks go back, the repeated hour is averaged. On the day
    clocks go forward, the skipped hour is left empty (NaN). Every calendar
    date between the first and last is present, so shifting rows by n gives
    exactly n days earlier.
    """
    local = series.index
    frame = pd.DataFrame({
        "date": local.date,
        "block": local.hour * 4 + local.minute // 15,
        "value": series.to_numpy(),
    })
    matrix = frame.pivot_table(index="date", columns="block", values="value",
                               aggfunc="mean")
    all_dates = pd.date_range(min(matrix.index), max(matrix.index), freq="D").date
    return matrix.reindex(index=all_dates, columns=range(BLOCKS_PER_DAY))


def make_features(series):
    """Build the model input table, one row per 15-minute block.

    Every feature for a block on day D uses only information known by the
    end of day D-1, so it is a fair day-ahead forecast input.

    Parameters
    ----------
    series : pandas.Series
        Regular 15-minute headcount series with a tz-aware local index
        (output of ``occupancy.data.to_regular_grid``).

    Returns
    -------
    pandas.DataFrame
        Indexed like ``series``. Columns:

        - ``headcount``: the target (actual people, time-weighted average)
        - ``block``: position in the day, 0 to 95
        - ``block_sin``, ``block_cos``: time of day as a point on a circle,
          so 23:45 and 00:00 come out close together (used by the linear
          model and neural network, which can't see that ``block`` wraps)
        - ``weekday``: 0 = Monday to 6 = Sunday
        - ``is_weekend``, ``is_holiday``: 0 or 1
        - ``period_<name>``: one 0/1 column per entry in ``PERIODS``
        - ``lag_1d``: same block yesterday
        - ``lag_7d``: same block, same weekday last week
        - ``mean_4wk``: average of the same block on the same weekday over the
          previous 4 weeks (a recent weekly profile that adapts to change)
        - ``mean_prev_day``: average headcount over all of yesterday
        - ``mean_prev_7d``: average headcount over the previous 7 days
          (the recent level of use)

        Lag columns are NaN where the history doesn't exist yet (the first
        4 weeks) or fell in the skipped daylight-saving hour.
    """
    matrix = _daily_matrix(series)
    daily_mean = matrix.mean(axis=1)

    lags = {
        "lag_1d": matrix.shift(1),
        "lag_7d": matrix.shift(7),
        "mean_4wk": (matrix.shift(7) + matrix.shift(14)
                     + matrix.shift(21) + matrix.shift(28)) / 4,
    }

    local = series.index
    days = local.date
    blocks = local.hour * 4 + local.minute // 15
    row = matrix.index.get_indexer(days)  # which matrix row each block uses

    table = pd.DataFrame(index=series.index)
    table["headcount"] = series.to_numpy()
    table["block"] = blocks
    angle = 2 * np.pi * blocks / BLOCKS_PER_DAY
    table["block_sin"] = np.sin(angle)
    table["block_cos"] = np.cos(angle)
    table["weekday"] = local.dayofweek
    table["is_weekend"] = (local.dayofweek >= 5).astype(int)
    table["is_holiday"] = np.isin(days, list(PUBLIC_HOLIDAYS)).astype(int)

    periods = pd.Series([academic_period(d) for d in days], index=series.index)
    for name in PERIODS:
        table[f"period_{name}"] = (periods == name).astype(int)

    for name, lagged in lags.items():
        table[name] = lagged.to_numpy()[row, blocks]

    table["mean_prev_day"] = daily_mean.shift(1).to_numpy()[row]
    table["mean_prev_7d"] = daily_mean.rolling(7).mean().shift(1).to_numpy()[row]
    return table


def feature_columns(table):
    """Return the names of the input columns in a ``make_features`` table."""
    return [c for c in table.columns if c != "headcount"]


def split_by_date(table, test_start):
    """Split a feature table into training and test parts by date.

    Never random: a random split would put future blocks in the training
    set and make the forecast look better than it could be in real use.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`make_features`.
    test_start : str or pandas.Timestamp
        First local date of the test period, e.g. ``"2025-03-03"``.

    Returns
    -------
    train, test : pandas.DataFrame
        Rows before ``test_start`` and rows from ``test_start`` on.

    Raises
    ------
    ValueError
        If either part would be empty.
    """
    start = pd.Timestamp(test_start)
    if start.tz is None:
        start = start.tz_localize(table.index.tz)
    train = table[table.index < start]
    test = table[table.index >= start]
    if train.empty or test.empty:
        raise ValueError(f"test_start {test_start} leaves an empty train or test set")
    return train, test
