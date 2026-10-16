"""Data stage: load, audit, clean and resample the occupancy sensor data.

The occupancy file holds about 9 million event rows for five floorspaces.
Each row is written when a sensor reports a reading, so the rows are
irregularly spaced in time. Functions here turn those raw events into a
clean headcount series on a regular time grid.
"""

from pathlib import Path

import pandas as pd

#: Time zone of the building. The raw timestamps carry a +10 or +11 offset.
LOCAL_TZ = "Australia/Melbourne"

#: Floorspace IDs for the rooms used in this project (from the location sheet).
FLOORSPACES = {
    "G25": "227f1068-9e41-469f-bef0-275f27a0bdba",  # building 60, makerspace
    "G20": "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",  # building 60, informal breakout
}

#: The three statuses the sensor can report. Any other value is stored as
#: missing (NaN) so the audit step can count it.
STATUSES = ["NotOccupied", "RecentlyOccupied", "CurrentlyOccupied"]

# Only these columns are needed, so the others are never loaded into memory.
_USECOLS = ["id", "floorspaceid", "occupancystatus", "headcount", "collecteddate"]
_DTYPES = {
    "id": "int64",
    "floorspaceid": "string",
    "occupancystatus": "string",
    "headcount": "int16",
    "collecteddate": "string",
}


def load_occupancy_events(csv_path, floorspace_id, chunksize=1_000_000):
    """Read the raw occupancy events for one floorspace.

    The file is read in chunks so the full 1.6 GB file never has to sit
    in memory at once. Each chunk is filtered to the requested floorspace
    before being kept.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to ``5occupancySensor_MayToDec2024_9MRows.csv``.
    floorspace_id : str
        Floorspace ID to keep, e.g. ``FLOORSPACES["G25"]``.
    chunksize : int, optional
        Number of rows read per chunk. Larger is faster but uses more memory.

    Returns
    -------
    pandas.DataFrame
        One row per raw event, sorted by time, with columns:

        - ``timestamp``: reading time, tz-aware in Australia/Melbourne
        - ``headcount``: people detected (int, 0 or more)
        - ``status``: one of ``STATUSES`` (NaN if the raw value was unknown)
        - ``id``: original row ID, kept so duplicates can be traced

        Duplicates are NOT removed here, so they can be counted in the audit.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the file has no rows for ``floorspace_id``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Occupancy file not found: {path}")

    kept = []
    reader = pd.read_csv(path, usecols=_USECOLS, dtype=_DTYPES, chunksize=chunksize)
    for chunk in reader:
        kept.append(chunk[chunk["floorspaceid"] == floorspace_id])

    events = pd.concat(kept, ignore_index=True)
    if events.empty:
        raise ValueError(f"No rows found for floorspace {floorspace_id}")

    # Strings look like "2024-05-01 10:00:00+10". Parse as UTC first so the
    # +10/+11 daylight-saving offsets are handled correctly, then convert.
    events["timestamp"] = pd.to_datetime(
        events["collecteddate"], utc=True, format="ISO8601"
    ).dt.tz_convert(LOCAL_TZ)

    # Unknown status strings become NaN instead of raising, so the audit
    # step can count them. Fixed categories keep the dtype stable.
    raw_status = events["occupancystatus"]
    events["status"] = pd.Categorical(
        raw_status.where(raw_status.isin(STATUSES)), categories=STATUSES
    )
    events = events.sort_values(["timestamp", "id"]).reset_index(drop=True)
    return events[["timestamp", "headcount", "status", "id"]]


def remove_duplicates(events):
    """Drop events that share a timestamp, keeping the last one.

    Two readings at the same instant can't both be true. The one with the
    higher row ``id`` was written later, so it is kept.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`load_occupancy_events` (sorted by timestamp then id).

    Returns
    -------
    clean : pandas.DataFrame
        Same columns, one row per timestamp.
    n_removed : int
        Number of rows dropped, for the audit table.
    """
    clean = events.drop_duplicates(subset="timestamp", keep="last")
    clean = clean.reset_index(drop=True)
    return clean, len(events) - len(clean)


def audit_events(events, dropout_limit="1h", quiet_day_events=100):
    """Count the data-quality issues in a set of raw events.

    Nothing is changed here. The counts go straight into the data audit
    table in the report, so every cleaning step can be justified.

    Parameters
    ----------
    events : pandas.DataFrame
        Raw events from :func:`load_occupancy_events` (before removing
        duplicates, so duplicates can be counted).
    dropout_limit : str or pandas.Timedelta, optional
        A gap longer than this that starts while people are present is
        counted as a likely sensor dropout.
    quiet_day_events : int, optional
        Days with fewer events than this are counted as "quiet days".

    Returns
    -------
    pandas.Series
        Named counts and values, one per check. Times are in hours.
    """
    limit = pd.Timedelta(dropout_limit)
    t = events["timestamp"]
    headcount = events["headcount"]
    status = events["status"]

    gap_after = t.shift(-1) - t  # how long each reading stands before the next
    long_gap = gap_after > limit
    occupied = headcount > 0

    # Share of the 1am to 5am window with someone detected, weighted by how
    # long each reading lasted (not by number of readings).
    seconds = gap_after.dt.total_seconds().fillna(0)
    night = t.dt.hour.isin([1, 2, 3, 4])
    night_seconds = seconds[night].sum()
    night_share = (
        (seconds[night] * occupied[night]).sum() / night_seconds
        if night_seconds > 0 else float("nan")
    )

    events_per_day = t.dt.date.value_counts()

    return pd.Series({
        "rows": len(events),
        "first_timestamp": t.min(),
        "last_timestamp": t.max(),
        "duplicate_timestamps": int(t.duplicated().sum()),
        "unknown_status": int(status.isna().sum()),
        "negative_headcount": int((headcount < 0).sum()),
        "status_headcount_mismatch": int(
            ((status == "CurrentlyOccupied") & ~occupied).sum()
            + ((status != "CurrentlyOccupied") & status.notna() & occupied).sum()
        ),
        "max_headcount": int(headcount.max()),
        "gaps_over_limit": int(long_gap.sum()),
        "gaps_over_limit_after_empty": int((long_gap & ~occupied).sum()),
        "likely_dropouts": int((long_gap & occupied).sum()),
        # A zero reading lasting a minute or less between two non-zero
        # readings is probably sensor flicker, not people leaving.
        "zero_blips_under_1min": int(
            (~occupied & occupied.shift(1, fill_value=False)
             & occupied.shift(-1, fill_value=False)
             & (gap_after <= pd.Timedelta("1min"))).sum()
        ),
        "longest_gap_hours": gap_after.max().total_seconds() / 3600,
        "days": len(events_per_day),
        "quiet_days": int((events_per_day < quiet_day_events).sum()),
        "night_share_occupied": round(float(night_share), 3),
    })


def to_regular_grid(events, freq="15min", dropout_limit="1h", agg="mean"):
    """Convert irregular events into headcount on a regular time grid.

    Concept: between two readings the headcount is assumed to stay at the
    last reported value (sample-and-hold), so the true signal is a step
    function. Each grid block then gets either

    - ``"mean"``: the time-weighted average of that step function, i.e. its
      exact integral over the block divided by the block length, or
    - ``"max"``: the highest headcount reached in the block.

    Linear interpolation is not used because it would invent fractional
    people between readings.

    Dropouts: if a non-zero reading is followed by a gap longer than
    ``dropout_limit``, the value is only held for ``dropout_limit`` and then
    set to 0. Zero readings are held for any length of gap, because the
    sensor reports little when a room is empty.

    The work is done in UTC so daylight-saving changes cannot create missing
    or repeated blocks. The result is converted back to local time.

    Parameters
    ----------
    events : pandas.DataFrame
        Events with ``timestamp`` (tz-aware) and ``headcount`` columns,
        one row per timestamp (run :func:`remove_duplicates` first).
    freq : str, optional
        Block length, e.g. ``"15min"`` or ``"1h"``. Must divide one hour.
    dropout_limit : str or pandas.Timedelta, optional
        Longest a non-zero reading is trusted without a new reading.
    agg : {"mean", "max"}, optional
        How to summarise each block.

    Returns
    -------
    pandas.Series
        Headcount per block (float), indexed by the block start time in
        Australia/Melbourne, named ``"headcount"``.

    Raises
    ------
    ValueError
        If ``events`` is empty, ``agg`` is not recognised, or timestamps
        repeat.
    """
    if agg not in ("mean", "max"):
        raise ValueError(f"agg must be 'mean' or 'max', got {agg!r}")
    if events.empty:
        raise ValueError("events is empty")
    if events["timestamp"].duplicated().any():
        raise ValueError("timestamps repeat; run remove_duplicates first")

    step = pd.Timedelta(freq)
    limit = pd.Timedelta(dropout_limit)
    t = events["timestamp"].dt.tz_convert("UTC").reset_index(drop=True)
    value = events["headcount"].astype(float).reset_index(drop=True)

    start = t.iloc[0].floor(step)
    end = t.iloc[-1].ceil(step)
    if end == t.iloc[-1]:
        end += step  # make sure the last reading falls inside a block

    # Points where the step function changes value:
    # 1. every reading
    # 2. the moment a held non-zero reading expires (dropout)
    next_t = t.shift(-1).fillna(end)
    expires = (value > 0) & (next_t - t > limit)
    changes = pd.concat([
        pd.DataFrame({"time": t, "value": value}),
        pd.DataFrame({"time": t[expires] + limit, "value": 0.0}),
    ])

    # Add every block edge so no segment crosses a block boundary. The
    # value at an edge is whatever was being held there (filled below).
    edges = pd.DataFrame({"time": pd.date_range(start, end, freq=step)})
    segments = (
        pd.concat([changes, edges])
        .sort_values("time", kind="stable")
        .drop_duplicates("time", keep="first")
        .reset_index(drop=True)
    )
    segments["value"] = segments["value"].ffill()  # sample-and-hold
    segments["duration"] = (
        segments["time"].shift(-1) - segments["time"]
    ).dt.total_seconds()
    segments = segments[segments["duration"] > 0]  # drop the final edge
    segments = segments[segments["time"] >= t.iloc[0]]  # nothing before first reading
    block = segments["time"].dt.floor(step)

    if agg == "mean":
        weighted = (segments["value"] * segments["duration"]).groupby(block).sum()
        covered = segments["duration"].groupby(block).sum()
        result = weighted / covered
    else:
        result = segments["value"].groupby(block).max()

    result.index = result.index.tz_convert(LOCAL_TZ)
    result.index.name = "block_start"
    return result.rename("headcount")


def drop_partial_days(series):
    """Remove the first and last local days, which are usually incomplete.

    Recording starts and stops part-way through a day, so those two days
    would look artificially quiet. Days in between are complete because
    :func:`to_regular_grid` produces a continuous series.

    Parameters
    ----------
    series : pandas.Series
        Regular series indexed by tz-aware local time.

    Returns
    -------
    pandas.Series
        The series without its first and last calendar days.
    """
    dates = series.index.date
    keep = (dates != dates.min()) & (dates != dates.max())
    return series[keep]


def zero_overnight(series, start_hour=1, end_hour=5):
    """Set every block from ``start_hour`` up to ``end_hour`` (local) to 0.

    Used only for the sensitivity check on the assumption that overnight
    detections are real people. If they are actually sensor false counts,
    the true overnight headcount would be 0.

    Parameters
    ----------
    series : pandas.Series
        Regular headcount series with a tz-aware local index.
    start_hour, end_hour : int, optional
        Hours of the day to clear, start included and end excluded.

    Returns
    -------
    pandas.Series
        A copy with those blocks set to 0.
    """
    hours = series.index.hour
    night = (hours >= start_hour) & (hours < end_hour)
    return series.where(~night, 0.0)


# ---------------------------------------------------------------------------
# Environmental sensors (used only for the CO2 cross-check)
# ---------------------------------------------------------------------------

#: CO2 sensor with usable data. Location sheet: building 60, room G.38
#: (Compactus_upper), a different room from G.25. Data Feb to Jun 2024.
CO2_SENSOR_G38 = "6012002000869"


def load_env_sensor(csv_path, sensor_id, variable="Carbon dioxide", chunksize=50_000):
    """Read one variable from one environmental sensor.

    Each row of the environmental CSV stores its readings as a JSON list,
    for example ``[{"values": [{"time": 1702273965, "value": 603}],
    "variable": {"name": "Carbon dioxide", "unit": "ppm"}}, ...]``. Times
    are Unix seconds (UTC). Many rows are exact duplicates, and some JSON
    entries have no value, so those are skipped.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to ``5EnvSensor_MayToDec2024_180kRows.csv``.
    sensor_id : str
        Sensor ID, e.g. ``CO2_SENSOR_G38``.
    variable : str, optional
        Variable name inside the JSON, e.g. ``"Carbon dioxide"`` (ppm) or
        ``"Temperature"`` (deg C).
    chunksize : int, optional
        Rows read at a time.

    Returns
    -------
    pandas.Series
        One value per reading time, sorted, tz-aware local index, duplicates
        removed. Named after ``variable``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If no readings of ``variable`` are found for ``sensor_id``.
    """
    import json  # only needed here

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Environmental file not found: {path}")

    times, values = [], []
    reader = pd.read_csv(path, usecols=["sensorid", "jsondata"],
                         dtype=str, chunksize=chunksize)
    for chunk in reader:
        for text in chunk.loc[chunk["sensorid"] == sensor_id, "jsondata"]:
            try:
                readings = json.loads(text)
            except (TypeError, ValueError):
                continue  # broken JSON row: skip it
            for item in readings:
                if item.get("variable", {}).get("name") != variable:
                    continue
                for reading in item.get("values", []):
                    if "value" in reading and "time" in reading:
                        times.append(reading["time"])
                        values.append(float(reading["value"]))

    if not times:
        raise ValueError(f"No '{variable}' readings for sensor {sensor_id}")
    index = pd.to_datetime(times, unit="s", utc=True).tz_convert(LOCAL_TZ)
    series = pd.Series(values, index=index, name=variable)
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.index.name = "time"
    return series
