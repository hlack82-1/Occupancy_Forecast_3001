"""Tests for occupancy.data."""

import pandas as pd
import pytest

from occupancy.data import LOCAL_TZ, load_occupancy_events

ROOM = "room-a"
OTHER = "room-b"


@pytest.fixture
def small_csv(tmp_path):
    """Write a tiny CSV with the same columns as the real export."""
    rows = [
        # id, floorspace, status, headcount, collecteddate (deliberately out of order)
        (3, ROOM, "CurrentlyOccupied", 2, "2024-04-07 02:30:00+10"),  # after DST ends
        (1, ROOM, "CurrentlyOccupied", 1, "2024-04-06 09:00:00+11"),  # before DST ends
        (2, OTHER, "CurrentlyOccupied", 5, "2024-04-06 09:00:00+11"),
        (4, ROOM, "NotOccupied", 0, "2024-04-07 18:00:00+10"),
    ]
    df = pd.DataFrame(
        rows, columns=["id", "floorspaceid", "occupancystatus", "headcount", "collecteddate"]
    )
    df["deviceid"] = "dev"  # extra column the loader should ignore
    path = tmp_path / "occ.csv"
    df.to_csv(path, index=False)
    return path


def test_keeps_only_requested_floorspace(small_csv):
    events = load_occupancy_events(small_csv, ROOM)
    assert list(events["id"]) == [1, 3, 4]


def test_sorted_by_time(small_csv):
    events = load_occupancy_events(small_csv, ROOM)
    assert events["timestamp"].is_monotonic_increasing


def test_timestamps_are_local_and_tz_aware(small_csv):
    events = load_occupancy_events(small_csv, ROOM)
    assert str(events["timestamp"].dt.tz) == LOCAL_TZ
    # 09:00+11 should stay 09:00 local time, not shift by an hour.
    assert events["timestamp"].iloc[0].hour == 9


def test_chunking_gives_same_result(small_csv):
    whole = load_occupancy_events(small_csv, ROOM, chunksize=100)
    chunked = load_occupancy_events(small_csv, ROOM, chunksize=1)
    pd.testing.assert_frame_equal(whole, chunked)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_occupancy_events(tmp_path / "nope.csv", ROOM)


def test_unknown_floorspace_raises(small_csv):
    with pytest.raises(ValueError):
        load_occupancy_events(small_csv, "not-a-room")


def test_unknown_status_becomes_missing(tmp_path):
    df = pd.DataFrame({
        "id": [1], "floorspaceid": [ROOM], "occupancystatus": ["Garbage"],
        "headcount": [1], "collecteddate": ["2024-04-06 09:00:00+11"],
    })
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    events = load_occupancy_events(path, ROOM)
    assert events["status"].isna().all()


# ---------------------------------------------------------------------------
# Cleaning, audit and resampling
# ---------------------------------------------------------------------------
from occupancy.data import (  # noqa: E402
    audit_events,
    drop_partial_days,
    remove_duplicates,
    to_regular_grid,
)


def make_events(rows):
    """Build an events table from (local time string, headcount) pairs."""
    times = pd.to_datetime([r[0] for r in rows]).tz_localize(LOCAL_TZ)
    counts = [r[1] for r in rows]
    return pd.DataFrame({
        "timestamp": times,
        "headcount": counts,
        "status": ["CurrentlyOccupied" if c > 0 else "NotOccupied" for c in counts],
        "id": range(len(rows)),
    })


def test_remove_duplicates_keeps_last_and_counts():
    events = make_events([
        ("2024-05-01 09:00", 1),
        ("2024-05-01 09:00", 3),  # same time, later id
        ("2024-05-01 09:05", 2),
    ])
    clean, n_removed = remove_duplicates(events)
    assert n_removed == 1
    assert list(clean["headcount"]) == [3, 2]


def test_grid_constant_value():
    events = make_events([("2024-05-01 09:00", 4), ("2024-05-01 09:30", 4)])
    grid = to_regular_grid(events)
    assert (grid == 4).all()


def test_grid_time_weighted_mean_and_max():
    # 0 people for 5 minutes, then 3 people for 10 minutes.
    events = make_events([
        ("2024-05-01 09:00", 0),
        ("2024-05-01 09:05", 3),
        ("2024-05-01 09:15", 3),
    ])
    mean = to_regular_grid(events, agg="mean")
    peak = to_regular_grid(events, agg="max")
    assert mean.iloc[0] == pytest.approx((0 * 5 + 3 * 10) / 15)
    assert peak.iloc[0] == 3


def test_grid_dropout_sets_value_to_zero_after_limit():
    # 2 people, then silence for 3 hours.
    events = make_events([("2024-05-01 09:00", 2), ("2024-05-01 12:00", 0)])
    grid = to_regular_grid(events, dropout_limit="1h")
    assert (grid["2024-05-01 09:00":"2024-05-01 09:45"] == 2).all()
    assert (grid["2024-05-01 10:00":"2024-05-01 11:45"] == 0).all()


def test_grid_zero_is_held_through_long_gap():
    events = make_events([("2024-05-01 18:00", 0), ("2024-05-02 08:00", 1)])
    grid = to_regular_grid(events)
    assert (grid[:"2024-05-02 07:45"] == 0).all()


def test_grid_has_no_missing_or_repeated_blocks_across_dst():
    # Clocks go back at 3am on 7 April 2024, so that local day has 25 hours.
    events = make_events([("2024-04-06 12:00", 0), ("2024-04-08 12:00", 0)])
    grid = to_regular_grid(events)
    gaps = grid.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta("15min")).all()
    day = grid[grid.index.date == pd.Timestamp("2024-04-07").date()]
    assert len(day) == 25 * 4


def test_grid_rejects_bad_input():
    events = make_events([("2024-05-01 09:00", 1), ("2024-05-01 09:00", 2)])
    with pytest.raises(ValueError):
        to_regular_grid(events)  # duplicate timestamps
    with pytest.raises(ValueError):
        to_regular_grid(events.iloc[:1], agg="median")
    with pytest.raises(ValueError):
        to_regular_grid(events.iloc[:0])


def test_drop_partial_days_removes_first_and_last_day():
    events = make_events([("2024-05-01 15:00", 0), ("2024-05-04 10:00", 0)])
    grid = drop_partial_days(to_regular_grid(events))
    days = sorted(set(grid.index.date))
    assert [str(d) for d in days] == ["2024-05-02", "2024-05-03"]


def test_audit_counts_issues():
    events = make_events([
        ("2024-05-01 09:00", 2),
        ("2024-05-01 09:00", 2),  # duplicate
        ("2024-05-01 12:00", 0),  # 3 h gap after people present: dropout
        ("2024-05-01 20:00", 1),  # 8 h gap after empty: normal
        ("2024-05-01 20:10", 0),
    ])
    audit = audit_events(events, dropout_limit="1h")
    assert audit["duplicate_timestamps"] == 1
    assert audit["likely_dropouts"] == 1
    assert audit["gaps_over_limit_after_empty"] == 1
    assert audit["status_headcount_mismatch"] == 0
    assert audit["max_headcount"] == 2


def test_audit_counts_zero_blips():
    events = make_events([
        ("2024-05-01 09:00:00", 3),
        ("2024-05-01 09:00:20", 0),  # 20 s blip between occupied readings
        ("2024-05-01 09:00:40", 3),
        ("2024-05-01 09:10:00", 0),  # genuine drop, lasts 10 min
        ("2024-05-01 09:20:00", 0),
    ])
    assert audit_events(events)["zero_blips_under_1min"] == 1


def test_zero_overnight_only_clears_night_blocks():
    from occupancy.data import zero_overnight
    index = pd.date_range("2024-05-01", periods=96, freq="15min", tz=LOCAL_TZ)
    grid = zero_overnight(pd.Series(2.0, index=index))
    assert (grid[grid.index.hour.isin([1, 2, 3, 4])] == 0).all()
    assert (grid[grid.index.hour.isin([0, 5, 12, 23])] == 2).all()
