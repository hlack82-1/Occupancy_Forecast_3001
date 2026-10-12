"""Run the full occupancy forecasting pipeline from raw data to results.

Usage (from the repository root, with the mma3001 environment active)::

    python main.py

Each stage is added here as it is built. Outputs:

- ``reports/audit_<room>.csv``: data-quality counts for the report
- ``data/processed/<room>_15min.parquet``: clean headcount series (not committed)
"""

from pathlib import Path

from occupancy.data import (
    FLOORSPACES,
    audit_events,
    drop_partial_days,
    load_occupancy_events,
    remove_duplicates,
    to_regular_grid,
)

RAW_OCCUPANCY = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

ROOM = "G25"
FREQ = "15min"          # grid block length
DROPOUT_LIMIT = "1h"    # longest a non-zero reading is trusted without a new one
AGG = "mean"            # time-weighted average headcount per block


def main():
    """Run each pipeline stage in order and print a short summary."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # 1. Load raw events for one room
    events = load_occupancy_events(RAW_OCCUPANCY, FLOORSPACES[ROOM])
    print(f"Loaded {len(events):,} raw events for {ROOM}")

    # 2. Audit before changing anything
    audit = audit_events(events, dropout_limit=DROPOUT_LIMIT)
    audit.to_csv(REPORTS / f"audit_{ROOM}.csv", header=["value"])
    print("\nData audit:")
    print(audit.to_string())

    # 3. Clean and convert to a regular grid
    events, n_dupes = remove_duplicates(events)
    series = to_regular_grid(events, freq=FREQ, dropout_limit=DROPOUT_LIMIT, agg=AGG)
    series = drop_partial_days(series)
    out_path = PROCESSED / f"{ROOM}_{FREQ}.parquet"
    series.to_frame().to_parquet(out_path)
    print(f"\nRemoved {n_dupes} duplicate rows")
    print(f"Saved {len(series):,} {FREQ} blocks to {out_path}")
    print(f"From {series.index.min()} to {series.index.max()}")


if __name__ == "__main__":
    main()
