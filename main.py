"""Run the full occupancy forecasting pipeline from raw data to results.

Usage (from the repository root, with the mma3001 environment active)::

    python main.py

Each stage is added here as it is built. Outputs:

- ``reports/audit_<room>.csv``: data-quality counts
- ``data/processed/<room>_15min.parquet``: clean headcount series (not committed)
- ``reports/baseline_results.csv``: baseline scores on the whole test period
- ``reports/baseline_results_by_period.csv``: the same, split by year and
  academic period
- ``reports/validation_results.csv``: every model setting tried on the
  validation window (used to pick one setting per model)
- ``reports/model_results.csv``: baselines and the three ML models on the
  test period, with training and prediction times
- ``reports/model_results_by_period.csv``: the same, by year and period
"""

import time
from pathlib import Path

import pandas as pd

from occupancy.data import (
    FLOORSPACES,
    audit_events,
    drop_partial_days,
    load_occupancy_events,
    remove_duplicates,
    to_regular_grid,
)
from occupancy.evaluate import score_table
from occupancy.features import PERIODS, make_features, split_by_date
from occupancy.models import (
    CANDIDATES,
    fit_weekly_profile,
    forecast_model,
    train_model,
    forecast_empty,
    forecast_fixed_profile,
    forecast_last_week,
    forecast_rolling_profile,
)

RAW_OCCUPANCY = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

ROOM = "G25"
FREQ = "15min"          # grid block length
DROPOUT_LIMIT = "1h"    # longest a non-zero reading is trusted without a new one
AGG = "mean"            # time-weighted average headcount per block
TEST_START = "2025-03-03"  # train on the first 12 months, test on the rest
VALIDATION_START = "2024-09-02"  # last 6 months of the training year pick settings


def build_clean_series():
    """Stage 1: load, audit, clean and resample. Returns the 15-minute series."""
    events = load_occupancy_events(RAW_OCCUPANCY, FLOORSPACES[ROOM])
    print(f"Loaded {len(events):,} raw events for {ROOM}")

    audit = audit_events(events, dropout_limit=DROPOUT_LIMIT)
    audit.to_csv(REPORTS / f"audit_{ROOM}.csv", header=["value"])
    print("\nData audit:")
    print(audit.to_string())

    events, n_dupes = remove_duplicates(events)
    series = to_regular_grid(events, freq=FREQ, dropout_limit=DROPOUT_LIMIT, agg=AGG)
    series = drop_partial_days(series)
    out_path = PROCESSED / f"{ROOM}_{FREQ}.parquet"
    series.to_frame().to_parquet(out_path)
    print(f"\nRemoved {n_dupes} duplicate rows")
    print(f"Saved {len(series):,} {FREQ} blocks to {out_path}")
    print(f"From {series.index.min()} to {series.index.max()}")
    return series


def run_baselines(series):
    """Stage 2: features, date split and baseline forecasts. Returns the scores."""
    table = make_features(series)
    train, test = split_by_date(table, TEST_START)
    print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} "
          f"({len(train):,} blocks)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} "
          f"({len(test):,} blocks)")

    forecasts = baseline_forecasts(train, test)
    results = score_table(test["headcount"], forecasts)
    results.to_csv(REPORTS / "baseline_results.csv")
    scores_by_group(test, forecasts).to_csv(
        REPORTS / "baseline_results_by_period.csv", index=False)

    print("\nBaseline results on the test period:")
    print(results.round(3).to_string())
    return table, train, test


def baseline_forecasts(train, test):
    """The four baseline forecasts for the test rows."""
    return {
        "always_empty": forecast_empty(test),
        "last_week": forecast_last_week(test),
        "fixed_profile": forecast_fixed_profile(fit_weekly_profile(train), test),
        "rolling_profile": forecast_rolling_profile(test),
    }


def scores_by_group(test, forecasts):
    """Score every forecast separately for each year and academic period."""
    period = test[[f"period_{p}" for p in PERIODS]].idxmax(axis=1).str[len("period_"):]
    groups = {f"year_{y}": test.index.year == y for y in sorted(set(test.index.year))}
    groups.update({f"period_{p}": (period == p).to_numpy() for p in PERIODS})
    parts = []
    for name, mask in groups.items():
        part = score_table(test["headcount"][mask],
                           {k: v[mask] for k, v in forecasts.items()})
        parts.append(part.assign(group=name))
    return pd.concat(parts).rename_axis("method").reset_index()


def choose_settings(train):
    """Stage 3a: try every candidate setting on the validation window.

    The training year is split again: models learn on the first 6 months
    and are scored on the last 6. The best setting per model kind (lowest
    MAE) is kept. The test period is not touched here.
    """
    fit_part, validation = split_by_date(train, VALIDATION_START)
    forecasts = {"rolling_profile": forecast_rolling_profile(validation)}
    for kind, options in CANDIDATES.items():
        for name, settings in options.items():
            model = train_model(kind, fit_part, **settings)
            forecasts[name] = forecast_model(kind, model, validation)
    results = score_table(validation["headcount"], forecasts)
    results.to_csv(REPORTS / "validation_results.csv")

    chosen = {}
    for kind, options in CANDIDATES.items():
        best = results.loc[list(options), "mae"].idxmin()
        chosen[kind] = (best, options[best])
    print(f"\nValidation ({validation.index.min().date()} to "
          f"{validation.index.max().date()}):")
    print(results.round(3).drop(columns="blocks_scored").to_string())
    print("Chosen:", {kind: name for kind, (name, _) in chosen.items()})
    return chosen


def run_models(train, test, chosen):
    """Stage 3b: retrain the chosen settings on the whole training year and
    score them once on the test period, next to the baselines."""
    forecasts = baseline_forecasts(train, test)
    timing = {}
    for kind, (name, settings) in chosen.items():
        start = time.perf_counter()
        model = train_model(kind, train, **settings)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        forecasts[name] = forecast_model(kind, model, test)
        timing[name] = {"train_seconds": fit_seconds,
                        "predict_seconds": time.perf_counter() - start}

    results = score_table(test["headcount"], forecasts)
    results = results.join(pd.DataFrame(timing).T)
    results.to_csv(REPORTS / "model_results.csv")
    scores_by_group(test, forecasts).to_csv(
        REPORTS / "model_results_by_period.csv", index=False)

    print("\nAll methods on the test period:")
    print(results.round(3).drop(columns="blocks_scored").to_string())
    return results


def main():
    """Run each pipeline stage in order and print a short summary."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    series = build_clean_series()
    _, train, test = run_baselines(series)
    chosen = choose_settings(train)
    run_models(train, test, chosen)


if __name__ == "__main__":
    main()
