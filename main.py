"""Run the full occupancy forecasting pipeline from raw data to results.

Usage (from the repository root, with the mma3001 environment active)::

    python main.py           # everything (about 15 to 20 minutes)
    python main.py --quick   # skips monthly retraining of the neural
                             # network, the slow part (about 2 minutes)

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
  (both model tables include the monthly-retrained versions)
- ``reports/sensitivity_results.csv``: how the results change when each
  cleaning assumption is changed
- ``reports/airflow_results.csv``: what each forecast would mean for
  fresh-air supply (needs ``LITRES_PER_PERSON`` set below)
- ``reports/co2_crosscheck.csv``: correlation between G.25 occupancy and CO2
  measured in room G.38
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from occupancy.data import (
    CO2_SENSOR_G38,
    FLOORSPACES,
    audit_events,
    drop_partial_days,
    load_env_sensor,
    load_occupancy_events,
    remove_duplicates,
    to_regular_grid,
    zero_overnight,
)
from occupancy.evaluate import airflow_summary, co2_crosscheck, score_table
from occupancy.features import PERIODS, make_features, split_by_date
from occupancy.models import (
    CANDIDATES,
    fit_weekly_profile,
    forecast_model,
    forecast_monthly_retrain,
    train_model,
    forecast_empty,
    forecast_fixed_profile,
    forecast_last_week,
    forecast_rolling_profile,
)

RAW_OCCUPANCY = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
RAW_ENVIRONMENT = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

ROOM = "G25"
FREQ = "15min"          # grid block length
DROPOUT_LIMIT = "1h"    # longest a non-zero reading is trusted without a new one
AGG = "mean"            # time-weighted average headcount per block
TEST_START = "2025-03-03"  # train on the first 12 months, test on the rest
VALIDATION_START = "2024-09-02"  # last 6 months of the training year pick settings

#: Minimum outdoor-air rate in L/s per person for this type of space, from
#: AS 1668.2. TODO (Harper): look up the value for G.25 (a makerspace /
#: workshop) in AS 1668.2 via the Monash library, enter it here and cite it in
#: the report. Left as None, the airflow stage is skipped.
LITRES_PER_PERSON = 10


def build_clean_series():
    """Stage 1: load, audit, clean and resample. Returns the 15-minute series."""
    events = load_occupancy_events(RAW_OCCUPANCY, FLOORSPACES[ROOM])
    print(f"Loaded {len(events):,} raw events for {ROOM}")

    audit = audit_events(events, dropout_limit=DROPOUT_LIMIT)
    audit.to_csv(REPORTS / f"audit_{ROOM}.csv", header=["value"])
    print("\nData audit:")
    print(audit.to_string())

    events, n_dupes = remove_duplicates(events)  # cleaned events kept for stage 4
    series = to_regular_grid(events, freq=FREQ, dropout_limit=DROPOUT_LIMIT, agg=AGG)
    series = drop_partial_days(series)
    out_path = PROCESSED / f"{ROOM}_{FREQ}.parquet"
    series.to_frame().to_parquet(out_path)
    print(f"\nRemoved {n_dupes} duplicate rows")
    print(f"Saved {len(series):,} {FREQ} blocks to {out_path}")
    print(f"From {series.index.min()} to {series.index.max()}")
    return events, series


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


def run_models(table, train, test, chosen, quick=False):
    """Stage 3b: score the chosen models on the test period, next to the
    baselines, in two ways:

    - trained once on the training year (``<name>``)
    - retrained at the start of every test month on all earlier data
      (``<name>_monthly``), to keep up with the drop in room use

    With ``quick=True`` the neural network is not retrained monthly (it is
    by far the slowest part).
    """
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

    for kind, (name, settings) in chosen.items():
        if quick and kind == "mlp":
            continue
        print(f"Monthly retraining: {name} ...", flush=True)
        forecast, seconds = forecast_monthly_retrain(kind, table, TEST_START, **settings)
        forecasts[f"{name}_monthly"] = forecast.reindex(test.index)
        timing[f"{name}_monthly"] = {"train_seconds": seconds}

    results = score_table(test["headcount"], forecasts)
    results = results.join(pd.DataFrame(timing).T)
    results.to_csv(REPORTS / "model_results.csv")
    scores_by_group(test, forecasts).to_csv(
        REPORTS / "model_results_by_period.csv", index=False)

    print("\nAll methods on the test period:")
    print(results.round(3).drop(columns="blocks_scored").to_string())
    return forecasts


#: Cleaning assumptions to vary, one at a time, from the main settings.
SENSITIVITY_CASES = {
    "main settings (15 min, mean, 1 h dropout)": {},
    "block value = maximum, not average": {"agg": "max"},
    "dropout limit 30 min": {"dropout_limit": "30min"},
    "dropout limit 2 h": {"dropout_limit": "2h"},
    "overnight (1am to 5am) set to 0": {"zero_nights": True},
    "hourly blocks instead of 15 min": {"freq": "1h"},
}


def run_sensitivity(events):
    """Stage 4: change one cleaning assumption at a time and re-score.

    Uses the fast methods only (rolling 4-week profile, linear regression
    trained once and retrained monthly). Each case changes the target data
    as well as the inputs, so the question is whether the conclusions hold,
    not whether one number is lower.
    """
    rows = []
    for case, change in SENSITIVITY_CASES.items():
        freq = change.get("freq", FREQ)
        tic = time.perf_counter()
        series = to_regular_grid(events, freq=freq,
                                 dropout_limit=change.get("dropout_limit", DROPOUT_LIMIT),
                                 agg=change.get("agg", AGG))
        series = drop_partial_days(series)
        if change.get("zero_nights"):
            series = zero_overnight(series)
        grid_seconds = time.perf_counter() - tic

        table = make_features(series)
        train, test = split_by_date(table, TEST_START)
        linear_monthly, _ = forecast_monthly_retrain("linear", table, TEST_START)
        forecasts = {
            "rolling_profile": forecast_rolling_profile(test),
            "linear": forecast_model("linear", train_model("linear", train), test),
            "linear_monthly": linear_monthly.reindex(test.index),
        }
        scores = score_table(test["headcount"], forecasts)
        scores["case"] = case
        scores["test_mean_headcount"] = test["headcount"].mean()
        scores["grid_seconds"] = grid_seconds
        rows.append(scores.rename_axis("method").reset_index())
        print(f"  done: {case}", flush=True)

    results = pd.concat(rows)
    results.to_csv(REPORTS / "sensitivity_results.csv", index=False)
    print("\nSensitivity (MAE by case):")
    print(results.pivot(index="case", columns="method", values="mae")
          .reindex(SENSITIVITY_CASES).round(3).to_string())
    return results


def run_airflow(test, forecasts):
    """Stage 5: convert every forecast into fresh-air terms (L/s and m3/day).

    Skipped if ``LITRES_PER_PERSON`` hasn't been set.
    """
    if LITRES_PER_PERSON is None:
        print("\nAirflow stage skipped: set LITRES_PER_PERSON at the top of main.py "
              "(AS 1668.2 rate for this space).")
        return None
    common = test["headcount"].notna()
    for forecast in forecasts.values():
        common &= forecast.notna()
    rows = {name: airflow_summary(test["headcount"][common], f[common], LITRES_PER_PERSON)
            for name, f in forecasts.items()}
    results = pd.DataFrame(rows).T
    results["litres_per_person"] = LITRES_PER_PERSON
    results.to_csv(REPORTS / "airflow_results.csv")
    print(f"\nAirflow on the test period ({LITRES_PER_PERSON} L/s per person):")
    print(results.drop(columns="litres_per_person").round(2).to_string())
    return results


def run_co2_crosscheck(series):
    """Stage 6: independent check of the occupancy data against CO2.

    CO2 sensor 6012002000869 is in room G.38 (building 60), not G.25, and only
    covers Feb to Jun 2024. So this tests whether the building's rhythm in
    the occupancy data is real, not G.25's exact counts.
    """
    co2 = load_env_sensor(RAW_ENVIRONMENT, CO2_SENSOR_G38)
    result = co2_crosscheck(series, co2)
    result.to_csv(REPORTS / "co2_crosscheck.csv", header=["value"])
    print("\nCO2 cross-check (G.25 occupancy vs CO2 in G.38):")
    print(result.to_string())
    return result


def main(quick=False):
    """Run each pipeline stage in order and print a short summary."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    events, series = build_clean_series()
    table, train, test = run_baselines(series)
    chosen = choose_settings(train)
    forecasts = run_models(table, train, test, chosen, quick=quick)
    print("\nSensitivity checks:")
    run_sensitivity(events)
    run_airflow(test, forecasts)
    run_co2_crosscheck(series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy forecasting pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="skip monthly retraining of the neural network")
    main(quick=parser.parse_args().quick)
