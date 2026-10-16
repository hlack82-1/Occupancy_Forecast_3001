"""Measure the speed and memory of the pipeline, and check FLOP estimates.

Usage (from the repository root, with the mma3001 environment active)::

    python benchmark.py

Run ``python main.py`` first (this script reads its cleaned data).
Takes a few minutes. Outputs:

- ``reports/performance.csv``: every timing, memory and FLOP figure
- ``reports/profile_load.txt``: profiler output for the loading step, showing
  where the time goes
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from pathlib import Path

import pandas as pd

from occupancy.data import (
    FLOORSPACES,
    drop_partial_days,
    load_occupancy_events,
    remove_duplicates,
    to_regular_grid,
)
from occupancy.features import make_features, split_by_date
from occupancy.models import train_model
from occupancy.performance import (
    linear_regression_flops,
    matmul_intensity,
    mlp_training_flops,
    mlp_weight_count,
    to_regular_grid_loop,
)

RAW_OCCUPANCY = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
REPORTS = Path("reports")
ROOM = "G25"
TEST_START = "2025-03-03"
CHUNK_SIZES = [100_000, 1_000_000]


def timed(function, *args, track_memory=False, **kwargs):
    """Run a function once; return (result, seconds, peak memory in MB).

    Peak memory uses ``tracemalloc``, which tracks NumPy/Python allocations
    but not memory held by Arrow-backed text columns, and slows the code
    down. So it is only switched on for steps that are pure NumPy/pandas
    arithmetic, and those steps are timed separately without it.
    """
    if track_memory:
        tracemalloc.start()
    tic = time.perf_counter()
    result = function(*args, **kwargs)
    seconds = time.perf_counter() - tic
    peak_mb = None
    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1e6
    return result, seconds, peak_mb


def main():
    """Run every measurement and save the results."""
    REPORTS.mkdir(exist_ok=True)
    rows = []

    def record(step, variant, seconds=None, peak_mb=None, **extra):
        rows.append({"step": step, "variant": variant, "seconds": seconds,
                     "peak_memory_mb": peak_mb, **extra})
        shown = f"{seconds:.2f} s" if seconds is not None else ""
        print(f"{step:28s} {variant:34s} {shown}", flush=True)

    # 1. Loading: effect of chunk size on time and peak memory
    for size in CHUNK_SIZES:
        events, s, _ = timed(load_occupancy_events, RAW_OCCUPANCY,
                             FLOORSPACES[ROOM], chunksize=size)
        record("load CSV", f"chunks of {size:,} rows", s,
               result_mb=events.memory_usage(deep=True).sum() / 1e6,
               file_mb=RAW_OCCUPANCY.stat().st_size / 1e6)

    # Profile the load to see which functions take the time
    profiler = cProfile.Profile()
    profiler.enable()
    load_occupancy_events(RAW_OCCUPANCY, FLOORSPACES[ROOM])
    profiler.disable()
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(15)
    (REPORTS / "profile_load.txt").write_text(text.getvalue())

    # 2. Event-to-grid conversion: Python loop vs vectorised pandas
    events, _ = remove_duplicates(events)
    grid, s_fast, _ = timed(to_regular_grid, events)
    _, _, mb_fast = timed(to_regular_grid, events, track_memory=True)
    record("event to 15-min grid", "vectorised (pandas)", s_fast, mb_fast)
    slow, s_loop, _ = timed(to_regular_grid_loop, events)
    _, _, mb_loop = timed(to_regular_grid_loop, events, track_memory=True)
    record("event to 15-min grid", "Python loop", s_loop, mb_loop,
           speedup=s_loop / s_fast,
           max_difference_people=float((grid - slow).abs().max()))

    # 3. Resolution: 15-minute vs hourly blocks
    for freq in ["15min", "1h"]:
        series, s_grid, _ = timed(to_regular_grid, events, freq=freq)
        series = drop_partial_days(series)
        table, s_feat, _ = timed(make_features, series)
        train, _ = split_by_date(table, TEST_START)
        _, s_fit, _ = timed(train_model, "linear", train)
        record("resolution", f"{freq} blocks: grid", s_grid, blocks=len(series))
        record("resolution", f"{freq} blocks: features", s_feat)
        record("resolution", f"{freq} blocks: train linear", s_fit, rows=len(train))

    # 4. Training: measured time vs rough FLOP estimate
    table = make_features(drop_partial_days(grid))
    train, _ = split_by_date(table, TEST_START)
    n = len(train)

    linear, s, _ = timed(train_model, "linear", train)
    p = linear[-1].coef_.size  # inputs after scaling and one-hot encoding
    flops = linear_regression_flops(n, p)
    record("train", "linear regression", s, rows=n, inputs=p,
           est_flops=flops, achieved_gflops=flops / s / 1e9)

    mlp, s, _ = timed(train_model, "mlp", train, hidden_layers=(64, 32), seeds=(0,))
    network = mlp.models_[0][-1]
    epochs = network.n_iter_
    flops = mlp_training_flops(n, p, (64, 32), epochs)
    record("train", "neural network (64, 32), 1 seed", s, rows=n, inputs=p,
           weights=mlp_weight_count(p, (64, 32)), epochs=epochs,
           est_flops=flops, achieved_gflops=flops / s / 1e9)

    # Arithmetic intensity of the network's biggest layer for one mini-batch
    batch = min(200, n)  # scikit-learn's default batch size
    record("arithmetic intensity", f"batch {batch} x {p} inputs -> 64 neurons",
           intensity_flops_per_byte=matmul_intensity(batch, p, 64))

    pd.DataFrame(rows).to_csv(REPORTS / "performance.csv", index=False)
    print(f"\nSaved {REPORTS / 'performance.csv'} and {REPORTS / 'profile_load.txt'}")


if __name__ == "__main__":
    main()
