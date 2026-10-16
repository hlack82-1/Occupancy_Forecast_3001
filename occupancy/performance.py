"""Performance stage: a slow reference implementation and FLOP estimates.

Two jobs:

1. ``to_regular_grid_loop`` does the same job as
   ``occupancy.data.to_regular_grid`` with a plain Python loop over events.
   It is the "baseline" for the speed comparison, and because it is written
   completely differently it also acts as an independent check that the
   fast version gives the right answer.
2. Rough floating-point operation (FLOP) counts for training the models, so
   measured times can be checked against how much arithmetic was needed.
   These are order-of-magnitude estimates, not exact counts.
"""

import numpy as np
import pandas as pd


def to_regular_grid_loop(events, freq="15min", dropout_limit="1h"):
    """Time-weighted average headcount per block, using a plain Python loop.

    Same rules as ``occupancy.data.to_regular_grid`` with ``agg="mean"``:
    sample-and-hold between readings, non-zero readings held for at most
    ``dropout_limit``, worked in UTC. Much slower; used for comparison and
    checking only.

    Parameters
    ----------
    events : pandas.DataFrame
        ``timestamp`` (tz-aware) and ``headcount`` columns, sorted, no
        repeated timestamps.
    freq : str, optional
        Block length.
    dropout_limit : str, optional
        Longest a non-zero reading is held without a new reading.

    Returns
    -------
    pandas.Series
        Headcount per block, indexed by block start in local time.
    """
    step = pd.Timedelta(freq).total_seconds()
    limit = pd.Timedelta(dropout_limit).total_seconds()
    tz = events["timestamp"].dt.tz
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    # seconds since 1970 (works whatever time resolution pandas stores)
    seconds = (events["timestamp"].dt.tz_convert("UTC") - epoch).dt.total_seconds().to_numpy()
    counts = events["headcount"].astype(float).to_numpy()

    first_block = np.floor(seconds[0] / step) * step
    last_edge = np.ceil(seconds[-1] / step) * step
    if last_edge == seconds[-1]:
        last_edge += step
    n_blocks = int(round((last_edge - first_block) / step))
    area = np.zeros(n_blocks)      # person-seconds in each block
    covered = np.zeros(n_blocks)   # seconds of known data in each block

    for i in range(len(seconds)):
        t0 = seconds[i]
        t1 = seconds[i + 1] if i + 1 < len(seconds) else last_edge
        value = counts[i]
        if value > 0 and t1 - t0 > limit:
            # held for the dropout limit, then 0 until the next reading
            pieces = [(t0, t0 + limit, value), (t0 + limit, t1, 0.0)]
        else:
            pieces = [(t0, t1, value)]
        for start, end, v in pieces:
            # spread this constant piece over the blocks it touches
            while start < end:
                b = int((start - first_block) // step)
                block_end = first_block + (b + 1) * step
                piece_end = min(end, block_end)
                area[b] += v * (piece_end - start)
                covered[b] += piece_end - start
                start = piece_end

    keep = covered > 0
    starts = pd.to_datetime((first_block + np.arange(n_blocks) * step)[keep], unit="s", utc=True)
    index = pd.DatetimeIndex(starts).tz_convert(tz)
    index.name = "block_start"
    return pd.Series(area[keep] / covered[keep], index=index, name="headcount")


def linear_regression_flops(n_rows, n_inputs):
    """Rough FLOPs to fit least squares on an n_rows x n_inputs matrix.

    Least squares by QR/SVD costs about 2 * n * p^2 operations.
    """
    return 2 * n_rows * n_inputs ** 2


def mlp_weight_count(n_inputs, hidden_layers):
    """Number of weights and biases in a fully connected network (1 output)."""
    sizes = [n_inputs, *hidden_layers, 1]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def mlp_training_flops(n_rows, n_inputs, hidden_layers, epochs):
    """Rough FLOPs to train an MLP.

    Each weight costs about 2 FLOPs per row going forward (multiply and add)
    and about 4 going backward, so roughly 6 per weight per row per epoch.
    """
    return 6 * mlp_weight_count(n_inputs, hidden_layers) * n_rows * epochs


def matmul_intensity(m, k, n, bytes_per_number=8):
    """Arithmetic intensity (FLOPs per byte) of multiplying (m x k) by (k x n).

    FLOPs = 2mkn. Bytes = reading both matrices and writing the result once.
    Used to explain why small neural-network layers run far below the
    processor's peak speed (they are memory-bound on the roofline).
    """
    flops = 2 * m * k * n
    moved = bytes_per_number * (m * k + k * n + m * n)
    return flops / moved
