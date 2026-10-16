"""Day-ahead occupancy forecasting for room G.25 (building 60, Monash Clayton).

The package forecasts how many people will be in the makerspace for every
15 minutes of the next day, so the air conditioning and fresh-air supply can
be scheduled ahead of time instead of running for an empty room.

The whole pipeline is run by ``main.py`` in the repository root. This
package holds the logic, split into one module per stage:

| Stage | Module | What it does |
|---|---|---|
| 1. Data | `occupancy.data` | Load the raw sensor events, audit them, remove duplicates, turn them into a regular 15-minute headcount series |
| 2. Features | `occupancy.features` | Add calendar inputs (time of day, weekday, holidays, semester period) and past headcounts; split by date |
| 3. Models | `occupancy.models` | Four simple baselines and three ML models (linear regression, decision tree, neural network), trained once or retrained monthly |
| 4. Evaluation | `occupancy.evaluate` | Error metrics, conversion to fresh-air flow, CO2 cross-check |
| Performance | `occupancy.performance` | Slow reference version of the grid step and FLOP estimates, used by ``benchmark.py`` |

## Top-level inputs

| Input | Where it comes from |
|---|---|
| Occupancy sensor events | ``data/raw/5occupancySensor_MayToDec2024_9MRows.csv`` |
| CO2 readings (cross-check only) | ``data/raw/5EnvSensor_MayToDec2024_180kRows.csv`` |
| Public holidays and semester dates | Written into `occupancy.features` |
| Settings | Constants at the top of ``main.py`` |

**Occupancy file:** CSV, one row per sensor event (a reading reported by a
sensor, not a person or a visit), about 9.1 million rows for 5 rooms, of
which about 1.5 million are for G.25. Columns used:

- ``headcount``: people detected, whole number, 0 or more
- ``collecteddate``: timestamp as text with a +10 or +11 offset (Melbourne
  standard or daylight time)
- ``floorspaceid``: room ID (`occupancy.data.FLOORSPACES`)
- ``occupancystatus``: ``NotOccupied``, ``RecentlyOccupied`` or
  ``CurrentlyOccupied``
- ``id``: row number, used to trace duplicates

**Environmental file:** CSV with each reading stored as JSON text. Only
CO2 (ppm) from the sensor in room G.38 is used.

**Settings in ``main.py``:** room (``"G25"``), block length (15 minutes),
dropout limit (1 hour), block value (time-weighted mean), test start date
(3 March 2025), validation start date (2 September 2024) and the
outdoor-air rate in L/s per person.

The raw data is not in the repository (see ``data/README.md``).

## Top-level outputs

| Output | Meaning and use |
|---|---|
| ``data/processed/G25_15min.parquet`` | Clean headcount series: time-weighted average people per 15-minute block, Melbourne time. Input to every later step |
| ``reports/audit_G25.csv`` | Data-quality counts (duplicates, likely dropouts, flicker) that justify each cleaning choice |
| ``reports/baseline_results.csv``, ``reports/model_results.csv`` (and ``_by_period``) | Error of every method on the test period, in people per block, plus under/over-supply in person-hours per day. Used to choose the method |
| ``reports/validation_results.csv`` | Model settings tried on the validation window, used to pick one setting per model |
| ``reports/sensitivity_results.csv`` | How the results change when each cleaning assumption is changed |
| ``reports/airflow_results.csv`` | Each forecast as outdoor-air flow (L/s, m³/day short or wasted). Shows the engineering consequence of forecast errors |
| ``reports/co2_crosscheck.csv`` | Correlation between G.25 headcount and CO2 in a nearby room, an independent check that the headcount is real |
| ``reports/performance.csv`` | Run time, memory and FLOP figures from ``benchmark.py`` |

A forecast itself is a ``pandas.Series`` of expected people per block
(a decimal, never below 0), indexed by block start time in Melbourne time.

## How invalid, missing or unsupported inputs are handled

- **Missing file or unknown room:** ``FileNotFoundError`` / ``ValueError``
  with a message naming the file or room.
- **Unknown status text:** stored as missing and counted in the audit.
- **Duplicate events:** removed (`occupancy.data.remove_duplicates`); the
  grid step refuses repeated timestamps.
- **Gaps in the data:** a non-zero reading is only trusted for 1 hour
  without a new reading, then treated as 0 (likely sensor dropout). Days
  only partly covered at the start and end of the data are dropped.
- **Daylight saving:** worked in UTC internally; past-headcount inputs are
  matched by local clock time.
- **Missing past headcounts** (first 4 weeks, skipped daylight-saving hour):
  left as NaN in the feature table. The ML models fill them with the
  training median; baseline forecasts stay NaN there, and blocks without
  both a forecast and an actual value are skipped when scoring.
- **Bad settings** (unknown model kind, aggregation, negative airflow rate,
  test date that leaves no training data): ``ValueError``.
- **Negative forecasts:** clipped to 0 people.

Every function's own inputs, outputs and errors are in its docstring
(NumPy style) on the module pages.
"""
