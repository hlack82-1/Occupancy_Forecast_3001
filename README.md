# Occupancy_Forecast_3001

Day-ahead occupancy forecasting for a university makerspace, so the air conditioning can be scheduled before people arrive. MMA3001 (Numerical Methods and Machine Learning) individual project, Monash University, Semester 2 2026.

## The problem

Predict the number of people in room G.25 (makerspace, building 60, Monash Clayton) for every 15 minutes of the next day, using past headcount data and calendar information such as time of day, weekday and semester week. The forecast would let the building's air conditioning be scheduled ahead of time, so it isn't wasting electricity on empty rooms but still has the room at a comfortable temperature and fresh-air level when people arrive.

**Input:** about 1.5 million raw occupancy sensor events for G.25 (March 2024 to April 2026), plus public holidays and semester dates.
**Output:** expected people in the room for each 15-minute block of tomorrow, and what that means in fresh-air flow (L/s).

Full details of inputs, outputs and how bad data is handled are on the front page of the HTML documentation (`docs/occupancy.html`).

## Results in brief

Tested day-ahead on 13.5 months the models never saw (3 Mar 2025 to 21 Apr 2026). Error is in people per 15-minute block, lower is better.

| Method | MAE | RMSE |
|---|---|---|
| Always empty (sanity check) | 0.508 | 0.974 |
| Rolling 4-week profile (best baseline) | 0.491 | 0.775 |
| Linear regression, trained once | 0.502 | 0.738 |
| Neural network, trained once | 0.546 | 0.943 |
| **Linear regression, retrained monthly (chosen)** | **0.478** | **0.718** |
| Neural network, retrained monthly | 0.491 | 0.806 |

Use of the room dropped a lot after 2024, so models trained once go out of date. Retraining every month fixes most of that. The best model beats the best simple baseline by about 3% on MAE and 7% on RMSE: a small gain, because the room is empty much of the time. All numbers are in `reports/`, and the figures are made by the notebooks.

## Repository layout

```
Occupancy_Forecast_3001/
├── main.py              Runs the whole pipeline, raw data to results
├── benchmark.py         Speed, memory and FLOP measurements
├── occupancy/           The code, one module per stage
│   ├── data.py          Load, audit, clean, resample to 15-minute blocks
│   ├── features.py      Calendar and past-headcount inputs, date split
│   ├── models.py        Baselines, linear regression, decision tree, neural network
│   ├── evaluate.py      Error metrics, airflow conversion, CO2 cross-check
│   └── performance.py   Slow reference version and FLOP estimates
├── tests/               pytest unit tests (49 tests)
├── notebooks/
│   ├── 01_data_audit.ipynb   What the raw data looks like and what cleaning does
│   └── 02_results.ipynb      Result figures and tables for the report
├── reports/             Result tables (CSV), figures, test report
├── docs/                HTML documentation (made by pdoc)
├── data/                Raw data goes here (not committed, see data/README.md)
├── environment.yml      Conda environment
├── pytest.ini           Test settings
└── LICENSE              MIT licence
```

## Setup

Needs [Anaconda](https://www.anaconda.com/download) or Miniconda. On Windows, use **Anaconda Prompt** for the commands below.

```
git clone https://github.com/hlack82-1/Occupancy_Forecast_3001.git
cd Occupancy_Forecast_3001
conda env create -f environment.yml
conda activate mma3001
```

This installs Python 3.12, NumPy, pandas, pyarrow, scikit-learn, matplotlib, pytest, pytest-html, pdoc and Jupyter.

Then put the raw data files in `data/raw/` (see `data/README.md` for the exact file names and where to get them).

## How to run

Run everything from the repository root with the `mma3001` environment active.

| Command | What it does | Time (laptop) |
|---|---|---|
| `python main.py` | Full pipeline: cleaning, baselines, model selection, models, sensitivity, airflow, CO2 check. Writes `data/processed/` and `reports/*.csv` | about 15 to 20 min |
| `python main.py --quick` | Same, but skips monthly retraining of the neural network (the slow part) | about 2 min |
| `python benchmark.py` | Speed and memory tests. Run `main.py` first | a few min |
| `pytest` | Runs the 49 unit tests (no raw data needed) | a few seconds |

The airflow results use a minimum outdoor-air rate of 10 L/s per person, set as `LITRES_PER_PERSON` at the top of `main.py`.

### Notebooks

Open with `jupyter lab` (or VS Code started from Anaconda Prompt with `code .`) and run all cells, after `main.py` has run.

- `01_data_audit.ipynb`: example week of raw vs cleaned data, average day, drop in use over time, CO2 cross-check (figures 1 to 4)
- `02_results.ipynb`: method comparison, one test week of forecasts, error by period, fresh-air trade-off (figures 5 to 8)

Figures are saved to `reports/figures/`.

## Tests

```
pytest
```

Each test runs a function on a small made-up input where the right answer is known by hand, and checks the output matches. They cover cleaning, resampling and daylight saving, features (including a check that no future data leaks into a forecast), models, metrics and the speed-comparison code.

To regenerate the HTML test report (`reports/test_report.html`):

```
pytest --html=reports/test_report.html --self-contained-html
```

## Documentation

HTML documentation is generated from the docstrings (NumPy style) with pdoc. Open `docs/index.html` in a browser. To regenerate it after changing the code:

```
pdoc -d numpy occupancy -o docs
```

## Data and assumptions

- The data is Dataset 2 (Monash Smart Infrastructure occupancy and environmental data) provided by the MMA3001 teaching team. It is not included here; see `data/README.md`.
- Headcounts detected overnight are treated as real.
- A non-zero reading with no update for over 1 hour is treated as a sensor dropout (the room is assumed empty after 1 hour).
- 2024 semester dates and the end of semester 2 2025 are estimated from the pattern of official dates.

## Use of AI

Generative AI (Claude) was used to help write and debug code, explain methods and draft documentation. All use is disclosed in the AI reflection section of the report.

## Licence

MIT, see `LICENSE`.
