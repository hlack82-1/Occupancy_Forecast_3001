# Data

The raw data is **not** in this repository. It is about 2 GB, and it is university building data provided to MMA3001 students for this unit only. `.gitignore` excludes everything in `data/` except this file.

## Where to get it

Dataset 2, "Monash Smart Infrastructure Occupancy and Environmental Data", provided by the MMA3001 teaching team on Moodle (Semester 2 2026). It contains sensor exports from buildings 60 and 69 at Monash Clayton.

## Where to put it

```
data/
├── raw/
│   ├── 5occupancySensor_MayToDec2024_9MRows.csv   (1.6 GB)
│   ├── 5EnvSensor_MayToDec2024_180kRows.csv       (353 MB)
│   └── Sensor ID and Locations.xlsx
└── processed/        (made by main.py)
```

The file names must match exactly, because `main.py` and `benchmark.py` look for them by name.

## What is in each file

| File | Contents | Used for |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | About 9.1 million sensor events for 5 floorspaces. One row each time a sensor reports a reading. Columns used: `id`, `floorspaceid`, `occupancystatus`, `headcount`, `collecteddate` | Everything. Only room G.25 (about 1.5 million events) is modelled |
| `5EnvSensor_MayToDec2024_180kRows.csv` | About 180,000 environmental readings (CO2, temperature and others) stored as JSON text. About 136,600 rows are duplicates, and only 2 of the 5 sensors have usable data | CO2 cross-check only (sensor in room G.38) |
| `Sensor ID and Locations.xlsx` | Which sensor and floorspace ID is in which room | Finding the G.25 floorspace ID |

## Things to know

- The file names say "MayToDec2024", but the data actually covers November 2023 to April 2026. G.25 has data from March 2024.
- Timestamps carry a +10 or +11 offset (Melbourne standard or daylight time). The code converts everything properly; see `occupancy.data`.
- `data/processed/G25_15min.parquet` is the cleaned 15-minute series written by `main.py`. It can be rebuilt any time, so it is not committed either. Parquet is a compressed table format that pandas reads with `pd.read_parquet`.
