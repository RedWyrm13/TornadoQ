
# TornadoQ
*This work is being actively developed, and is not ready for industry use*

This repository deploys a tool to help government officals and other appropriate authorities classify tornadoes on the EF scale so as to improve the quality of decisions made in response to these emergencies.

## Setting up TornadoQ
This project was set up with Python 3.10.14. Certain necessary dependencies may not be compatible with newer versions of Python.

1. Create a virtural environment
   To initialize the enviroment on Linux, MacOS, or Windows open a terminal window and run
   ```
   python -m venv .venv
   ```
   Note for certain distributions of linux and MacOS, replace ```python``` with ```python3```.

   For the appropriate OS, the enviroment is activated as follows:
   ### Windows (command prompt)
   ```
   .venv\Scripts\activate
   ```

   ### Windows (Powershell)
   ```
   .venv\Scripts\Activate.ps1

   ```

   ### MacOS
   ```
   source .venv/bin/activate

   ```

   ### Linux
   ```
   source .venv/bin/activate

   ```
2. Install the repository as a python package

```
pip install -e .
pip install -r requirements.txt
```

# Tornado Environment Dataset Builder (NOAA Storm Events + ERA5)

The build_dataset.py script located in TornadoQ/Data/NOAA/_generate_data/ recreates the tornado-environment dataset used in this project.
*Note: The StormEvents*.csv files are required for this script to work. They were obtained from this link: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/*

It:
- Reads **NOAA Storm Events** CSV rows (tornado events + labels / metadata)
- Downloads matching **ERA5 environmental variables** (via the Copernicus Climate Data Store / CDS API)
- Extracts features at each tornado’s **begin time (rounded to nearest hour) and location**
- Outputs **one row per tornado event** with environmental features + `ef_category` / `ef_binary`

## What you get (outputs)

For each input NOAA CSV, the script writes a dataset CSV like:

- `tornado_env_dataset0.csv`
- `tornado_env_dataset1.csv`
- ...

Each output row contains:

**Event metadata**
- `event_id`
- `begin_time_utc`
- `begin_lat`, `begin_lon`
- `cz_timezone`, `begin_date_time_raw`, `tor_f_scale_raw`

**Labels**
- `ef_category` (0–5 if parseable)
- `ef_binary` (1 if EF/F >= 2 else 0)

**ERA5 single-level features**
- `cape`, `cin`
- `temp_2m`, `dewpoint_2m`
- `surface_pressure`
- `tcwv`
- `u10`, `v10` *(used internally; not currently written as columns unless you add them)*

**ERA5 pressure-level derived features**
- `shear_0_1km`
- `shear_0_3km`

It also creates/uses a cache folder:
- `era5_cache/` containing downloaded NetCDF files (`.nc`)

---

After running build_dataset.py to generate the different tornado_env files, the parse_data.py script located in the same directory parse the generated data to create the "tornado_data_for_model_training.csv" file located one directory higher.
This utility script takes the **per-file tornado environment datasets** produced by the ERA5/NOAA builder (e.g. `tornado_env_dataset0.csv`, `tornado_env_dataset1.csv`, …), concatenates them, removes non-training metadata columns, and writes a single **model-training-ready** CSV.

## What it does

1. Finds input files matching a glob pattern (default: `*dataset*.csv`)
2. Reads and concatenates them into one DataFrame
3. Drops metadata columns (IDs, timestamps, raw strings, lat/lon, etc.)
4. Reorders feature + label columns into a consistent order
5. Writes `tornado_data_for_model_training.csv`

It also adds a `source_file` column during concatenation (useful for debugging / provenance).  
If you don’t want that, delete the line that adds it in the script.