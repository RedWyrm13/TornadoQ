# TornadoQ
*This work is actively under development and is **not** intended for industry or operational use.*

TornadoQ is a research-focused toolkit for building and organizing tornado–environment datasets intended to support machine learning models that classify tornadoes on the Enhanced Fujita (EF) scale. The goal is to improve post-event classification and analysis to support government officials and other appropriate authorities in making better-informed decisions during and after severe weather events.

---

## Repository Setup

This project was developed using **Python 3.10.14**.  
Some dependencies may not be compatible with newer Python versions.

### 1. Create a Virtual Environment

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

> On some Linux and macOS systems, replace `python` with `python3`.

Activate the environment for your operating system:

#### Windows (Command Prompt)
```bash
.venv\Scripts\activate
```

#### Windows (PowerShell)
```bash
.venv\Scripts\Activate.ps1
```

#### macOS
```bash
source .venv/bin/activate
```

#### Linux
```bash
source .venv/bin/activate
```

---

### 2. Install Dependencies

Install the repository in editable mode and install required packages:

```bash
pip install -e .
pip install -r requirements.txt
```

---

## `build_dataset.py`  
### Tornado Environment Dataset Builder (NOAA Storm Events + ERA5)

**Location:**  
```
TornadoQ/Data/NOAA/_generate_data/build_dataset.py
```

This script recreates the tornado–environment dataset used throughout the project.

> **Important:**  
> The script requires NOAA Storm Events Details CSV files, already included. (`StormEvents_details*.csv`).  
> These files were obtained from:
> https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

### What the Script Does

- Reads **NOAA Storm Events** CSV files and filters for tornado events
- Parses EF/F-scale labels and converts them into:
  - `ef_category` (0–5)
  - `ef_binary` (EF/F ≥ 2)
- Downloads corresponding **ERA5 environmental data** via the Copernicus Climate Data Store (CDS)
- Samples environmental variables at each tornado’s:
  - Begin time (rounded to the nearest hour)
  - Geographic location
- Outputs **one row per tornado event** with environmental features and labels

### ERA5 Variables Used

**Single-level variables**
- CAPE (`cape`)
- CIN (`cin`)
- 2 m temperature (`temp_2m`)
- 2 m dewpoint (`dewpoint_2m`)
- Surface pressure (`surface_pressure`)
- Total column water vapor (`tcwv`)
- 10 m winds (`u10`, `v10` — used internally)

**Pressure-level derived variables**
- 0–1 km wind shear (`shear_0_1km`)
- 0–3 km wind shear (`shear_0_3km`)

### Outputs

For each input NOAA CSV file, the script produces:

- `tornado_env_dataset0.csv`
- `tornado_env_dataset1.csv`
- …

Each row contains:

**Event metadata**
- `event_id`
- `begin_time_utc`
- `begin_lat`, `begin_lon`
- `cz_timezone`
- `begin_date_time_raw`
- `tor_f_scale_raw`

**Labels**
- `ef_category`
- `ef_binary`

**Environmental features**
- ERA5 single-level and pressure-level variables listed above

### ERA5 Cache

Downloaded ERA5 NetCDF files are cached in:

```
era5_cache/
```

This prevents repeated downloads and ensures reproducibility across runs.

---

## `parse_data.py`  
### Dataset Aggregation and Training-Ready Formatting

**Location:**  
```
TornadoQ/Data/NOAA/_generate_data/parse_data.py
```

After running `build_dataset.py`, multiple per-file datasets (`tornado_env_dataset*.csv`) will exist in the directory.  
This script aggregates and formats them into a **single training-ready dataset**.

### What the Script Does

1. Finds all CSV files matching `*dataset*.csv`
2. Concatenates them into a single DataFrame
3. Drops non-training metadata columns:
   - Event IDs
   - Raw timestamps
   - Lat/lon
   - Timezone and raw string fields
4. Reorders columns into a consistent feature + label layout
5. Writes a final dataset:

```
tornado_data_for_model_training.csv
```

The output file is written **one directory level above** the script location.

### Final Training Dataset Columns

The default column order is:

- `cape`
- `cin`
- `dewpoint_2m`
- `temp_2m`
- `tcwv`
- `surface_pressure`
- `shear_0_1km`
- `shear_0_3km`
- `ef_category`
- `ef_binary`

This format is intended to be directly ingestible by downstream machine learning models.

---

## Development Status

This repository is under active development.  
APIs, data formats, and modeling workflows may change as research progresses.

---

## License & Usage

This project is intended for **research and educational purposes only**.  
It is not certified for operational forecasting or emergency response use.
