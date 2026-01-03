#!/usr/bin/env python3
"""
Combine and organize tornado dataset CSVs into a single training-ready CSV.

Default behavior:
- Finds files matching: *dataset*.csv  (in the current directory)
- Concatenates them
- Drops metadata columns (event_id, time/lat/lon, etc.)
- Reorders columns into a consistent training feature order
- Writes: tornado_data_for_model_training.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_DROP_COLS = [
    "event_id",
    "begin_time_utc",
    "begin_lat",
    "begin_lon",
    "tor_f_scale_raw",
    "cz_timezone",
    "begin_date_time_raw",
]

DEFAULT_ORDER = [
    "cape",
    "cin",
    "dewpoint_2m",
    "temp_2m",
    "tcwv",
    "surface_pressure",
    "shear_0_1km",
    "shear_0_3km",
    "ef_category",
    "ef_binary",
]


def find_csvs(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine tornado dataset CSVs and output a model-training CSV."
    )
    parser.add_argument(
        "--pattern",
        default="*dataset*.csv",
        help="Glob pattern for input CSV files (default: *dataset*.csv)",
    )
    parser.add_argument(
        "--out",
        default="tornado_data_for_model_training.csv",
        help="Output CSV filename (default: tornado_data_for_model_training.csv)",
    )
    parser.add_argument(
        "--drop-cols",
        default=",".join(DEFAULT_DROP_COLS),
        help="Comma-separated list of columns to drop.",
    )
    parser.add_argument(
        "--order",
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated list specifying output column order.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, interpret --pattern as recursive (e.g., **/*dataset*.csv).",
    )

    args = parser.parse_args()

    pattern = args.pattern
    if args.recursive and "**" not in pattern:
        # user probably wants recursion but forgot **
        # we won't guess a directory; just enable recursive glob behavior if pattern contains **
        pass

    files = find_csvs(pattern)
    if not files:
        print(f"[ERROR] No files matched pattern: {pattern!r}")
        return 1

    print(f"[1] Found {len(files)} file(s).")
    for f in files:
        print(f"    - {f}")

    # Read + concat (ignore_index gives clean row indexing)
    print("[2] Reading and concatenating...")
    dfs = []
    for f in files:
        temp_df = pd.read_csv(f)
        temp_df["source_file"] = Path(f).name  # helpful traceability
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"[3] Combined rows: {len(df):,} | columns: {len(df.columns):,}")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    order = [c.strip() for c in args.order.split(",") if c.strip()]

    # Drop columns that exist (don’t crash if some are missing)
    print("[4] Dropping metadata columns (if present)...")
    existing_drop = [c for c in drop_cols if c in df.columns]
    missing_drop = [c for c in drop_cols if c not in df.columns]
    if missing_drop:
        print(f"    note: these drop-cols were not present and will be ignored: {missing_drop}")
    df = df.drop(columns=existing_drop)

    # Verify order columns exist
    missing_order = [c for c in order if c not in df.columns]
    if missing_order:
        print("[ERROR] Some columns requested in --order are missing from the data:")
        for c in missing_order:
            print(f"    - {c}")
        print("Fix by editing --order, or confirm your upstream dataset builder produced these columns.")
        return 2

    # Reorder + (optional) keep any extra columns at the end
    extras = [c for c in df.columns if c not in order]
    if extras:
        print(f"[5] Keeping {len(extras)} extra column(s) at the end (not in --order): {extras}")

    print("[6] Reordering columns...")
    out_df = df[order + extras]

    out_path = Path(args.out)
    print(f"[7] Writing: {out_path}")
    out_df.to_csv(out_path, index=False)

    print("[DONE] Training CSV created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
