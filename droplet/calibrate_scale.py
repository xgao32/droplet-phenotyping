#!/usr/bin/env python3
"""
Calculate micrometers-per-pixel from multiple Leica ruler measurements.

Input CSV must contain:
    pixel_length,um_length

Example calibration.csv:
    pixel_length,um_length
    217.3,97.33
    204.1,91.61
    211.4,94.61

Usage:
    python Tools/calibrate_scale.py calibration.csv

The scale is estimated as the slope forced through the origin:
    um_per_pixel = sum(pixel * um) / sum(pixel^2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if not {"pixel_length", "um_length"}.issubset(df.columns):
        raise ValueError("CSV needs columns: pixel_length, um_length")

    x = pd.to_numeric(df["pixel_length"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["um_length"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Provide at least two valid calibration measurements.")

    slope = float(np.sum(x * y) / np.sum(x * x))
    predicted = slope * x
    residual = y - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))

    print(f"Measurements:       {len(x)}")
    print(f"Scale:              {slope:.6f} µm/pixel")
    print(f"Pixels per µm:      {1/slope:.4f}")
    print(f"Calibration RMSE:    {rmse:.3f} µm")


if __name__ == "__main__":
    main()
