#!/usr/bin/env python3
"""
Downstream analysis of droplet_measurements.csv.

Produces:
    - summary CSV
    - histogram of all valid droplets
    - histogram of accepted droplets
    - size statistics CSV
    - optional per-image summary

Example:
    python droplet/droplet_analysis.py \
        --input results/droplet_size/droplet_measurements.csv \
        --output results/droplet_size/analysis \
        --min-diameter 90 \
        --max-diameter 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze detected droplet sizes.")
    parser.add_argument("--input", type=Path, required=True, help="droplet_measurements.csv")
    parser.add_argument("--output", type=Path, default=Path("results/droplet_size/analysis"))
    parser.add_argument("--min-diameter", type=float, default=None)
    parser.add_argument("--max-diameter", type=float, default=None)
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def safe_stats(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "n": 0,
            "mean_um": np.nan,
            "median_um": np.nan,
            "std_um": np.nan,
            "cv_percent": np.nan,
            "min_um": np.nan,
            "max_um": np.nan,
        }

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return {
        "n": int(len(values)),
        "mean_um": mean,
        "median_um": float(values.median()),
        "std_um": std,
        "cv_percent": (std / mean * 100.0) if mean else np.nan,
        "min_um": float(values.min()),
        "max_um": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {"image", "diameter_um", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Only use complete droplets for population-level size statistics.
    valid = df.loc[df["status"] != "PARTIAL_EDGE"].copy()
    valid["diameter_um"] = pd.to_numeric(valid["diameter_um"], errors="coerce")
    valid = valid.dropna(subset=["diameter_um"])

    keep = valid.loc[valid["status"] == "KEEP"].copy()

    # If no limits were supplied here, infer the KEEP group from the detector output.
    # Otherwise explicitly reclassify using the requested limits.
    if args.min_diameter is not None or args.max_diameter is not None:
        conditions = []
        for d in valid["diameter_um"]:
            if args.min_diameter is not None and d < args.min_diameter:
                conditions.append("TOO_SMALL")
            elif args.max_diameter is not None and d > args.max_diameter:
                conditions.append("TOO_LARGE")
            else:
                conditions.append("KEEP")
        valid["analysis_status"] = conditions
        keep = valid.loc[valid["analysis_status"] == "KEEP"].copy()
    else:
        valid["analysis_status"] = valid["status"]

    # Global statistics.
    global_stats = safe_stats(valid["diameter_um"])
    keep_stats = safe_stats(keep["diameter_um"])

    summary = pd.DataFrame([
        {"population": "all_valid", **global_stats},
        {"population": "accepted", **keep_stats},
    ])

    summary.to_csv(args.output / "droplet_size_summary.csv", index=False)

    # Per-image statistics.
    per_image_rows = []
    for image_name, group in valid.groupby("image", sort=True):
        stats = safe_stats(group["diameter_um"])
        accepted = group.loc[group["analysis_status"] == "KEEP", "diameter_um"]
        accepted_stats = safe_stats(accepted)

        per_image_rows.append({
            "image": image_name,
            "n_valid": stats["n"],
            "mean_um": stats["mean_um"],
            "median_um": stats["median_um"],
            "std_um": stats["std_um"],
            "cv_percent": stats["cv_percent"],
            "accepted_n": accepted_stats["n"],
            "accepted_fraction": (
                accepted_stats["n"] / stats["n"] if stats["n"] else np.nan
            ),
        })

    pd.DataFrame(per_image_rows).to_csv(
        args.output / "per_image_summary.csv",
        index=False,
    )

    # Histogram: all valid droplets.
    plt.figure(figsize=(8, 5))
    plt.hist(valid["diameter_um"], bins=args.bins)
    if args.min_diameter is not None:
        plt.axvline(args.min_diameter, linestyle="--", linewidth=1.5)
    if args.max_diameter is not None:
        plt.axvline(args.max_diameter, linestyle="--", linewidth=1.5)
    plt.xlabel("Droplet diameter (µm)")
    plt.ylabel("Number of droplets")
    plt.title("Droplet size distribution — all valid droplets")
    plt.tight_layout()
    plt.savefig(args.output / "droplet_size_histogram_all.png", dpi=300)
    plt.close()

    # Histogram: accepted droplets only.
    if len(keep):
        plt.figure(figsize=(8, 5))
        plt.hist(keep["diameter_um"], bins=args.bins)
        plt.xlabel("Droplet diameter (µm)")
        plt.ylabel("Number of droplets")
        plt.title("Droplet size distribution — accepted droplets")
        plt.tight_layout()
        plt.savefig(args.output / "droplet_size_histogram_accepted.png", dpi=300)
        plt.close()

    # Print useful summary.
    print("\nDroplet size analysis")
    print("=====================")
    print(f"Valid droplets:       {len(valid)}")
    print(f"Accepted droplets:    {len(keep)}")
    if len(valid):
        print(f"Accepted fraction:     {len(keep) / len(valid) * 100:.2f}%")

    print("\nAll valid droplets:")
    print(
        f"  mean = {global_stats['mean_um']:.2f} µm\n"
        f"  median = {global_stats['median_um']:.2f} µm\n"
        f"  SD = {global_stats['std_um']:.2f} µm\n"
        f"  CV = {global_stats['cv_percent']:.2f}%"
    )

    print("\nAccepted droplets:")
    print(
        f"  mean = {keep_stats['mean_um']:.2f} µm\n"
        f"  median = {keep_stats['median_um']:.2f} µm\n"
        f"  SD = {keep_stats['std_um']:.2f} µm\n"
        f"  CV = {keep_stats['cv_percent']:.2f}%"
    )

    print(f"\nSaved analysis to: {args.output}")


if __name__ == "__main__":
    main()
