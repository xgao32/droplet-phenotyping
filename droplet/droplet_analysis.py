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
    python Analysis/droplet_analysis.py \
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

    # Save per-droplet sizes (filtered set used for statistics)
    valid.to_csv(args.output / "droplet_sizes.csv", index=False)

    # Try to annotate per-image outputs using available image files. The CSV typically
    # contains only the image file name, so search for a matching file under the
    # input CSV directory, its parent, or the current working directory. Also prefer
    # using any previously generated annotated images if present.
    annotated_out = args.output / "annotated_analysis"
    annotated_out.mkdir(parents=True, exist_ok=True)

    search_roots = [args.input.parent, args.input.parent.parent, Path.cwd()]
    annotated_candidate_dir = args.input.parent / "annotated"

    def find_image_file(image_name: str) -> Path | None:
        # 1) If detector already produced an annotated image, use it (it already has circles/labels).
        if annotated_candidate_dir.exists():
            cand = annotated_candidate_dir / f"{Path(image_name).stem}_annotated.png"
            if cand.exists():
                return cand

        # 2) Try to find exact filename under search roots.
        for root in search_roots:
            try:
                for p in root.rglob(image_name):
                    return p
            except Exception:
                pass

        # 3) Try to match by stem (name without extension).
        stem = Path(image_name).stem
        for root in search_roots:
            try:
                for p in root.rglob(f"{stem}.*"):
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        return p
            except Exception:
                pass

        return None

    for image_name, group in valid.groupby("image", sort=True):
        img_path = find_image_file(image_name)
        if img_path is None:
            print(f"Could not find image file for {image_name}; skipping annotation")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read image {img_path}; skipping")
            continue

        # Ensure BGR 3-channel for annotation
        if img.ndim == 2:
            annotated_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            annotated_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            annotated_img = img.copy()

        for _, row in group.iterrows():
            try:
                x = int(round(float(row.get("x_px", row.get("cx", 0)))))
                y = int(round(float(row.get("y_px", row.get("cy", 0)))))
                r = int(round(float(row.get("radius_px", row.get("r", row.get("diameter_px", 0) / 2.0)))))
            except Exception:
                continue

            status = row.get("analysis_status", row.get("status", ""))

            if status == "KEEP":
                color = (0, 200, 0)
            elif status == "TOO_SMALL":
                color = (255, 180, 0)
            elif status == "TOO_LARGE":
                color = (0, 0, 255)
            else:
                color = (160, 160, 160)

            # Draw circle and centroid
            cv2.circle(annotated_img, (x, y), r, color, 2, cv2.LINE_AA)
            cv2.circle(annotated_img, (x, y), 2, color, -1, cv2.LINE_AA)

            label = f"{float(row['diameter_um']):.1f} um"
            # Put a filled rectangle for better contrast of text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = max(3, x - r)
            ty = max(th + 3, y)
            cv2.rectangle(annotated_img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(annotated_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = annotated_out / f"{Path(image_name).stem}_analysis_annotated.png"
        cv2.imwrite(str(out_path), annotated_img)
        print(f"Wrote analysis annotation: {out_path}")

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
