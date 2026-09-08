#!/usr/bin/env python3
"""
Automatic droplet diameter detection for Leica DMI8 TIFF/PNG images.

Method:
    1. Load image and convert to grayscale.
    2. Normalize to 8-bit for OpenCV processing.
    3. Detect approximately circular droplet boundaries with Hough circles.
    4. Convert radius/diameter from pixels to micrometers.
    5. Classify droplets as KEEP / TOO_SMALL / TOO_LARGE.
    6. Exclude partial droplets touching the image border by default.
    7. Write a combined CSV and annotated PNG images.

Example:
    python droplet/droplet_detection.py \
        --input images/ \
        --output results/ \
        --um-per-pixel 0.447 \
        --min-diameter 90 \
        --max-diameter 100


"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import sys


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_images(input_path: Path) -> list[Path]:
    """Return image files from a file or recursively from a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(f"No TIFF/PNG/JPEG images found in {input_path}")

    return files


def load_grayscale(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image.

    Returns:
        original_bgr: image for drawing/export
        gray8: 8-bit grayscale image for image processing
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    # Keep a BGR image for annotation.
    if image.ndim == 2:
        gray = image
        original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        original_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        original_bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize any bit depth to uint8.
    if gray.dtype != np.uint8:
        gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray8 = gray

    # Mild smoothing suppresses pixel noise while retaining the droplet edge.
    gray8 = cv2.GaussianBlur(gray8, (9, 9), 2.0)

    return original_bgr, gray8


def detect_circles(
    gray8: np.ndarray,
    min_radius_px: float,
    max_radius_px: float,
    min_dist_px: float = 110.0,
    dp: float = 1.2,
    param1: float = 100.0,
    param2: float = 35.0,
) -> np.ndarray:
    """Detect droplets using Hough gradient circles."""
    circles = cv2.HoughCircles(
        gray8,
        method=cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist_px,
        param1=param1,
        param2=param2,
        minRadius=max(1, int(round(min_radius_px))),
        maxRadius=max(1, int(round(max_radius_px))),
    )

    if circles is None:
        return np.empty((0, 3), dtype=float)

    return circles[0].astype(float)


def classify_diameter(
    diameter_um: float,
    min_diameter_um: float | None,
    max_diameter_um: float | None,
) -> str:
    """Assign a size class."""
    if min_diameter_um is not None and diameter_um < min_diameter_um:
        return "TOO_SMALL"
    if max_diameter_um is not None and diameter_um > max_diameter_um:
        return "TOO_LARGE"
    return "KEEP"


def analyze_image(
    path: Path,
    um_per_pixel: float,
    min_radius_px: float,
    max_radius_px: float,
    min_dist_px: float,
    dp: float,
    param1: float,
    param2: float,
    min_diameter_um: float | None,
    max_diameter_um: float | None,
    include_partial: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Detect droplets in one image and return a measurement table + annotated image."""
    original_bgr, gray8 = load_grayscale(path)
    height, width = gray8.shape

    circles = detect_circles(
        gray8,
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
        min_dist_px=min_dist_px,
        dp=dp,
        param1=param1,
        param2=param2,
    )

    rows = []

    for i, (x, y, r) in enumerate(circles, start=1):
        diameter_px = 2.0 * r
        diameter_um = diameter_px * um_per_pixel

        # A circle is considered partial if its fitted radius crosses the image border.
        partial = (
            x - r < 0 or
            y - r < 0 or
            x + r >= width or
            y + r >= height
        )

        status = classify_diameter(
            diameter_um,
            min_diameter_um,
            max_diameter_um,
        )

        if partial and not include_partial:
            status = "PARTIAL_EDGE"

        rows.append(
            {
                "image": path.name,
                "droplet_id": i,
                "x_px": round(float(x), 2),
                "y_px": round(float(y), 2),
                "radius_px": round(float(r), 2),
                "diameter_px": round(diameter_px, 2),
                "diameter_um": round(float(diameter_um), 3),
                "partial_edge": bool(partial),
                "status": status,
            }
        )

    df = pd.DataFrame(rows)

    # Annotation colors are only used for the image, not for charts.
    annotated = original_bgr.copy()

    for row in rows:
        x = int(round(row["x_px"]))
        y = int(round(row["y_px"]))
        r = int(round(row["radius_px"]))
        status = row["status"]

        if status == "KEEP":
            color = (0, 200, 0)       # green
        elif status == "TOO_SMALL":
            color = (255, 180, 0)     # orange-ish in BGR
        elif status == "TOO_LARGE":
            color = (0, 0, 255)       # red
        else:
            color = (160, 160, 160)  # gray

        cv2.circle(annotated, (x, y), r, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 2, color, -1, cv2.LINE_AA)

        label = f'{row["diameter_um"]:.1f} um'
        cv2.putText(
            annotated,
            label,
            (max(3, x - r), max(15, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # Add a compact legend.
    legend = [
        ("KEEP", (0, 200, 0)),
        ("TOO_SMALL", (255, 180, 0)),
        ("TOO_LARGE", (0, 0, 255)),
        ("PARTIAL_EDGE", (160, 160, 160)),
    ]

    y0 = 25
    for label, color in legend:
        cv2.circle(annotated, (15, y0 - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (28, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        y0 += 22

    cv2.putText(
        annotated,
        f"Scale: {um_per_pixel:.5f} um/px",
        (15, y0 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return df, annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect droplets and measure their diameter."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input TIFF/PNG/JPEG file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/droplet_size"),
        help="Output directory.",
    )
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=None,
        help="Physical scale in micrometers per image pixel. If omitted, the script will attempt to estimate the scale automatically from ruler annotations embedded in the images (requires OCR).",
    )
    parser.add_argument(
        "--auto-output",
        action="store_true",
        help="When input is a folder named 'input', place results under a sibling 'output' folder (e.g. droplet/example/output).",
    )
    parser.add_argument(
        "--min-diameter",
        type=float,
        default=None,
        help="Minimum accepted diameter in micrometers.",
    )
    parser.add_argument(
        "--max-diameter",
        type=float,
        default=None,
        help="Maximum accepted diameter in micrometers.",
    )
    parser.add_argument(
        "--min-radius-px",
        type=float,
        default=70,
        help="Minimum Hough-circle radius in pixels. Used only if radius is not derived from size limits.",
    )
    parser.add_argument(
        "--max-radius-px",
        type=float,
        default=130,
        help="Maximum Hough-circle radius in pixels. Used only if radius is not derived from size limits.",
    )
    parser.add_argument(
        "--min-dist-px",
        type=float,
        default=110,
        help="Minimum center-to-center distance between detected circles.",
    )
    parser.add_argument(
        "--dp",
        type=float,
        default=1.2,
        help="Hough accumulator resolution ratio.",
    )
    parser.add_argument(
        "--param1",
        type=float,
        default=100,
        help="Upper threshold for the internal Canny edge detector.",
    )
    parser.add_argument(
        "--param2",
        type=float,
        default=35,
        help="Hough circle detection threshold. Higher = fewer/more conservative detections.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Keep droplets that intersect the image border.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine output path. If user requested auto-output and input looks like
    # a folder named 'input', set output to sibling 'output' folder.
    if args.auto_output and args.input.is_dir() and args.input.name == 'input' and args.output == Path('results/droplet_size'):
        args.output = args.input.parent / 'output'

    output = args.output

    # If um_per_pixel was not supplied, try to estimate it from the first image using OCR/ruler.
    if args.um_per_pixel is None:
        try:
            from .estimate_scale import estimate_scale_from_image, interactive_measure
        except Exception:
            # fall back to relative import for direct script execution
            try:
                from estimate_scale import estimate_scale_from_image, interactive_measure
            except Exception:
                estimate_scale_from_image = None
                interactive_measure = None

        um_per_pixel = None
        if estimate_scale_from_image is not None:
            try:
                images = find_images(args.input)
                if images:
                    # Try up to the first 3 images to estimate scale and average results
                    estimates = []
                    n_try = min(3, len(images))
                    for img in images[:n_try]:
                        try:
                            print(f"Attempting to estimate scale from image: {img}")
                            r = estimate_scale_from_image(img)
                            if r and r.get('um_per_pixel'):
                                estimates.append(float(r['um_per_pixel']))
                                print(f"  -> image estimate: {estimates[-1]:.6f} µm/px")
                            else:
                                print(f"  -> no automatic estimate from {img}")
                        except Exception as e:
                            print(f"  -> estimation failed for {img}: {e}")

                    if estimates:
                        um_per_pixel = float(sum(estimates) / len(estimates))
                        print(f"Aggregated estimated um/pixel = {um_per_pixel:.6f} from {len(estimates)} image(s)")
                    else:
                        print("Could not estimate scale automatically from the first images.")
                        # Try interactive fallback once on the first image
                        try:
                            if interactive_measure is not None:
                                print("Falling back to interactive ruler measurement. A window will open for you to select/measure a ruler line.")
                                r2 = interactive_measure(images[0])
                                um_per_pixel = float(r2.get('um_per_pixel'))
                                print(f"Interactive estimate: um/pixel = {um_per_pixel:.6f}")
                            else:
                                print("Interactive measurement not available. Please provide --um-per-pixel manually.")
                        except Exception as ie:
                            print(f"Interactive measurement failed: {ie}")
            except Exception as exc:
                print(f"Scale estimation failed: {exc}")

        if um_per_pixel is None:
            raise ValueError("--um-per-pixel must be provided or automatic estimation must succeed")
        args.um_per_pixel = um_per_pixel

    if args.um_per_pixel <= 0:
        raise ValueError("--um-per-pixel must be > 0")

    # When size limits are supplied, use a wider search window around them.
    # This keeps detection independent of the later PASS/FAIL filter.
    min_radius = args.min_radius_px
    max_radius = args.max_radius_px

    if args.min_diameter is not None:
        min_radius = max(1.0, (args.min_diameter / args.um_per_pixel) / 2.0 * 0.75)

    if args.max_diameter is not None:
        max_radius = (args.max_diameter / args.um_per_pixel) / 2.0 * 1.25

    if min_radius >= max_radius:
        raise ValueError("Computed min radius is >= max radius; check calibration/size limits.")

    images = find_images(args.input)

    output = args.output
    annotated_dir = output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for image_path in images:
        print(f"Processing: {image_path}")
        df, annotated = analyze_image(
            image_path,
            um_per_pixel=args.um_per_pixel,
            min_radius_px=min_radius,
            max_radius_px=max_radius,
            min_dist_px=args.min_dist_px,
            dp=args.dp,
            param1=args.param1,
            param2=args.param2,
            min_diameter_um=args.min_diameter,
            max_diameter_um=args.max_diameter,
            include_partial=args.include_partial,
        )

        all_results.append(df)

        annotated_path = annotated_dir / f"{image_path.stem}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)

        print(f"  detected: {len(df)} droplets")
        print(f"  annotation: {annotated_path}")

    combined = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame()
    )

    csv_path = output / "droplet_measurements.csv"
    combined.to_csv(csv_path, index=False)

    print(f"\nSaved measurements: {csv_path}")
    print(f"Total detections: {len(combined)}")

    if not combined.empty and "status" in combined:
        print("\nStatus counts:")
        print(combined["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
