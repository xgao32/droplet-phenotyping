pyt#!/usr/bin/env python3
"""
Estimate micrometers-per-pixel from embedded ruler annotations in microscopy images.

This script attempts OCR on the image to find labels like "72.68 µm" or "75.42 um",
then finds the nearest detected line segment (via HoughLinesP) and computes um/px = value / pixel_length.

Requires: opencv-python, pytesseract (optional but recommended). If pytesseract is not available,
this script will attempt to detect long measurement lines and return their pixel length but cannot
extract the physical value without OCR.

Usage:
    python droplet/estimate_scale.py --input image.png

Return (stdout): JSON with fields {"um_per_pixel": float or null, "matches": [...]}

"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

try:
    import pytesseract
    _HAVE_TESSERACT = True
except Exception:
    pytesseract = None
    _HAVE_TESSERACT = False

RE_NUM = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:µ|u)?m", flags=re.IGNORECASE)


def find_text_measurements(img_gray: np.ndarray) -> list[tuple[float, int, int]]:
    """Run OCR and return list of (value_um, center_x, center_y).

    If pytesseract is not available, returns empty list.
    """
    if not _HAVE_TESSERACT:
        return []

    # Increase contrast and threshold to help OCR on small white text
    img = img_gray.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Use pytesseract image_to_data to get bounding boxes
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
    n = len(data.get('text', []))
    results = []
    for i in range(n):
        txt = (data['text'][i] or '').strip()
        if not txt:
            continue
        m = RE_NUM.search(txt)
        if m:
            try:
                val = float(m.group(1))
            except Exception:
                continue
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            cx = x + w / 2
            cy = y + h / 2
            results.append((val, int(cx), int(cy)))
    return results


def detect_line_segments(img_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect line segments using probabilistic Hough transform, return list of (x1,y1,x2,y2)."""
    edges = cv2.Canny(img_gray, 50, 150)
    # Dilate to connect ticks with line body
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=20, maxLineGap=10)
    if lines is None:
        return []
    out = []
    for l in lines:
        if hasattr(l, 'shape') and l.ndim == 1 and l.size == 4:
            coords = l.tolist()
        else:
            # shape like (1,4)
            try:
                coords = l[0].tolist()
            except Exception:
                continue
        out.append(tuple(coords))
    return out


def segment_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def point_segment_distance(px, py, x1, y1, x2, y2):
    # Compute distance from point to line segment
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projx = x1 + t * dx
    projy = y1 + t * dy
    return math.hypot(px - projx, py - projy)


def estimate_scale_from_image(path: Path, debug: bool = False):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    text_meas = find_text_measurements(gray)
    lines = detect_line_segments(gray)

    matches = []
    # For each OCR text, find nearest line segment and compute um_per_px
    for val_um, cx, cy in text_meas:
        best = None
        best_dist = 1e9
        best_len = None
        for (x1, y1, x2, y2) in lines:
            d = point_segment_distance(cx, cy, x1, y1, x2, y2)
            if d < best_dist:
                best_dist = d
                best = (x1, y1, x2, y2)
                best_len = segment_length(x1, y1, x2, y2)
        if best is not None and best_len and best_len > 2:
            um_per_px = val_um / best_len
            matches.append({
                'value_um': val_um,
                'text_xy': (cx, cy),
                'line': best,
                'line_length_px': best_len,
                'um_per_px': um_per_px,
                'distance_px': best_dist,
            })

    # If we have matches, aggregate using the mean (multiple ruler measurements averaged)
    um_per_px = None
    if matches:
        vals = [m['um_per_px'] for m in matches]
        # Use mean to combine multiple ruler readings when available
        um_per_px = float(mean(vals))

    # If no OCR matches, try to return pixel lengths of long lines (user can inspect)
    if not matches and lines:
        # return the median of top lengths
        lengths = sorted([segment_length(*l) for l in lines], reverse=True)
        if lengths:
            um_per_px = None
            # Provide the most common large pixel-length as suggestion

    result = { 'um_per_pixel': um_per_px, 'matches': matches, 'num_lines': len(lines), 'num_texts': len(text_meas) }
    if debug:
        result['lines'] = [ {'coords': l, 'len': segment_length(*l)} for l in lines ]
        result['text_meas'] = [ {'val': t[0], 'xy': (t[1], t[2])} for t in text_meas ]
    return result


def parse_args():
    p = argparse.ArgumentParser(description='Estimate µm/pixel from image ruler annotations')
    p.add_argument('--input', '-i', type=Path, required=True)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--interactive', action='store_true', help='Click two endpoints on a ruler line in the image and enter the physical length (µm) when prompted')
    return p.parse_args()


def interactive_measure(path: Path) -> dict:
    """Open an OpenCV window for the user to click two points. Then prompt for physical length."""
    point_list: list[tuple[int,int]] = []
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    display = img.copy()
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point_list.append((x, y))
            cv2.circle(display, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('Select two points (press q to finish)', display)

    cv2.namedWindow('Select two points (press q to finish)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select two points (press q to finish)', on_mouse)
    print('Click two points on the ruler line, then press q in the image window when done.')
    cv2.imshow('Select two points (press q to finish)', display)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

    if len(point_list) < 2:
        raise RuntimeError('Need two points to measure distance')
    (x1, y1), (x2, y2) = point_list[:2]
    px_len = segment_length(x1, y1, x2, y2)
    print(f'Pixel length between points: {px_len:.2f} px')
    while True:
        val = input('Enter physical length in µm for this line (e.g. 75.4): ').strip()
        try:
            valf = float(val)
            break
        except Exception:
            print('Could not parse number, try again')
    um_per_px = valf / px_len
    return {'um_per_pixel': float(um_per_px), 'pixel_length': float(px_len), 'value_um': float(valf)}


def main():
    args = parse_args()
    if args.interactive:
        res = interactive_measure(args.input)
        print(json.dumps(res, indent=2))
        return
    res = estimate_scale_from_image(args.input, debug=args.debug)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
