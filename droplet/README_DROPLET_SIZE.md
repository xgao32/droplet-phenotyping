# Droplet size analysis

These scripts are designed for the Leica DMI8 droplet images used in the
`phenotyping-updates` branch. The repository contains tools to detect droplet
boundaries, estimate the image scale from embedded ruler annotations, and
produce publication-ready size histograms and per-image summaries.

## 1. Quick overview (automatic scale estimation)

New workflow highlights:

- The pixel → micrometer scale is estimated automatically from ruler
  annotations inside the image using `droplet/estimate_scale.py`.
- `droplet/droplet_detection.py` accepts `--um-per-pixel` optionally; when
  omitted it will try to estimate the scale from the first image.
- Use `--auto-output` with an input folder named `input` to write results to a
  sibling `output` directory (e.g. `droplet/example/output`).

This lets you run the detector without manually looking up the µm/px value.

## 2. Estimate the image scale (recommended)

If you want to inspect or validate the automatic estimate before running the
full detector, run the scale estimator on one representative image:

```bash
python droplet/estimate_scale.py --input /Users/xiangxigao/Desktop/git/droplet-phenotyping/droplet/example/input/TECH_XG_014_1_300_100_12hr_ch00.png --debug
```

Notes:

- The estimator uses OCR (pytesseract) to find labels like `75.4 µm` in the
  image and pairs each found label with the nearest detected ruler line. When
  more than one ruler measurement is found it averages the inferred µm/px
  values (arithmetic mean) to produce a single `um_per_pixel` estimate.
- If OCR/Tesseract is not available, run the estimator in interactive mode
  to click two endpoints of a visible ruler and type the physical length:

```bash
python droplet/estimate_scale.py --input /path/to/image.png --interactive
```

To enable fully automatic estimation, install Tesseract and pytesseract:

- macOS: `brew install tesseract`
- Python: `pip install pytesseract`

## 3. Detect droplets (no manual µm/px required)

Run the detector on the example folder and place all outputs under
`droplet/example/output` (use `--auto-output`):

```bash
python droplet/droplet_detection.py \
    --input droplet/example/input \
    --auto-output \
    --min-diameter 60 \
    --max-diameter 120 \
    --param2 35 --min-dist-px 110 --min-radius-px 40 --max-radius-px 200
```

Behavior:

- If `--um-per-pixel` is omitted the script attempts to estimate the scale
  from the first image using `droplet/estimate_scale.py` (OCR required for
  fully automatic behavior).
- The detector writes:

```text
droplet/example/output/
├── droplet_measurements.csv
├── annotated/                  # per-image detection annotations
│   └── <image>_annotated.png
└── analysis/                   # downstream analysis (created by the separate analysis step)
    ├── droplet_size_summary.csv
    ├── per_image_summary.csv
    ├── droplet_size_histogram_all.png
    ├── droplet_size_histogram_accepted.png
    └── annotated_analysis/     # per-image analysis overlays created by droplet_analysis
        └── <image>_analysis_annotated.png
```

Example (paths on this machine):

Input image (before):

![Input image](/Users/xiangxigao/Desktop/git/droplet-phenotyping/droplet/example/input/TECH_XG_014_1_300_100_12hr_ch00.png)

Annotated detection (after):

![Annotated detection](/Users/xiangxigao/Desktop/git/droplet-phenotyping/droplet/example/output/annotated/TECH_XG_014_1_300_100_12hr_ch00_annotated.png)

Annotated analysis overlay (after analysis):

![Annotated analysis](/Users/xiangxigao/Desktop/git/droplet-phenotyping/droplet/example/output/analysis/annotated_analysis/TECH_XG_014_1_300_100_12hr_ch00_analysis_annotated.png)

## 4. Downstream analysis (histograms & CSV)

Run the downstream analysis to produce population statistics, histograms, and
per-image annotated overlays. If you used `--auto-output` above the input CSV
will be at `droplet/example/output/droplet_measurements.csv`.

```bash
python droplet/droplet_analysis.py \
    --input droplet/example/output/droplet_measurements.csv \
    --output droplet/example/output/analysis \
    --min-diameter 60 \
    --max-diameter 120
```

Outputs (example):

- `droplet/example/output/analysis/droplet_size_summary.csv`
- `droplet/example/output/analysis/per_image_summary.csv`
- `droplet/example/output/analysis/droplet_sizes.csv` (per-droplet table)
- `droplet/example/output/analysis/droplet_size_histogram_all.png`
- `droplet/example/output/analysis/droplet_size_histogram_accepted.png`
- `droplet/example/output/analysis/annotated_analysis/<image>_analysis_annotated.png`

## 5. Notes & troubleshooting

- Multiple rulers: the estimator averages multiple detected ruler measurements
  (arithmetic mean) to improve robustness when several ruler labels are
  present in an image.
- OCR reliability: automatic estimation relies on pytesseract. If OCR does not
  find ruler labels the detector will not be able to infer µm/px and you will
  need to provide `--um-per-pixel` or use `--interactive` to measure a ruler
  manually.
- Install Tesseract for best results (see above).
- Tuning Hough parameters: if droplets are missing or false positives appear,
  adjust `--param2`, `--min-dist-px`, and radius bounds once on a
  representative image and reuse those parameters for a batch.

If you want, the README can be extended with a short script that runs the
estimator and detector in one command and validates the produced scale before
processing the full dataset.
