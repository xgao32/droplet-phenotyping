# Droplet size analysis

These scripts are designed for the Leica DMI8 droplet images used in the
`phenotyping-updates` branch.
```

## 2. Calibrate the image scale

The detector needs the physical scale in µm/pixel.

For a particular Leica imaging configuration, make a CSV such as:

```csv
pixel_length,um_length
217.3,97.33
204.1,91.61
211.4,94.61
220.0,98.77
```

Then run:

```bash
python Tools/calibrate_scale.py calibration.csv
```

Use the reported `Scale` as `--um-per-pixel`.

For the two example images supplied during development, approximately
0.44–0.45 µm/pixel is expected. For production analysis, use calibration
measurements from the exact exported image/magnification rather than copying
this value blindly.

## 3. Detect droplets

Example for a target range of 90–100 µm:

```bash
python Tools/droplet_detection.py \
    --input images/ \
    --output results/droplet_size \
    --um-per-pixel 0.447 \
    --min-diameter 90 \
    --max-diameter 100
```

Outputs:

```text
results/droplet_size/
├── droplet_measurements.csv
└── annotated/
    ├── image1_annotated.png
    ├── image2_annotated.png
    └── ...
```

The default behavior excludes droplets touching the image border because
their apparent diameter is incomplete.

To keep border droplets:

```bash
--include-partial
```

### Important Hough parameters

For images similar to the supplied Leica images:

```text
--param2 35
--min-dist-px 110
--min-radius-px 70
--max-radius-px 130
```

`--param2` controls detection stringency:

- increase it if there are false positives
- decrease it if real droplets are being missed

`--min-dist-px` prevents multiple detections of the same droplet.

## 4. Downstream analysis

```bash
python Analysis/droplet_analysis.py \
    --input results/droplet_size/droplet_measurements.csv \
    --output results/droplet_size/analysis \
    --min-diameter 90 \
    --max-diameter 100
```

Outputs:

```text
results/droplet_size/analysis/
├── droplet_size_summary.csv
├── per_image_summary.csv
├── droplet_size_histogram_all.png
└── droplet_size_histogram_accepted.png
```

The summary contains:

- number of droplets
- mean diameter
- median diameter
- standard deviation
- CV%
- minimum/maximum
- accepted fraction

## 5. Recommended workflow for many images

Keep original Leica images untouched:

```text
images/
    experiment_01/
        image001.tif
        image002.tif
    experiment_02/
        image003.tif
```

Then run the detector on the whole directory. It recursively processes all
supported TIFF/PNG/JPEG files and combines results into one CSV.

## Notes on the current detector

The current implementation uses Hough-gradient circle detection because the
supplied images have strong, approximately circular droplet boundaries and
densely packed droplets.

The annotations exported by Leica (ruler lines, labels, etc.) can remain in
the image; the circle detector searches for circular boundaries rather than
trying to threshold the whole image.

For a large dataset, manually inspect a random subset of annotated images
before trusting the population statistics. The `--param2`, radius limits,
and minimum-distance parameters should be tuned once on representative
images and then kept fixed for a batch.
