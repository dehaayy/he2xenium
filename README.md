# H&E → Xenium alignment

Transfers pathologist annotations drawn on an H&E whole-slide image (Hamamatsu `.ndpi` + `.ndpa`) onto the cells of a 10x Xenium run on the same tissue section. Each cell gets the region label of the annotation it falls in. The alignment can also be mapped back, so every cell has coordinates on the original H&E slide.

The alignment is fully automatic and uses only tissue shape. The H&E and the Xenium cell centroids are both turned into binary tissue masks. Their outlines are matched with radial contour profiles, and a similarity transform (rotation, uniform scale and translation) is fitted with RANSAC. A genetic algorithm searches the contour-matching parameters to find the transform with the best overlap.

## Repository layout

| File | Contents |
|---|---|
| `run_alignment.py` | Command-line entry point that runs the whole pipeline for one sample |
| `mal_load.py` | NDPI/NDPA loading, H&E tissue mask (k-means), tissue crop, Xenium centroid rasterisation |
| `auto_crop.py` | Frame detection and multi-sample detection on a slide, selecting one sample with its annotations |
| `allign_w_genetic.py` | Contour matching, GA optimiser, applying the transform, QC plot, region labelling, mapping back to the H&E |

## Installation

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Inputs

| Input | Description |
|---|---|
| H&E slide | Hamamatsu `.ndpi`. The pipeline works on one pyramid level (default 5). |
| Annotations | Matching `.ndpa` file (optional; without it cells are aligned but not labelled). |
| Xenium cells | `cells.parquet` from the Xenium output, with `cell_id` and `centroid_x`, `centroid_y` in µm. |

If the slide holds several tissue sections, set `--auto-crop` and choose one with `--location` (see below).

### Choosing the sample with `--location`

With `--auto-crop`, every tissue section on the slide gets a number in reading order: **left to right, then top to bottom**, like reading a page. `--location` (or the `location` column of the sample table) is that number, starting at 1.

For example, a slide with 5 samples laid out like this:

```
X  X  X
X  X
```

is numbered:

```
1  2  3
4  5
```

So `--location 4` selects the first sample in the second row.

- **Rows:** samples are grouped into a row when their vertical centres fall within the same band. Sections that are slightly staggered up or down still count as one row.
- **Check the numbering:** `sample_detection.png` in the output folder shows every detected sample with its number. Look at it after the first run to confirm that the right one was used.
- **Number of samples:** by default the number of samples is detected automatically. If two sections touch or sit very close together and get merged, pass the true count with `--sample-count` (or the `sample_count` column).

## Usage

### One sample, explicit paths

```bash
python run_alignment.py \
  --he "/path/to/slide.ndpi" \
  --annotation "/path/to/slide.ndpi.ndpa" \
  --xenium "/path/to/cells.parquet" \
  --auto-crop --location 2 \
  --out "/path/to/results/2018-50"
```

Quote every path. Slide names often contain spaces, `&` or `#`.

### One row of a sample table

The table can be CSV, TSV, parquet or Excel, with one row per sample:

| sample_name | xenium_path | H&E_path | annotation_path | auto_crop | location | sample_count |
|---|---|---|---|---|---|---|
| 2018-50 | …/cells.parquet | …/slide.ndpi | …/slide.ndpi.ndpa | True | 2 | None |

- **`sample_count`:** use `None` or leave it empty to detect the number of samples automatically. The misspelled `sampe_count` column is also accepted.
- **Sample name:** the name picks the row and becomes the output folder name (under `--out-root`, default `./results`).

```bash
python run_alignment.py --manifest samples.csv --name-col sample_name --row 2018-50
python run_alignment.py --manifest samples.csv --name-col sample_name --row-number 0   # 0-based position
```

### Many samples on SLURM

```bash
#!/bin/bash
#SBATCH --job-name=he_xenium_align
#SBATCH --array=0-9                 # one task per manifest row
#SBATCH --cpus-per-task=90
#SBATCH --mem=64G
#SBATCH --time=12:00:00

python run_alignment.py --manifest samples.csv --name-col sample_name \
    --row-number $SLURM_ARRAY_TASK_ID --out-root /path/to/results
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--population` | 1000 | GA population size |
| `--generations` | 20 | GA generations |
| `--workers` | allocated CPUs | parallel GA worker processes |
| `--level` | 5 | NDPI pyramid level used for alignment |
| `--out` / `--out-root` | `./results/<sample>` | output folder |
| `--module-root` | folder of `run_alignment.py` | where the three modules live |

Run time grows roughly with population × generations ÷ workers.

The remaining GA settings (elite, immigrant and mating fractions, mutation schedule, diversity floor, stagnation limit) and the preprocessing settings are in the `GA` and `PREP` dictionaries at the top of `run_alignment.py`. Run `python run_alignment.py --help` for the full list of options.

### From Python

```python
from run_alignment import run_pipeline

final_df = run_pipeline(
    he_path="slide.ndpi", annotation_path="slide.ndpi.ndpa", xenium_path="cells.parquet",
    out_dir="results/2018-50", auto_crop=True, location=2,
    ga=dict(population_size=1500, generations=40),
)
```

## Outputs

Each sample folder contains:

| File | Content |
|---|---|
| `sample_detection.png` | Detected samples and their numbers (auto-crop only) |
| `sample_selected.png` | The selected H&E sample crop (auto-crop only) |
| `sample_annotations.json` | Annotations in `sample_selected.png` pixels; read with `mal_load.load_annotations()` |
| `alignment_qc.png` | Alignment quality: difference overlays before/after, bidirectional Chamfer error maps, cumulative error curves |
| `cell_labels.png` | Cells coloured by region, with per-region counts |
| `xenium_coord_df.parquet` | Final per-cell table, indexed by `cell_id` |
| `ga_results.csv` | Every parameter set the GA evaluated, with its score |

### Columns of `xenium_coord_df.parquet`

All coordinate pairs describe the same cell positions, measured on different grids:

| Columns | Coordinate system | Lines up with |
|---|---|---|
| `centroid_x`, `centroid_y` | Xenium µm | Xenium outputs (`cells.parquet`, `transcripts.parquet`), Xenium Explorer |
| `x_new`, `y_new` | Rasterised Xenium mask (internal) | – |
| `x_transformed`, `y_transformed` | Tissue-cropped H&E mask the alignment runs on (internal) | – |
| `x_he`, `y_he` | Loaded NDPI level (default 5), full slide | The image returned by `load_ndpi_with_annotations(level=5)` and the original annotations |
| `x_he_l0`, `y_he_l0` | NDPI level 0 (full resolution) | QuPath / NDP.view pixel coordinates, full-resolution tiles |
| `region` | Annotation title (`unannotated` if none) | – |
| `region_id`, `region_area` | Index and area (px²) of the assigned annotation | – |

Cells inside nested annotations get the smallest enclosing one.

To map other Xenium data (for example transcripts) onto the H&E, build the full affine from Xenium µm to H&E pixels. This uses the intermediate results of the pipeline steps (`r`, `xen_meta`, `crop_coordinates`, `info` inside `run_pipeline`), so run it where those are available, e.g. in a notebook:

```python
import allign_w_genetic as awg
T = awg.xenium_to_he_affine(r["affine"], xen_meta, crop_coordinates, info["total_offset"])
awg.apply_affine_to_df(transcripts, T, in_cols=("x_location", "y_location"))
```

## How it works

1. **Load.** One NDPI pyramid level is read, and the NDPA annotations are converted from slide nanometres to pixels of that level.
2. **Select the sample** (optional). The slide frame is detected by its colour. Tissue sections are split by cutting along the widest empty gaps (XY-cut). The chosen section is cropped with padding, and the annotations are shifted with it.
3. **Binarise.**
   - **H&E:** 2-cluster k-means separates tissue from background, then the mask is cropped to the tissue with padding.
   - **Xenium:** centroids are rasterised at a matching scale, chosen by comparing the diagonals of the two tissue extents.
4. **Align.** For each parameter set, both masks are morphologically closed at several kernel sizes, and each outline becomes a z-scored radial-distance profile. Profiles are matched across kernel sizes by FFT cross-correlation. The best-correlated pairs give point correspondences for a RANSAC similarity fit.
   - **Search space:** the GA tunes the kernel range, correlation threshold, number of chunks, RANSAC threshold, downsampling factor and kernel shape.
   - **Scoring:** 1 − Dice between the transformed cells and the H&E tissue.
5. **Label and map back.** Annotations are moved onto the alignment grid and each cell gets its region. The coordinates are then shifted back to the full slide and scaled to level 0.

## Notes

- **Thread limits on HPC:** the GA runs many worker processes. `run_alignment.py` sets `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` before importing numpy, so each worker uses one BLAS thread. In a Jupyter notebook, put this in the first cell, before any import, and restart the kernel:

  ```python
  import os
  for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
      os.environ[v] = "1"
  ```

  If you skip it, `run_genetic_algorithm` stops with a `RuntimeError` explaining the fix. Otherwise the job would exceed the process limit (`RLIMIT_NPROC`). Raising the limit (`ulimit -u 65536`) also works.
- **Wrong sample:** check `sample_detection.png` and pass the correct `--location`. If sections touch, set `--sample-count`.
- **Poor alignment:** look at `alignment_qc.png`. The Chamfer distances and the cumulative-error curve should improve clearly over "before". Try a larger `--population` or `--generations`, and check `ga_results.csv` for frequent failure reasons.
- **Transform model:** it covers rotation, uniform scale and translation only. It cannot correct mirrored sections or strong non-rigid tissue deformation.