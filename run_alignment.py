#!/usr/bin/env python
"""
H&E annotations -> Xenium cells: sample crop, GA alignment, region labels, mapping back to the H&E.

Single sample, paths on the command line:
    python run_alignment.py --he slide.ndpi --annotation slide.ndpa --xenium cells.parquet \
        --out /home/day/Desktop/deha_base/auto_allignment/S1 --auto-crop --location 2 --sample-count 3

One row of a sample table (columns: H&E_path, annotation_path, xenium_path, auto_crop, location,
sample_count; the row name becomes the output folder name):
    python run_alignment.py --manifest samples.csv --row S1
    python run_alignment.py --manifest samples.csv --row-number $SLURM_ARRAY_TASK_ID

Writes to the output folder: sample_detection.png, sample_selected.png, sample_annotations.json
(auto-crop only), alignment_qc.png, cell_labels.png, xenium_coord_df.parquet, ga_results.csv.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # before numpy is imported: forked GA workers stay single-threaded

import matplotlib
matplotlib.use("Agg")             # headless: figures are saved, never shown

import argparse
import importlib
import sys
import time
from pathlib import Path

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent     # the 3 modules sit next to this script
OUT_ROOT = Path("results")                        # default when neither --out nor --out-root is given

PREP = dict(level=5, crop_pad_frac=0.05,
            he_ratio=0.01, he_pad_frac=0.2, he_sigma=1.0,
            xen_ratio=0.015, xen_pad_frac=0.2, xen_cols=("centroid_x", "centroid_y"))

GA = dict(population_size=1000, generations=20,
          elite_frac=0.03, immigrant_frac=0.08, mating_frac=0.65, tournament_size=3,
          mutation_rate_hi=0.40, mutation_rate_lo=0.08, sigma_hi=0.20, sigma_lo=0.02,
          diversity_floor=0.12, stagnation_limit=8, n_workers=None)


# ---------------------------------------------------------------------------
def import_modules(root=MODULE_ROOT):
    """Put every folder under `root` holding a .py file on sys.path; return (mall, ac, awg)."""
    root = Path(root)
    dirs = sorted({str(p.parent) for p in root.rglob("*.py")
                   if not any(x.startswith(".") for x in p.relative_to(root).parts)})
    sys.path[:0] = [d for d in dirs if d not in sys.path]
    return tuple(importlib.import_module(m) for m in ("mal_load", "auto_crop", "allign_w_genetic"))


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(v) if pd.notna(v) else False


def _as_int_or_none(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "none", "nan", "na", "auto"):
        return None
    return int(float(v))


# ---------------------------------------------------------------------------
def run_pipeline(he_path, xenium_path, out_dir, annotation_path=None, auto_crop=False,
                 location=1, sample_count=None, prep=None, ga=None, modules=None):
    """Run the whole pipeline for one sample. Returns the final per-cell DataFrame."""
    mall, ac, awg = modules or import_modules()
    prep = {**PREP, **(prep or {})}
    ga = {**GA, **(ga or {})}
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # 1. load
    image, annotations, ndpi_meta = mall.load_ndpi_with_annotations(
        he_path, annotation_path, level=prep["level"], return_meta=True)
    xenium_obs = pd.read_parquet(xenium_path)
    if "cell_id" in xenium_obs.columns:
        xenium_obs = xenium_obs.set_index("cell_id")
    log(f"loaded H&E {image.shape}, {len(annotations)} annotations, {len(xenium_obs):,} cells")

    # 2. select the sample
    if auto_crop:
        sample_img, sample_anns, info = ac.crop_sample_with_annotations(
            image, annotations, location=int(location), n_samples=sample_count,
            pad_frac=prep["crop_pad_frac"], plot=False, out_dir=out_dir)
        log(f"auto-crop: sample {location}, offset {info['total_offset']}, {len(sample_anns)} annotations")
    else:
        sample_img, sample_anns, info = image, annotations, {"total_offset": (0, 0)}

    # 3. binarise + crop both modalities
    binary_img, _ = mall.kmeans_binary(sample_img, plot=False)
    ref_img, img_diag, crop_coordinates = mall.contour_crop_binary(
        binary_img, ratio=prep["he_ratio"], pad_frac=prep["he_pad_frac"], sigma=prep["he_sigma"], plot=False)
    coord_df, xenium_img, xen_meta = mall.xenium_binarize(
        xenium_obs, pad_frac=prep["xen_pad_frac"], ratio=prep["xen_ratio"], img_diag_size=img_diag,
        plot=False, column_names=list(prep["xen_cols"]))
    log(f"reference {ref_img.shape}, xenium raster {xenium_img.shape}")

    # 4. GA alignment
    result = awg.run_genetic_algorithm(xenium_img, ref_img, coord_df,
                                       output_csv=os.path.join(out_dir, "ga_results.csv"), **ga)
    r = awg.apply_ga_result(result, xenium_img, ref_img, coord_df)
    qc = awg.plot_chamfer_qc(ref_img, coord_df, display_px=600, show=False, out_dir=out_dir)
    a = r["affine_summary"]
    log(f"aligned: GA score {r['ga_score']:.5f}, Chamfer sum {qc['chamfer_sum']:.2f} px, "
        f"scale {a['scale']:.3f}, rotation {a['rotation_deg']:.2f} deg, shift ({a['tx']:.0f}, {a['ty']:.0f})")

    # 5. region labels + back to the original H&E, final table saved
    counts, _, _, _ = awg.label_cells_from_annotations(
        coord_df, sample_anns, binary_img, ref_img, crop_coordinates=crop_coordinates,
        plot=False, out_dir=out_dir)
    final_df = awg.to_original_he(coord_df, crop_coordinates, info["total_offset"],
                                  level_downsample=ndpi_meta["downsample"], out_dir=out_dir)
    log(f"done -> {out_dir}")
    return final_df


# ---------------------------------------------------------------------------
def _read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=0)
    return pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",", index_col=0,
                       encoding="utf-8-sig")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input: a manifest row OR explicit paths")
    src.add_argument("--manifest", help="sample table (csv/tsv/parquet/xlsx); first column = sample name")
    src.add_argument("--name-col", help="manifest column holding the sample name (default: the index)")
    row = src.add_mutually_exclusive_group()
    row.add_argument("--row", help="sample name (manifest row label)")
    row.add_argument("--row-number", type=int, help="0-based manifest row, e.g. $SLURM_ARRAY_TASK_ID")
    src.add_argument("--he", help="H&E .ndpi")
    src.add_argument("--annotation", help="annotation .ndpa (optional)")
    src.add_argument("--xenium", help="Xenium cells parquet (cell_id + centroid columns)")
    src.add_argument("--auto-crop", action="store_true", help="crop one sample out of a multi-sample slide")
    src.add_argument("--location", type=int, default=1, help="sample id to keep (see sample_detection.png)")
    src.add_argument("--sample-count", type=int, help="number of samples on the slide (default: auto)")

    out = p.add_argument_group("output")
    out.add_argument("--out", help="output folder (default: --out-root/<sample name>)")
    out.add_argument("--out-root", default=str(OUT_ROOT))

    run = p.add_argument_group("run")
    run.add_argument("--module-root", default=str(MODULE_ROOT), help="folder containing the 3 modules")
    run.add_argument("--level", type=int, default=PREP["level"], help="NDPI pyramid level")
    run.add_argument("--population", type=int, default=GA["population_size"])
    run.add_argument("--generations", type=int, default=GA["generations"])
    run.add_argument("--workers", type=int, default=None, help="GA workers (default: allocated CPUs)")
    args = p.parse_args(argv)

    if args.manifest is None and not (args.he and args.xenium):
        p.error("give --manifest with --row/--row-number, or --he and --xenium")
    if args.manifest is not None and args.row is None and args.row_number is None:
        p.error("--manifest needs --row or --row-number")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.manifest:
        table = _read_table(args.manifest)
        table.columns = table.columns.str.strip().str.lstrip("\ufeff")
        if args.name_col:
            table = table.set_index(args.name_col)
        table.index = table.index.astype(str)
        inst = table.iloc[args.row_number] if args.row_number is not None else table.loc[str(args.row)]
        name = str(inst.name)
        sample = dict(he_path=inst["H&E_path"], xenium_path=inst["xenium_path"],
                      annotation_path=inst.get("annotation_path"),
                      auto_crop=_as_bool(inst.get("auto_crop", False)),
                      location=_as_int_or_none(inst.get("location")) or 1,
                      sample_count=_as_int_or_none(inst.get("sample_count", inst.get("sampe_count"))))
    else:
        name = Path(args.xenium).parent.name or Path(args.xenium).stem
        sample = dict(he_path=args.he, xenium_path=args.xenium, annotation_path=args.annotation,
                      auto_crop=args.auto_crop, location=args.location, sample_count=args.sample_count)

    out_dir = Path(args.out) if args.out else Path(args.out_root) / name
    print(f"sample {name} -> {out_dir}\n{sample}", flush=True)

    run_pipeline(**sample, out_dir=out_dir, prep=dict(level=args.level),
                 ga=dict(population_size=args.population, generations=args.generations,
                         n_workers=args.workers),
                 modules=import_modules(args.module_root))


if __name__ == "__main__":
    main()