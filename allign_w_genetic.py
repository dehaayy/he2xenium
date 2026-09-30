"""
GA optimiser for contour-based Xenium -> reference-image affine alignment.

Import-only module. Typical use:

    import allign_w_genetic as ga
    result = ga.run_genetic_algorithm(xenium_image, CROPPED_ORIGINAL_IMG, xenium_coord_df)
    r = ga.apply_ga_result(result, xenium_image, CROPPED_ORIGINAL_IMG, xenium_coord_df)
    # xenium_coord_df now has 'x_transformed', 'y_transformed'; r["affine"] is the 2x3 matrix

    # annotations (in sample-crop px) -> per-cell region labels (exact offset from crop_coordinates)
    ga.label_cells_from_annotations(xenium_coord_df, sample_anns, binary_img, CROPPED_ORIGINAL_IMG,
                                    crop_coordinates=crop_coordinates)

    # back to the original H&E (IMAGE): writes 'x_he', 'y_he' (+ level-0 cols if downsample given)
    ga.to_original_he(xenium_coord_df, crop_coordinates, info["total_offset"], out_dir=OUT)
    T = ga.xenium_to_he_affine(r["affine"], meta, crop_coordinates, info["total_offset"])

Saving (out_dir): plot_chamfer_qc -> alignment_qc.png, label_cells_from_annotations -> cell_labels.png,
to_original_he -> xenium_coord_df.parquet.

Inputs:
    xenium_image          2D array, foreground > 0
    CROPPED_ORIGINAL_IMG  2D array, foreground > 0
    xenium_coord_df       DataFrame with 'x_new', 'y_new' in xenium_image pixel coords
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # only effective if imported before numpy

import copy
import multiprocessing as mp
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
cv2.setNumThreads(1)

BAD_SCORE = 1e9
N_POINTS = 500     # samples per contour profile
MIN_CHUNKS = 1     # min matched chunks; each chunk gives N_POINTS correspondences
RNG_SEED = 0       # fixes cv2 RANSAC so a refit reproduces the GA result

PARAM_BOUNDS = {
    'ratio_start':       (0.001, 0.010),
    'ratio_stop':        (0.020, 0.040),
    'ratio_num':         (5, 25),
    'min_corr':          (0.60, 0.99),
    'n_chunks':          (1, 10),
    'ransac_thresh':     (1.0, 30.0),
    'scale':             (2, 4),          # integer downsample factor before contouring
    'elliptical_kernel': [False, True],   # ellipse vs rectangular closing kernel
}


# ===========================================================================
# 1. Contours + radial profiles
# ===========================================================================

def _ratio_to_k(h, w, ratio):
    return max(3, int((h * h + w * w) ** 0.5 * ratio) | 1)


def _downscale(img, scale):
    """Binarise + downsample; return mask and per-axis factors back to full res."""
    b = (np.asarray(img) > 0).astype(np.uint8) * 255
    if scale <= 1:
        return b, 1.0, 1.0
    s = cv2.resize(b, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA)
    s = (s > 127).astype(np.uint8) * 255
    return s, b.shape[1] / s.shape[1], b.shape[0] / s.shape[0]


def contour_morphology(mask, k, elliptical_kernel=True):
    """Close + hole-fill a binary mask, return its largest external contour."""
    shape = cv2.MORPH_ELLIPSE if elliptical_kernel else cv2.MORPH_RECT
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(shape, (k, k)))
    p = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = cv2.bitwise_not(p)
    cv2.floodFill(ff, np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8), (0, 0), 0)
    filled = cv2.bitwise_or(p, ff)[1:-1, 1:-1]
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def radial_distance_profile(contour, n=360):
    """CW arc-length resampling from the max-y point; z-scored centroid distances."""
    pts = contour.reshape(-1, 2).astype(float)
    x, y = pts[:, 0], pts[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pts = pts[::-1]

    M = cv2.moments(pts.astype(np.float32))
    cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else pts.mean(0)

    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg[:-1])]
    total = arc[-1] + seg[-1]
    arc = (arc - arc[np.argmax(pts[:, 1])]) % total

    order = np.argsort(arc)
    t = np.linspace(0, total, n, endpoint=False)
    xi = np.interp(t, arc[order], pts[order, 0], period=total)
    yi = np.interp(t, arc[order], pts[order, 1], period=total)

    r = np.hypot(xi - cx, yi - cy)
    return (r - r.mean()) / r.std(), np.column_stack([xi, yi])


def _profiles_for_image(img, ratios, n_points, elliptical_kernel, scale):
    small, fx, fy = _downscale(img, scale)
    h, w = small.shape
    cache = {}
    for r in ratios:
        k = _ratio_to_k(h, w, r)
        if k in cache:
            continue
        cnt = contour_morphology(small, k, elliptical_kernel)
        if cnt is None or len(cnt) < 3:
            raise ValueError(f"no contour found (k={k})")
        radi, xy = radial_distance_profile(cnt, n=n_points)
        cache[k] = (radi, np.column_stack([(xy[:, 0] + 0.5) * fx - 0.5,
                                           (xy[:, 1] + 0.5) * fy - 0.5]))
    return cache


def run_contour_comparison(xenium_img, ref_img, ratios, n_chunks=5, n_points=N_POINTS,
                           elliptical_kernel=True, scale=1):
    """All-pairs FFT cross-correlation of radial profiles across kernel sizes."""
    x_cache = _profiles_for_image(xenium_img, ratios, n_points, elliptical_kernel, scale)
    i_cache = _profiles_for_image(ref_img,    ratios, n_points, elliptical_kernel, scale)

    x_keys, i_keys = list(x_cache), list(i_cache)
    X = np.stack([x_cache[k][0] for k in x_keys])
    I = np.stack([i_cache[k][0] for k in i_keys])
    Nx, Ni, n = X.shape[0], I.shape[0], n_points

    prod = np.fft.rfft(I, axis=1)[None] * np.conj(np.fft.rfft(X, axis=1))[:, None]
    shifts = np.argmax(np.fft.irfft(prod, n=n, axis=-1), axis=-1).astype(np.int64)

    idx = (np.arange(n)[None, None, :] - shifts[:, :, None]) % n
    aligned = np.take_along_axis(np.broadcast_to(X[:, None, :], (Nx, Ni, n)), idx, axis=-1)

    mse = np.mean((aligned - I[None]) ** 2, axis=-1)
    a_c = aligned - aligned.mean(-1, keepdims=True)
    i_c = I[None] - I.mean(-1)[None, :, None]
    den = np.sqrt((a_c * a_c).sum(-1) * (i_c * i_c).sum(-1))
    corr = (a_c * i_c).sum(-1) / np.where(den == 0, 1, den)

    df = pd.DataFrame([{'k_x': kx, 'k_i': ki, 'avg_k': (kx + ki) / 2,
                        'mse': float(mse[a, b]), 'corr': float(corr[a, b]),
                        'shift': int(shifts[a, b]),
                        'x_coords': x_cache[kx][1], 'i_coords': i_cache[ki][1]}
                       for a, kx in enumerate(x_keys) for b, ki in enumerate(i_keys)])
    df['chunk'] = pd.qcut(df['avg_k'], n_chunks, labels=False, duplicates='drop')
    return df.sort_values('mse').reset_index(drop=True)


def get_best_per_chunk(df, n_chunks, min_corr=0.9):
    """Lowest-MSE pair per k-chunk with corr > min_corr, with shift-aligned coords."""
    rows = []
    for c in range(n_chunks):
        sub = df[(df['chunk'] == c) & (df['corr'] > min_corr)]
        if not sub.empty:
            b = sub.iloc[0]
            rows.append({'chunk': c,
                         'x_aligned': np.roll(b['x_coords'], int(b['shift']), axis=0),
                         'i_coords': b['i_coords']})
    return pd.DataFrame(rows)


def safe_get_best_per_chunk(df, n_chunks, min_corr, min_rows=MIN_CHUNKS,
                            fallback_corrs=(0.95, 0.90, 0.85, 0.80, 0.70, 0.60)):
    for corr in [min_corr] + [c for c in fallback_corrs if c < min_corr]:
        best = get_best_per_chunk(df, n_chunks, min_corr=corr)
        if len(best) >= min_rows:
            return best, corr
    return None, None


# ===========================================================================
# 2. Affine fit for one parameter set  (shared by GA and apply_ga_result)
# ===========================================================================

class FitFailed(Exception):
    def __init__(self, reason, **info):
        super().__init__(reason)
        self.reason, self.info = reason, info


def fit_affine(xenium_img, ref_img, ind):
    """Run the contour pipeline for one individual -> (M 2x3, info dict)."""
    ratios = np.linspace(float(ind["ratio_start"]), float(ind["ratio_stop"]),
                         int(ind["ratio_num"]))
    if len(ratios) < 2:
        raise FitFailed("too_few_ratios")

    n_chunks = int(ind["n_chunks"])
    df = run_contour_comparison(xenium_img, ref_img, ratios, n_chunks=n_chunks,
                                elliptical_kernel=bool(ind["elliptical_kernel"]),
                                scale=int(ind["scale"]))
    if len(df) == 0:
        raise FitFailed("df1_empty")

    best, used_corr = safe_get_best_per_chunk(df, n_chunks, float(ind["min_corr"]))
    if best is None:
        raise FitFailed("no_best_df")

    src = np.float32(np.vstack(best["x_aligned"].values))
    dst = np.float32(np.vstack(best["i_coords"].values))
    cv2.setRNGSeed(RNG_SEED)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=float(ind["ransac_thresh"]))
    info = dict(used_corr=used_corr, n_matches=len(best),
                n_inliers=int(inl.sum()) if inl is not None else 0)
    if M is None:
        raise FitFailed("affine_failed", **info)
    if info["n_inliers"] < 3:
        raise FitFailed("too_few_inliers", **info)
    return M, info


def transform_coords(coord_df, M, coord_cols=('x_new', 'y_new'),
                     out_cols=('x_transformed', 'y_transformed')):
    """Apply a 2x3 affine to coord_df[coord_cols]; writes out_cols in place."""
    xy = coord_df[list(coord_cols)].to_numpy().astype(np.float32).reshape(-1, 1, 2)
    t = cv2.transform(xy, np.asarray(M, np.float64)).reshape(-1, 2)
    coord_df[out_cols[0]], coord_df[out_cols[1]] = t[:, 0], t[:, 1]
    return coord_df


def fast_alignment_score(ref_img, xy):
    """1 - Dice between reference foreground and rasterised points (lower = better)."""
    H, W = ref_img.shape[:2]
    xy = np.rint(xy).astype(np.int64)
    m = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    pts = np.zeros((H, W), dtype=bool)
    pts[xy[m, 1], xy[m, 0]] = True
    fg = ref_img > 0
    return 1 - 2 * (fg & pts).sum() / (fg.sum() + pts.sum() + 1e-8)


# ===========================================================================
# 3. GA worker evaluation
# ===========================================================================

def make_meta(status, failure_reason=None, used_corr=None, n_matches=0, n_inliers=0,
              error=None, affine=None):
    return dict(status=status, failure_reason=failure_reason, used_corr=used_corr,
                n_matches=n_matches, n_inliers=n_inliers, error=error, affine=affine)


_G = {}


def _available_cpus():
    """CPUs this process may actually use (respects SLURM/cgroup affinity, unlike os.cpu_count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _check_fork_threads(n_workers):
    """With 'fork', every worker inherits the notebook's BLAS thread pool size. If that is > 1,
    n_workers x threads can exceed the user process limit (RLIMIT_NPROC) and the pool breaks or
    hangs. Fail fast with instructions instead."""
    try:
        import resource
        from threadpoolctl import threadpool_info
    except ImportError:
        return
    blas_threads = max([i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"] or [1])
    soft = resource.getrlimit(resource.RLIMIT_NPROC)[0]
    if blas_threads > 1 and soft != resource.RLIM_INFINITY and n_workers * blas_threads > 0.8 * soft:
        raise RuntimeError(
            f"BLAS in this kernel already uses {blas_threads} threads, so {n_workers} forked workers "
            f"would need ~{n_workers * blas_threads:,} threads (process limit {soft:,}).\n"
            "Fix: make this the FIRST cell (before importing numpy/pandas/etc.), restart the kernel:\n"
            "    import os\n"
            "    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):\n"
            "        os.environ[v] = '1'\n"
            "or raise the limit (ulimit -u 65536 before starting Jupyter).")


class _SingleThreadWorkers:
    """While active: env forces 1 BLAS/OpenMP thread, and this module's folder is on sys.path.
    Workers are started with 'spawn', i.e. fresh interpreters that import numpy AFTER these
    variables are set, so each worker runs exactly 1 BLAS thread. ('fork' copies the notebook's
    already-initialised OpenBLAS, which re-creates its 96-thread pool in every worker and exhausts
    RLIMIT_NPROC - env vars and threadpoolctl cannot undo that after the fact.)
    The notebook's own numpy keeps its threads, so other steps (e.g. kmeans_binary) stay fast."""

    def __enter__(self):
        self._env = {v: os.environ.get(v) for v in _THREAD_VARS}
        for v in _THREAD_VARS:
            os.environ[v] = "1"
        here = os.path.dirname(os.path.abspath(__file__))  # workers must be able to import this module
        self._path_added = here not in sys.path
        if self._path_added:
            sys.path.insert(0, here)
        return self

    def __exit__(self, *exc):
        for v, old in self._env.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
        if self._path_added and os.path.dirname(os.path.abspath(__file__)) in sys.path:
            sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
        return False


def init_worker(x_img, crop_img, pts):
    """pts: (N, 2) xenium_image coords (x_new, y_new); a DataFrame with those columns also works."""
    cv2.setNumThreads(0)
    try:                                   # guard BLAS oversubscription in forked workers
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    if hasattr(pts, "columns"):
        pts = pts[["x_new", "y_new"]].values
    _G.update(xen=x_img, ref=crop_img, pts=np.float32(pts).reshape(-1, 1, 2))


def evaluate_pipeline(ind):
    try:
        M, info = fit_affine(_G["xen"], _G["ref"], ind)
        score = float(fast_alignment_score(_G["ref"], cv2.transform(_G["pts"], M).reshape(-1, 2)))
        if not np.isfinite(score):
            return ind, BAD_SCORE, make_meta("failed", "bad_score", **info)
        if info["used_corr"] < float(ind["min_corr"]):          # penalise relaxed threshold
            score += 0.01 * (float(ind["min_corr"]) - info["used_corr"])
        return ind, score, make_meta("ok", **info, affine=M.tolist())
    except FitFailed as e:
        return ind, BAD_SCORE, make_meta("failed", e.reason, **e.info)
    except Exception as e:
        return ind, BAD_SCORE, make_meta("failed", "exception", error=repr(e))


# ===========================================================================
# 4. Genetic operators
# ===========================================================================

def _fix_ratio_order(ind):
    if ind['ratio_start'] >= ind['ratio_stop']:
        ind['ratio_start'], ind['ratio_stop'] = ind['ratio_stop'], ind['ratio_start']
    return ind


def generate_individual():
    ind = {}
    for key, b in PARAM_BOUNDS.items():
        if isinstance(b[0], bool):
            ind[key] = random.choice(b)
        elif isinstance(b[0], int):
            ind[key] = random.randint(b[0], b[-1])
        else:
            ind[key] = random.uniform(b[0], b[-1])
    return ind


def mutate(individual, mutation_rate=0.25, sigma_scale=0.10):
    m = copy.deepcopy(individual)
    for key, b in PARAM_BOUNDS.items():
        if random.random() >= mutation_rate:
            continue
        if isinstance(b[0], bool):
            m[key] = not m[key]
        else:
            nudge = random.gauss(0, (b[-1] - b[0]) * sigma_scale)
            if isinstance(b[0], int):
                nudge = int(round(nudge))
            m[key] = max(b[0], min(b[-1], m[key] + nudge))
    return _fix_ratio_order(m)


def crossover(p1, p2):
    c1, c2 = {}, {}
    for key, b in PARAM_BOUNDS.items():
        if isinstance(b[0], float):                              # BLX-alpha
            lo, hi = sorted((p1[key], p2[key]))
            span = (hi - lo) * 0.5
            b_lo, b_hi = max(b[0], lo - span), min(b[-1], hi + span)
            c1[key], c2[key] = random.uniform(b_lo, b_hi), random.uniform(b_lo, b_hi)
        elif random.random() > 0.5:
            c1[key], c2[key] = p1[key], p2[key]
        else:
            c1[key], c2[key] = p2[key], p1[key]
    return _fix_ratio_order(c1), _fix_ratio_order(c2)


def tournament_select(ranked_pool, tournament_size=4):
    comp = random.sample(ranked_pool, min(tournament_size, len(ranked_pool)))
    return min(comp, key=lambda x: x[0])[1]


def compute_population_diversity(population):
    keys = [k for k, b in PARAM_BOUNDS.items() if isinstance(b[0], float)]
    if not keys or len(population) < 2:
        return 0.0
    vecs = np.array([[(ind[k] - PARAM_BOUNDS[k][0]) / (PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0])
                      for k in keys] for ind in population])
    n = len(vecs)
    d = [np.linalg.norm(vecs[i] - vecs[j]) / np.sqrt(len(keys))
         for i, j in (random.sample(range(n), 2) for _ in range(min(200, n * (n - 1) // 2)))]
    return float(np.mean(d))


# ===========================================================================
# 5. GA driver
# ===========================================================================

def run_genetic_algorithm(
    xenium_image, CROPPED_ORIGINAL_IMG, xenium_coord_df,
    population_size=144, generations=20,
    elite_frac=0.05, immigrant_frac=0.05, mating_frac=0.50, tournament_size=4,
    mutation_rate_hi=0.35, mutation_rate_lo=0.10, sigma_hi=0.15, sigma_lo=0.03,
    diversity_floor=0.08, stagnation_limit=4,
    output_csv="ga_all_configurations_results.csv",
    n_workers=None, seed=None, mp_start="fork",
):
    """Returns (best_ind, best_score, best_meta, df_results). best_meta['affine'] is the 2x3 matrix.
    mp_start: worker start method. 'fork' (default): fast, workers share the images with the
              notebook, but BLAS must be single-threaded in the kernel (env cell before importing
              numpy) - checked before the pool starts. 'spawn': works without that cell, but every
              worker re-imports numpy/cv2 and receives its own copy of the images (slow startup)."""
    if seed is not None:
        random.seed(seed)
    n_workers = n_workers or min(population_size, _available_cpus())
    print(f"Initializing process pool: {n_workers} workers, population {population_size}")

    population = [generate_individual() for _ in range(population_size)]
    all_history = []
    best_score_ever, best_ind_ever, best_meta_ever = BAD_SCORE, None, None
    stagnation_ctr = 0

    pts = np.float32(xenium_coord_df[["x_new", "y_new"]].values)   # only what workers need
    if mp_start == "fork":
        _check_fork_threads(n_workers)
    with _SingleThreadWorkers():
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context(mp_start),
                                 initializer=init_worker,
                                 initargs=(xenium_image, CROPPED_ORIGINAL_IMG, pts)) as ex:
            for gen in range(generations):
                progress = gen / max(generations - 1, 1)
                mutation_rate = mutation_rate_hi + progress * (mutation_rate_lo - mutation_rate_hi)
                sigma_scale = sigma_hi + progress * (sigma_lo - sigma_hi)
                print(f"\n{'=' * 54}\n  Generation {gen + 1:>3} / {generations}  |  "
                      f"mut={mutation_rate:.3f}  σ={sigma_scale:.3f}\n{'=' * 54}")

                futures = [ex.submit(evaluate_pipeline, ind) for ind in population]
                results = []
                for f in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Eval Gen {gen + 1}", leave=False, colour="green"):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        results.append(({}, BAD_SCORE, make_meta("failed", "future_exception",
                                                                 error=repr(e))))
                results.sort(key=lambda r: r[1])

                for ind, score, meta in results:
                    all_history.append({**ind, "generation": gen + 1, "score": score, **meta})
                pd.DataFrame(all_history).to_csv(output_csv, index=False)

                best_ind_gen, best_score_gen, best_meta_gen = results[0]
                if best_score_gen < best_score_ever - 1e-6:
                    best_score_ever = best_score_gen
                    best_ind_ever, best_meta_ever = copy.deepcopy(best_ind_gen), copy.deepcopy(best_meta_gen)
                    stagnation_ctr = 0
                else:
                    stagnation_ctr += 1

                diversity = compute_population_diversity(population)
                n_ok = sum(m["status"] == "ok" for _, _, m in results)
                print(f"  Best score : {best_score_gen:.6f}  (all-time: {best_score_ever:.6f})")
                print(f"  Valid runs : {n_ok}/{len(results)}  |  Failed: {len(results) - n_ok}")
                print(f"  Best status: {best_meta_gen['status']}  |  Reason: {best_meta_gen['failure_reason']}")
                print(f"  Diversity  : {diversity:.4f}  |  Stagnation: {stagnation_ctr}/{stagnation_limit}")
                print("  Top failure reasons:")
                print(pd.Series([m["failure_reason"] for _, _, m in results])
                      .value_counts(dropna=False).head(5).to_string())

                # ---- next generation ----
                n_elite = max(1, int(population_size * elite_frac))
                next_pop = [copy.deepcopy(r[0]) for r in results[:n_elite] if r[0]]

                n_imm = max(1, int(population_size * immigrant_frac))
                if diversity < diversity_floor or stagnation_ctr >= stagnation_limit:
                    n_imm = max(n_imm, int(population_size * 0.20))
                    if stagnation_ctr >= stagnation_limit:
                        print(f"  ⚠  Stagnation detected – injecting {n_imm} immigrants")
                        stagnation_ctr = 0
                    else:
                        print(f"  ⚠  Low diversity ({diversity:.4f}) – boosting immigration")
                next_pop += [generate_individual() for _ in range(n_imm)]

                n_mating = max(2, int(population_size * mating_frac))
                pool = [(i, r[0]) for i, r in enumerate(results[:n_mating]) if r[0]]
                while len(next_pop) < population_size:
                    if len(pool) < 2:
                        next_pop.append(generate_individual())
                        continue
                    c1, c2 = crossover(tournament_select(pool, tournament_size),
                                       tournament_select(pool, tournament_size))
                    next_pop.append(mutate(c1, mutation_rate, sigma_scale))
                    if len(next_pop) < population_size:
                        next_pop.append(mutate(c2, mutation_rate, sigma_scale))
                population = next_pop[:population_size]

    df_results = pd.DataFrame(all_history)
    lead = ["generation", "score", "status", "failure_reason",
            "used_corr", "n_matches", "n_inliers", "error"]
    df_results = df_results[[c for c in lead if c in df_results]
                            + [c for c in df_results if c not in lead]]
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Optimisation complete.\nBest score ever: {best_score_ever:.6f}\n"
          f"Results saved to: {output_csv}")
    return best_ind_ever, best_score_ever, best_meta_ever, df_results


# ===========================================================================
# 6. Post-run: apply the GA result
# ===========================================================================

def _stats(a):
    v = a[~np.isnan(a)]
    return (np.nan,) * 3 if v.size == 0 else (v.mean(), np.median(v), np.percentile(v, 95))


def _metrics(aligned, target):
    a, t = aligned > 0, target > 0
    err_a2b = np.where(a, _dist_to_mask(t), np.nan)
    err_b2a = np.where(t, _dist_to_mask(a), np.nan)
    return {
        "mae":     np.mean(np.abs(aligned.astype(np.float32) - target.astype(np.float32))),
        "dice":    2 * np.logical_and(a, t).sum() / (a.sum() + t.sum() + 1e-9),
        "signed":  target.astype(np.float32) / 255 - aligned.astype(np.float32) / 255,
        "err_a2b": err_a2b, "a2b_stats": _stats(err_a2b),
        "err_b2a": err_b2a, "b2a_stats": _stats(err_b2a),
    }


def affine_summary(M):
    """Scale, rotation (deg), translation of a 2x3 similarity matrix."""
    M = np.asarray(M, float)
    return dict(scale=float(np.hypot(M[0, 0], M[1, 0])),
                rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                tx=float(M[0, 2]), ty=float(M[1, 2]))


def apply_ga_result(result, xenium_image, cropped_img, xenium_coord_df=None,
                    coord_cols=('x_new', 'y_new'), out_cols=('x_transformed', 'y_transformed'),
                    refit=False):
    """
    Turn the output of run_genetic_algorithm() into an alignment, run_alignment()-style.

    result : the 4-tuple from run_genetic_algorithm, or just the best_ind dict.
    refit  : recompute the affine from the parameters instead of using the stored matrix.
    If xenium_coord_df is given, out_cols are written into it in place.
    """
    if isinstance(result, dict):
        best_ind, best_score, best_meta = result, None, {}
    else:
        best_ind, best_score, best_meta = result[0], result[1], result[2] or {}
    if best_ind is None:
        raise ValueError("GA found no valid individual (best_ind is None)")

    if best_meta.get("affine") is not None and not refit:
        M = np.asarray(best_meta["affine"], np.float64)
        info = {k: best_meta.get(k) for k in ("used_corr", "n_matches", "n_inliers")}
    else:
        M, info = fit_affine(xenium_image, cropped_img, best_ind)

    img1 = (xenium_image > 0).astype(np.uint8) * 255
    img2 = (cropped_img > 0).astype(np.uint8) * 255
    h, w = img2.shape
    aligned = cv2.warpAffine(img1, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if xenium_coord_df is not None:
        transform_coords(xenium_coord_df, M, coord_cols, out_cols)

    return {
        "img1": img1, "img2": img2,
        "n_matches": info["n_matches"], "n_inliers": info["n_inliers"],
        "used_corr": info["used_corr"],
        "affine": M, "affine_summary": affine_summary(M),
        "aligned": aligned, "metrics": _metrics(aligned, img2),
        "params": best_ind, "ga_score": best_score,
    }


# ===========================================================================
# 7. Chamfer visual QC
# ===========================================================================

def _dist_to_mask(mask):
    """Exact Euclidean distance from every pixel to the nearest True pixel (cv2, float32)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _raster(p, shape):
    m = np.zeros(shape, bool)
    m[p[:, 1], p[:, 0]] = True
    return m


def chamfer_distances(ref_img, xy):
    """
    Exact per-point nearest-neighbour distances.
      d_i2x : for every reference-foreground pixel, distance to nearest Xenium point
      d_x2i : for every in-bounds Xenium point, distance to nearest reference-foreground pixel
    """
    fg = ref_img > 0
    H, W = fg.shape
    xy_i = np.rint(np.asarray(xy)).astype(np.int64)
    inb = (xy_i[:, 0] >= 0) & (xy_i[:, 0] < W) & (xy_i[:, 1] >= 0) & (xy_i[:, 1] < H)
    p = xy_i[inb]
    return _dist_to_mask(_raster(p, fg.shape))[fg], _dist_to_mask(fg)[p[:, 1], p[:, 0]], inb


def _dstats(d, thr):
    if d.size == 0:
        return dict(mean=np.nan, median=np.nan, p95=np.nan, within=np.nan, n=0)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)), within=float(100 * (d <= thr).mean()), n=int(d.size))


def _pool(a, f, how, fill=0):
    """Block-reduce a 2D array by integer factor f ('max' or 'mean')."""
    if f == 1:
        return a
    h, w = a.shape
    hd, wd = -(-h // f), -(-w // f)
    b = np.full((hd * f, wd * f), fill, a.dtype)
    b[:h, :w] = a
    b = b.reshape(hd, f, wd, f)
    return b.max(axis=(1, 3)) if how == 'max' else b.mean(axis=(1, 3), dtype=np.float32)


def _grid_max(p, vals, f, shape):
    """Max of vals per display cell (p in full-res pixel coords); -1 where empty."""
    hd, wd = shape
    lin = (p[:, 1] // f) * wd + (p[:, 0] // f)
    o = np.lexsort((vals, lin))
    lin, vals = lin[o], vals[o]
    last = np.r_[lin[1:] != lin[:-1], True]
    g = np.full(hd * wd, -1, np.float32)
    g[lin[last]] = vals[last]
    return g.reshape(hd, wd)


def plot_chamfer_qc(ref_img, coord_df, x_col='x_transformed', y_col='y_transformed',
                    before_cols=('x_new', 'y_new'), thr=5.0, px_size=None,
                    img_name='IMG', xen_name='XENIUM', overlay_sigma=3.0, display_px=900,
                    save_path=None, dpi=150, show=True, out_dir=None):
    """
    Visual proof of alignment quality using bidirectional Chamfer distances.
    out_dir: shortcut for save_path=out_dir/alignment_qc.png

    Row 1: difference overlay (after) | IMG→XEN error map | XEN→IMG error map
    Row 2: difference overlay (before, optional) | cumulative error curves | summary table

    Distances/stats are exact at full resolution; panels are rendered at ~display_px
    (max-pooled, so small errors are never averaged away).

    before_cols   : untransformed coords shown as a baseline (None to skip)
    thr           : distance threshold (px) reported as '% within thr'
    px_size       : µm per pixel; if given, stats are also reported in µm
    overlay_sigma : Gaussian smoothing (full-res px) for the difference overlay
    display_px    : longest side of each rendered panel; lower = faster
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if save_path is None and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "alignment_qc.png")
    fg_full = ref_img > 0
    H, W = fg_full.shape

    # --- integer coords, in-bounds masks ---
    sets = {'after': coord_df[[x_col, y_col]].to_numpy()}
    if before_cols is not None and all(c in coord_df for c in before_cols):
        sets['before'] = coord_df[list(before_cols)].to_numpy()
    pix = {}
    for k, xy in sets.items():
        xy_i = np.rint(xy).astype(np.int64)
        inb = (xy_i[:, 0] >= 0) & (xy_i[:, 0] < W) & (xy_i[:, 1] >= 0) & (xy_i[:, 1] < H)
        pix[k] = (xy_i[inb], 100 * (1 - inb.mean()))

    # --- crop to bbox of tissue ∪ all points (exact: nothing relevant lies outside) ---
    bx, by, bw, bh = cv2.boundingRect(fg_full.astype(np.uint8))
    x0, y0, x1, y1 = bx, by, bx + bw, by + bh
    for p, _ in pix.values():
        if len(p):
            x0, y0 = min(x0, p[:, 0].min()), min(y0, p[:, 1].min())
            x1, y1 = max(x1, p[:, 0].max() + 1), max(y1, p[:, 1].max() + 1)
    pad = max(5, int(0.02 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    fg = fg_full[y0:y1, x0:x1]
    dt_fg = _dist_to_mask(fg)

    R = {}
    for k, (p, oob) in pix.items():
        p = p - [x0, y0]
        dt_x = _dist_to_mask(_raster(p, fg.shape))
        d_i2x, d_x2i = dt_x[fg], dt_fg[p[:, 1], p[:, 0]]
        R[k] = dict(p=p, oob=oob, dt_x=dt_x, d_i2x=d_i2x, d_x2i=d_x2i,
                    s_i2x=_dstats(d_i2x, thr), s_x2i=_dstats(d_x2i, thr))
    A, B = R['after'], R.get('before')
    s_i2x, s_x2i = A['s_i2x'], A['s_x2i']
    vmax = max(s_i2x['p95'], s_x2i['p95'], thr)

    # --- display grids ---
    f = max(1, int(np.ceil(max(fg.shape) / display_px)))
    fg_d = _pool(fg, f, 'max', False)
    fg_mean = _pool(fg.astype(np.float32), f, 'mean')
    hd, wd = fg_d.shape
    sig = max(overlay_sigma / f, 0.5)

    C_IMG, C_XEN = '#e08214', '#8073ac'           # PuOr ends: orange = IMG, purple = XEN
    cmap_err = plt.get_cmap('magma_r').copy(); cmap_err.set_bad(alpha=0)

    def _frame(ax, title):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11, fontweight='bold')

    def _overlay(ax, p, title):
        cnt = np.bincount((p[:, 1] // f) * wd + p[:, 0] // f,
                          minlength=hd * wd).reshape(hd, wd).astype(np.float32)
        xen_s = cv2.GaussianBlur(cnt, (0, 0), sig)
        nz = xen_s[xen_s > 1e-6]
        xen_s = np.clip(xen_s / (np.percentile(nz, 50) if nz.size else 1), 0, 1)
        img_s = np.clip(cv2.GaussianBlur(fg_mean, (0, 0), sig), 0, 1)
        a_img, a_xen = img_s > 0.5, xen_s > 0.5
        union = max((a_img | a_xen).sum(), 1)
        ax.imshow(xen_s - img_s, cmap='PuOr', vmin=-1, vmax=1, interpolation='nearest')
        _frame(ax, f'{title}\n{img_name}-only {100 * (a_img & ~a_xen).sum() / union:.1f}%  │  '
                   f'{xen_name}-only {100 * (a_xen & ~a_img).sum() / union:.1f}%  │  '
                   f'matched {100 * (a_img & a_xen).sum() / union:.1f}%')
        ax.legend(handles=[Patch(fc=C_IMG, label=f'{img_name} only'),
                           Patch(fc=C_XEN, label=f'{xen_name} only'),
                           Patch(fc='#f7f7f7', ec='0.6', label='matched')],
                  loc='lower right', fontsize=8, framealpha=0.9)

    def _cdf(ax, d, **kw):
        if d.size:
            q = np.linspace(0, 100, 401)
            ax.plot(np.percentile(d, q), q, **kw)

    fig = plt.figure(figsize=(21, 12), facecolor='white')
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1], hspace=0.25, wspace=0.15)

    # Row 1 ------------------------------------------------------------
    _overlay(fig.add_subplot(gs[0, 0]), A['p'], 'Difference overlay after alignment')

    ax = fig.add_subplot(gs[0, 1])
    e = np.where(fg, A['dt_x'], -1).astype(np.float32)
    im = ax.imshow(np.ma.masked_less(_pool(e, f, 'max', -1), 0), cmap=cmap_err,
                   vmin=0, vmax=vmax, interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label('distance (px)')
    _frame(ax, f'{img_name}→{xen_name}: tissue pixels far from any transcript\n'
               f'mean={s_i2x["mean"]:.2f}  med={s_i2x["median"]:.2f}  p95={s_i2x["p95"]:.2f} px')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(fg_d, cmap='Greys', vmin=0, vmax=3, interpolation='nearest')
    g = _grid_max(A['p'], A['d_x2i'], f, (hd, wd))
    im = ax.imshow(np.ma.masked_less(g, 0), cmap=cmap_err, vmin=0, vmax=vmax,
                   interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label('distance (px)')
    _frame(ax, f'{xen_name}→{img_name}: transcripts outside tissue\n'
               f'mean={s_x2i["mean"]:.2f}  med={s_x2i["median"]:.2f}  p95={s_x2i["p95"]:.2f} px')

    # Row 2 ------------------------------------------------------------
    if B is not None:
        _overlay(fig.add_subplot(gs[1, 0]), B['p'], 'Difference overlay before alignment')
        ax_cdf = fig.add_subplot(gs[1, 1])
    else:
        ax_cdf = fig.add_subplot(gs[1, :2])

    _cdf(ax_cdf, A['d_i2x'], color=C_IMG, lw=2.2, label=f'{img_name}→{xen_name} (after)')
    _cdf(ax_cdf, A['d_x2i'], color=C_XEN, lw=2.2, label=f'{xen_name}→{img_name} (after)')
    if B is not None:
        _cdf(ax_cdf, B['d_i2x'], color=C_IMG, lw=1.4, ls='--', alpha=0.7, label='… before')
        _cdf(ax_cdf, B['d_x2i'], color=C_XEN, lw=1.4, ls='--', alpha=0.7, label='… before')
    ax_cdf.axvline(thr, color='k', ls=':', lw=1.2)
    ax_cdf.text(thr, 3, f'  {thr:g} px', fontsize=9, va='bottom')
    ax_cdf.set_xlim(0, max(vmax * 2, thr * 2)); ax_cdf.set_ylim(0, 101)
    ax_cdf.set_xlabel('distance to nearest partner (px)')
    ax_cdf.set_ylabel('% of points within distance')
    ax_cdf.set_title('Cumulative error (higher & further left = better)',
                     fontsize=11, fontweight='bold')
    ax_cdf.grid(alpha=0.3); ax_cdf.legend(fontsize=9, loc='lower right', frameon=False)
    ax_cdf.spines[['top', 'right']].set_visible(False)

    ax_t = fig.add_subplot(gs[1, 2]); ax_t.axis('off')
    u = f' ({px_size:g} µm/px)' if px_size else ''
    fmt = (lambda v: f'{v:.2f} px / {v * px_size:.2f} µm') if px_size else (lambda v: f'{v:.2f} px')
    cols, stats_sets = [f'{img_name}→{xen_name}', f'{xen_name}→{img_name}'], [s_i2x, s_x2i]
    if B is not None:
        cols += ['before I→X', 'before X→I']; stats_sets += [B['s_i2x'], B['s_x2i']]
    rows = [('mean', lambda s: fmt(s['mean'])), ('median', lambda s: fmt(s['median'])),
            ('p95', lambda s: fmt(s['p95'])), (f'% ≤ {thr:g} px', lambda s: f"{s['within']:.1f}%"),
            ('n points', lambda s: f"{s['n']:,}")]
    tbl = ax_t.table(cellText=[[fn(s) for s in stats_sets] for _, fn in rows],
                     rowLabels=[r for r, _ in rows], colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    ax_t.set_title(f'Summary{u}\n{xen_name} points out of image bounds: {A["oob"]:.2f}%',
                   fontsize=11, fontweight='bold')

    chamfer_sum = s_i2x['mean'] + s_x2i['mean']
    title = f'{img_name} vs {xen_name}  │  Chamfer sum = {chamfer_sum:.2f} px'
    if B is not None:
        title += f'  (before: {B["s_i2x"]["mean"] + B["s_x2i"]["mean"]:.2f} px)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {'IMG_to_XENIUM': s_i2x, 'XENIUM_to_IMG': s_x2i, 'chamfer_sum': chamfer_sum,
            'pct_out_of_bounds': A['oob'], 'dist_IMG_to_XENIUM': A['d_i2x'],
            'dist_XENIUM_to_IMG': A['d_x2i'],
            'before': None if B is None else
            {'IMG_to_XENIUM': B['s_i2x'], 'XENIUM_to_IMG': B['s_x2i']}}



# ===========================================================================
# 8. Annotations -> per-cell region labels
# ===========================================================================

def shift_annotations(annotations, dx, dy):
    """Translate annotation points by (-dx, -dy)."""
    return [{**a, "points": np.asarray(a["points"], float) - [dx, dy]} for a in annotations]


def find_crop_offset(parent, child, coarse=4, min_score=0.9):
    """
    Locate `child` (CROPPED_ORIGINAL_IMG) inside `parent` (binary_img) at the same pixel scale,
    including children padded past the parent's edges.
    Returns ((dx, dy), score) with  parent_px = child_px + (dx, dy).
    """
    P = (np.asarray(parent) > 0).astype(np.float32)
    C = np.asarray(child, np.float32)
    C = C / C.max() if C.max() > 0 else C
    ch, cw = C.shape[:2]
    Pp = cv2.copyMakeBorder(P, ch, ch, cw, cw, cv2.BORDER_CONSTANT, value=0)

    f = max(1, int(coarse))
    small = lambda a: cv2.resize(a, (max(1, a.shape[1] // f), max(1, a.shape[0] // f)),
                                 interpolation=cv2.INTER_AREA)
    _, _, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(small(Pp), small(C), cv2.TM_CCOEFF_NORMED))

    x, y, m = x * f, y * f, 2 * f                      # refine at full resolution
    x0 = min(max(0, x - m), Pp.shape[1] - cw)
    y0 = min(max(0, y - m), Pp.shape[0] - ch)
    win = Pp[y0:min(Pp.shape[0], y + m + ch), x0:min(Pp.shape[1], x + m + cw)]
    _, score, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(win, C, cv2.TM_CCOEFF_NORMED))

    if score < min_score:
        warnings.warn(f"find_crop_offset: match score {score:.3f} < {min_score}; child may be "
                      f"rescaled/rotated relative to parent - use the crop coordinates instead.")
    return (x0 + bx - cw, y0 + by - ch), float(score)


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def label_cells_by_annotation(coord_df, annotations, x_col="x_transformed", y_col="y_transformed",
                              out_col="region", unlabeled="unannotated"):
    """
    Label each cell with the polygon it falls in; nested polygons -> the SMALLEST wins.
    Writes out_col (title), out_col+'_id' (annotation index, -1 = none), out_col+'_area' (px^2).
    Returns value counts of out_col.
    """
    from matplotlib.path import Path
    xy = coord_df[[x_col, y_col]].to_numpy(float)
    labels = np.full(len(xy), unlabeled, dtype=object)
    ids = np.full(len(xy), -1, np.int64)
    areas_out = np.full(len(xy), np.nan)

    areas = [_poly_area(np.asarray(a["points"], float)) for a in annotations]
    for i in np.argsort(areas)[::-1]:                   # largest first; smaller overwrite
        p = np.asarray(annotations[i]["points"], float)
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        cand = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        inside = cand[Path(p).contains_points(xy[cand])]
        labels[inside], ids[inside], areas_out[inside] = annotations[i]["title"], i, areas[i]

    coord_df[out_col], coord_df[out_col + "_id"], coord_df[out_col + "_area"] = labels, ids, areas_out
    return coord_df[out_col].value_counts()


def label_cells_from_annotations(coord_df, sample_anns, binary_img, cropped_img,
                                 x_col="x_transformed", y_col="y_transformed", out_col="region",
                                 plot=True, min_score=0.9, crop_coordinates=None, out_dir=None):
    """
    One call: sample-crop annotations -> cropped_img grid -> per-cell labels.
    binary_img must share the pixel grid of the sample crop the annotations live in.
    crop_coordinates : (y0, y1, x0, x1) from mall.contour_crop_binary. If given, the offset is
                       exact and find_crop_offset (template matching) is skipped (score = 1.0).
    out_dir          : saves the label plot (with per-region cell counts) as out_dir/cell_labels.png.
    Returns (counts, crop_anns, (dx, dy), score).
    """
    if crop_coordinates is not None:
        dy, _, dx, _ = (int(v) for v in crop_coordinates)
        score = 1.0
    else:
        (dx, dy), score = find_crop_offset(binary_img, cropped_img, min_score=min_score)
    crop_anns = shift_annotations(sample_anns, dx, dy)
    counts = label_cells_by_annotation(coord_df, crop_anns, x_col, y_col, out_col)
    print(f"crop offset=({dx}, {dy})  match score={score:.3f}\n{counts.to_string()}")
    if plot or out_dir is not None:
        save_path = None
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "cell_labels.png")
        plot_cell_labels(cropped_img, coord_df, crop_anns, x_col, y_col, out_col,
                         save_path=save_path, show=plot)
    return counts, crop_anns, (dx, dy), score


def annotations_to_xenium(annotations, M):
    """Map annotations from cropped_img px to Xenium px (M maps Xenium -> image; inverse applied)."""
    M_inv = cv2.invertAffineTransform(np.asarray(M, np.float64))
    return [{**a, "points": cv2.transform(np.float32(a["points"]).reshape(-1, 1, 2), M_inv).reshape(-1, 2)}
            for a in annotations]


def plot_cell_labels(img, coord_df, annotations, x_col="x_transformed", y_col="y_transformed",
                     label_col="region", unlabeled="unannotated", n_plot=200_000, s=0.5, seed=0,
                     save_path=None, show=True):
    """Left: annotation outlines on the image. Right: cells coloured by region label.
    save_path: also save the figure there; show=False closes it instead of displaying."""
    import matplotlib.pyplot as plt
    titles = sorted({a["title"] for a in annotations})
    colours = {**dict(zip(titles, plt.cm.tab10.colors)), unlabeled: (0.8, 0.8, 0.8)}
    sub = coord_df.iloc[np.random.default_rng(seed).permutation(len(coord_df))[:n_plot]]
    counts = coord_df[label_col].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a in ax:
        a.imshow(img, cmap="gray", alpha=0.35)
    for lab in [unlabeled] + titles:                    # unlabeled underneath
        d = sub[sub[label_col] == lab]
        ax[1].scatter(d[x_col], d[y_col], s=s, color=colours[lab], edgecolors="none",
                      rasterized=True, label=f"{lab} ({counts.get(lab, 0):,})")
    for an in annotations:
        p = np.vstack([an["points"], an["points"][:1]])
        for a in ax:
            a.plot(p[:, 0], p[:, 1], color=colours[an["title"]], lw=1.5)
    ax[0].set_title("Annotations on alignment image")
    ax[1].set_title("Cells labelled (smallest enclosing region wins)")
    ax[1].legend(markerscale=12, fontsize=9, loc="upper right")
    for a in ax:
        a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0); a.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.show() if show else plt.close(fig)


# ===========================================================================
# 9. Map the alignment back to the original H&E
# ===========================================================================
#
#   IMAGE px  =  CROPPED_ORIGINAL_IMG px  +  (crop x0, crop y0)  +  sample_offset
#   (x_transformed lives on the CROPPED_ORIGINAL_IMG grid)
#
#   crop_coordinates : (y0, y1, x0, x1) from mall.contour_crop_binary
#   sample_offset    : info["total_offset"] from ac.crop_sample_with_annotations,
#                      (0, 0) when no auto-crop was done
#   level_downsample : ndpi_meta["downsample"] from
#                      mall.load_ndpi_with_annotations(..., return_meta=True)

def crop_to_image_offset(crop_coordinates, sample_offset=(0, 0)):
    """(dx, dy) with  IMAGE_px = CROPPED_ORIGINAL_IMG_px + (dx, dy)."""
    y0, _, x0, _ = crop_coordinates
    return int(x0) + int(sample_offset[0]), int(y0) + int(sample_offset[1])


def _as_xy(v):
    return np.broadcast_to(np.asarray(v, float), (2,))


def _to_level0(v, ds):
    return (v + 0.5) * ds - 0.5                         # pixel-centre convention


def to_original_he(coord_df, crop_coordinates, sample_offset=(0, 0),
                   in_cols=('x_transformed', 'y_transformed'), out_cols=('x_he', 'y_he'),
                   level_downsample=None, level0_cols=('x_he_l0', 'y_he_l0'), out_dir=None):
    """
    Shift CROPPED_ORIGINAL_IMG coords (x_transformed) into the H&E level you loaded (IMAGE).
    Writes out_cols in place; if level_downsample is given also writes level0_cols (full-res px).
    Consistent with the coords used for the region labels (same rounding).
    out_dir: saves the final table as out_dir/xenium_coord_df.parquet (.csv if parquet is unavailable).
    """
    dx, dy = crop_to_image_offset(crop_coordinates, sample_offset)
    coord_df[out_cols[0]] = coord_df[in_cols[0]] + dx
    coord_df[out_cols[1]] = coord_df[in_cols[1]] + dy
    if level_downsample is not None:
        sx, sy = _as_xy(level_downsample)
        coord_df[level0_cols[0]] = _to_level0(coord_df[out_cols[0]], sx)
        coord_df[level0_cols[1]] = _to_level0(coord_df[out_cols[1]], sy)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "xenium_coord_df.parquet")
        try:
            coord_df.to_parquet(path)
        except Exception as e:
            warnings.warn(f"parquet failed ({e!r}); writing csv instead")
            coord_df.to_csv(os.path.splitext(path)[0] + ".csv")
    return coord_df


def xenium_to_he_affine(M, xen_meta, crop_coordinates, sample_offset=(0, 0), level_downsample=None):
    """
    Single 2x3 affine: raw Xenium coords (the µm columns given to mall.xenium_binarize, e.g.
    centroid_x / centroid_y, or transcript x_location / y_location) -> IMAGE px
    (or level-0 px if level_downsample is given).

    M        : GA affine (r["affine"] / result[2]["affine"]), xenium_image px -> CROPPED_ORIGINAL_IMG px
    xen_meta : 3rd output of mall.xenium_binarize (needs 'scale' and 'shift_xy')

    Sub-pixel: skips the integer rounding inside xenium_binarize, so it can differ from
    to_original_he by ~0.5 xenium_image px (times the GA scale).
    Inverse (H&E -> Xenium µm): cv2.invertAffineTransform(T).
    """
    if 'scale' not in xen_meta:
        raise KeyError("xen_meta has no 'scale' - rerun mall.xenium_binarize with the updated mal_load")
    s = float(xen_meta['scale'])
    sx, sy = xen_meta['shift_xy']
    A = np.array([[1 / s, 0, sx], [0, 1 / s, sy], [0, 0, 1]], float)   # µm -> xenium_image px
    B = np.vstack([np.asarray(M, float), [0, 0, 1]])                    # -> CROPPED_ORIGINAL_IMG px
    dx, dy = crop_to_image_offset(crop_coordinates, sample_offset)
    C = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)            # -> IMAGE px
    T = C @ B @ A
    if level_downsample is not None:
        ds_x, ds_y = _as_xy(level_downsample)
        D = np.array([[ds_x, 0, 0.5 * ds_x - 0.5], [0, ds_y, 0.5 * ds_y - 0.5], [0, 0, 1]])
        T = D @ T                                                       # -> level-0 px
    return T[:2]


def apply_affine_to_df(df, T, in_cols=('x_location', 'y_location'), out_cols=('x_he', 'y_he')):
    """Apply a 2x3 affine (e.g. from xenium_to_he_affine) to any DataFrame, in place (float64)."""
    xy = df[list(in_cols)].to_numpy(np.float64)
    T = np.asarray(T, np.float64)
    df[out_cols[0]] = xy @ T[0, :2] + T[0, 2]
    df[out_cols[1]] = xy @ T[1, :2] + T[1, 2]
    return df