import os
import json
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt


def _view(IMAGE, max_side=1600):
    img = np.asarray(IMAGE.convert("RGB") if hasattr(IMAGE, "convert") else IMAGE)[..., :3]
    s = min(1, max_side / max(img.shape[:2]))
    return img, (cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)


def _show_image(img_arr: np.ndarray, title: str | None = None) -> None:
    """Displays a 3-axis (H, W, C) color image array."""
    if img_arr.ndim != 3:
        raise ValueError(f"Expected a 3-axis array (H, W, C), got {img_arr.ndim} dimensions.")

    # If RGBA, slice down to RGB; if BGR, assume user passed standard RGB
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_arr)
    plt.axis("off")  # Hide axis ticks and numbers
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def _detect_frame(IMAGE, plot=False, target_color="#E0D6DF"):
    _, view = _view(IMAGE)
    target = np.array([int(target_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
    m = (np.linalg.norm(view.astype(np.float32) - target, axis=2) < 16).astype(np.uint8)
    H, W = m.shape
    k = max(7, round(min(H, W) * 0.07)) | 1

    def runs(close_k, open_k):
        r = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones(close_k, np.uint8))
        return cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones(open_k, np.uint8))

    hor = runs((1, k), (1, max(8, round(W * .07))))
    ver = runs((k, 1), (max(8, round(H * .07)), 1))

    def peaks(scores, cutoff):
        ids = np.flatnonzero(scores >= cutoff)
        groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
        best = [int(max(g, key=lambda i: int(scores[i]))) for g in groups if len(g)]
        return sorted(best, key=lambda i: int(scores[i]), reverse=True)[:16]

    xs, ys = peaks(ver.sum(0), H * .13), peaks(hor.sum(1), W * .13)
    t = max(2, round(min(H, W) * .01))
    best = (0, None)

    for x1 in xs:
        for x2 in xs:
            if x2 - x1 < W * .15:
                continue
            for y1 in ys:
                for y2 in ys:
                    if y2 - y1 < H * .15 or (x2 - x1) * (y2 - y1) < W * H * .1:
                        continue
                    sides = [hor[max(0, y - t):min(H, y + t + 1), x1:x2 + 1].max(0).mean() for y in (y1, y2)] + \
                            [ver[y1:y2 + 1, max(0, x - t):min(W, x + t + 1)].max(1).mean() for x in (x1, x2)]
                    score = min(sides) + .1 * np.mean(sides) + .01 * (x2 - x1) * (y2 - y1) / (W * H)
                    if score > best[0]:
                        best = (score, (x1, y1, x2, y2))

    if best[1] is None:
        return False, None

    x1, y1, x2, y2 = best[1]

    if plot:
        prev = view.copy()
        cv2.rectangle(prev, (x1, y1), (x2, y2), (255, 0, 0), 3)
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(hor + 2 * ver, cmap="viridis")
        ax[0].set_title("Long horizontal and vertical color runs")
        ax[1].imshow(prev)
        ax[1].set_title(f"Detected frame: {(x1, y1, x2, y2)}")
        for a in ax:
            a.axis("off")
        plt.show()

    return True, (x1, x2, y1, y2)


def _crop_frame(IMAGE, coords, path=None):
    """Crop using coords from the resized view. Saves to `path` if given."""
    if coords is None:
        return None

    img, view = _view(IMAGE)
    H, W = img.shape[:2]
    h, w = view.shape[:2]
    x1, x2, y1, y2 = coords
    cropped = img[round(y1 * H / h):round(y2 * H / h), round(x1 * W / w):round(x2 * W / w)]

    if path:
        cv2.imwrite(str(path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    return cropped


def crop_frame(IMAGE, path=None, plot=False, plot_debug=False, target_color="#E0D6DF",
               return_offset=False):
    """Detect the frame and return the cropped image (None if not detected).
    Saves to `path` only when a frame is detected.
    plot: show the cropped image. plot_debug: show the detection debug plot.
    return_offset: also return the (x, y) top-left of the crop in IMAGE pixels ((0, 0) if not detected)."""
    detected, coords = _detect_frame(IMAGE, plot=plot_debug, target_color=target_color)
    if not detected:
        return (None, (0, 0)) if return_offset else None
    cropped = _crop_frame(IMAGE, coords, path=path)
    if plot:
        plt.imshow(cropped)
        plt.axis("off")
        plt.show()
    if return_offset:
        img, view = _view(IMAGE)
        x1, _, y1, _ = coords
        return cropped, (round(x1 * img.shape[1] / view.shape[1]), round(y1 * img.shape[0] / view.shape[0]))
    return cropped


def find_samples(IMAGE, plot=True, sat_thresh=None, hue_ranges=((0, 20), (120, 180)),
                 merge_frac=0.02, min_area_frac=0.005, max_side=1000):
    """Count H&E tissue samples and map their density.
    Tissue = saturated pixels (Otsu on S unless sat_thresh given) with red/pink/purple hue
    (OpenCV hue 0-180). merge_frac: gap (fraction of image) bridged to join pieces of one
    sample. Returns dict: n, samples (bbox/centroid in input pixels), mask, density, sat_thresh."""
    img = np.asarray(IMAGE)[..., :3]
    s = min(1, max_side / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    H, W = view.shape[:2]

    hsv = cv2.cvtColor(cv2.GaussianBlur(view, (5, 5), 0), cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    thr = sat_thresh if sat_thresh is not None else cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    hue_ok = np.zeros(hue.shape, bool)
    for lo, hi in hue_ranges:
        hue_ok |= (hue >= lo) & (hue <= hi)
    tissue = ((sat > thr) & hue_ok).astype(np.uint8)

    k = max(3, round(min(H, W) * merge_frac)) | 1
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    blob = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, ell(k))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, ell(max(3, k // 2) | 1))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(blob)
    keep = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * H * W),
                  key=lambda i: cents[i][0])  # left to right

    density = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), min(H, W) * .02)
    samples = []
    for j, i in enumerate(keep, 1):
        x, y, w, h, a = stats[i]
        samples.append(dict(id=j, bbox=tuple(round(v / s) for v in (x, y, x + w, y + h)),  # x1,y1,x2,y2
                            centroid=tuple(round(v / s) for v in cents[i]),
                            area_frac=float(a / (H * W)), density=float(tissue[labels == i].mean())))
    mask = np.isin(labels, keep)

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0, 0].imshow(view)
        ax[0, 0].contour(mask, levels=[.5], colors="lime", linewidths=1.5)
        for d in samples:
            ax[0, 0].text(d["centroid"][0] * s, d["centroid"][1] * s, str(d["id"]), color="yellow",
                          fontsize=16, weight="bold", ha="center", va="center")
        ax[0, 0].set_title(f"{len(samples)} samples detected")
        ax[0, 1].hist(sat.ravel(), 128, color="gray")
        ax[0, 1].axvline(thr, color="r", ls="--", label=f"threshold = {thr:.0f}")
        ax[0, 1].set(title="Saturation (background vs tissue)", yscale="log"); ax[0, 1].legend()
        hv = hue[tissue.astype(bool)]
        counts, _ = np.histogram(hv, bins=180, range=(0, 180))
        cols = cv2.cvtColor(np.array([[[b, 255, 255] for b in range(180)]], np.uint8), cv2.COLOR_HSV2RGB)[0] / 255
        ax[1, 0].bar(range(180), counts, color=cols, width=1)
        ax[1, 0].set_title("Hue of tissue pixels (OpenCV 0-180: red ~0, purple ~130, pink ~165)")
        im = ax[1, 1].imshow(density, cmap="magma")
        ax[1, 1].set_title("Tissue density"); plt.colorbar(im, ax=ax[1, 1], fraction=.046)
        for a in (ax[0, 0], ax[1, 1]):
            a.axis("off")
        plt.tight_layout(); plt.show()

    return dict(n=len(samples), samples=samples, mask=mask, density=density, sat_thresh=float(thr))


def hue_profile_x(IMAGE, plot=True, sat_thresh=None, hue_ranges=((0, 20), (120, 180)), bins=None):
    """1D profile along x: fraction of pixels in each column that are H&E-colored
    (saturated + red/pink/purple hue). bins=None gives one value per pixel column."""
    img = np.asarray(IMAGE)[..., :3]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    thr = sat_thresh if sat_thresh is not None else cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ok = np.zeros(hue.shape, bool)
    for lo, hi in hue_ranges:
        ok |= (hue >= lo) & (hue <= hi)
    prof = ((sat > thr) & ok).mean(axis=0)

    if bins:
        prof = np.array([c.mean() for c in np.array_split(prof, bins)])

    if plot:
        x = np.linspace(0, img.shape[1], len(prof))
        plt.figure(figsize=(12, 3))
        plt.fill_between(x, prof, color="crimson", alpha=.6)
        plt.xlim(0, img.shape[1]); plt.xlabel("x (px)"); plt.ylabel("H&E pixel density")
        plt.show()

    return prof


def _tissue_mask(view, sat_thresh=None, hue_ranges=((0, 20), (120, 180))):
    hsv = cv2.cvtColor(cv2.GaussianBlur(view, (5, 5), 0), cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    thr = sat_thresh if sat_thresh is not None else cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ok = np.zeros(hue.shape, bool)
    for lo, hi in hue_ranges:
        ok |= (hue >= lo) & (hue <= hi)
    return ((sat > thr) & ok).astype(np.uint8)


def _clean_mask(mask, close_frac=0.015, min_frac=0.002):
    """Drop debris: close small gaps, then remove blobs smaller than min_frac of the largest blob."""
    k = max(3, round(min(mask.shape) * close_frac)) | 1
    blob = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    if n < 2:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_frac * areas.max()) + 1
    return mask * np.isin(lab, keep).astype(np.uint8)


def _widest_gap(sub, ax):
    """Widest empty run in the x ('x') or y ('y') projection of a binary mask -> (width, (start, end))."""
    idx = np.flatnonzero(sub.sum(axis=0 if ax == "x" else 1))
    if len(idx) < 2:
        return 0, None
    d = np.diff(idx)
    j = int(d.argmax())
    w = int(d[j]) - 1
    return (w, (int(idx[j]) + 1, int(idx[j + 1]))) if w > 0 else (0, None)


def count_samples(IMAGE, n_samples=None, plot=True, gap_frac=0.03, sat_thresh=None,
                  hue_ranges=((0, 20), (120, 180)), min_frac=0.002, max_side=1200, save_path=None):
    """Find H&E samples by recursively cutting the tissue mask at the widest empty gaps in its
    x / y projections (XY-cut).
      n_samples given -> make the (n_samples - 1) widest cuts (supervised).
      n_samples None  -> cut every gap wider than gap_frac * max(H, W) (automatic).
    Returns dict: n, samples (bbox x1,y1,x2,y2 / centroid in input pixels, ordered left->right then
    top->bottom), cuts, next_gap (widest gap NOT cut, in view px - compare to cut widths), mask.
    save_path: also save the detection figure there (works with plot=False)."""
    img = np.asarray(IMAGE)[..., :3]
    s = min(1, max_side / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    H, W = view.shape[:2]
    mask = _clean_mask(_tissue_mask(view, sat_thresh, hue_ranges), min_frac=min_frac)
    if not mask.any():
        return dict(n=0, samples=[], cuts=[], next_gap=0, mask=mask)

    def tight(x0, y0, x1, y1):
        sub = mask[y0:y1, x0:x1]
        xs, ys = np.flatnonzero(sub.any(0)), np.flatnonzero(sub.any(1))
        return (x0 + xs[0], y0 + ys[0], x0 + xs[-1] + 1, y0 + ys[-1] + 1)

    def best_gap(regions):
        best = (0, None, None, None)
        for i, (x0, y0, x1, y1) in enumerate(regions):
            for ax in "xy":
                w, g = _widest_gap(mask[y0:y1, x0:x1], ax)
                if g and w > best[0]:
                    best = (w, i, ax, g)
        return best

    min_gap = gap_frac * max(H, W)
    regions, cuts = [tight(0, 0, W, H)], []
    while n_samples is None or len(regions) < n_samples:
        w, i, ax, g = best_gap(regions)
        if i is None or (n_samples is None and w < min_gap):
            break
        a, b = g
        x0, y0, x1, y1 = regions.pop(i)
        if ax == "x":
            regions += [tight(x0, y0, x0 + a, y1), tight(x0 + b, y0, x1, y1)]
            cuts.append(dict(axis="x", start=x0 + a, end=x0 + b, width=w))
        else:
            regions += [tight(x0, y0, x1, y0 + a), tight(x0, y0 + b, x1, y1)]
            cuts.append(dict(axis="y", start=y0 + a, end=y0 + b, width=w))
    next_gap = best_gap(regions)[0]

    if n_samples is not None and len(regions) < n_samples:
        warnings.warn(f"Only {len(regions)} separable samples found (asked for {n_samples}); "
                      f"try a larger n_samples only if the samples are truly separated by empty gaps.")

    samples = []
    for x0, y0, x1, y1 in regions:
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        samples.append(dict(bbox=tuple(round(v / s) for v in (x0, y0, x1, y1)),
                            centroid=(round((xs.mean() + x0) / s), round((ys.mean() + y0) / s)),
                            area_frac=float(len(xs) / (H * W)), density=float(len(xs) / ((x1 - x0) * (y1 - y0)))))
    # reading order: group into rows (a sample joins a row if its y-centre falls inside the row's
    # y-span), rows top -> bottom, samples left -> right within each row
    rows = []
    for d in sorted(samples, key=lambda d: d["centroid"][1]):
        cy = d["centroid"][1]
        row = next((r for r in rows if r["y0"] <= cy <= r["y1"]), None)
        if row is None:
            rows.append(dict(y0=d["bbox"][1], y1=d["bbox"][3], items=[d]))
        else:
            row["items"].append(d)
            row["y0"], row["y1"] = min(row["y0"], d["bbox"][1]), max(row["y1"], d["bbox"][3])
    rows.sort(key=lambda r: r["y0"])
    samples = [d for r in rows for d in sorted(r["items"], key=lambda d: d["centroid"][0])]
    for j, d in enumerate(samples, 1):
        d["id"] = j

    if plot or save_path:
        fig = plt.figure(figsize=(13, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[4, 1], hspace=.05, wspace=.05)
        axi = fig.add_subplot(gs[0, 0])
        axx = fig.add_subplot(gs[1, 0], sharex=axi)
        axy = fig.add_subplot(gs[0, 1], sharey=axi)
        axi.imshow(view)
        for d in samples:
            x0, y0, x1, y1 = (v * s for v in d["bbox"])
            axi.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
            axi.text(x0 + 6, y0 + 6, str(d["id"]), color="yellow", fontsize=16, weight="bold", va="top")
        axi.set_title(f"{len(samples)} samples ({'supervised' if n_samples else 'auto'})")
        axx.fill_between(np.arange(W), mask.mean(0), color="crimson", alpha=.6)
        axy.fill_betweenx(np.arange(H), mask.mean(1), color="crimson", alpha=.6)
        for c in cuts:
            (axx.axvspan if c["axis"] == "x" else axy.axhspan)(c["start"], c["end"], color="dodgerblue", alpha=.3)
        axx.set_xlabel("x (px, resized view)  |  blue = cut gaps"); axx.set_ylabel("density")
        plt.setp(axi.get_xticklabels(), visible=False)
        axy.tick_params(labelleft=False)
        if save_path:
            fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor="white")
        plt.show() if plot else plt.close(fig)

    return dict(n=len(samples), samples=samples, cuts=cuts, next_gap=next_gap, mask=mask)


def crop_samples(IMAGE, n_samples=None, res=None, pad_frac=0.05, path=None, plot=True, **count_kw):
    """Crop each sample found by count_samples with a breathing-room buffer.
    Buffer = pad_frac * max(H, W) per side, but a side facing a neighbouring sample stops at the
    midpoint between the two boxes, so no neighbour tissue enters the crop.
    res: reuse a count_samples result (else it is computed with n_samples / **count_kw).
    path: e.g. "out/slide.png" -> saves out/slide_s1.png, out/slide_s2.png, ...
    Returns dict {1: {"bbox": (L, T, R, B), "image": array}, 2: {...}, ...} in reading order."""
    from pathlib import Path
    img = np.asarray(IMAGE)[..., :3]
    H, W = img.shape[:2]
    if res is None:
        res = count_samples(IMAGE, n_samples=n_samples, plot=False, **count_kw)
    boxes = [d["bbox"] for d in res["samples"]]
    pad = round(pad_frac * max(H, W))
    out = []

    for d, (x0, y0, x1, y1) in zip(res["samples"], boxes):
        others = [b for b in boxes if b != (x0, y0, x1, y1)]
        py0, py1 = max(0, y0 - pad), min(H, y1 + pad)
        L, R = max(0, x0 - pad), min(W, x1 + pad)
        for ox0, oy0, ox1, oy1 in others:              # limit x using padded y-range
            if oy0 < py1 and oy1 > py0:
                if ox0 >= x1: R = min(R, (x1 + ox0) // 2)
                if ox1 <= x0: L = max(L, (x0 + ox1) // 2)
        T, B = py0, py1
        for ox0, oy0, ox1, oy1 in others:              # limit y using final x-range
            if ox0 < R and ox1 > L:
                if oy0 >= y1: B = min(B, (y1 + oy0) // 2)
                if oy1 <= y0: T = max(T, (y0 + oy1) // 2)
        crop = img[T:B, L:R]
        out.append(dict(id=d["id"], bbox=(L, T, R, B), image=crop))

        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p.with_name(f"{p.stem}_s{d['id']}{p.suffix or '.png'}")),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    if plot and out:
        fig, ax = plt.subplots(1, len(out) + 1, figsize=(4 * (len(out) + 1), 4))
        ax = np.atleast_1d(ax)
        ax[0].imshow(img)
        for o, d in zip(out, res["samples"]):
            L, T, R, B = o["bbox"]
            ax[0].add_patch(plt.Rectangle((L, T), R - L, B - T, fill=False, ec="cyan", lw=2))
            bx0, by0, bx1, by1 = d["bbox"]
            ax[0].add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="lime", lw=1, ls="--"))
        ax[0].set_title("green = sample, cyan = crop")
        for a, o in zip(ax[1:], out):
            a.imshow(o["image"]); a.set_title(f"sample {o['id']}")
        for a in ax:
            a.axis("off")
        plt.tight_layout(); plt.show()

    return {o["id"]: dict(bbox=o["bbox"], image=o["image"]) for o in out}


# ---------------------------------------------------------------------------
# Annotations: crops are pure translations, so points shift by the crop offsets
# ---------------------------------------------------------------------------

def shift_annotations(annotations, dx, dy, crop_shape=None, min_inside=0.0):
    """Translate annotation points by (-dx, -dy). If crop_shape is given, drop annotations whose
    fraction of vertices inside the crop is <= min_inside (0.0 = keep if any vertex is inside).
    Adds 'frac_inside' so partially-cut polygons can be spotted."""
    out = []
    for a in annotations:
        p = np.asarray(a["points"], float) - [dx, dy]
        frac = 1.0
        if crop_shape is not None:
            h, w = crop_shape[:2]
            frac = float(((p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)).mean())
            if frac <= min_inside:
                continue
        out.append({**a, "points": p, "frac_inside": frac})
    return out


def crop_sample_with_annotations(IMAGE, annotations, location, n_samples=None, pad_frac=0.05,
                                 target_color="#E0D6DF", min_inside=0.0, plot=False, out_dir=None):
    """crop_frame -> crop_samples -> select `location`, carrying annotations along.
    Returns (sample_image, sample_annotations, info) with info = offsets used.
    out_dir: writes
        sample_detection.png      which samples were found and their ids (to check `location`)
        sample_selected.png       the selected H&E crop (= the returned sample_image)
        sample_annotations.json   its annotations in crop px (mal_load.load_annotations reads it)"""
    save = out_dir is not None
    if save:
        os.makedirs(out_dir, exist_ok=True)
    cropped, (fx, fy) = crop_frame(IMAGE, target_color=target_color, return_offset=True)
    if cropped is None:
        cropped = _view(IMAGE)[0]
    res = count_samples(cropped, n_samples=n_samples, plot=False,
                        save_path=os.path.join(out_dir, "sample_detection.png") if save else None)
    sel = crop_samples(cropped, res=res, pad_frac=pad_frac, plot=False)[int(location)]
    L, T = sel["bbox"][:2]
    anns = shift_annotations(annotations, fx + L, fy + T, sel["image"].shape, min_inside)
    if plot:
        plot_annotations(sel["image"], anns, title=f"sample {location}: {len(anns)} annotations")
    if save:
        cv2.imwrite(os.path.join(out_dir, "sample_selected.png"), cv2.cvtColor(sel["image"], cv2.COLOR_RGB2BGR))
        with open(os.path.join(out_dir, "sample_annotations.json"), "w") as f:
            json.dump([{**a, "points": np.asarray(a["points"], float).tolist()} for a in anns], f, indent=1)
    return sel["image"], anns, dict(frame_offset=(fx, fy), sample_bbox=sel["bbox"],
                                    total_offset=(fx + L, fy + T))


def annotations_to_xenium(annotations, M):
    """Map annotations from H&E-crop pixels to Xenium pixels (M maps Xenium -> H&E, inverse is applied)."""
    M_inv = cv2.invertAffineTransform(np.asarray(M, np.float64))
    return [{**a, "points": cv2.transform(np.float32(a["points"]).reshape(-1, 1, 2), M_inv).reshape(-1, 2)}
            for a in annotations]


def plot_annotations(img, annotations, title=None, ax=None):
    """Draw closed annotation polygons, one colour per title."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    if img is not None:
        ax.imshow(img)
    titles = sorted({a["title"] for a in annotations})
    colours = dict(zip(titles, plt.cm.tab10.colors))
    for t in titles:
        for i, a in enumerate(x for x in annotations if x["title"] == t):
            p = np.vstack([a["points"], a["points"][:1]])
            ax.plot(p[:, 0], p[:, 1], color=colours[t], lw=1.8, label=t if i == 0 else None)
    if titles:
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title or ""); ax.axis("off")
    plt.show()
    return ax


# ---------------------------------------------------------------------------
# Annotations -> alignment image -> per-cell region labels
# ---------------------------------------------------------------------------

def find_crop_offset(parent, child, coarse=4, min_score=0.9):
    """
    Locate `child` (e.g. CROPPED_ORIGINAL_IMG) inside `parent` (e.g. binary_img), same pixel scale.
    Handles children that were padded past the parent's edges.
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

    x, y, m = x * f, y * f, 2 * f                       # refine at full resolution
    x0 = min(max(0, x - m), Pp.shape[1] - cw)
    y0 = min(max(0, y - m), Pp.shape[0] - ch)
    win = Pp[y0:min(Pp.shape[0], y + m + ch), x0:min(Pp.shape[1], x + m + cw)]
    _, score, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(win, C, cv2.TM_CCOEFF_NORMED))

    if score < min_score:
        warnings.warn(f"find_crop_offset: match score {score:.3f} < {min_score}. The child may be "
                      f"rescaled/rotated relative to the parent - check crop_coordinates instead.")
    return (x0 + bx - cw, y0 + by - ch), float(score)


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def label_cells_by_annotation(coord_df, annotations, x_col="x_transformed", y_col="y_transformed",
                              out_col="region", unlabeled="unannotated"):
    """
    Label each cell with the annotation polygon it falls in. Cells inside several polygons get the
    SMALLEST one (nested region wins). Writes, in place:
      out_col          annotation title
      out_col + '_id'  index into `annotations` (-1 = none), distinguishes same-title regions
      out_col + '_area' area (px^2) of the assigned polygon
    Returns value counts of out_col.
    """
    from matplotlib.path import Path
    xy = coord_df[[x_col, y_col]].to_numpy(float)
    n = len(xy)
    labels = np.full(n, unlabeled, dtype=object)
    ids = np.full(n, -1, np.int64)
    areas_out = np.full(n, np.nan)

    areas = [_poly_area(np.asarray(a["points"], float)) for a in annotations]
    for i in np.argsort(areas)[::-1]:                     # largest first, smaller ones overwrite
        p = np.asarray(annotations[i]["points"], float)
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        cand = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        inside = cand[Path(p).contains_points(xy[cand])]
        labels[inside], ids[inside], areas_out[inside] = annotations[i]["title"], i, areas[i]

    coord_df[out_col] = labels
    coord_df[out_col + "_id"] = ids
    coord_df[out_col + "_area"] = areas_out
    return coord_df[out_col].value_counts()


def plot_cell_labels(img, coord_df, annotations, x_col="x_transformed", y_col="y_transformed",
                     label_col="region", unlabeled="unannotated", n_plot=200_000, s=0.5, seed=0):
    """Cells coloured by region label over the alignment image, with annotation outlines."""
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    titles = sorted({a["title"] for a in annotations})
    colours = dict(zip(titles, plt.cm.tab10.colors))
    colours[unlabeled] = (0.8, 0.8, 0.8)

    idx = np.random.default_rng(seed).permutation(len(coord_df))[:n_plot]
    sub = coord_df.iloc[idx]
    for a in ax:
        a.imshow(img, cmap="gray", alpha=0.35)
    for lab in [unlabeled] + titles:                      # unlabeled drawn first (underneath)
        d = sub[sub[label_col] == lab]
        ax[1].scatter(d[x_col], d[y_col], s=s, color=colours[lab], label=f"{lab} ({(coord_df[label_col] == lab).sum():,})",
                      edgecolors="none", rasterized=True)
    for a_ in annotations:
        p = np.vstack([a_["points"], a_["points"][:1]])
        for a in ax:
            a.plot(p[:, 0], p[:, 1], color=colours[a_["title"]], lw=1.5)
    ax[0].set_title("Annotations on alignment image")
    ax[1].set_title("Cells labelled (smallest enclosing region wins)")
    ax[1].legend(markerscale=12, fontsize=9, loc="upper right")
    for a in ax:
        a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0); a.axis("off")
    plt.tight_layout(); plt.show()