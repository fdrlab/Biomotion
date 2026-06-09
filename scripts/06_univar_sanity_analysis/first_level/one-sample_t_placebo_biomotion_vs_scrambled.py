#!/usr/bin/env python3
"""
one-sample_t_placebo_biomotion_vs_scrambled.py
Subject-level first-level-style Z from GLMsingle trial-wise betas (bio − scrambled).

What it does
• Uses GLMsingle TRIAL-WISE betas (one observation per event) as data.
• Builds a trial-level design: 6 condition columns (cell-means coding),
  optional run-02 dummy (omit when --demean-by-run to absorb run offsets).
• Per-voxel OLS across trials: β̂ = (X'X)^(-1)X'y; contrast c'β̂; SE from residual variance.
• t → Z with configurable sidedness (--side two | one_pos | one_neg). Also saves the effect (contrast estimate).
• Optional pre-GLM smoothing of betas (like nilearn FirstLevelModel.smoothing_fwhm).
• MASK: If --mask-img is NOT provided, uses the fMRIPrep FIRST-RUN (run-01) brain mask per subject:
        <preproc-root>/<sub>/func/<sub>_task-PLDs_run-01_space-MNI152NLin2009cAsym_res-1.7_desc-brain_mask.nii.gz
        (resampled to the betas grid if needed).

Key robustness
• Tighten mask by excluding voxels with ANY non-finite trial beta.
• Drop remaining bad voxels after Y assembly (safety).
• Sanitize outputs (effect, Z) to be finite before saving/plotting.
• Clamp p-values at 1e-16 to avoid inf Z.

Outputs per subject
  sub-XXX_*_firstlevelZ_bio_minus_scrambled.nii.gz   (Z map)
  sub-XXX_*_firstlevelZ_bio_minus_scrambled.png      (Z quicklook; hard-coded Z≥1.96)
  sub-XXX_*_firstlevelEff_bio_minus_scrambled.nii.gz (effect-size map)
  sub-XXX_*_firstlevelEff_bio_minus_scrambled.png    (effect quicklook; robust symmetric scale)
  kept_subjects.csv / skipped_subjects.csv (bookkeeping)
"""

import os, argparse, sys, json
import numpy as np
import pandas as pd
import nibabel as nb
from numpy.linalg import pinv, matrix_rank
from scipy.stats import t as t_dist, norm
from typing import Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import smooth_img, resample_to_img

# -------------------- Task constants --------------------
CANON_LABELS = [
    'happy_females','happy_males','angry_females',
    'angry_males','neutral_females','scrambled'
]
BIO_LABELS = ['happy_females','happy_males','angry_females','angry_males','neutral_females']

# -------------------- Fixed visualization slices (mm MNI Z-coords) --------------------
FIXED_Z_SLICES = [-25, -15, 0, 15, 30, 45, 60]

# -------------------- Args --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Subject-level first-level-style Z from GLMsingle betas")

    # Subject selection
    ap.add_argument("--subject", action="append", default=None,
                    help="Process ONLY this subject (repeat to pass multiple). Accepts 'sub-XXX' or 'XXX'.")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Process ONLY these subjects (space-separated).")

    # GLMsingle branch / data
    ap.add_argument("--branch", choices=["A_up3x","B_up3x"], default="A_up3x",
                    help="GLMsingle branch to read (A_up3x baseline; B_up3x aCompCor).")
    ap.add_argument("--datatype", default="unsmoothed", choices=["unsmoothed","smoothed"],
                    help="Which GLMsingle betas to read under data/<sub>/<datatype>/")
    ap.add_argument("--mask-img", default=None,
                    help="(Optional) Analysis mask (e.g., MNI mask). If omitted, use subject's fMRIPrep run-01 mask.")
    ap.add_argument("--preproc-root", default="/lustre/scratch/data/.../preprocessed",
                    help="fMRIPrep derivatives root (for auto first-run mask when --mask-img is not given).")
    ap.add_argument("--outdir", required=True, help="Output directory (per-subject maps).")

    # If subjects not given, fall back to this CSV (placebo-only)
    ap.add_argument("--placebo-csv",
        default="/lustre/scratch/data/.../unblinding_lorazepam_placebo_no_sub-159.csv",
        help="CSV with columns exactly: subject,group. Only rows with group=='placebo' are included if no subjects were specified.")

    ap.add_argument("--design-order-json", required=True,
        help="Path to design_order_map.json (from infer_design_order.py) for provenance/logging.")

    # Data handling / run pooling
    ap.add_argument("--demean-by-run", action="store_true",
                    help="Per-run voxelwise de-mean across trials before GLM (then no run dummy).")
    ap.add_argument("--base-glm", default="/lustre/scratch/data/.../GLMsingle")

    # Contrast options
    ap.add_argument("--contrast-mode", choices=["mean","sum5"], default="mean",
                    help="mean: [+1/5 on BIO, -1 on scrambled]; sum5: [+1 on BIO, -5 on scrambled].")
    ap.add_argument("--contrast-sign", choices=["pos","neg"], default="pos",
                    help="Flip sign for display preference.")

    # Sidedness
    ap.add_argument("--side", choices=["two", "one_pos", "one_neg"], default="two",
                    help=("Tail of the test: 'two' (two-sided, default), "
                          "'one_pos' (bio>scrambled), or 'one_neg' (bio<scrambled). "
                          "The Z map is oriented so positive Z indicates the chosen direction."))

    # PNG quicklook flags (kept for CLI compatibility but ignored for Z PNG threshold)
    ap.add_argument("--png-unc-p", type=float, default=0.05,
                    help="(Ignored for Z PNG; threshold is hard-coded to Z≥1.96).")
    ap.add_argument("--png-z", type=float, default=None,
                    help="(Ignored for Z PNG; threshold is hard-coded to Z≥1.96).")
    ap.add_argument("--png-cut-coords", type=int, default=7,
                    help="(Ignored when using fixed slices) Number of z cuts in the stat map PNGs.")

    # Pre-GLM smoothing
    ap.add_argument("--smooth", action="store_true",
                    help="Smooth trial-wise beta images BEFORE the OLS (like nilearn FirstLevelModel).")
    ap.add_argument("--smooth-fwhm", type=float, default=6.0,
                    help="Gaussian FWHM in mm used when --smooth is set.")

    return ap.parse_args()

# -------------------- Paths & I/O helpers --------------------
def branch_dir(BASE_GLM: str, branch: str, sub: str, dtype: str) -> str:
    return (os.path.join(BASE_GLM, "output_A_conf", "data", sub, dtype)
            if branch == "A_up3x"
            else os.path.join(BASE_GLM, "output_B_winner", "data", sub, dtype))

def betas_run_paths(BASE_GLM: str, sub: str, dtype: str, branch: str) -> Tuple[str, str]:
    base = branch_dir(BASE_GLM, branch, sub, dtype)
    f1a = os.path.join(base, f"{sub}_{dtype}_{branch}_betas_by_run-01.nii.gz")
    f2a = os.path.join(base, f"{sub}_{dtype}_{branch}_betas_by_run-02.nii.gz")
    f1b = os.path.join(base, f"{sub}_{dtype}_A_up3x_betas_by_run-01.nii.gz")
    f2b = os.path.join(base, f"{sub}_{dtype}_A_up3x_betas_by_run-02.nii.gz")
    f1 = f1a if os.path.isfile(f1a) else f1b
    f2 = f2a if os.path.isfile(f2a) else f2b
    return f1, f2

def betas_concat_path(BASE_GLM: str, sub: str, dtype: str, branch: str) -> str:
    base = branch_dir(BASE_GLM, branch, sub, dtype)
    p1 = os.path.join(base, f"{sub}_{dtype}_{branch}_betas.nii.gz")
    p2 = os.path.join(base, f"{sub}_{dtype}_A_up3x_betas.nii.gz")
    return p1 if os.path.isfile(p1) else p2

def events_path(BASE_GLM: str, sub: str, run: str) -> str:
    return os.path.join(BASE_GLM, "events", sub, "func",
                        f"{sub}_task-PLDs_run-{run}_events.tsv")

def load_events_by_run(BASE_GLM: str, sub: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evs = []
    for run in ["01","02"]:
        tsv = events_path(BASE_GLM, sub, run)
        if not os.path.isfile(tsv):
            raise FileNotFoundError(f"Events missing: {tsv}")
        ev = pd.read_csv(tsv, sep="\t")
        if "trial_type" not in ev.columns:
            raise ValueError(f"'trial_type' missing in {tsv}")
        ev = ev.loc[ev["trial_type"].isin(CANON_LABELS)].copy()
        evs.append(ev)
    return evs[0], evs[1]

def read_placebo_subjects(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if not {"subject","group"}.issubset(df.columns):
        raise SystemExit(f"[ERROR] CSV must have columns exactly: subject,group (got {df.columns.tolist()})")
    mask = df["group"].astype(str).str.strip().str.lower().eq("placebo")
    subs = df.loc[mask, "subject"].astype(str).str.strip().str.lower().unique().tolist()
    subs = [s if s.startswith("sub-") else f"sub-{s}" for s in subs]
    return sorted(subs)

def normalize_sub(s: str) -> str:
    s = str(s).strip().lower()
    return s if s.startswith("sub-") else f"sub-{s}"

def first_run_mask_path(preproc_root: str, sub: str) -> str:
    """Build fMRIPrep FIRST-RUN (run-01) brain mask path for given subject."""
    return os.path.join(
        preproc_root, sub, "func",
        f"{sub}_task-PLDs_run-01_space-MNI152NLin2009cAsym_res-1.7_desc-brain_mask.nii.gz"
    )

# -------------------- Design / Contrast helpers --------------------
def build_trial_design(ev1: pd.DataFrame, ev2: pd.DataFrame,
                       demean_by_run: bool, include_run_dummy: bool) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray, int]:
    labs1 = ev1["trial_type"].astype(str).values
    labs2 = ev2["trial_type"].astype(str).values
    labels = np.concatenate([labs1, labs2])
    T1, T2 = len(labs1), len(labs2)
    T = T1 + T2

    colnames = list(CANON_LABELS)
    P = len(colnames)
    X = np.zeros((T, P), dtype=np.float32)
    lab_to_idx = {lab: i for i, lab in enumerate(colnames)}
    for i, lab in enumerate(labels):
        if lab in lab_to_idx:
            X[i, lab_to_idx[lab]] = 1.0

    run_flags = np.zeros(T, dtype=np.int8)
    if T2 > 0:
        run_flags[T1:] = 1

    if include_run_dummy:
        X = np.concatenate([X, run_flags[:, None].astype(np.float32)], axis=1)
        colnames.append("run02_dummy")

    rnk = matrix_rank(X.astype(np.float64))
    df_model = int(T - rnk)
    if df_model <= 0:
        raise ValueError(f"Design rank deficiency: T={T}, rank={rnk}, df={df_model} ≤ 0.")
    return X, colnames, labels, run_flags, df_model

def get_contrast_vector(colnames: list, mode: str, sign: str) -> np.ndarray:
    w = {}
    if mode == "mean":
        for lab in BIO_LABELS: w[lab] = 1.0/len(BIO_LABELS)
        w['scrambled'] = -1.0
    elif mode == "sum5":
        for lab in BIO_LABELS: w[lab] = 1.0
        w['scrambled'] = -5.0
    else:
        raise ValueError("Unknown contrast mode.")
    if sign == "neg":
        for k in w: w[k] = -w[k]

    c = np.zeros(len(colnames), dtype=np.float64)
    for i, name in enumerate(colnames):
        c[i] = w.get(name, 0.0)  # nuisance gets 0
    return c

# -------------------- Core math --------------------
def fit_glm_and_contrast(beta4d: np.ndarray, mask_bool: np.ndarray,
                         X: np.ndarray, c: np.ndarray, df: int, side: str,
                         aff, hdr, out_prefix: str,
                         png_unc_p: Optional[float], png_z: Optional[float], png_cuts: int):
    """
    beta4d: (X,Y,Z,T) GLMsingle trial-wise betas
    mask_bool: (X,Y,Z) boolean analysis mask (already tightened)
    side: 'two', 'one_pos' (bio>scrambled), 'one_neg' (bio<scrambled)
    """
    # Flatten mask indices
    idx = np.where(mask_bool.ravel() > 0)[0]
    if idx.size == 0:
        raise ValueError("Mask has zero voxels after tightening.")
    T = beta4d.shape[3]

    # Assemble Y over in-mask voxels
    Y = beta4d.reshape(-1, T)[idx, :].T.astype(np.float64)  # (T, V)

    # Drop any remaining voxels with non-finite trial betas (safety)
    finite_vox = np.isfinite(Y).all(axis=0)
    if not finite_vox.all():
        kept = int(finite_vox.sum())
        dropped = int((~finite_vox).sum())
        print(f"[INFO] Dropping {dropped} voxel(s) with non-finite trial betas; keeping {kept}.", file=sys.stderr)
        Y = Y[:, finite_vox]
        idx = idx[finite_vox]

    # OLS
    XtX = X.T @ X
    XtX_inv = pinv(XtX)
    X_pinv = XtX_inv @ X.T
    B = X_pinv @ Y  # (P, V)

    # Residual variance
    Y_hat = X @ B
    E = Y - Y_hat
    RSS = np.sum(E * E, axis=0)
    sigma2 = RSS / float(df)

    # Contrast
    c = c.reshape(-1, 1)
    effect = (c.T @ B).ravel()

    c_cov = (c.T @ XtX_inv @ c).item()
    if c_cov <= 0:
        raise ValueError("Non-positive contrast covariance (c'(X'X)^(-1)c).")
    se = np.sqrt(np.maximum(sigma2, 0.0) * c_cov)

    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = np.divide(effect, se, out=np.zeros_like(effect), where=se>0)

    # p→Z with clamp to avoid inf; orient so positive Z = effect in the hypothesized direction
    if side == "two":
        p = 2.0 * t_dist.sf(np.abs(tvals), df=df)
        p = np.clip(p, 1e-16, 1.0)
        zmap = np.sign(tvals) * norm.isf(p / 2.0)
        side_label = "two-sided"
    elif side == "one_pos":  # bio > scrambled
        p = t_dist.sf(tvals, df=df)          # Pr(T >= t)
        p = np.clip(p, 1e-16, 1.0)
        zmap = norm.isf(p)                   # positive Z means bio>scrambled
        side_label = "one-sided (bio>scrambled)"
    else:  # "one_neg": bio < scrambled
        p = t_dist.cdf(tvals, df=df)         # Pr(T <= t)
        p = np.clip(p, 1e-16, 1.0)
        zmap = norm.isf(p)                   # positive Z means bio<scrambled
        side_label = "one-sided (bio<scrambled)"

    # Sanitize outputs to finite values
    effect = np.nan_to_num(effect, nan=0.0, posinf=0.0, neginf=0.0)
    zmap   = np.nan_to_num(zmap,   nan=0.0, posinf=0.0, neginf=0.0)

    # Write volumes
    shape3 = mask_bool.shape
    eff_vol = np.zeros(shape3, dtype=np.float32).ravel()
    z_vol   = np.zeros(shape3, dtype=np.float32).ravel()
    eff_vol[idx] = effect.astype(np.float32)
    z_vol[idx]   = zmap.astype(np.float32)

    eff_img = nb.Nifti1Image(eff_vol.reshape(shape3), aff, hdr)
    z_img   = nb.Nifti1Image(z_vol.reshape(shape3),   aff, hdr)

    eff_path = f"{out_prefix}_firstlevelEff_bio_minus_scrambled.nii.gz"
    z_path   = f"{out_prefix}_firstlevelZ_bio_minus_scrambled.nii.gz"
    eff_img.to_filename(eff_path)
    z_img.to_filename(z_path)

    # ---- PNGs ----
    # Hard-code Z threshold to 1.96 for display, ignoring --png-unc-p and --png-z and sidedness.
    zthr = 1.96
    thr_label = "Z≥1.96 (hard-coded)"

    try:
        disp = plotting.plot_stat_map(
            z_img,
            threshold=zthr,
            display_mode="z",
            cut_coords=FIXED_Z_SLICES,  # fixed set of slices
            colorbar=True,
            title=f"First-level Z (bio − scrambled) — {side_label}; {thr_label}"
        )
        png_path = f"{out_prefix}_firstlevelZ_bio_minus_scrambled.png"
        disp.savefig(png_path, dpi=150, bbox_inches="tight"); disp.close(); plt.close('all')
    except Exception as e:
        print(f"[WARN] Z PNG plotting failed: {e}", file=sys.stderr)

    # Effect PNG: robust symmetric color scale (±99th pct of |effect|)
    try:
        abs_eff = np.abs(effect[np.isfinite(effect)])
        vmax_eff = float(np.percentile(abs_eff, 99.0)) if abs_eff.size else 1e-6
        disp2 = plotting.plot_stat_map(
            eff_img, threshold=None, display_mode="z",
            cut_coords=FIXED_Z_SLICES,  # fixed set of slices
            colorbar=True, symmetric_cbar=True, vmax=vmax_eff,
            title="First-level effect (bio − scrambled) — robust scale (±99th pct)"
        )
        png_path2 = f"{out_prefix}_firstlevelEff_bio_minus_scrambled.png"
        disp2.savefig(png_path2, dpi=150, bbox_inches="tight"); disp2.close(); plt.close('all')
    except Exception as e:
        print(f"[WARN] Effect PNG plotting failed: {e}", file=sys.stderr)

    return eff_path, z_path

# -------------------- Main --------------------
def main():
    args = parse_args()
    BASE_GLM = args.base_glm
    os.makedirs(args.outdir, exist_ok=True)

    # Avoid double-smoothing if pointing at already smoothed betas
    if args.smooth and args.datatype.lower().startswith("smoot"):
        print("[WARN] --smooth used while --datatype=smoothed; beware of double smoothing.", file=sys.stderr)

    # Subjects
    if args.subjects or args.subject:
        raw = []
        if args.subjects: raw.extend(args.subjects)
        if args.subject:  raw.extend(args.subject)
        subjects = sorted({normalize_sub(s) for s in raw})
        print(f"[INFO] Explicit subjects: N={len(subjects)}")
    else:
        subjects = read_placebo_subjects(args.placebo_csv)
        print(f"[INFO] From CSV (placebo): N={len(subjects)}")

    kept, skipped = [], []

    # Provenance (optional)
    try:
        order_map = json.load(open(args.design_order_json, "r"))
        n_disagree = int(order_map.get("n_runs_disagree_with_global", -1))
        print(f"[INFO] Loaded design order map: runs_disagree_with_global={n_disagree}")
    except Exception as e:
        print(f"[WARN] Could not read design-order JSON: {e}")
        order_map = None

    for sub in subjects:
        try:
            ev1, ev2 = load_events_by_run(BASE_GLM, sub)
            n1, n2 = len(ev1), len(ev2)

            # Trial-wise betas
            p1, p2 = betas_run_paths(BASE_GLM, sub, args.datatype, args.branch)
            has_r1, has_r2 = os.path.isfile(p1), os.path.isfile(p2)

            if has_r1 and has_r2:
                img1 = nb.load(p1); img2 = nb.load(p2)
                if args.smooth:
                    img1 = smooth_img(img1, fwhm=float(args.smooth_fwhm))
                    img2 = smooth_img(img2, fwhm=float(args.smooth_fwhm))
                data1 = img1.get_fdata(dtype=np.float32)
                data2 = img2.get_fdata(dtype=np.float32)
                if data1.shape[:3] != data2.shape[:3]:
                    raise RuntimeError("Run grids differ.")
                T1, T2 = data1.shape[3], data2.shape[3]
                if (T1 != n1) or (T2 != n2):
                    raise RuntimeError(f"Strict mismatch: run-01 betas={T1} vs events={n1}; run-02 betas={T2} vs events={n2}")
                trial4d = np.concatenate([data1, data2], axis=3).astype(np.float32)
                ref_img = img1  # reference for resampling mask
                aff, hdr = img1.affine, img1.header
            else:
                pc = betas_concat_path(BASE_GLM, sub, args.datatype, args.branch)
                if not os.path.isfile(pc):
                    raise RuntimeError("No per-run and no concat betas found.")
                img = nb.load(pc)
                if args.smooth:
                    img = smooth_img(img, fwhm=float(args.smooth_fwhm))
                trial4d = img.get_fdata(dtype=np.float32)
                if trial4d.shape[3] != (n1 + n2):
                    raise RuntimeError(f"Concat mismatch: betas T={trial4d.shape[3]} vs events_total={n1+n2}")
                ref_img = img
                aff, hdr = img.affine, img.header

            # Optional run de-mean (voxelwise)
            if args.demean_by_run:
                if n1 > 0:
                    seg1 = trial4d[..., :n1]
                    seg1 -= seg1.mean(axis=3, keepdims=True)
                    trial4d[..., :n1] = seg1
                if n2 > 0:
                    seg2 = trial4d[..., n1:n1+n2]
                    seg2 -= seg2.mean(axis=3, keepdims=True)
                    trial4d[..., n1:n1+n2] = seg2

            include_run_dummy = (not args.demean_by_run)
            X, colnames, _, _, df_model = build_trial_design(
                ev1, ev2, demean_by_run=args.demean_by_run, include_run_dummy=include_run_dummy
            )
            c = get_contrast_vector(colnames, args.contrast_mode, args.contrast_sign)

            # ----------------- MASK (first-run fMRIPrep or explicit) -----------------
            if args.mask_img is not None:
                mask_img = nb.load(args.mask_img)
            else:
                mask_path = first_run_mask_path(args.preproc_root, sub)
                if not os.path.isfile(mask_path):
                    raise RuntimeError(f"First-run fMRIPrep mask not found: {mask_path}")
                mask_img = nb.load(mask_path)

            # If mask grid != betas grid, resample mask to betas (nearest)
            if mask_img.shape != ref_img.shape or not np.allclose(mask_img.affine, ref_img.affine):
                mask_img = resample_to_img(mask_img, ref_img, interpolation="nearest")

            mask_data = mask_img.get_fdata()
            mask_bool_base = (mask_data > 0.5)

            # Tighten mask by requiring all trials finite in each voxel
            finite_trials = np.isfinite(trial4d).all(axis=3)
            if finite_trials.shape != mask_bool_base.shape:
                # should not happen after resample, but guard anyway
                raise RuntimeError("Mask shape mismatch after resampling.")
            mask_bool = mask_bool_base & finite_trials
            # ------------------------------------------------------------------------

            out_prefix = os.path.join(args.outdir, f"{sub}_{args.datatype}_{args.branch}")
            eff_path, z_path = fit_glm_and_contrast(
                trial4d, mask_bool, X.astype(np.float64), c, df_model, args.side,
                aff, hdr, out_prefix,
                png_unc_p=args.png_unc_p, png_z=args.png_z, png_cuts=args.png_cut_coords
            )

            kept.append({"subject": sub, "branch": args.branch, "datatype": args.datatype,
                         "T_trials": int(trial4d.shape[3]), "df_model": int(df_model),
                         "Z_path": z_path, "Eff_path": eff_path, "smoothed": bool(args.smooth),
                         "fwhm": (float(args.smooth_fwhm) if args.smooth else 0.0),
                         "side": args.side,
                         "mask": (args.mask_img if args.mask_img else first_run_mask_path(args.preproc_root, sub))})
            print(f"[OK] {sub}: wrote\n  Z → {z_path}\n  Eff → {eff_path}")

        except Exception as e:
            skipped.append({"subject": sub, "reason": str(e)})
            print(f"[SKIP] {sub}: {e}", file=sys.stderr)

    if kept:
        pd.DataFrame(kept).to_csv(os.path.join(args.outdir, "kept_subjects.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(args.outdir, "skipped_subjects.csv"), index=False)

if __name__ == "__main__":
    main()
