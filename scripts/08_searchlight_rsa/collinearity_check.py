#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether the past "see-saw" collinearity issue is fixed after orthogonalizing model RDMs.

What it does per subject (radius=3 only):
  • Loads provenance JSON (prefers metric=crossnobis) to read:
      - glm_predictors (names, in order)
      - design_corr (P×P correlation between model RDM vectors actually used)
      - design_vif  (length-P VIF from (X'X)^-1)
  • Loads per-predictor maps from disk (beta-, t-, pr2-), masks to coverage>0, flattens, and computes:
      - Pairwise Pearson correlations between predictors’ β maps
      - Pairwise Pearson correlations between predictors’ t maps  (the "see-saw" can show as strong negative)
      - Pairwise Pearson correlations between predictors’ partial-R² maps
  • Summarizes key diagnostics:
      - max |offdiag(design_corr)|, mean |offdiag|
      - condition number (via eigenvalues of design_corr, robust to scale when columns were z-scored)
      - mean/median VIF
      - worst (most negative) t-map correlation across predictor pairs
  • Writes one CSV row per subject + an optional group summary row.

Usage:
  python check_collinearity_orthogonalization_rad3.py \
    --root /lustre/scratch/data/.../searchlight_RSA \
    --datatype unsmoothed
  # Optional subsets:
  #   --subjects sub-102 sub-114
  #   --subjects-file /path/to/subs.txt
"""

import os, glob, json, argparse
import numpy as np
import nibabel as nb
import pandas as pd

# ---------- helpers ----------

RADIUS = 3
PREFER_METRIC = "crossnobis"

def first_or_none(lst): return lst[0] if lst else None

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_provenance(sub_dir, radius=RADIUS, prefer_metric=PREFER_METRIC):
    pats = sorted(glob.glob(os.path.join(sub_dir, f"*rad{radius}_*_GLM_provenance.json")))
    if not pats:
        return None, None
    def score(p):
        try:
            d = load_json(p)
            met = str(d.get("metric",""))
            prefer = 1 if (prefer_metric and met == prefer_metric) else 0
            mtime = os.path.getmtime(p)
            return (prefer, mtime)
        except Exception:
            return (0,0)
    pats.sort(key=score, reverse=True)
    p = pats[0]
    return p, load_json(p)

def load_nii(path): return nb.load(path).get_fdata()

def pairwise_indices(n):
    out=[]
    for i in range(n):
        for j in range(i+1,n):
            out.append((i,j))
    return out

def safe_flat_corr(a, b):
    """Pearson corr of two same-shaped arrays after masking to finite in both."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:  # too few points, return nan
        return np.nan
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    den = np.sqrt((aa*aa).sum() * (bb*bb).sum())
    return float((aa @ bb) / den) if den>0 else np.nan

def mask_to_coverage(vol, cov):
    """Return vol with non-covered voxels set to NaN; keep existing NaNs."""
    out = np.array(vol, dtype=np.float32, copy=True)
    out[~(np.isfinite(cov) & (cov>0))] = np.nan
    return out

def design_cond_number_from_corr(R):
    """Condition number from correlation matrix eigenvalues (scale-invariant)."""
    try:
        w = np.linalg.eigvalsh(R)
        w = np.sort(np.maximum(w, 1e-12))
        return float(w[-1] / w[0])
    except Exception:
        return np.nan

# ---------- main per-subject work ----------

def analyze_subject(root, datatype, sub):
    sub_dir = os.path.join(root, datatype, sub)
    if not os.path.isdir(sub_dir):
        return dict(subject=sub, note="missing subject dir")

    prov_path, prov = find_provenance(sub_dir)
    if prov is None:
        return dict(subject=sub, note="missing provenance rad3")

    preds = prov.get("glm_predictors", []) or []
    P = len(preds)
    if P < 2:
        return dict(subject=sub, note="need >=2 predictors to test collinearity")

    files = prov.get("files", {}) or {}

    # Design diagnostics (from when GLM was fit)
    R = np.asarray(prov.get("design_corr", []), dtype=float) if ("design_corr" in prov) else None
    VIF = np.asarray(prov.get("design_vif", []), dtype=float) if ("design_vif" in prov) else None

    # Core maps used for masking
    cov_path = files.get("coverage") or first_or_none(glob.glob(os.path.join(sub_dir, f"*rad{RADIUS}_*_GLM_coverage.nii.gz")))
    if not (cov_path and os.path.isfile(cov_path)):
        return dict(subject=sub, note="missing coverage map for masking")
    cov = load_nii(cov_path)

    # Load per-predictor maps
    beta_vols = []
    t_vols = []
    pr2_vols = []
    have_t = True
    have_pr2 = True
    for nm in preds:
        bpath = files.get(f"beta-{nm}") or first_or_none(glob.glob(os.path.join(sub_dir, f"*rad{RADIUS}_*_GLM_beta-{nm}.nii.gz")))
        tpath = files.get(f"t-{nm}")    or first_or_none(glob.glob(os.path.join(sub_dir, f"*rad{RADIUS}_*_GLM_t-{nm}.nii.gz")))
        p2path = files.get(f"pr2-{nm}") or first_or_none(glob.glob(os.path.join(sub_dir, f"*rad{RADIUS}_*_GLM_pr2-{nm}.nii.gz")))
        if not (bpath and os.path.isfile(bpath)):
            return dict(subject=sub, note=f"missing beta map for {nm}")
        beta_vols.append(mask_to_coverage(load_nii(bpath), cov))
        if tpath and os.path.isfile(tpath):
            t_vols.append(mask_to_coverage(load_nii(tpath), cov))
        else:
            have_t = False
            t_vols.append(None)
        if p2path and os.path.isfile(p2path):
            pr2_vols.append(mask_to_coverage(load_nii(p2path), cov))
        else:
            have_pr2 = False
            pr2_vols.append(None)

    # Pairwise correlations across predictors
    pairs = pairwise_indices(P)
    rows_pairs = []
    worst_t_corr = np.nan
    for (i,j) in pairs:
        rec = dict(subject=sub, pred_i=preds[i], pred_j=preds[j])

        # β-map correlation
        r_b = safe_flat_corr(beta_vols[i], beta_vols[j])
        rec["corr_beta"] = r_b

        # t-map correlation (may be NaN if ridge)
        if have_t and t_vols[i] is not None and t_vols[j] is not None:
            r_t = safe_flat_corr(t_vols[i], t_vols[j])
        else:
            r_t = np.nan
        rec["corr_t"] = r_t

        # pr2 correlation
        if have_pr2 and pr2_vols[i] is not None and pr2_vols[j] is not None:
            r_p = safe_flat_corr(pr2_vols[i], pr2_vols[j])
        else:
            r_p = np.nan
        rec["corr_pr2"] = r_p

        rows_pairs.append(rec)

        if np.isfinite(r_t):
            if np.isnan(worst_t_corr) or (r_t < worst_t_corr):
                worst_t_corr = r_t

    # Design-level summaries
    if R is not None and R.size == P*P:
        R = R.reshape(P, P)
        off = R[~np.eye(P, dtype=bool)]
        max_abs_off = float(np.nanmax(np.abs(off))) if off.size else np.nan
        mean_abs_off = float(np.nanmean(np.abs(off))) if off.size else np.nan
        cond_num = design_cond_number_from_corr(R)
    else:
        max_abs_off = mean_abs_off = cond_num = np.nan

    if VIF is not None and VIF.size == P:
        vif_mean = float(np.nanmean(VIF))
        vif_median = float(np.nanmedian(VIF))
        vif_max = float(np.nanmax(VIF))
    else:
        vif_mean = vif_median = vif_max = np.nan

    # Subject summary
    subj_row = dict(
        subject=sub,
        n_predictors=P,
        predictors=",".join(preds),
        design_max_abs_offdiag_corr=max_abs_off,
        design_mean_abs_offdiag_corr=mean_abs_off,
        design_condition_number=cond_num,
        vif_mean=vif_mean,
        vif_median=vif_median,
        vif_max=vif_max,
        worst_pair_tmap_corr=worst_t_corr,
        note=""
    )

    return subj_row, rows_pairs

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of searchlight_RSA (contains <datatype>/sub-XXX/).")
    ap.add_argument("--datatype", default="unsmoothed", choices=["unsmoothed","smoothed"])
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional explicit list (e.g., sub-102 sub-114).")
    ap.add_argument("--subjects-file", default="", help="Optional text file (one subject per line).")
    args = ap.parse_args()

    root = args.root
    dt = args.datatype
    base = os.path.join(root, dt)

    # subject discovery
    if args.subjects:
        subs = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
    elif args.subjects_file:
        with open(args.subjects_file, "r") as f:
            subs = []
            for line in f:
                s = line.strip()
                if s:
                    subs.append(s if s.startswith("sub-") else f"sub-{s}")
        subs = sorted(list(dict.fromkeys(subs)))
    else:
        subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(base, "sub-*")) if os.path.isdir(p)])

    if not subs:
        raise SystemExit("No subjects found.")

    # run
    subj_rows = []
    pair_rows = []
    for sub in subs:
        out = analyze_subject(root, dt, sub)
        if isinstance(out, tuple):
            srow, prows = out
            subj_rows.append(srow)
            pair_rows.extend(prows)
        else:
            subj_rows.append(out)

    # write outputs
    outdir = base
    os.makedirs(outdir, exist_ok=True)

    df_subj = pd.DataFrame(subj_rows).sort_values("subject")
    path_subj = os.path.join(outdir, "collinearity_check_rad3_summary.csv")
    df_subj.to_csv(path_subj, index=False)
    print(f"[WROTE] {path_subj}")

    if pair_rows:
        df_pairs = pd.DataFrame(pair_rows).sort_values(["subject","pred_i","pred_j"])
        path_pairs = os.path.join(outdir, "collinearity_check_rad3_pairwise_map_corrs.csv")
        df_pairs.to_csv(path_pairs, index=False)
        print(f"[WROTE] {path_pairs}")

    # console verdict
    if not df_subj.empty:
        bad_design = df_subj["design_max_abs_offdiag_corr"].dropna()
        bad_t = df_subj["worst_pair_tmap_corr"].dropna()
        if not bad_design.empty:
            print(f"[DESIGN] max |offdiag corr|: mean={bad_design.mean():.3f}, max={bad_design.max():.3f}")
        if not bad_t.empty:
            print(f"[TMAPS ] worst pair corr across subjects: mean={bad_t.mean():.3f}, min={bad_t.min():.3f}")
        print("[OK] If orthogonalization worked, off-diagonal design correlations should be ~0 "
              "and t-map pairwise correlations should cluster near 0 (no strong negatives).")

if __name__ == "__main__":
    main()
