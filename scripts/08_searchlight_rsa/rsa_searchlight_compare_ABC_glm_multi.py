#!/usr/bin/env python3
"""
rsa_searchlight_compare_ABCD_glm_multi.py

Searchlight RSA across multiple beta-generation approaches + post-RSA split-half reliability.

KEY CHANGE (FIX MISSINGNESS FOR GROUP TFCE WITHOUT IMPUTATION):
- Use a *group-level cortex universe mask* for BOTH centers and neighbor pool.
- Use *adaptive-K neighborhoods* (KNN) inside that universe, with NO subject-specific GM/spillover masks.
- Candidate neighbor voxels are restricted to voxels that are *pattern-valid* for the subject
  (all conditions finite in both runs, non-zero variance, in run-01 brainmask, optional --mask).

This removes the classic failure mode:
    "sphere near edges has < min_neigh valid voxels -> NaN holes vary by subject"
because every center gets exactly K neighbors (unless you explicitly enforce a hard max distance).

TFCE readiness:
- Beta/t/p/pr2/r2 maps are initialized to finite defaults INSIDE the universe mask:
    beta=0, t=0, p=1, pr2=0, r2=0
  so numeric degeneracies (e.g., y variance ~0) do NOT create NaN holes.
- QC maps (rdm reliability, pattern r, offdiag corr, kth distance) may still contain NaNs; they are QC.

Important:
- If you turn on --min-rdm-reliability >= 0, you reintroduce missingness by design.
  For TFCE, keep it at -1 and do QC-based masking separately at the group level.

"""

import os, json, glob, argparse
import numpy as np
import pandas as pd
import nibabel as nb
import numpy.linalg as npl
from scipy.stats import spearmanr, t as student_t, rankdata
from scipy.spatial import cKDTree

# plotting (optional)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as sch

# optional resampling (for group masks)
try:
    from nilearn.image import resample_to_img
except Exception:
    resample_to_img = None


# -------------------- Constants / paths --------------------
BASE_GLM   = "/lustre/scratch/data/.../GLMsingle"
EVENTS_DIR = os.path.join(BASE_GLM, "events")
OUT_BASE   = "/lustre/scratch/data/.../searchlight_RSA"

DEFAULT_PREPROC_DIR = "/lustre/scratch/data/.../preprocessed"
LEGACY_PREPROC_DIR  = DEFAULT_PREPROC_DIR


COND_ORDER = [
    "happy_females", "happy_males", "angry_females",
    "angry_males", "neutral_females", "scrambled"
]
K = len(COND_ORDER)
IU = np.triu_indices(K, 1)
N_PAIRS = IU[0].size

APPROACHES_ALL = ["A_up3x", "B_up3x", "nilearn", "D_up3x_legacy"]

APPROACH_SPECS = {
    "A_up3x": {
        "data_root": os.path.join(BASE_GLM, "output_A_conf", "data"),
        "betas_tag": "A_up3x",
        "is_legacy": False,
        "is_nilearn": False,
        "equalized_runs": True,
    },
    "B_up3x": {
        "data_root": os.path.join(BASE_GLM, "output_B_aCompCor", "data"),
        "betas_tag": "B_up3x",
        "is_legacy": False,
        "is_nilearn": False,
        "equalized_runs": True,
    },
    "D_up3x_legacy": {
        "data_root": os.path.join(BASE_GLM, "output", "data"),
        "betas_tag": "A_up3x",  # legacy filename tag
        "is_legacy": True,
        "is_nilearn": False,
        "equalized_runs": False,
    },
}


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--datatype",
        nargs="+",
        default=None,
        choices=["unsmoothed", "smoothed"],
        help="If omitted, runs BOTH datatypes (unsmoothed + smoothed).",
    )
    ap.add_argument("--approaches", nargs="+", choices=APPROACHES_ALL, default=APPROACHES_ALL)

    ap.add_argument("--model-rdms", nargs="+", required=True)
    ap.add_argument("--glm-rank", action="store_true")
    ap.add_argument("--ridge-lambda", type=float, default=0.0)

    # Searchlight neighborhood mode
    ap.add_argument(
        "--neigh-mode",
        choices=["adaptiveK", "radius"],
        default="adaptiveK",
        help=("adaptiveK = K nearest neighbors (recommended to avoid subject-specific holes). "),
    )
    ap.add_argument("--k-neigh", type=int, default=60, help="K for adaptiveK neighborhoods.")
    ap.add_argument(
        "--k-hard-maxdist",
        action="store_true",
        help=("If set, enforce a hard max distance for adaptiveK via --k-maxdist-vox. "
              "This can reintroduce holes (missingness)."),
    )
    ap.add_argument(
        "--k-maxdist-vox",
        type=float,
        default=0.0,
        help=("Hard max distance (in voxel units) for adaptiveK if --k-hard-maxdist is set. "
              "0 disables."),
    )

    # Optional gating (WARNING: creates holes if used)
    ap.add_argument("--min-rdm-reliability", type=float, default=-1.0)

    # Optional additional mask (group-level; applied to centers + neighbors)
    ap.add_argument("--mask", default="", help="Optional additional mask applied to BOTH centers and neighbor pool.")

    # Group masks
    ap.add_argument(
        "--center-mask",
        required=True,
        help=("Group cortex universe mask (binary NIfTI). "
              "Used for BOTH centers and neighbor pool unless --neighbor-mask is provided. "
              "Will be resampled (nearest) to beta grid if needed."),
    )
    ap.add_argument(
        "--neighbor-mask",
        default="",
        help=("Optional *group-level* alternative neighbor pool mask (binary NIfTI). "
              "If provided, neighbors are drawn from this (still group-level, not subject-specific)."),
    )

    # QC / gating helpers (unchanged)
    ap.add_argument("--tsnr-dir", default="")
    ap.add_argument("--gate-rel-spear", type=float, default=0.2)
    ap.add_argument("--gate-rel-pear", type=float, default=0.2)
    ap.add_argument("--gate-tsnr", type=float, default=24.0)

    ap.add_argument("--metric", default="crossnobis", choices=["crossnobis", "xcorrdiff"])
    ap.add_argument("--shrink-alpha", type=float, default=0.10)

    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--debug", type=int, default=0)

    ap.add_argument("--save-searchlight-rdmvec", action="store_true")
    ap.add_argument("--save-summary-rdms", action="store_true")
    ap.add_argument("--make-figures", action="store_true")
    ap.add_argument("--fig-dpi", type=int, default=120)

    ap.add_argument("--out-tag", default="compare_ABCD_glm_multi")

    # modeled-event window controls
    ap.add_argument("--preproc-dir", default=DEFAULT_PREPROC_DIR)
    ap.add_argument("--drop-trs", type=int, default=5)
    ap.add_argument("--tr-override", type=float, default=0.0)
    ap.add_argument("--no-time-window-filter", action="store_true")
    ap.add_argument("--events-label-col", default="trial_type")
    ap.add_argument("--events-onset-col", default="onset")

    ap.add_argument(
        "--equalize-runs",
        default="auto",
        choices=["auto", "yes", "no"],
        help="Whether to enforce common modeled window across run-01/run-02.",
    )

    ap.add_argument("--max-split-mismatch", type=int, default=0)

    ap.add_argument("--nilearn-outdir", default=NILEARN_DEFAULT_OUTDIR)

    ap.add_argument("--allow-missing-conditions", action="store_true",
                    help="Allow a run to have 0 modeled events for a condition (not recommended).")

    return ap.parse_args()


# -------------------- Small helpers --------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _normalize_sub(s: str) -> str:
    s = str(s).strip()
    if s.startswith("sub-"):
        return s
    if s.isdigit():
        return f"sub-{int(s):03d}"
    return f"sub-{s}"

def _safe_spearman(a, b, min_n=10):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return np.nan
    return float(spearmanr(a[m], b[m])[0])

def _safe_pearson(a, b, min_n=10):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    den = np.sqrt((aa*aa).sum() * (bb*bb).sum())
    return float((aa @ bb) / den) if den > 0 else np.nan

def _glob_best(cands, prefer_substrs=()):
    if not cands:
        return None
    if prefer_substrs:
        scored = []
        for p in cands:
            score = 0
            bn = os.path.basename(p)
            for s in prefer_substrs:
                if s in bn or s in p:
                    score += 1
            scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1]
    return sorted(cands)[0]

def _as_3d_ref_img(img_ref, ref_shape):
    if len(img_ref.shape) == 3:
        return img_ref
    if len(img_ref.shape) == 4:
        return img_ref.slicer[:, :, :, 0]
    return nb.Nifti1Image(np.zeros(ref_shape, np.uint8), affine=img_ref.affine)

def _load_bool_mask_maybe_resample(path, img_ref, ref_shape, label, debug=0):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")

    m_img = nb.load(path)
    if m_img.shape == ref_shape:
        dat = m_img.get_fdata(dtype=np.float32)
        return (dat > 0.5) & np.isfinite(dat)

    if resample_to_img is None:
        raise ValueError(
            f"{label} shape {m_img.shape} != ref {ref_shape} and nilearn resampling is unavailable. "
            f"Provide a mask already on the beta grid or install nilearn. Mask: {path}"
        )

    ref3d = _as_3d_ref_img(img_ref, ref_shape)
    if debug:
        print(f"[info] resampling {label} to beta grid (nearest): {path} | {m_img.shape} -> {ref_shape}")
    m_rs = resample_to_img(m_img, ref3d, interpolation="nearest")
    dat = m_rs.get_fdata(dtype=np.float32)
    if dat.shape != ref_shape:
        raise ValueError(f"{label} resampled shape {dat.shape} != ref {ref_shape}: {path}")
    return (dat > 0.5) & np.isfinite(dat)


# -------------------- Model RDMs → design matrix --------------------
def load_one_rdm(path):
    M = np.load(path) if path.lower().endswith(".npy") else np.loadtxt(path, delimiter=",")
    M = np.asarray(M, float)
    if M.shape != (K, K):
        raise ValueError(f"Model RDM must be {K}x{K}, got {M.shape} for {path}.")
    M = 0.5*(M + M.T)
    np.fill_diagonal(M, 0.0)
    return M[IU]

def standardize_columns(X):
    X = np.asarray(X, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def load_model_design(paths, do_rank=False):
    names, cols, raw_cols = [], [], []
    for p in paths:
        v = load_one_rdm(p)
        raw_cols.append(v.copy())
        if do_rank:
            v = rankdata(v, method="average")
        cols.append(v)
        base = os.path.splitext(os.path.basename(p))[0]
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
        names.append(safe)
    X = np.column_stack(cols)
    X = standardize_columns(X)
    return X, names, raw_cols


# -------------------- Approach paths --------------------
def betas_path(sub, dtype, rec):
    spec = APPROACH_SPECS[rec]
    if spec.get("is_nilearn", False):
        return None
    return os.path.join(
        spec["data_root"], sub, dtype,
        f"{sub}_{dtype}_{spec['betas_tag']}_betas.nii.gz"
    )

def legacy_npz_path(sub, dtype):
    tag = APPROACH_SPECS["D_up3x_legacy"]["betas_tag"]  # "A_up3x"
    return os.path.join(
        APPROACH_SPECS["D_up3x_legacy"]["data_root"], sub, dtype,
        f"{sub}_{dtype}_{tag}_glmsingle_results.npz"
    )

def legacy_run01_mask_path(sub):
    return os.path.join(
        LEGACY_PREPROC_DIR, sub, "func",
        f"{sub}_task-PLDs_run-01_space-MNI152NLin2009cAsym_res-1.7_desc-brain_mask.nii.gz"
    )


# -------------------- Preproc discovery (BOLD length/TR + masks) --------------------
def find_preproc_bold(preproc_dir, sub, run):
    cands = []
    funcdir = os.path.join(preproc_dir, sub, "func")
    if os.path.isdir(funcdir):
        pats = [
            os.path.join(funcdir, f"{sub}_task-PLDs_run-{run}_*desc-preproc_bold.nii*"),
            os.path.join(funcdir, f"{sub}_task-PLDs_run-{run}_*bold.nii*"),
        ]
        for pat in pats:
            cands.extend(glob.glob(pat))

    smdir = os.path.join(preproc_dir, sub, "smoothed")
    if not cands and os.path.isdir(smdir):
        pats = [
            os.path.join(smdir, f"{sub}_task-PLDs_run-{run}_*desc-smoothed_bold.nii*"),
            os.path.join(smdir, f"{sub}_task-PLDs_run-{run}_*bold.nii*"),
        ]
        for pat in pats:
            cands.extend(glob.glob(pat))

    cands_preproc = [p for p in cands if ("desc-preproc_bold" in os.path.basename(p))]
    cands = cands_preproc or cands

    return _glob_best(
        cands,
        prefer_substrs=("space-MNI152NLin2009cAsym", "res-1.7", "desc-preproc_bold", "desc-smoothed_bold"),
    )

def find_brain_mask(preproc_dir, sub, run="01"):
    funcdir = os.path.join(preproc_dir, sub, "func")
    if not os.path.isdir(funcdir):
        return None
    cands = glob.glob(os.path.join(funcdir, f"{sub}_task-PLDs_run-{run}_*desc-brain_mask.nii*"))
    return _glob_best(cands, prefer_substrs=("space-MNI152NLin2009cAsym", "res-1.7", "desc-brain_mask"))

def get_run_timing(preproc_dir, sub, run, drop_trs, tr_override=0.0):
    bold_p = find_preproc_bold(preproc_dir, sub, run)
    if bold_p is None or (not os.path.isfile(bold_p)):
        return None
    img = nb.load(bold_p)
    shape = img.shape
    if len(shape) != 4:
        return None
    nvol = int(shape[3])
    tr = float(tr_override) if (tr_override and tr_override > 0) else float(img.header.get_zooms()[3])
    nvol_after = nvol - int(drop_trs)
    if nvol_after <= 0:
        raise RuntimeError(f"{sub} run-{run}: nvol={nvol} - drop_trs={drop_trs} => {nvol_after} (<=0)")
    T_model = float(nvol_after) * tr
    return dict(bold_path=bold_p, tr=tr, nvol=nvol, nvol_after_trim=nvol_after, drop_trs=int(drop_trs), T_model=T_model)


# -------------------- Events (MODELED) --------------------
def load_modeled_events_by_run(sub, args, equalize_runs=False):
    sub = _normalize_sub(sub)
    out = []
    meta = {"subject": sub, "runs": {}, "equalize_runs": bool(equalize_runs), "T_common": None}

    timing = {}
    T_common = None
    if not args.no_time_window_filter:
        for run in ["01", "02"]:
            timing[run] = get_run_timing(args.preproc_dir, sub, run, args.drop_trs, tr_override=args.tr_override)
            if timing[run] is None:
                raise FileNotFoundError(
                    f"{sub} run-{run}: could not find BOLD for modeled-window filtering under {os.path.join(args.preproc_dir, sub)}. "
                    "Fix --preproc-dir or use --no-time-window-filter (not recommended)."
                )
        if equalize_runs:
            T_common = float(min(timing["01"]["T_model"], timing["02"]["T_model"]))
            meta["T_common"] = T_common

    label_to_idx = {c: i for i, c in enumerate(COND_ORDER)}
    lbl = args.events_label_col
    onc = args.events_onset_col

    for run in ["01", "02"]:
        tsv = os.path.join(EVENTS_DIR, sub, "func", f"{sub}_task-PLDs_run-{run}_events.tsv")
        if not os.path.isfile(tsv):
            raise FileNotFoundError(f"Events missing: {tsv}")
        ev = pd.read_csv(tsv, sep="\t")

        if lbl not in ev.columns:
            raise ValueError(f"{sub} run-{run}: '{lbl}' missing in {tsv}.")
        if (not args.no_time_window_filter) and (onc not in ev.columns):
            raise ValueError(f"{sub} run-{run}: '{onc}' missing in {tsv} (needed for modeled-window filtering).")

        raw_n = int(len(ev))
        ev = ev.copy()
        ev[lbl] = ev[lbl].astype(str)

        keep_cond = ev[lbl].isin(COND_ORDER).to_numpy(dtype=bool)

        if args.no_time_window_filter:
            keep_time = np.ones(len(ev), dtype=bool)
            T_model_raw = np.nan
            T_used = np.nan
        else:
            T_model_raw = float(timing[run]["T_model"])
            T_used = float(T_common if (T_common is not None) else T_model_raw)
            onset = pd.to_numeric(ev[onc], errors="coerce").to_numpy(dtype=float)
            keep_time = np.isfinite(onset) & (onset >= 0.0) & (onset < T_used)

        keep = keep_cond & keep_time
        ev_kept = ev.loc[keep, :].copy().reset_index(drop=True)

        ev_kept["trial_idx"] = np.arange(len(ev_kept), dtype=np.int32)
        ev_kept["cond_idx"] = ev_kept[lbl].map(label_to_idx).astype("Int64")
        if ev_kept["cond_idx"].isna().any():
            bad = ev_kept.loc[ev_kept["cond_idx"].isna(), lbl].unique().tolist()[:20]
            raise RuntimeError(f"{sub} run-{run}: cond_idx mapping failed for labels: {bad}")

        counts_per_cond = {int(k): int(v) for k, v in ev_kept["cond_idx"].value_counts().items()}
        missing = [COND_ORDER[k] for k in range(K) if counts_per_cond.get(k, 0) == 0]
        if missing and (not args.allow_missing_conditions):
            raise RuntimeError(
                f"{sub} run-{run}: modeled events are missing conditions {missing}. "
                "This breaks 6-condition RSA / split-half. "
                "Fix truncation/windowing or rerun with --allow-missing-conditions (not recommended)."
            )

        meta["runs"][run] = dict(
            events_tsv=tsv,
            raw_events_n=raw_n,
            n_modeled=int(len(ev_kept)),
            counts_per_cond=counts_per_cond,
            missing_conditions=missing,
            T_model=float(T_used) if np.isfinite(T_used) else None,
            T_model_raw=float(T_model_raw) if np.isfinite(T_model_raw) else None,
            timing=(timing.get(run) if run in timing else None),
        )

        if len(ev_kept) == 0:
            raise RuntimeError(f"{sub} run-{run}: 0 modeled events after filtering.")

        out.append(ev_kept)

    return out, meta


# -------------------- Condition collapsing / z-scoring --------------------
def collapse_trial_betas_to_conditions(drun, cond_idx_T):
    X, Y, Z, N = drun.shape
    cond_idx_T = np.asarray(cond_idx_T, dtype=int)
    if cond_idx_T.shape[0] != N:
        raise ValueError(f"cond_idx length {cond_idx_T.shape[0]} != drun trials {N}")
    CM = np.full((X, Y, Z, K), np.nan, np.float32)
    for k in range(K):
        idx = np.where(cond_idx_T == k)[0]
        if idx.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            CM[..., k] = np.nanmean(drun[..., idx], axis=3).astype(np.float32, copy=False)
    return CM

def zscore_across_conditions(CM):
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(CM, axis=3, keepdims=True)
        sd = np.nanstd(CM, axis=3, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (CM - mu) / sd


# -------------------- Distances & QC helpers --------------------
def mean_offdiag_corr(P):
    # P: (K,F) full finite assumed
    Pz = (P - P.mean(axis=1, keepdims=True))
    s  = Pz.std(axis=1, keepdims=True); s[s == 0] = 1.0
    Pz = Pz / s
    C = np.corrcoef(Pz)
    iu = np.triu_indices(Pz.shape[0], 1)
    return float(np.mean(C[iu]))

def xcorrdiff_vec(P1, P2):
    # P1,P2: (K,F) full finite assumed
    K_, F = P1.shape
    assert P2.shape == (K_, F)
    iu = np.triu_indices(K_, 1)
    n = iu[0].size
    dcv = np.empty(n, np.float32)
    d1  = np.empty(n, np.float32)
    d2  = np.empty(n, np.float32)

    P1z = (P1 - P1.mean(axis=1, keepdims=True))
    s1  = P1z.std(axis=1, keepdims=True); s1[s1 == 0] = 1.0
    P1z = P1z / s1

    P2z = (P2 - P2.mean(axis=1, keepdims=True))
    s2  = P2z.std(axis=1, keepdims=True); s2[s2 == 0] = 1.0
    P2z = P2z / s2

    for m, (i, j) in enumerate(zip(iu[0], iu[1])):
        d1[m] = 1.0 - np.clip(np.dot(P1z[i], P1z[j]) / F, -1.0, 1.0)
        d2[m] = 1.0 - np.clip(np.dot(P2z[i], P2z[j]) / F, -1.0, 1.0)

        diff1 = P1z[i] - P1z[j]
        diff2 = P2z[i] - P2z[j]
        r_num = np.dot(diff1 - diff1.mean(), diff2 - diff2.mean())
        r_den = np.sqrt(((diff1 - diff1.mean())**2).sum() * ((diff2 - diff2.mean())**2).sum())
        r = r_num / r_den if r_den > 0 else 0.0
        dcv[m] = 1.0 - np.clip(r, -1.0, 1.0)

    return dcv, d1, d2


# ---- TRUE crossnobis helpers ----
def shrink_cov(S, alpha=0.10):
    if not np.all(np.isfinite(S)):
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.diag(np.diag(S))
    return (1.0 - alpha) * S + alpha * d

def safe_inv(S):
    try:
        return npl.inv(S)
    except npl.LinAlgError:
        u, s, vh = npl.svd(S, full_matrices=False)
        s = np.where(s > 1e-6, s, 1e-6)
        return (vh.T * (1.0 / s)) @ u.T

def estimate_noise_cov_from_trials(trials_run_FxT, cond_idx_T, alpha=0.10):
    F, T = trials_run_FxT.shape
    cond_idx_T = np.asarray(cond_idx_T, dtype=int)
    if cond_idx_T.shape[0] != T:
        raise ValueError("cond_idx_T length != trials T")
    Rs = []
    for k in range(K):
        t_idx = np.where(cond_idx_T == k)[0]
        if t_idx.size < 2:
            continue
        X = trials_run_FxT[:, t_idx]
        mu = np.mean(X, axis=1, keepdims=True)
        Rs.append(X - mu)
    if not Rs:
        return np.eye(F, dtype=np.float32)
    R = np.concatenate(Rs, axis=1)
    S = np.cov(R, bias=False)
    S = shrink_cov(S, alpha=alpha).astype(np.float32)
    return S

def crossnobis_vec(P1_raw, P2_raw, Sigma_inv):
    K_, F = P1_raw.shape
    iu = np.triu_indices(K_, 1)
    out = np.empty(iu[0].size, np.float32)
    for m, (i, j) in enumerate(zip(iu[0], iu[1])):
        d1 = P1_raw[i] - P1_raw[j]
        d2 = P2_raw[i] - P2_raw[j]
        out[m] = float(d1 @ Sigma_inv @ d2)
    return out


# -------------------- Visualization helpers --------------------
def square_from_vec(v):
    M = np.zeros((K, K), float)
    M[IU] = v
    M[(IU[1], IU[0])] = v
    return M

def save_rdm_png(M, labels, path, dpi=120):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    im = ax.imshow(M, interpolation="nearest")
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(labels)
    ax.set_title("RDM")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def classical_mds(D, n_components=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]; v = v[:, idx]
    w = np.maximum(w, 0)
    L = np.diag(np.sqrt(w[:n_components]))
    X = v[:, :n_components] @ L
    return X

def save_mds_png(M, labels, path, dpi=120):
    D = M.copy(); np.fill_diagonal(D, 0.0)
    X = classical_mds(D, 2)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    ax.scatter(X[:, 0], X[:, 1])
    for i, lab in enumerate(labels):
        ax.text(X[i, 0], X[i, 1], lab)
    ax.set_title("MDS (classical)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def save_dendrogram_png(M, labels, path, dpi=120):
    from scipy.spatial.distance import squareform
    dvec = squareform(M, checks=False)
    Z = sch.linkage(dvec, method="average")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    sch.dendrogram(Z, labels=labels, orientation="right", ax=ax)
    ax.set_title("Dendrogram (avg-link)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# -------------------- Legacy NPZ recovery (if needed) --------------------
def _get_typed_field(typed_dict, key):
    if key in typed_dict:
        return typed_dict[key]
    for k in typed_dict.keys():
        if k.lower() == key.lower():
            return typed_dict[k]
    raise KeyError(f"Field '{key}' not found in results['typed'].")

def _unmask_betas_2d_to_4d(betas_2d, mask_bool, xyz):
    betas_2d = np.asarray(betas_2d)
    if betas_2d.ndim != 2:
        raise ValueError(f"Expected 2D betas, got {betas_2d.ndim}D.")
    nvox0 = betas_2d.shape[0]
    nmask = int(mask_bool.sum())
    ngrid = int(np.prod(xyz))
    if nvox0 == nmask:
        out = np.full(xyz + (betas_2d.shape[1],), np.nan, np.float32)
        out[mask_bool] = betas_2d.astype(np.float32, copy=False)
        return out
    if nvox0 == ngrid:
        return betas_2d.reshape(xyz + (betas_2d.shape[1],)).astype(np.float32, copy=False)
    raise ValueError(f"Cannot unmask betas: rows={nvox0}, mask={nmask}, grid={ngrid}")

def _try_extract_run_by_condition_from_betas_array(betas_arr, mask_bool, xyz):
    A = np.asarray(betas_arr)
    if A.ndim == 4 and A.shape[:3] == tuple(xyz):
        ncond = A.shape[3]
        if ncond == 2*K:
            return A[..., :K], A[..., K:2*K]
        return None

    if A.ndim == 2:
        vol4d = _unmask_betas_2d_to_4d(A, mask_bool, xyz)
        ncond = vol4d.shape[3]
        if ncond == 2*K:
            return vol4d[..., :K], vol4d[..., K:2*K]
        return None

    if A.ndim == 3:
        if A.shape[1] == K and A.shape[2] == 2:
            A2 = np.concatenate([A[:, :, 0], A[:, :, 1]], axis=1)
            vol4d = _unmask_betas_2d_to_4d(A2, mask_bool, xyz)
            return vol4d[..., :K], vol4d[..., K:2*K]
        if A.shape[1] == 2 and A.shape[2] == K:
            A2 = np.concatenate([A[:, 0, :], A[:, 1, :]], axis=1)
            vol4d = _unmask_betas_2d_to_4d(A2, mask_bool, xyz)
            return vol4d[..., :K], vol4d[..., K:2*K]
        return None

    return None

def try_load_legacy_runbetas_from_npz(sub, dtype, xyz, debug=0):
    npz_p = legacy_npz_path(sub, dtype)
    if not os.path.isfile(npz_p):
        if debug:
            print(f"[legacy] NPZ not found: {npz_p}")
        return None

    mpath = legacy_run01_mask_path(sub)
    if not os.path.isfile(mpath):
        if debug:
            print(f"[legacy] run-01 mask not found (needed for unmasking): {mpath}")
        return None
    mask_bool = nb.load(mpath).get_fdata().astype(bool)
    if mask_bool.shape != tuple(xyz):
        if debug:
            print(f"[legacy] mask shape {mask_bool.shape} != target xyz {tuple(xyz)}")
        return None

    try:
        with np.load(npz_p, allow_pickle=True) as z:
            results = z["results"].item() if "results" in z else z[z.files[0]].item()
    except Exception as e:
        if debug:
            print(f"[legacy] failed to load NPZ: {e}")
        return None

    if not isinstance(results, dict) or "typed" not in results:
        if debug:
            print("[legacy] NPZ does not look like a GLMsingle results dict with results['typed'].")
        return None

    typed = results["typed"]
    candidates = [k for k in typed.keys() if "betas" in k.lower()]
    key_order = (["betasmd"] if "betasmd" in typed else []) + [k for k in candidates if k != "betasmd"]

    for k in key_order:
        try:
            arr = _get_typed_field(typed, k)
            got = _try_extract_run_by_condition_from_betas_array(arr, mask_bool, xyz)
            if got is not None:
                if debug:
                    print(f"[legacy] recovered run-split betas from NPZ field '{k}' ({npz_p})")
                return got
        except Exception as e:
            if debug:
                print(f"[legacy] tried field '{k}' but failed: {e}")

    return None


# -------------------- Nilearn betas loader --------------------
def _read_nilearn_run_betas(sub, run, dtype, nilearn_outdir):
    if dtype != "unsmoothed":
        raise RuntimeError("Nilearn betas expected only for datatype=unsmoothed here.")
    betas_dir = os.path.join(nilearn_outdir, "betas")
    img_p = os.path.join(betas_dir, f"{sub}_run-{run}_{dtype}_betas_allconds_4d.nii.gz")
    tsv_p = os.path.join(betas_dir, f"{sub}_run-{run}_{dtype}_betas_allconds_columns.tsv")
    if not os.path.isfile(img_p):
        raise FileNotFoundError(f"[nilearn] Missing per-run betas 4D: {img_p}")
    if not os.path.isfile(tsv_p):
        raise FileNotFoundError(f"[nilearn] Missing columns TSV: {tsv_p}")
    img = nb.load(img_p)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"[nilearn] Expected 4D betas, got {data.shape} for {img_p}")
    cols = pd.read_csv(tsv_p, sep="\t")
    if "condition" in cols.columns:
        names = cols["condition"].astype(str).tolist()
    elif cols.shape[1] >= 1:
        names = cols.iloc[:, -1].astype(str).tolist()
    else:
        raise ValueError(f"[nilearn] Columns TSV has no usable column: {tsv_p}")
    if len(names) != data.shape[3]:
        raise ValueError(
            f"[nilearn] Column count mismatch for {sub} run-{run}: "
            f"TSV has {len(names)} names but image has {data.shape[3]} volumes."
        )
    return img, data, names

def _canon_label(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s

def _reorder_to_cond_order(data4d, names, sub, run, debug=0):
    names_l = [_canon_label(n) for n in names]
    want_l  = [_canon_label(c) for c in COND_ORDER]
    out = np.full(data4d.shape[:3] + (K,), np.nan, np.float32)
    hit = {}
    for k, w in enumerate(want_l):
        if w in names_l:
            j = names_l.index(w)
            out[..., k] = data4d[..., j]
            hit[w] = j
    if debug:
        missing = [COND_ORDER[i] for i, w in enumerate(want_l) if w not in hit]
        if missing:
            print(f"[nilearn] {sub} run-{run}: missing conditions -> NaN: {missing}")
    return out


# -------------------- Betas loader --------------------
def load_betas_split_runs(sub, rec, args, E_by_run, events_meta):
    # --- Nilearn ---
    if APPROACH_SPECS[rec].get("is_nilearn", False):
        if args.datatype != "unsmoothed":
            raise RuntimeError("[nilearn] only valid for datatype=unsmoothed in this pipeline.")
        sub = _normalize_sub(sub)
        img1, d1, n1 = _read_nilearn_run_betas(sub, "01", args.datatype, args.nilearn_outdir)
        img2, d2, n2 = _read_nilearn_run_betas(sub, "02", args.datatype, args.nilearn_outdir)
        if d1.shape[:3] != d2.shape[:3]:
            raise ValueError(f"[nilearn] {sub}: run-01 vs run-02 beta shapes differ: {d1.shape} vs {d2.shape}")
        r1 = _reorder_to_cond_order(d1, n1, sub, "01", debug=args.debug)
        r2 = _reorder_to_cond_order(d2, n2, sub, "02", debug=args.debug)
        split_note = "Loaded Nilearn per-run betas and reordered to COND_ORDER."
        return r1, r2, False, {"betas_file": "(nilearn per-run betas_4d)", "layout": "nilearn_run_by_condition_reordered"}, img1, split_note

    # --- GLMsingle ---
    p = betas_path(sub, args.datatype, rec)
    if p is None or (not os.path.isfile(p)):
        raise FileNotFoundError(f"[{rec}] Betas missing for {sub}: {p}")

    img = nb.load(p)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"[{rec}] Betas must be 4D, got shape {data.shape} for {p}")

    N1, N2 = len(E_by_run[0]), len(E_by_run[1])
    need = N1 + N2
    T = int(data.shape[3])
    split_note = ""

    if T == 2 * K:
        return data[..., :K], data[..., K:2*K], False, {"betas_file": p, "layout": "run_by_condition_2K"}, img, split_note

    if T == need:
        return data[..., :N1], data[..., N1:N1+N2], True, {"betas_file": p, "layout": "trial_concat_by_modeled_events_strict"}, img, split_note

    if args.max_split_mismatch and args.max_split_mismatch > 0:
        if T > need and (T - need) <= int(args.max_split_mismatch):
            data = data[..., :need]
            split_note = f"Trimmed EXTRA betas: T={T} -> {need} to match modeled events (N1={N1}, N2={N2})."
            if args.debug:
                print(f"  ⚠ {sub} [{rec}] {args.datatype}: {split_note}")
            return data[..., :N1], data[..., N1:N1+N2], True, {"betas_file": p, "layout": "trial_concat_trim_extra_betas_only"}, img, split_note

    if T == K and APPROACH_SPECS[rec]["is_legacy"]:
        xyz = data.shape[:3]
        got = try_load_legacy_runbetas_from_npz(sub, args.datatype, xyz, debug=args.debug)
        if got is not None:
            data1, data2 = got
            return data1.astype(np.float32), data2.astype(np.float32), False, {
                "betas_file": p,
                "layout": "legacy_npz_run_by_condition_2K_recovered",
                "npz_file": legacy_npz_path(sub, args.datatype),
            }, img, split_note
        raise ValueError(
            f"[{rec}] {sub}: betas file has only K={K} volumes (shape={data.shape}); split-half impossible. "
            f"Tried NPZ recovery but failed: {legacy_npz_path(sub, args.datatype)}"
        )

    raise ValueError(
        f"[{rec}] {sub}: Unrecognized betas 4th-dim length T={T}. "
        f"Expected 2*K={2*K} or EXACT N1+N2={need}. "
        f"If betas have extra volumes, set --max-split-mismatch > 0. File: {p}"
    )


# -------------------- KNN neighborhood builder --------------------
def build_knn_table(universe_mask, candidate_mask, k_neigh, debug=0):
    """
    universe_mask: (X,Y,Z) bool, group-level centers universe
    candidate_mask: (X,Y,Z) bool, subject-specific allowed neighbor voxels (must be subset of pool)
    Returns:
      centers_lin: (nC,) int linear indices of centers (order used in loop)
      neigh_lin:   (nC,k) int linear indices of neighbors (in candidate voxel set)
      kth_dist:    (nC,) float distance (vox units) to the k-th neighbor
      n_cand: int
    """
    dims = universe_mask.shape
    u_lin = np.flatnonzero(universe_mask.reshape(-1))
    c_lin = np.flatnonzero(candidate_mask.reshape(-1))
    nC = int(u_lin.size)
    nCand = int(c_lin.size)
    if nCand < int(k_neigh):
        raise RuntimeError(f"candidate_mask has only {nCand} voxels, but k_neigh={k_neigh}")

    # xyz coords in voxel index space
    u_xyz = np.column_stack(np.unravel_index(u_lin, dims)).astype(np.float32, copy=False)
    c_xyz = np.column_stack(np.unravel_index(c_lin, dims)).astype(np.float32, copy=False)

    tree = cKDTree(c_xyz)

    # workers=-1 supported on newer SciPy; fall back if not
    try:
        dist, idx = tree.query(u_xyz, k=int(k_neigh), workers=-1)
    except TypeError:
        dist, idx = tree.query(u_xyz, k=int(k_neigh))

    # Ensure 2D
    if int(k_neigh) == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    neigh_lin = c_lin[idx]  # map KDTree indices -> linear indices in full grid
    kth_dist = dist[:, -1].astype(np.float32, copy=False)

    if debug:
        print(f"[knn] centers={nC} candidate_vox={nCand} k={k_neigh} "
              f"kth_dist: median={float(np.median(kth_dist)):.3f} p95={float(np.percentile(kth_dist,95)):.3f} max={float(np.max(kth_dist)):.3f}")

    return u_lin.astype(np.int64, copy=False), neigh_lin.astype(np.int64, copy=False), kth_dist, nCand


# -------------------- Core per-subject processing --------------------
def process_subject(sub, rec, args, design_X, model_names, out_root, model_raw_vecs=None):
    sub = _normalize_sub(sub)

    # equalize runs?
    spec = APPROACH_SPECS[rec]
    if args.equalize_runs == "yes":
        equalize = True
    elif args.equalize_runs == "no":
        equalize = False
    else:
        equalize = bool(spec.get("equalized_runs", False))

    # events
    E_by_run, ev_meta = load_modeled_events_by_run(sub, args, equalize_runs=equalize)

    # betas split
    data1, data2, have_trials, input_desc, img_ref, split_note = load_betas_split_runs(sub, rec, args, E_by_run, ev_meta)

    # condition maps run1/run2
    if have_trials:
        cond_idx_run1 = E_by_run[0]["cond_idx"].to_numpy(dtype=int)
        cond_idx_run2 = E_by_run[1]["cond_idx"].to_numpy(dtype=int)
        B1_raw = collapse_trial_betas_to_conditions(data1, cond_idx_run1)
        B2_raw = collapse_trial_betas_to_conditions(data2, cond_idx_run2)
    else:
        if data1.shape[3] != K or data2.shape[3] != K:
            raise ValueError(f"[{rec}] {sub}: expected (..,K), got {data1.shape}, {data2.shape}")
        B1_raw = data1
        B2_raw = data2

    Xv, Yv, Zv, _ = B1_raw.shape
    ref_shape = (Xv, Yv, Zv)

    # ---------------- Group-level universe mask (centers) ----------------
    universe_mask = _load_bool_mask_maybe_resample(args.center_mask, img_ref, ref_shape, label="--center-mask", debug=args.debug)

    # Optional group neighbor pool mask (still group-level)
    if args.neighbor_mask:
        neighbor_pool = _load_bool_mask_maybe_resample(args.neighbor_mask, img_ref, ref_shape, label="--neighbor-mask", debug=args.debug)
        pool_mask = universe_mask & neighbor_pool
    else:
        pool_mask = universe_mask

    # Optional extra mask (must be on-grid)
    if args.mask:
        extra_m = nb.load(args.mask).get_fdata().astype(bool)
        if extra_m.shape != ref_shape:
            raise ValueError(f"--mask shape {extra_m.shape} != beta grid {ref_shape}")
        pool_mask &= extra_m
        universe_mask &= extra_m

    # Apply run-01 brain mask for safety (subject-specific but not GM)
    bm_path = find_brain_mask(args.preproc_dir, sub, run="01")
    if bm_path and os.path.isfile(bm_path):
        bm = nb.load(bm_path).get_fdata().astype(bool)
        if bm.shape == ref_shape:
            pool_mask &= bm
        elif args.debug:
            print(f"[warn] {sub} [{rec}]: brain mask shape mismatch: {bm.shape} vs {ref_shape}")
    elif args.debug:
        print(f"[warn] {sub} [{rec}]: no run-01 brain mask found under {args.preproc_dir} (continuing).")

    # ---------------- Subject candidate mask for neighbors ----------------
    # We only allow neighbors where patterns are well-defined for the subject.
    # This is NOT a GM spillover mask; it is strictly "data are finite + non-degenerate".
    with np.errstate(invalid="ignore"):
        var1 = np.nanvar(B1_raw, axis=3)
        var2 = np.nanvar(B2_raw, axis=3)
    finite_all = np.all(np.isfinite(B1_raw), axis=3) & np.all(np.isfinite(B2_raw), axis=3)
    nondeg = np.isfinite(var1) & np.isfinite(var2) & (var1 > 0) & (var2 > 0)
    candidate_mask = pool_mask & finite_all & nondeg

    nU = int(universe_mask.sum())
    nPool = int(pool_mask.sum())
    nCand = int(candidate_mask.sum())
    if nU < 100:
        raise RuntimeError(f"[{rec}] {sub}: too few voxels in universe_mask after masking ({nU}).")
    if nCand < max(200, int(args.k_neigh)):
        raise RuntimeError(f"[{rec}] {sub}: too few candidate neighbor voxels ({nCand}); need at least {max(200, int(args.k_neigh))}.")

    if args.debug:
        print(f"[mask] {sub} [{rec}]: universe={nU} pool={nPool} candidate={nCand} | neigh_mode={args.neigh_mode}")

    # z-score across conditions
    B1 = zscore_across_conditions(B1_raw.copy())
    B2 = zscore_across_conditions(B2_raw.copy())

    # Flatten patterns (K, V)
    V = Xv * Yv * Zv
    R1z = np.moveaxis(B1,     3, 0).reshape(K, V)
    R2z = np.moveaxis(B2,     3, 0).reshape(K, V)
    R1r = np.moveaxis(B1_raw, 3, 0).reshape(K, V)
    R2r = np.moveaxis(B2_raw, 3, 0).reshape(K, V)

    # Flatten trial betas if needed
    use_true_crossnobis = (args.metric == "crossnobis") and have_trials
    if args.metric == "crossnobis" and not have_trials and args.debug:
        print(f"[warn] {sub} [{rec}]: crossnobis requested but no trials -> using xcorrdiff distances for RSA target.")

    if use_true_crossnobis:
        cond_idx_run1 = E_by_run[0]["cond_idx"].to_numpy(dtype=int)
        cond_idx_run2 = E_by_run[1]["cond_idx"].to_numpy(dtype=int)
        T1 = int(data1.shape[3]); T2 = int(data2.shape[3])
        trials1_flat = data1.reshape(V, T1)
        trials2_flat = data2.reshape(V, T2)
    else:
        trials1_flat = None
        trials2_flat = None

    # ---------------- Build neighborhoods ----------------
    if args.neigh_mode == "adaptiveK":
        centers_lin, neigh_lin, kth_dist, _ = build_knn_table(universe_mask, candidate_mask, int(args.k_neigh), debug=args.debug)

        if args.k_hard_maxdist and args.k_maxdist_vox and args.k_maxdist_vox > 0:
            hard_ok = kth_dist <= float(args.k_maxdist_vox)
            if args.debug:
                print(f"[knn] {sub} [{rec}]: hard_maxdist={args.k_maxdist_vox:g} -> centers_ok={int(hard_ok.sum())}/{int(hard_ok.size)}")
        else:
            hard_ok = None
    else:
        raise RuntimeError("radius mode was removed from this improved version. Use --neigh-mode adaptiveK.")

    # ---------------- Global QC (computed on candidate voxels within universe) ----------------
    qc_lin = np.flatnonzero((candidate_mask & universe_mask).reshape(-1))
    if qc_lin.size < 50:
        raise RuntimeError(f"[{rec}] {sub}: too few QC voxels for global QC ({int(qc_lin.size)}).")

    P1_all = R1z[:, qc_lin]
    P2_all = R2z[:, qc_lin]
    dcv_all, d1_all, d2_all = xcorrdiff_vec(P1_all, P2_all)
    rel_spear = _safe_spearman(d1_all, d2_all, min_n=10)
    rel_pear  = _safe_pearson(d1_all, d2_all, min_n=10)
    offdiag_mean = 0.5 * (mean_offdiag_corr(P1_all) + mean_offdiag_corr(P2_all))

    def load_tsnr(subj, run):
        if not args.tsnr_dir:
            return np.nan
        path = os.path.join(args.tsnr_dir, subj, "func", f"{subj}_task-PLDs_run-{run}_desc-tsnr_bold.nii.gz")
        if not os.path.isfile(path):
            return np.nan
        tsn = nb.load(path).get_fdata()
        m = universe_mask & np.isfinite(tsn)
        return np.nanmedian(tsn[m]) if np.any(m) else np.nan

    tsnr1 = load_tsnr(sub, "01")
    tsnr2 = load_tsnr(sub, "02")
    tsnr_med = np.nanmedian([tsnr1, tsnr2])

    passed_gate = True
    if np.isfinite(rel_spear) and rel_spear < args.gate_rel_spear:
        passed_gate = False
    if np.isfinite(rel_pear) and rel_pear < args.gate_rel_pear:
        passed_gate = False
    if np.isfinite(tsnr_med) and tsnr_med < args.gate_tsnr:
        passed_gate = False

    if args.debug:
        print(f"[qc] {sub} [{rec}]: global rel_spear={rel_spear:.3f}, rel_pear={rel_pear:.3f}, "
              f"offdiag_mean={offdiag_mean:.3f}, tsnr_med={tsnr_med:.2f}, passed_gate={passed_gate}")
        if split_note:
            print(f"[qc] {sub} [{rec}]: split_note: {split_note}")

    # ---------------- Searchlight outputs ----------------
    P = design_X.shape[1]
    aff = img_ref.affine
    hdr = img_ref.header

    # Initialize inference-relevant maps to finite defaults INSIDE universe_mask
    def init_map(default_inside, nan_outside=True, dtype=np.float32):
        m = np.full(ref_shape, np.nan if nan_outside else default_inside, dtype=dtype)
        m[universe_mask] = default_inside
        return m

    beta_maps = [init_map(0.0) for _ in range(P)]
    t_maps    = [init_map(0.0) for _ in range(P)]
    p_maps    = [init_map(1.0) for _ in range(P)]
    pr2_maps  = [init_map(0.0) for _ in range(P)]
    r2_full   = init_map(0.0)

    # QC maps (NaNs are fine)
    rel_map   = np.full(ref_shape, np.nan, np.float32)
    pat_map   = np.full(ref_shape, np.nan, np.float32)
    off_map   = np.full(ref_shape, np.nan, np.float32)
    cov_map   = init_map(float(args.k_neigh))  # constant K in adaptiveK
    kth_map   = np.full(ref_shape, np.nan, np.float32)  # effective radius (distance to Kth neighbor)
    fit_ok    = init_map(0.0)  # 1.0 where GLM solved cleanly

    rdmvec_4d = None
    if args.save_searchlight_rdmvec:
        rdmvec_4d = np.full(ref_shape + (N_PAIRS,), np.nan, np.float32)

    # GLM precompute
    X_std = design_X.copy()
    X_aug = np.column_stack([np.ones(N_PAIRS, dtype=np.float32), X_std])
    XtX = X_aug.T @ X_aug
    if args.ridge_lambda > 0.0:
        reg = np.zeros_like(XtX); np.fill_diagonal(reg, 1.0); reg[0, 0] = 0.0
        XtX_eff = XtX + (args.ridge_lambda * reg)
    else:
        XtX_eff = XtX
    XtX_inv = safe_inv(XtX_eff)
    Xt = X_aug.T

    # Loop over centers in the fixed universe
    dims = ref_shape
    cx, cy, cz = np.unravel_index(centers_lin, dims)

    for i in range(centers_lin.size):
        if hard_ok is not None and (not hard_ok[i]):
            continue  # leaves defaults; NOTE: this reintroduces holes only if you later treat these as missing

        x = int(cx[i]); y = int(cy[i]); z = int(cz[i])
        neigh_i = neigh_lin[i, :]

        kth_map[x, y, z] = float(kth_dist[i])

        # Extract patterns (K, F) with F=Kneigh; candidate_mask ensures finite + nondeg
        P1z = R1z[:, neigh_i]
        P2z = R2z[:, neigh_i]
        P1r = R1r[:, neigh_i]
        P2r = R2r[:, neigh_i]
        Fvox = int(P1z.shape[1])

        # Pattern split-half (median across conditions)
        rks = []
        for k in range(K):
            a = P1z[k]; b = P2z[k]
            aa = a - a.mean()
            bb = b - b.mean()
            den = np.sqrt((aa*aa).sum() * (bb*bb).sum())
            rks.append(float((aa @ bb) / den) if den > 0 else np.nan)
        pat_map[x, y, z] = np.nanmedian(np.asarray(rks, float))

        # Off-diagonal mean corr (conditions)
        off_map[x, y, z] = np.nanmean([mean_offdiag_corr(P1z), mean_offdiag_corr(P2z)])

        # Distances + reliability
        if use_true_crossnobis:
            trials1 = trials1_flat[neigh_i, :].astype(np.float32, copy=False)  # (F,T1)
            trials2 = trials2_flat[neigh_i, :].astype(np.float32, copy=False)  # (F,T2)
            S1 = estimate_noise_cov_from_trials(trials1, cond_idx_run1, alpha=args.shrink_alpha)
            S2 = estimate_noise_cov_from_trials(trials2, cond_idx_run2, alpha=args.shrink_alpha)
            Spool = 0.5 * (S1 + S2)
            Sigma_inv = safe_inv(Spool)
            dcv = crossnobis_vec(P1r, P2r, Sigma_inv)
            _, d1, d2 = xcorrdiff_vec(P1z, P2z)  # within-run distances for reliability
        else:
            dcv, d1, d2 = xcorrdiff_vec(P1z, P2z)

        rho_rel = _safe_spearman(d1, d2, min_n=10)
        rel_map[x, y, z] = np.float32(rho_rel) if np.isfinite(rho_rel) else np.nan

        # Optional reliability gating (WARNING: creates holes)
        if args.min_rdm_reliability >= 0 and np.isfinite(rho_rel) and (rho_rel < args.min_rdm_reliability):
            continue

        # GLM RSA
        yv = np.asarray(dcv, float)
        if args.glm_rank:
            yv = rankdata(yv, method="average").astype(float)

        ymu = np.nanmean(yv); ysd = np.nanstd(yv, ddof=1)
        if (not np.isfinite(ysd)) or (ysd <= 0):
            # Degenerate: no variance in target. Best honest default is beta=0 (already set).
            continue

        yz = (yv - ymu) / ysd

        beta = XtX_inv @ (Xt @ yz)
        fit = X_aug @ beta
        resid = yz - fit
        df = N_PAIRS - (P + 1)
        if df <= 0:
            continue

        RSS = float(np.dot(resid, resid))
        fit_ok[x, y, z] = 1.0

        if args.ridge_lambda > 0.0:
            for j in range(P):
                beta_maps[j][x, y, z] = np.float32(beta[j+1])
        else:
            sigma2 = RSS / df
            cov_beta = sigma2 * XtX_inv
            se = np.sqrt(np.clip(np.diag(cov_beta), 0, np.inf))
            for j in range(P):
                b = float(beta[j+1]); s = float(se[j+1])
                if (not np.isfinite(s)) or (s <= 0):
                    # leave defaults: t=0, p=1, pr2=0
                    beta_maps[j][x, y, z] = np.float32(b) if np.isfinite(b) else 0.0
                    continue
                tval = b / s
                pr2  = (tval*tval) / (tval*tval + df)
                pval = 2.0 * float(student_t.sf(abs(tval), df))
                beta_maps[j][x, y, z] = np.float32(b)
                t_maps[j][x, y, z]    = np.float32(tval)
                p_maps[j][x, y, z]    = np.float32(pval)
                pr2_maps[j][x, y, z]  = np.float32(pr2)

        # R2 (kept finite)
        r2_full[x, y, z] = np.float32(1.0 - RSS / (N_PAIRS - 1))

        if rdmvec_4d is not None:
            rdmvec_4d[x, y, z, :] = dcv

    # ---------------- Save outputs ----------------
    out_dir = os.path.join(out_root, sub)
    os.makedirs(out_dir, exist_ok=True)

    tag = (f"{args.datatype}_approach{rec}_{args.neigh_mode}"
           f"_k{int(args.k_neigh)}_{args.metric}"
           + (f"_ridge{args.ridge_lambda:g}" if args.ridge_lambda > 0 else ""))

    def outpath(name):
        return os.path.join(out_dir, f"{sub}_{tag}_{name}.nii.gz")

    # Save the aligned universe mask once per subject dir for sanity
    nb.Nifti1Image(universe_mask.astype(np.uint8), aff, hdr).to_filename(os.path.join(out_dir, f"{sub}_{tag}_universe_mask.nii.gz"))
    nb.Nifti1Image(candidate_mask.astype(np.uint8), aff, hdr).to_filename(os.path.join(out_dir, f"{sub}_{tag}_candidate_mask.nii.gz"))

    nb.Nifti1Image(r2_full, aff, hdr).to_filename(outpath("r2_full"))
    nb.Nifti1Image(rel_map, aff, hdr).to_filename(outpath("rdm_rel_spearman"))
    nb.Nifti1Image(pat_map, aff, hdr).to_filename(outpath("pattern_r_median"))
    nb.Nifti1Image(off_map, aff, hdr).to_filename(outpath("offdiag_mean_corr"))
    nb.Nifti1Image(cov_map, aff, hdr).to_filename(outpath("coverage_neighbors"))
    nb.Nifti1Image(kth_map, aff, hdr).to_filename(outpath("knn_kth_distance_vox"))
    nb.Nifti1Image(fit_ok,  aff, hdr).to_filename(outpath("glm_fit_ok"))

    for j, name in enumerate(model_names):
        nb.Nifti1Image(beta_maps[j], aff, hdr).to_filename(outpath(f"beta-{name}"))
        nb.Nifti1Image(t_maps[j],    aff, hdr).to_filename(outpath(f"t-{name}"))
        nb.Nifti1Image(p_maps[j],    aff, hdr).to_filename(outpath(f"p-{name}"))
        nb.Nifti1Image(pr2_maps[j],  aff, hdr).to_filename(outpath(f"pr2-{name}"))

    if rdmvec_4d is not None:
        nb.Nifti1Image(rdmvec_4d, aff, hdr).to_filename(os.path.join(out_dir, f"{sub}_{tag}_rdmvec4d.nii.gz"))

    if args.save_summary_rdms or args.make_figures:
        M_run1 = square_from_vec(d1_all)
        M_run2 = square_from_vec(d2_all)
        M_sess = (M_run1 + M_run2) / 2.0
        M_cv   = square_from_vec(dcv_all)

        if args.save_summary_rdms:
            np.save(os.path.join(out_dir, f"{sub}_{tag}_RDM_run1.npy"), M_run1)
            np.save(os.path.join(out_dir, f"{sub}_{tag}_RDM_run2.npy"), M_run2)
            np.save(os.path.join(out_dir, f"{sub}_{tag}_RDM_sessionAvg.npy"), M_sess)
            np.save(os.path.join(out_dir, f"{sub}_{tag}_RDM_crossval.npy"), M_cv)

        if args.make_figures:
            fig_dir = os.path.join(out_dir, "figs"); os.makedirs(fig_dir, exist_ok=True)
            save_rdm_png(M_sess, COND_ORDER, os.path.join(fig_dir, f"{sub}_{tag}_RDM_sessionAvg.png"), dpi=args.fig_dpi)
            save_mds_png(M_sess, COND_ORDER, os.path.join(fig_dir, f"{sub}_{tag}_MDS_sessionAvg.png"), dpi=args.fig_dpi)
            save_dendrogram_png(M_sess, COND_ORDER, os.path.join(fig_dir, f"{sub}_{tag}_Dendro_sessionAvg.png"), dpi=args.fig_dpi)

    # -------- Post-RSA reliability summary (over universe centers) --------
    good_centers = universe_mask & np.isfinite(rel_map)
    rel_vals = rel_map[good_centers].astype(float)
    rsa_rel_median = float(np.nanmedian(rel_vals)) if rel_vals.size else np.nan
    rsa_rel_mean   = float(np.nanmean(rel_vals))   if rel_vals.size else np.nan
    rsa_rel_p75    = float(np.nanpercentile(rel_vals, 75)) if rel_vals.size else np.nan
    rsa_rel_p95    = float(np.nanpercentile(rel_vals, 95)) if rel_vals.size else np.nan
    rsa_rel_nvox   = int(rel_vals.size)

    kth_vals = kth_map[universe_mask & np.isfinite(kth_map)].astype(float)
    kth_med = float(np.nanmedian(kth_vals)) if kth_vals.size else np.nan
    kth_p95 = float(np.nanpercentile(kth_vals, 95)) if kth_vals.size else np.nan
    kth_max = float(np.nanmax(kth_vals)) if kth_vals.size else np.nan

    prov = {
        "subject": sub,
        "approach": rec,
        "datatype": args.datatype,
        "neigh_mode": args.neigh_mode,
        "k_neigh": int(args.k_neigh),
        "k_hard_maxdist": bool(args.k_hard_maxdist),
        "k_maxdist_vox": float(args.k_maxdist_vox),
        "metric": args.metric,
        "have_trials": bool(have_trials),
        "input_desc": input_desc,
        "split_note": split_note,
        "events_meta": ev_meta,

        "mask_universe_path": args.center_mask,
        "mask_neighbor_pool_path": args.neighbor_mask or args.center_mask,
        "mask_extra_path": args.mask or "",

        "n_universe_vox": int(universe_mask.sum()),
        "n_candidate_vox": int(candidate_mask.sum()),

        "knn_kth_dist_median_vox": kth_med,
        "knn_kth_dist_p95_vox": kth_p95,
        "knn_kth_dist_max_vox": kth_max,

        "global_rel_spear": float(rel_spear) if np.isfinite(rel_spear) else np.nan,
        "global_rel_pear": float(rel_pear) if np.isfinite(rel_pear) else np.nan,
        "offdiag_mean": float(offdiag_mean),

        "tSNR_run1": float(tsnr1) if np.isfinite(tsnr1) else np.nan,
        "tSNR_run2": float(tsnr2) if np.isfinite(tsnr2) else np.nan,
        "tSNR_median": float(tsnr_med) if np.isfinite(tsnr_med) else np.nan,
        "passed_gate": bool(passed_gate),

        "rsa_rel_median": rsa_rel_median,
        "rsa_rel_mean": rsa_rel_mean,
        "rsa_rel_p75": rsa_rel_p75,
        "rsa_rel_p95": rsa_rel_p95,
        "rsa_rel_nvox": rsa_rel_nvox,
    }
    with open(os.path.join(out_dir, f"{sub}_{tag}_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    row = {
        "subject": sub,
        "approach": rec,
        "datatype": args.datatype,
        "neigh_mode": args.neigh_mode,
        "k_neigh": int(args.k_neigh),
        "metric": args.metric,
        "have_trials": bool(have_trials),
        "passed_gate": bool(passed_gate),
        "n_universe_vox": int(universe_mask.sum()),
        "n_candidate_vox": int(candidate_mask.sum()),
        "knn_kth_med_vox": kth_med,
        "knn_kth_p95_vox": kth_p95,
        "knn_kth_max_vox": kth_max,
        "global_rel_spear": prov["global_rel_spear"],
        "global_rel_pear": prov["global_rel_pear"],
        "offdiag_mean": prov["offdiag_mean"],
        "tSNR_median": prov["tSNR_median"],
        "rsa_rel_median": rsa_rel_median,
        "rsa_rel_mean": rsa_rel_mean,
        "rsa_rel_p75": rsa_rel_p75,
        "rsa_rel_p95": rsa_rel_p95,
        "rsa_rel_nvox": rsa_rel_nvox,
        "out_dir": out_dir,
    }
    return row


# -------------------- Subject listing --------------------
def list_subjects_auto(args):
    if args.subjects:
        return [_normalize_sub(s) for s in args.subjects]

    subs = []
    if os.path.isdir(EVENTS_DIR):
        subs = sorted([d for d in os.listdir(EVENTS_DIR)
                       if d.startswith("sub-") and os.path.isdir(os.path.join(EVENTS_DIR, d))])
    if subs:
        return subs

    rec0 = args.approaches[0]
    base = APPROACH_SPECS[rec0]["data_root"]
    if os.path.isdir(base):
        return sorted([d for d in os.listdir(base)
                       if d.startswith("sub-") and os.path.isdir(os.path.join(base, d))])

    raise RuntimeError("Could not auto-detect subjects. Use --subjects explicitly.")


# -------------------- Main --------------------
def main():
    args = parse_args()

    if args.neigh_mode == "adaptiveK":
        if args.k_neigh < 10:
            raise RuntimeError("--k-neigh is unrealistically small; use something like 40–120.")
    else:
        raise RuntimeError("Only --neigh-mode adaptiveK is supported in this improved version.")

    design_X, model_names, model_raw_vecs = load_model_design(args.model_rdms, do_rank=args.glm_rank)
    subjects = list_subjects_auto(args)
    datatypes = args.datatype if (args.datatype is not None) else ["unsmoothed", "smoothed"]

    for dtype in datatypes:
        args.datatype = dtype

        suffix = (f"{args.neigh_mode}_k{int(args.k_neigh)}_{args.metric}"
                  + (f"_ridge{args.ridge_lambda:g}" if args.ridge_lambda > 0 else ""))

        out_root_by_rec = {}
        compare_dir_by_rec = {}
        for rec in args.approaches:
            out_root_by_rec[rec] = ensure_dir(os.path.join(OUT_BASE, rec, args.out_tag, args.datatype, suffix, "runs"))
            compare_dir_by_rec[rec] = ensure_dir(os.path.join(OUT_BASE, rec, args.out_tag, args.datatype, suffix, "compare"))

        rows = []
        for rec in args.approaches:
            if rec == "nilearn" and args.datatype != "unsmoothed":
                print(f"\n=== SKIP {rec} | datatype={args.datatype} (nilearn runs only on unsmoothed) ===")
                continue

            out_root = out_root_by_rec[rec]
            for sub in subjects:
                try:
                    print(f"\n=== {sub} | {rec} | {args.datatype} | {args.neigh_mode} k={int(args.k_neigh)} ===")
                    row = process_subject(sub, rec, args, design_X, model_names, out_root, model_raw_vecs=model_raw_vecs)
                    rows.append(row)
                except Exception as e:
                    print(f"[ERROR] {sub} [{rec}] ({args.datatype}): {e}")
                    rows.append({
                        "subject": sub, "approach": rec, "datatype": args.datatype,
                        "neigh_mode": args.neigh_mode, "k_neigh": int(args.k_neigh), "metric": args.metric,
                        "have_trials": np.nan, "passed_gate": False,
                        "n_universe_vox": 0, "n_candidate_vox": 0,
                        "knn_kth_med_vox": np.nan, "knn_kth_p95_vox": np.nan, "knn_kth_max_vox": np.nan,
                        "global_rel_spear": np.nan, "global_rel_pear": np.nan,
                        "offdiag_mean": np.nan, "tSNR_median": np.nan,
                        "rsa_rel_median": np.nan, "rsa_rel_mean": np.nan,
                        "rsa_rel_p75": np.nan, "rsa_rel_p95": np.nan,
                        "rsa_rel_nvox": 0, "out_dir": ""
                    })

        df = pd.DataFrame(rows)

        # write compare CSVs under each approach
        for rec in args.approaches:
            compdir = compare_dir_by_rec[rec]
            subj_csv = os.path.join(compdir, f"QC_postRSA_compare_{args.datatype}_{suffix}.csv")
            df.to_csv(subj_csv, index=False)
            print(f"\n[OK] Wrote subject-level comparison CSV:\n  {subj_csv}")

        # summary over ALL results (no gating)
        agg_all = (df
            .groupby("approach", dropna=False)
            .agg(
                n_subjects=("subject", "nunique"),
                rsa_rel_median_mean=("rsa_rel_median", "mean"),
                rsa_rel_median_median=("rsa_rel_median", "median"),
                rsa_rel_mean_mean=("rsa_rel_mean", "mean"),
                rsa_rel_nvox_mean=("rsa_rel_nvox", "mean"),
                n_universe_vox_mean=("n_universe_vox", "mean"),
                n_candidate_vox_mean=("n_candidate_vox", "mean"),
                knn_kth_p95_vox_mean=("knn_kth_p95_vox", "mean"),
            )
            .reset_index()
            .sort_values("rsa_rel_median_mean", ascending=False))

        # gated-only summary (convenience)
        df_gate = df[df["passed_gate"] == True].copy()
        agg_gate = (df_gate
            .groupby("approach", dropna=False)
            .agg(
                n_subjects=("subject", "nunique"),
                rsa_rel_median_mean=("rsa_rel_median", "mean"),
                rsa_rel_median_median=("rsa_rel_median", "median"),
                rsa_rel_mean_mean=("rsa_rel_mean", "mean"),
                rsa_rel_nvox_mean=("rsa_rel_nvox", "mean"),
            )
            .reset_index()
            .sort_values("rsa_rel_median_mean", ascending=False)) if (not df_gate.empty) else pd.DataFrame()

        for rec in args.approaches:
            compdir = compare_dir_by_rec[rec]
            agg_csv = os.path.join(compdir, f"QC_postRSA_compare_{args.datatype}_{suffix}_APPROACH_SUMMARY_ALL.csv")
            agg_all.to_csv(agg_csv, index=False)
            print(f"\n[OK] Wrote approach-level summary (ALL):\n  {agg_csv}")
            if not agg_gate.empty:
                agg_csv2 = os.path.join(compdir, f"QC_postRSA_compare_{args.datatype}_{suffix}_APPROACH_SUMMARY_PASSEDGATE.csv")
                agg_gate.to_csv(agg_csv2, index=False)
                print(f"[OK] Wrote approach-level summary (passed_gate):\n  {agg_csv2}")

        print("\n=== Approach ranking (higher = better post-RSA split-half reliability; ALL subjects) ===")
        for _, r in agg_all.iterrows():
            print(f"  {r['approach']}: mean(median rel)={r['rsa_rel_median_mean']:.4f} | "
                  f"median(median rel)={r['rsa_rel_median_median']:.4f} | n_subs={int(r['n_subjects'])} | "
                  f"mean(knn kth p95 vox)={r['knn_kth_p95_vox_mean']:.2f}")

if __name__ == "__main__":
    main()
