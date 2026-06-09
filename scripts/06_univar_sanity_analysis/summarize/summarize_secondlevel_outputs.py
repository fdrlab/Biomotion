#!/usr/bin/env python3
"""
summarize_secondlevel_outputs.py

Scans selected subtrees of a second_level/ tree for outputs produced by
second_level_compare_new.py and summarizes + compares them.

DISCOVERY UNIT:
  Each (analysis_dir, tag) where analysis_dir contains:
    group_zmap_<tag>.nii.gz

SCAN ROOTS (NEW):
  Only scans the directories in --scan-roots (defaults to):
    - .../second_level/GLMsingle_new/
    - .../second_level/nilearn_new_smoothed/

MASKING:
  Uses mask_used.nii.gz if present in analysis_dir; else uses finite voxels in zmap.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nb


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def data_f32(img):
    return img.get_fdata(dtype=np.float32)


def voxel_volume_mm3(img):
    zooms = img.header.get_zooms()[:3]
    return float(zooms[0] * zooms[1] * zooms[2])


def mm3_to_ml(mm3):
    return float(mm3) / 1000.0


def pick_one(glob_list, prefer_regex=None):
    if not glob_list:
        return None
    gl = sorted(set(glob_list))
    if prefer_regex:
        rx = re.compile(prefer_regex)
        pref = [p for p in gl if rx.search(os.path.basename(p))]
        if pref:
            return pref[0]
    return gl[0]


def parse_fwhm_from_tag(tag):
    m = re.search(r"_fwhm([0-9]+(?:\.[0-9]+)?)", tag)
    if m:
        return float(m.group(1))
    if tag.endswith("_unsmoothed"):
        return 0.0
    return np.nan


def infer_pipeline_from_tag(tag):
    if tag.startswith("nilearn__"):
        return "nilearn"
    if tag.startswith("glmsingle__"):
        return "glmsingle"
    return "unknown"


def infer_name_from_tag(tag):
    t = tag
    t = re.sub(r"^(nilearn__|glmsingle__)", "", t)
    t = re.sub(r"_(unsmoothed|fwhm[0-9]+(?:\.[0-9]+)?)$", "", t)
    return t


def get_mask_bool(analysis_dir, like_img):
    mpath = os.path.join(analysis_dir, "mask_used.nii.gz")
    if not os.path.isfile(mpath):
        return None, None
    mimg = nb.load(mpath)

    if mimg.shape != like_img.shape:
        try:
            from nilearn.image import resample_to_img
            mimg = resample_to_img(mimg, like_img, interpolation="nearest")
        except Exception:
            return None, None

    m = data_f32(mimg) > 0
    if not np.any(m):
        return None, None
    return m, mimg


def summarize_cluster_csv(csv_path):
    out = dict(
        clusters_csv=os.path.basename(csv_path),
        n_clusters=np.nan,
        max_cluster_size_mm3=np.nan,
        max_peak_stat=np.nan,
    )
    try:
        df = pd.read_csv(csv_path)
        if df.shape[0] == 0:
            out["n_clusters"] = 0
            return out

        out["n_clusters"] = int(df.shape[0])

        size_cols = [c for c in df.columns if re.search(r"(cluster.*size|size)", c, flags=re.I)]
        if size_cols:
            s = pd.to_numeric(df[size_cols[0]], errors="coerce")
            out["max_cluster_size_mm3"] = float(np.nanmax(s.values))

        peak_cols = [c for c in df.columns if re.search(r"(peak.*stat|peak.*z|stat)", c, flags=re.I)]
        if peak_cols:
            p = pd.to_numeric(df[peak_cols[0]], errors="coerce")
            out["max_peak_stat"] = float(np.nanmax(p.values))

        return out
    except Exception:
        return out


def infer_glmsingle_input_from_kept_subjects(analysis_dir, pipeline):
    if pipeline != "glmsingle":
        return "n/a"

    kept_csv = os.path.join(analysis_dir, "kept_subjects.csv")
    if not os.path.isfile(kept_csv):
        return "unknown"

    try:
        kdf = pd.read_csv(kept_csv)
        if "path" not in kdf.columns or kdf.empty:
            return "unknown"
        paths = kdf["path"].astype(str).tolist()
    except Exception:
        return "unknown"

    fn_unsm = 0
    fn_sm = 0
    for p in paths:
        b = os.path.basename(p).lower()
        if "_unsmoothed_" in b:
            fn_unsm += 1
        if "_smoothed_" in b:
            fn_sm += 1

    if fn_unsm > 0 and fn_sm > 0:
        return "mixed"
    if fn_unsm > 0:
        return "unsmoothed"
    if fn_sm > 0:
        return "smoothed"

    # fallback: directory heuristics
    dir_unsm = 0
    dir_sm = 0
    for p in paths:
        pl = p.lower()
        if re.search(r"(/|\\)(unsmoothed)(/|\\)", pl) or re.search(r"(/|\\)no[_-]?smoothing(/|\\)", pl):
            dir_unsm += 1
        if re.search(r"(/|\\)(smoothed)(/|\\)", pl) or re.search(r"(/|\\)smooth[-_]?6(/|\\)", pl):
            dir_sm += 1

    if dir_unsm > 0 and dir_sm > 0:
        return "mixed"
    if dir_unsm > 0:
        return "unsmoothed"
    if dir_sm > 0:
        return "smoothed"

    return "unknown"


def find_summary_txt(start_dir, stop_dir):
    cur = os.path.abspath(start_dir)
    stop = os.path.abspath(stop_dir)

    while True:
        cand = os.path.join(cur, "summary.txt")
        if os.path.isfile(cand):
            return cand

        if cur == stop:
            break

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return None


def read_mask_arg_from_summary(summary_txt_path):
    if not summary_txt_path or not os.path.isfile(summary_txt_path):
        return "unknown"
    try:
        with open(summary_txt_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Mask:"):
                    return line.split("Mask:", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def is_no_mask_by_arg(mask_arg):
    s = str(mask_arg).strip().lower()
    return s in {"none", "null", ""}


def is_no_mask_by_path(s):
    sl = str(s).lower()
    return ("no-mask" in sl) or ("no_mask" in sl)


# -------------------------
# Core per-analysis metrics
# -------------------------
def compute_metrics_for_analysis(analysis_dir, tag, rel_root):
    rec = dict(
        rel_path=rel_root,
        analysis_dir=analysis_dir,
        tag=tag,
        pipeline=infer_pipeline_from_tag(tag),
        approach=infer_name_from_tag(tag),
        fwhm=parse_fwhm_from_tag(tag),
    )

    zmap_path = os.path.join(analysis_dir, f"group_zmap_{tag}.nii.gz")
    if not os.path.isfile(zmap_path):
        rec["status"] = "missing_zmap"
        return rec

    fdr_paths = glob.glob(os.path.join(analysis_dir, f"group_zmap_FDR0.05*_{tag}.nii.gz"))
    fdr_path = pick_one(fdr_paths)

    pmap_path = os.path.join(analysis_dir, f"group_pmap_parametric_{tag}.nii.gz")
    bonf_neglogp_path = os.path.join(analysis_dir, f"group_neglog10p_bonf_parametric_{tag}.nii.gz")

    perm_neglogp_paths = glob.glob(os.path.join(analysis_dir, f"group_perm_*_neglog10_vfwe_*n_{tag}.nii.gz"))
    perm_neglogp_path = pick_one(perm_neglogp_paths)
    perm_sig_paths = glob.glob(os.path.join(analysis_dir, f"group_perm_*_vfwe_p_lt0p05_*n_{tag}.nii.gz"))
    perm_sig_path = pick_one(perm_sig_paths)

    clus_fdr_paths = glob.glob(os.path.join(analysis_dir, f"group_clusters_FDR0.05*_{tag}.csv"))
    clus_unc_paths = glob.glob(os.path.join(analysis_dir, f"group_clusters_Zgt*_unc_{tag}.csv"))
    clus_fdr_path = pick_one(clus_fdr_paths)
    clus_unc_path = pick_one(clus_unc_paths)

    rec.update(dict(
        zmap_path=zmap_path,
        fdr_path=(fdr_path or ""),
        pmap_path=(pmap_path if os.path.isfile(pmap_path) else ""),
        bonf_neglogp_path=(bonf_neglogp_path if os.path.isfile(bonf_neglogp_path) else ""),
        perm_neglogp_path=(perm_neglogp_path or ""),
        perm_sig_path=(perm_sig_path or ""),
        clus_fdr_path=(clus_fdr_path or ""),
        clus_unc_path=(clus_unc_path or ""),
    ))

    zimg = nb.load(zmap_path)
    z = data_f32(zimg)
    vv = voxel_volume_mm3(zimg)

    mask_bool, _ = get_mask_bool(analysis_dir, zimg)
    if mask_bool is None:
        mask_bool = np.isfinite(z)

    inmask = mask_bool & np.isfinite(z)
    nmask = int(np.sum(inmask))
    rec["n_vox_mask"] = nmask
    rec["mask_vol_ml"] = mm3_to_ml(nmask * vv)

    if nmask > 0:
        zv = z[inmask]
        rec["z_max"] = float(np.nanmax(zv))
        rec["z_min"] = float(np.nanmin(zv))
        rec["z_max_abs"] = float(np.nanmax(np.abs(zv)))
        rec["z_mean"] = float(np.nanmean(zv))
        rec["z_mean_abs"] = float(np.nanmean(np.abs(zv)))
        rec["z_p95_abs"] = float(np.nanpercentile(np.abs(zv), 95))
    else:
        rec.update(dict(
            z_max=np.nan, z_min=np.nan, z_max_abs=np.nan,
            z_mean=np.nan, z_mean_abs=np.nan, z_p95_abs=np.nan
        ))

    # FDR
    rec["fdr_sig_vox"] = np.nan
    rec["fdr_sig_ml"] = np.nan
    rec["fdr_max_abs"] = np.nan
    if fdr_path and os.path.isfile(fdr_path):
        fimg = nb.load(fdr_path)
        f = data_f32(fimg)
        f_in = np.isfinite(f) & (f != 0)
        nf = int(np.sum(f_in))
        rec["fdr_sig_vox"] = nf
        rec["fdr_sig_ml"] = mm3_to_ml(nf * vv)
        if nf > 0:
            rec["fdr_max_abs"] = float(np.nanmax(np.abs(f[f_in])))

    # Parametric p-map
    rec["p_min"] = np.nan
    rec["neglog10p_max"] = np.nan
    if rec["pmap_path"]:
        pimg = nb.load(rec["pmap_path"])
        p = data_f32(pimg)
        p_in = inmask & np.isfinite(p)
        if np.any(p_in):
            pv = p[p_in]
            rec["p_min"] = float(np.nanmin(pv))
            rec["neglog10p_max"] = float(np.nanmax(-np.log10(np.maximum(pv, 1e-300))))

    # Bonferroni -log10(p)
    rec["neglog10p_bonf_max"] = np.nan
    if rec["bonf_neglogp_path"]:
        bimg = nb.load(rec["bonf_neglogp_path"])
        b = data_f32(bimg)
        b_in = inmask & np.isfinite(b)
        if np.any(b_in):
            rec["neglog10p_bonf_max"] = float(np.nanmax(b[b_in]))

    # Permutation outputs
    rec["perm_sig_vox"] = np.nan
    rec["perm_sig_ml"] = np.nan
    rec["perm_neglog10p_max"] = np.nan

    if rec["perm_sig_path"]:
        simg = nb.load(rec["perm_sig_path"])
        s = data_f32(simg)
        s_in = (s > 0.5) & inmask
        ns = int(np.sum(s_in))
        rec["perm_sig_vox"] = ns
        rec["perm_sig_ml"] = mm3_to_ml(ns * vv)

    if rec["perm_neglogp_path"]:
        nimg = nb.load(rec["perm_neglogp_path"])
        nlp = data_f32(nimg)
        nlp_in = inmask & np.isfinite(nlp)
        if np.any(nlp_in):
            rec["perm_neglog10p_max"] = float(np.nanmax(nlp[nlp_in]))

    # Cluster tables
    rec.update(dict(
        fdr_n_clusters=np.nan,
        fdr_max_cluster_mm3=np.nan,
        fdr_max_peak_stat=np.nan,
        unc_n_clusters=np.nan,
        unc_max_cluster_mm3=np.nan,
        unc_max_peak_stat=np.nan,
    ))
    if rec["clus_fdr_path"]:
        c = summarize_cluster_csv(rec["clus_fdr_path"])
        rec["fdr_n_clusters"] = c.get("n_clusters", np.nan)
        rec["fdr_max_cluster_mm3"] = c.get("max_cluster_size_mm3", np.nan)
        rec["fdr_max_peak_stat"] = c.get("max_peak_stat", np.nan)
    if rec["clus_unc_path"]:
        c = summarize_cluster_csv(rec["clus_unc_path"])
        rec["unc_n_clusters"] = c.get("n_clusters", np.nan)
        rec["unc_max_cluster_mm3"] = c.get("max_cluster_size_mm3", np.nan)
        rec["unc_max_peak_stat"] = c.get("max_peak_stat", np.nan)

    rec["status"] = "ok"
    return rec


def compute_strength_score(df):
    def nz(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        1e6 * nz(df["perm_neglog10p_max"]) +
        1e3 * nz(df["perm_sig_vox"]) +
        1e2 * nz(df["fdr_sig_vox"]) +
        1e1 * nz(df["neglog10p_bonf_max"]) +
        1.0 * nz(df["z_max_abs"])
    )


# -------------------------
# CLI + main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # Keep base as common parent for relpaths + summary search stop + default outdir
    ap.add_argument(
        "--base",
        default="/lustre/scratch/data/.../second_level",
        help="Common parent directory (used for relpaths and default outdir).",
    )

    # NEW: restrict scan to these roots
    ap.add_argument(
        "--scan-roots",
        nargs="+",
        default=[
            "/lustre/scratch/data/.../second_level/GLMsingle_new",
            "/lustre/scratch/data/.../second_level/nilearn_new_smoothed",
        ],
        help="Only scan these directories recursively for group_zmap_<tag>.nii.gz.",
    )

    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to write outputs. Default: <base>/_summary",
    )
    ap.add_argument(
        "--out-prefix",
        default="secondlevel_compare",
        help="Prefix for output CSVs/TSV.",
    )
    ap.add_argument(
        "--filter-regex",
        default=None,
        help="Optional regex; only keep analyses whose tag matches it.",
    )
    ap.add_argument(
        "--min-subjects",
        type=int,
        default=0,
        help="Optional: drop analyses where kept_subjects.csv has fewer than this many rows (if present).",
    )
    ap.add_argument(
        "--topn",
        type=int,
        default=50,
        help="How many top-ranked rows to write into the TSV table.",
    )

    # cov20 volume heuristic
    ap.add_argument(
        "--cov20-mask-vol-ml",
        type=float,
        default=1312.724,
        help="Target mask_vol_ml (mL) for cov20 masked runs.",
    )
    ap.add_argument(
        "--cov20-mask-vol-tol-ml",
        type=float,
        default=2.0,
        help="Tolerance (mL) around cov20 mask volume.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        raise SystemExit(f"Base directory not found: {base}")

    scan_roots = [os.path.abspath(p) for p in args.scan_roots]
    for r in scan_roots:
        if not os.path.isdir(r):
            raise SystemExit(f"Scan root not found: {r}")
        # sanity: require within base (prevents accidentally scanning elsewhere)
        if os.path.commonpath([base, r]) != base:
            raise SystemExit(f"Scan root is not under --base:\n  base={base}\n  root={r}")

    outdir = args.outdir or os.path.join(base, "_summary")
    ensure_dir(outdir)

    print("[INFO] scanning only these roots:")
    for r in scan_roots:
        print("  -", r)

    # Discover all group_zmap_<tag>.nii.gz (excluding FDR images) ONLY within scan_roots
    zpaths = []
    for scan_root in scan_roots:
        for root, _, files in os.walk(scan_root):
            for fn in files:
                if fn.startswith("group_zmap_") and fn.endswith(".nii.gz") and "FDR0.05" not in fn:
                    zpaths.append(os.path.join(root, fn))

    rx = re.compile(r"^group_zmap_(.+)\.nii\.gz$")
    analyses = []
    for zp in sorted(set(zpaths)):
        m = rx.match(os.path.basename(zp))
        if not m:
            continue
        tag = m.group(1)
        if args.filter_regex and not re.search(args.filter_regex, tag):
            continue
        analyses.append((os.path.dirname(zp), tag))

    if not analyses:
        raise SystemExit("No analyses found in scan roots (no group_zmap_<tag>.nii.gz discovered).")

    rows = []
    for analysis_dir, tag in analyses:
        rel_root = os.path.relpath(analysis_dir, base)
        rec = compute_metrics_for_analysis(analysis_dir, tag, rel_root)

        # n subjects (if kept_subjects.csv present)
        kept_csv = os.path.join(analysis_dir, "kept_subjects.csv")
        rec["n_subjects"] = np.nan
        if os.path.isfile(kept_csv):
            try:
                kdf = pd.read_csv(kept_csv)
                rec["n_subjects"] = int(kdf.shape[0])
            except Exception:
                pass

        # summary.txt search upwards (stop at --base)
        s_path = find_summary_txt(analysis_dir, base)
        rec["summary_txt"] = s_path or ""
        rec["mask_arg"] = read_mask_arg_from_summary(s_path)

        rows.append(rec)

    df = pd.DataFrame(rows)

    # Infer GLMsingle input smoothing from kept_subjects.csv paths
    df["glmsingle_input"] = [
        infer_glmsingle_input_from_kept_subjects(adir, pip)
        for adir, pip in zip(df["analysis_dir"].astype(str), df["pipeline"].astype(str))
    ]

    # Optional filter by n_subjects
    if args.min_subjects > 0 and "n_subjects" in df.columns:
        df = df[(df["n_subjects"].isna()) | (df["n_subjects"] >= args.min_subjects)].copy()

    # KEEP ONLY: cov20 mask OR no-mask (by arg OR by path)
    target = float(args.cov20_mask_vol_ml)
    tol = float(args.cov20_mask_vol_tol_ml)

    mask_vol = pd.to_numeric(df.get("mask_vol_ml", np.nan), errors="coerce").values
    cov20_by_vol = np.isfinite(mask_vol) & (np.abs(mask_vol - target) <= tol)

    mask_arg = df.get("mask_arg", "unknown").astype(str)
    cov20_by_text = mask_arg.str.lower().str.contains("cov20", na=False).values

    no_mask_by_arg = mask_arg.apply(is_no_mask_by_arg).values
    no_mask_by_path = (
        df["analysis_dir"].astype(str).apply(is_no_mask_by_path).values
        | df["rel_path"].astype(str).apply(is_no_mask_by_path).values
    )

    keep = cov20_by_vol | cov20_by_text | no_mask_by_arg | no_mask_by_path
    before = df.shape[0]
    df = df.loc[keep].copy()
    after = df.shape[0]
    if after == 0:
        raise SystemExit(
            f"[ERROR] After keeping cov20 (vol≈{target}±{tol} mL OR mask_arg contains 'cov20') "
            f"OR no-mask (Mask: None OR path contains no-mask/no_mask), 0 analyses remain."
        )
    print(f"[INFO] mask filter: kept {after}/{before} (cov20 OR no-mask).")

    # EXCLUDE fwhm == 8.0
    if "fwhm" in df.columns:
        before = df.shape[0]
        fwhm_vals = pd.to_numeric(df["fwhm"], errors="coerce").values
        drop = np.isfinite(fwhm_vals) & np.isclose(fwhm_vals, 8.0, atol=1e-6)
        df = df.loc[~drop].copy()
        after = df.shape[0]
        print(f"[INFO] fwhm filter: dropped {before - after} rows with fwhm==8.0; kept {after}.")
        if after == 0:
            raise SystemExit("[ERROR] After excluding fwhm==8.0, 0 analyses remain.")
    else:
        print("[WARN] fwhm column missing; cannot exclude fwhm==8.0.")

    # Rank
    df["strength_score"] = compute_strength_score(df)
    df_rank = df.sort_values(
        ["strength_score", "perm_neglog10p_max", "perm_sig_vox", "fdr_sig_vox", "z_max_abs"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    suffix = "cov20ORnomask_noFWHM8"
    summary_csv = os.path.join(outdir, f"{args.out_prefix}_ALL_{suffix}.csv")
    ranked_csv = os.path.join(outdir, f"{args.out_prefix}_RANKED_{suffix}.csv")
    top_tsv = os.path.join(outdir, f"{args.out_prefix}_TOP{args.topn}_{suffix}.tsv")

    df.to_csv(summary_csv, index=False)
    df_rank.to_csv(ranked_csv, index=False)

    table_cols = [
        "pipeline",
        "approach",
        "glmsingle_input",
        "fwhm",
        "mask_vol_ml",
        "mask_arg",
        "z_max_abs",
        "fdr_sig_vox",
        "strength_score",
        "rel_path",
        "analysis_dir",
    ]
    table_cols = [c for c in table_cols if c in df_rank.columns]
    df_rank.loc[:, table_cols].head(args.topn).to_csv(top_tsv, sep="\t", index=False)

    print("\n=== TOP 10 (ranked; cov20 OR no-mask; fwhm!=8.0) ===")
    print(df_rank[table_cols].head(10).to_string(index=False))
    print(f"\nWrote:\n  {summary_csv}\n  {ranked_csv}\n  {top_tsv}")


if __name__ == "__main__":
    main()
