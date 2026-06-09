#!/usr/bin/env python3
"""
group_tfce_two_group.py — Two-sample TFCE (FSL randomise) for searchlight RSA beta maps.

AorB-style layout:
  BASE/searchlight_RSA/<datatype>/sub-*/<files>

Example file:
  sub-102_unsmoothed_branchB_up3x_adaptiveK_k60_crossnobis_beta-model6_bio_vs_scrambled_rankortho.nii.gz

Robustness:
- For each subject map:
    (1) NaNs -> 0
    (2) apply mask (outside-mask -> 0)
    (3) optional smoothing
    (4) re-apply mask after smoothing (outside-mask -> 0)
"""

import os, sys, argparse, json, subprocess, shutil, csv
from glob import glob
import numpy as np
import nibabel as nb

# nilearn smoothing
try:
    from nilearn.image import smooth_img
except Exception:
    smooth_img = None

# nilearn plotting (optional; script still runs if unavailable)
try:
    import matplotlib
    matplotlib.use("Agg")
    from nilearn import plotting, datasets
except Exception:
    plotting = None
    datasets = None

# ---------- Marvin paths ----------
BASE = "/lustre/scratch/data/..."
RSA_ROOT = os.path.join(BASE, "searchlight_RSA")

# ---------- helpers ----------
def run(cmd, check=True):
    print("[cmd]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def merge_4d(in_files, out_path):
    run(["fslmerge", "-t", out_path, *in_files])
    run(["fslmaths", out_path, "-nan", out_path])  # randomise dislikes NaNs

def mask_voxels_and_volume(mask_img_path):
    img = nb.load(mask_img_path)
    data = img.get_fdata()
    n_vox = int(np.count_nonzero(data > 0.5))
    vox_mm3 = abs(np.linalg.det(img.affine[:3, :3]))  # mm^3 per voxel
    vol_ml = (n_vox * vox_mm3) / 1000.0
    return n_vox, float(vol_ml), float(vox_mm3)

def write_fsl_mat_con(design_mat_path, design_con_path, group_assignments):
    """
    Two-column design with group indicators, and two contrasts:
      tstat1: A > B  => [1, -1]
      tstat2: B > A  => [-1, 1]
    """
    n = len(group_assignments)
    rows = []
    for g in group_assignments:
        if g == "A":
            rows.append([1, 0])
        elif g == "B":
            rows.append([0, 1])
        else:
            raise ValueError("Group assignments must be 'A' or 'B'.")

    with open(design_mat_path, "w") as f:
        f.write("/NumWaves\t2\n")
        f.write(f"/NumPoints\t{n}\n")
        f.write("/PPheights\t1\t1\n")
        f.write("/Matrix\n")
        for r in rows:
            f.write(f"{r[0]:.6f}\t{r[1]:.6f}\n")

    with open(design_con_path, "w") as f:
        f.write("/NumWaves\t2\n")
        f.write("/NumContrasts\t2\n")
        f.write("/PPheights\t1\t1\n")
        f.write("/Matrix\n")
        f.write("1\t-1\n")
        f.write("-1\t1\n")

def parse_groups_csv(path, subject_col, group_col, groupA_label, groupB_label):
    """
    Returns mapping: sub_id ('sub-XXX') -> 'A' or 'B'.
    Accepts subject IDs with or without 'sub-' prefix in the CSV.
    """
    to_group = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = str(row[subject_col]).strip()
            sid = sid_raw if sid_raw.startswith("sub-") else f"sub-{sid_raw}"
            grp = str(row[group_col]).strip().lower()
            if grp == groupA_label.lower():
                to_group[sid] = "A"
            elif grp == groupB_label.lower():
                to_group[sid] = "B"
            else:
                continue
    return to_group

def load_mask_bool(mask_path):
    mimg = nb.load(mask_path)
    mdat = mimg.get_fdata(dtype=np.float32)
    m = np.isfinite(mdat) & (mdat > 0.5)
    return mimg, m

def nan0_and_mask_to_zero(in_path, mask_bool, out_path):
    img = nb.load(in_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI for {in_path}, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(data, dtype=np.float32)
    out[mask_bool] = data[mask_bool]
    nb.save(nb.Nifti1Image(out, img.affine, img.header), out_path)
    return out_path

def smooth_and_remask(in_path, fwhm, mask_bool, out_path):
    if smooth_img is None:
        raise RuntimeError("nilearn is not available but --smooth-fwhm > 0 was requested.")
    simg = smooth_img(in_path, fwhm=float(fwhm))
    sdat = simg.get_fdata(dtype=np.float32)
    sdat = np.nan_to_num(sdat, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(sdat, dtype=np.float32)
    out[mask_bool] = sdat[mask_bool]
    nb.save(nb.Nifti1Image(out, simg.affine, simg.header), out_path)
    return out_path

def _pick_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def _save_like(ref_img, data, out_path, dtype=np.float32):
    out = nb.Nifti1Image(data.astype(dtype, copy=False), ref_img.affine, ref_img.header)
    out.set_data_dtype(dtype)
    nb.save(out, out_path)
    return out_path

def postprocess_and_qc(out_root, two_dir, mask_path, groupA_label, groupB_label, corrp_thr=0.95):
    """
    Creates:
      - significant masks (corrp >= corrp_thr) for tstat1 and tstat2
      - thresholded t-stat maps
      - summary TSV + JSON
      - PNG visualizations if nilearn.plotting available
    """
    prefix = os.path.join(two_dir, "two_sample")

    corrp1 = _pick_existing([
        prefix + "_tfce_corrp_tstat1.nii.gz",
        prefix + "_tfce_corrp_tstat1.nii",
    ])
    corrp2 = _pick_existing([
        prefix + "_tfce_corrp_tstat2.nii.gz",
        prefix + "_tfce_corrp_tstat2.nii",
    ])

    # Prefer TFCE tstat if present; else fall back to plain tstat
    t1 = _pick_existing([
        prefix + "_tfce_tstat1.nii.gz",
        prefix + "_tstat1.nii.gz",
        prefix + "_tfce_tstat1.nii",
        prefix + "_tstat1.nii",
    ])
    t2 = _pick_existing([
        prefix + "_tfce_tstat2.nii.gz",
        prefix + "_tstat2.nii.gz",
        prefix + "_tfce_tstat2.nii",
        prefix + "_tstat2.nii",
    ])

    if not corrp1 or not corrp2:
        print("[warn] Could not find TFCE corrp maps; skipping QC/summary.")
        return

    mask_img = nb.load(mask_path)
    mask_bool = mask_img.get_fdata(dtype=np.float32) > 0.5
    _, _, vox_mm3 = mask_voxels_and_volume(mask_path)

    qc_dir = ensure_dir(os.path.join(out_root, "qc"))
    summary_rows = []
    summary = {
        "corrp_threshold": float(corrp_thr),
        "contrasts": {},
        "qc_dir": qc_dir,
    }

    def process_one(label, corrp_path, t_path, pretty):
        cimg = nb.load(corrp_path)
        cdat = np.nan_to_num(cimg.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        timg = nb.load(t_path) if (t_path and os.path.isfile(t_path)) else None
        tdat = np.nan_to_num(timg.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) if timg is not None else None

        sig = (cdat >= corrp_thr) & mask_bool & np.isfinite(cdat)
        n_sig = int(sig.sum())
        vol_ml = (n_sig * vox_mm3) / 1000.0

        sig_mask_path = os.path.join(qc_dir, f"sigmask_{label}_corrp_ge{int(corrp_thr*100):02d}.nii.gz")
        _save_like(mask_img, sig.astype(np.uint8), sig_mask_path, dtype=np.uint8)

        # thresholded t-map within sig
        thr_t_path = ""
        peak = {"peak_abs_t": None, "peak_t": None, "peak_corrp": None, "peak_xyz_mm": None}
        if (tdat is not None) and (timg is not None):
            thr = np.zeros_like(tdat, dtype=np.float32)
            thr[sig] = tdat[sig]
            thr_t_path = os.path.join(qc_dir, f"sig_tstat_{label}.nii.gz")
            _save_like(timg, thr, thr_t_path, dtype=np.float32)

            if n_sig > 0:
                idx = np.unravel_index(int(np.argmax(np.abs(thr))), thr.shape)
                peak_t = float(tdat[idx])
                peak_abs_t = float(abs(peak_t))
                peak_corrp = float(cdat[idx])
                xyz = nb.affines.apply_affine(timg.affine, idx)
                peak = {
                    "peak_abs_t": peak_abs_t,
                    "peak_t": peak_t,
                    "peak_corrp": peak_corrp,
                    "peak_xyz_mm": [float(x) for x in xyz],
                }

        # PNGs (optional)
        png_glass = ""
        png_stat = ""
        if plotting is not None and datasets is not None:
            try:
                tmpl = datasets.load_mni152_template()

                png_glass = os.path.join(qc_dir, f"glass_{label}.png")
                plotting.plot_glass_brain(
                    sig_mask_path,
                    title=f"{pretty} (corrp ≥ {corrp_thr:.2f})",
                    output_file=png_glass
                )

                if thr_t_path:
                    png_stat = os.path.join(qc_dir, f"stat_{label}.png")
                    plotting.plot_stat_map(
                        thr_t_path,
                        bg_img=tmpl,
                        threshold=0.0,
                        display_mode="ortho",
                        cut_coords=(0, -20, 20),
                        title=f"{pretty} (thresholded by corrp ≥ {corrp_thr:.2f})",
                        output_file=png_stat
                    )
            except Exception as e:
                print(f"[warn] QC plotting failed for {label}: {e}")

        row = {
            "contrast": label,
            "pretty": pretty,
            "corrp_map": corrp_path,
            "tstat_map": t_path or "",
            "sigmask_path": sig_mask_path,
            "sig_tstat_path": thr_t_path,
            "n_sig_vox": n_sig,
            "sig_vol_ml": float(vol_ml),
            "png_glass": png_glass,
            "png_stat": png_stat,
            "peak_abs_t": peak["peak_abs_t"],
            "peak_t": peak["peak_t"],
            "peak_corrp": peak["peak_corrp"],
            "peak_x_mm": (peak["peak_xyz_mm"][0] if peak["peak_xyz_mm"] else None),
            "peak_y_mm": (peak["peak_xyz_mm"][1] if peak["peak_xyz_mm"] else None),
            "peak_z_mm": (peak["peak_xyz_mm"][2] if peak["peak_xyz_mm"] else None),
        }

        summary["contrasts"][label] = row
        summary_rows.append(row)

    process_one("tstat1", corrp1, t1, f"{groupA_label} > {groupB_label}")
    process_one("tstat2", corrp2, t2, f"{groupB_label} > {groupA_label}")

    # write summary files
    sum_json = os.path.join(out_root, "RESULTS_SUMMARY.json")
    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    sum_tsv = os.path.join(out_root, "RESULTS_SUMMARY.tsv")
    cols = [
        "contrast","pretty","n_sig_vox","sig_vol_ml",
        "corrp_map","tstat_map","sigmask_path","sig_tstat_path",
        "png_glass","png_stat",
        "peak_abs_t","peak_t","peak_corrp","peak_x_mm","peak_y_mm","peak_z_mm",
    ]
    with open(sum_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    print(f"[ok] Wrote results summary:\n  {sum_tsv}\n  {sum_json}")
    if any(r.get("png_glass") for r in summary_rows):
        print(f"[ok] Wrote QC PNGs under:\n  {qc_dir}")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()

    # NOTE: allow arbitrary datatype folder (e.g., unsmoothed_cov40)
    ap.add_argument("--datatype", default="unsmoothed",
                    help="Folder under searchlight_RSA/ to scan (e.g., unsmoothed, smoothed, unsmoothed_cov40, ...).")

    ap.add_argument("--predictor", required=True, help="e.g., model6_bio_vs_scrambled_rankortho")
    ap.add_argument("--metric", default="crossnobis", choices=["crossnobis", "xcorrdiff", "pearson"])
    ap.add_argument("--smooth-fwhm", type=float, default=0.0,
                    help="Per-subject smoothing FWHM (mm). 0 = none. (NaNs cleaned + mask applied before smoothing.)")
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--tfce-name", required=True, help="Creates searchlight_RSA/<datatype>/group_tfce/<tfce-name>/")
    ap.add_argument("--out-root", default=None)

    # File matching (AorB layout)
    ap.add_argument(
        "--fname-template",
        default="{sub}_{datatype}_branch*_{metric}*_beta-{predictor}.nii.gz",
        help=("Template relative to subject dir under RSA_ROOT/<datatype>/sub-XXX/. "
              "Fields: {sub},{datatype},{metric},{predictor}. Wildcards allowed.")
    )

    # Mask
    ap.add_argument("--mask", default="", help="Analysis mask; will be regridded + binarized unless --assume-mask-aligned.")
    ap.add_argument("--assume-mask-aligned", action="store_true", help="Skip mask regridding; use as-is.")

    # Groups CSV
    ap.add_argument("--groups-csv", required=True, help="CSV with subject→group labels (unblinding).")
    ap.add_argument("--subject-col", default="subject", help="Column with subject IDs (e.g., 'sub-101' or '101').")
    ap.add_argument("--group-col", default="group", help="Column with group labels (e.g., 'lorazepam'/'placebo').")
    ap.add_argument("--groupA-label", default="lorazepam", help="Label treated as Group A (tstat1 = A>B).")
    ap.add_argument("--groupB-label", default="placebo", help="Label treated as Group B (tstat2 = B>A).")

    ap.add_argument("--parallel", action="store_true",
                    help="Use randomise_parallel instead of randomise (falls back to randomise if unavailable).")

    return ap.parse_args()

def main():
    args = parse_args()

    out_root = args.out_root or os.path.join(RSA_ROOT, args.datatype, "group_tfce", args.tfce_name)
    two_dir   = ensure_dir(os.path.join(out_root, "two_sample"))
    stage_dir = ensure_dir(os.path.join(out_root, "inputs"))
    mask_path = os.path.join(out_root, "mask_aligned.nii.gz")
    design_mat = os.path.join(out_root, "design.mat")
    design_con = os.path.join(out_root, "design.con")

    # AorB layout: RSA_ROOT/<datatype>/sub-*
    datadir = os.path.join(RSA_ROOT, args.datatype)
    if not os.path.isdir(datadir):
        sys.exit(f"[error] datatype dir not found: {datadir}")

    all_subdirs = sorted([d for d in os.listdir(datadir)
                          if d.startswith("sub-") and os.path.isdir(os.path.join(datadir, d))])

    # Parse group labels
    sub_to_group = parse_groups_csv(args.groups_csv, args.subject_col, args.group_col,
                                    args.groupA_label, args.groupB_label)

    # IMPORTANT: filenames may still say "_unsmoothed_" even if folder is "unsmoothed_cov40"
    if "unsmoothed" in args.datatype.lower():
        fname_dt = "unsmoothed"
    elif "smoothed" in args.datatype.lower():
        fname_dt = "smoothed"
    else:
        fname_dt = args.datatype

    collected, groups_used, missing = [], [], []
    picked_inputs = []

    for sub in all_subdirs:
        if sub not in sub_to_group:
            continue

        sub_dir = os.path.join(datadir, sub)
        patt = args.fname_template.format(
            sub=sub, datatype=fname_dt, metric=args.metric, predictor=args.predictor
        )
        hits = sorted(glob(os.path.join(sub_dir, patt)))
        if not hits:
            missing.append((sub, patt))
            continue

        in_path = hits[-1]
        picked_inputs.append(in_path)
        collected.append(in_path)
        groups_used.append(sub_to_group[sub])

    if missing:
        print("[warn] Subjects with no matching map (skipped):")
        for sub, patt in missing:
            print(f"   {sub}: {patt}")

    n_total = len(collected)
    nA = sum(1 for g in groups_used if g == "A")
    nB = sum(1 for g in groups_used if g == "B")
    if n_total < 8 or nA < 4 or nB < 4:
        sys.exit(f"[error] Not enough subjects per group (A={nA}, B={nB}, total={n_total}); need >=4 per group and >=8 total.")

    # ---- Mask: align/bin ----
    if not args.mask or (not os.path.isfile(args.mask)):
        sys.exit("[error] You must supply --mask (your 100% intersection TFCE-safe mask).")

    if args.assume_mask_aligned:
        print("[info] Using provided mask as-is (assumed aligned).")
        shutil.copy2(args.mask, mask_path)
    else:
        print("[info] Regridding provided mask to data grid (flirt -applyxfm -usesqform -interp nearestneighbour)…")
        ref = collected[0]
        run(["flirt", "-in", args.mask, "-ref", ref,
             "-applyxfm", "-usesqform", "-interp", "nearestneighbour",
             "-out", mask_path])

    run(["fslmaths", mask_path, "-thr", "0.5", "-bin", mask_path])

    _, mask_bool = load_mask_bool(mask_path)

    mask_voxels, mask_volume_ml, _ = mask_voxels_and_volume(mask_path)
    print(f"[info] Mask size after align/bin: voxels={mask_voxels}  volume={mask_volume_ml:.3f} mL")
    if mask_voxels < 1000:
        print("[warn] Mask is very small; double-check you pointed to the intended TFCE mask.")

    # ---- Stage per-subject inputs ----
    staged_files = []
    for in_path in collected:
        bn = os.path.basename(in_path)
        base_noext = bn[:-7] if bn.endswith(".nii.gz") else os.path.splitext(bn)[0]

        pre = os.path.join(stage_dir, f"{base_noext}_nan0_premask.nii.gz")
        nan0_and_mask_to_zero(in_path, mask_bool, pre)

        if args.smooth_fwhm and args.smooth_fwhm > 0:
            out = os.path.join(stage_dir, f"{base_noext}_nan0_premask_s{int(args.smooth_fwhm)}mm_remask.nii.gz")
            smooth_and_remask(pre, args.smooth_fwhm, mask_bool, out)
            staged_files.append(out)
        else:
            staged_files.append(pre)

    # ---- Merge staged -> 4D ----
    all_4d = os.path.join(out_root, "all_subjects_4D.nii.gz")
    merge_4d(staged_files, all_4d)

    # ---- Design files ----
    write_fsl_mat_con(design_mat, design_con, groups_used)

    # ---- Choose randomise executable ----
    exe = "randomise_parallel" if args.parallel else "randomise"
    if shutil.which(exe) is None:
        if args.parallel:
            print("[warn] randomise_parallel not found on PATH — falling back to randomise.")
        exe = "randomise"

    # ---- Run randomise ----
    rand_cmd = [
        exe,
        "-i", all_4d,
        "-o", os.path.join(two_dir, "two_sample"),
        "-m", mask_path,
        "-d", design_mat,
        "-t", design_con,
        "-n", str(args.permutations),
        "-T",
        "--uncorrp",
        "-R",
    ]
    run(rand_cmd)

    # ---- Provenance ----
    subjects_in_order = [os.path.basename(p).split("_")[0] for p in collected]
    groups_for_subjects = []
    for s, g in zip(subjects_in_order, groups_used):
        label = args.groupA_label if g == "A" else args.groupB_label
        groups_for_subjects.append([s, label])

    meta = {
        "mode": "two-sample",
        "datatype_folder": args.datatype,
        "datatype_token_in_filenames": fname_dt,
        "predictor": args.predictor,
        "metric": args.metric,
        "smooth_fwhm_mm": float(args.smooth_fwhm),
        "permutations": int(args.permutations),
        "tfce_name": args.tfce_name,

        "n_subjects_total": int(n_total),
        "n_groupA": int(nA),
        "n_groupB": int(nB),
        "groupA_label": args.groupA_label,
        "groupB_label": args.groupB_label,
        "subjects_and_groups": groups_for_subjects,

        "fname_template": args.fname_template,
        "picked_inputs": picked_inputs,
        "staged_inputs": staged_files,

        "merged_4D": all_4d,
        "mask_path": mask_path,
        "mask_source": args.mask,
        "mask_voxels": int(mask_voxels),
        "mask_volume_ml": float(mask_volume_ml),

        "design_mat": design_mat,
        "design_con": design_con,

        "randomise": exe,
        "randomise_cmd": " ".join(rand_cmd),

        "notes": (
            "Per-subject staging: NaNs->0, premask->0, "
            f"{'smooth->remask' if (args.smooth_fwhm and args.smooth_fwhm>0) else 'no_smoothing'}, "
            "then merge and randomise TFCE. "
            "tstat1 = A>B, tstat2 = B>A; corrp>=0.95 corresponds to p<0.05 FWER."
        ),
    }
    with open(os.path.join(out_root, "provenance.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- NEW: summary + visualization ----
    postprocess_and_qc(
        out_root=out_root,
        two_dir=two_dir,
        mask_path=mask_path,
        groupA_label=args.groupA_label,
        groupB_label=args.groupB_label,
        corrp_thr=0.95,
    )

    print("\n[done] Two-sample TFCE outputs (FWER-corrected corrp):")
    print("  ", os.path.join(two_dir, "two_sample_tfce_corrp_tstat1.nii.gz"),
          f"  [{args.groupA_label} > {args.groupB_label}]")
    print("  ", os.path.join(two_dir, "two_sample_tfce_corrp_tstat2.nii.gz"),
          f"  [{args.groupB_label} > {args.groupA_label}]")
    print("[done] Mask size:", f"voxels={mask_voxels}", f"volume={mask_volume_ml:.3f} mL")
    print("[done] Randomise:", " ".join(rand_cmd))
    print("[done] Log:", os.path.join(out_root, "provenance.json"))
    print("[done] Summary:", os.path.join(out_root, "RESULTS_SUMMARY.tsv"))

if __name__ == "__main__":
    main()
