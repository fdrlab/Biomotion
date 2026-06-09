#!/usr/bin/env python3
"""
second_level_compare_new.py — Run group (second-level) GLM for multiple first-level pipelines.
- Auto-mask by default (or honor --mask-img, with a "FULL" option to include all voxels).
- Explicitly apply the fitted-model mask to Z and FDR images before saving (prevents out-of-brain signal).
- Save mask_used.nii.gz per analysis for QA/overlay in Mango.
"""

import os, sys, argparse, json, gc
import numpy as np
import pandas as pd
import nibabel as nb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table

try:
    from nilearn.glm.second_level import non_parametric_inference
    _HAS_PERM = True
except Exception:
    _HAS_PERM = False

# -------------------- Visualization constants --------------------
ZPNG_THR = 1.96  # display threshold for the Z PNGs
MOSAIC_NCUTS = (10, 10, 10)

def save_display(disp, path, dpi=150):
    disp.savefig(path, dpi=dpi, bbox_inches="tight"); disp.close()
    plt.close('all'); gc.collect()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def plot_mosaic_safe(img, title, out_png, threshold=None, colorbar=True):
    try:
        disp = plotting.plot_stat_map(
            img, display_mode="mosaic",
            cut_coords=MOSAIC_NCUTS, colorbar=colorbar,
            title=title, threshold=threshold
        )
    except Exception as e:
        print(f"[WARN] mosaic tuple cut_coords failed ({e}); falling back to cut_coords=10.", file=sys.stderr)
        disp = plotting.plot_stat_map(
            img, display_mode="mosaic",
            cut_coords=10, colorbar=colorbar,
            title=title, threshold=threshold
        )
    save_display(disp, out_png)

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Second-level comparison across multiple pipelines.")
    ap.add_argument(
        "--mask-img", default=None,
        help=("Optional group/analysis mask NIfTI. If omitted, nilearn auto-masks. "
              'If set to "FULL", a full-volume ones mask is created from the first contrast '
              "to include ALL voxels.")
    )
    ap.add_argument("--outdir", required=True, help="Output directory (group-level results live under subfolders).")

    # subjects
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="List of subjects like sub-105 sub-106 ...")
    ap.add_argument("--subject", action="append", default=None,
                    help="Repeatable single-subject flag; can be combined with --subjects.")
    ap.add_argument("--placebo-csv", default=None,
                    help="Optional fallback CSV (columns: subject,group) to pull subjects from if none provided.")

    # first-level sources (repeatable)
    ap.add_argument("--nilearn-dir", action="append", default=[],
                    help="Repeatable: a Nilearn first-level output folder.")
    ap.add_argument("--glms-dir", action="append", default=[],
                    help="Repeatable: a GLMsingle OLS first-level output folder.")

    # naming pieces for file discovery
    ap.add_argument("--runs-tag", default="01-02", help="e.g., 01-02 (used in Nilearn filenames).")
    ap.add_argument("--datatype", default="unsmoothed", choices=["unsmoothed","smoothed"],
                    help="Used by both pipelines’ file names.")
    ap.add_argument("--branch", default="A_up3x", choices=["A_up3x","B_up3x"],
                    help="GLMsingle branch used in file names.")

    # stats / viz
    ap.add_argument("--one-sided", action="store_true", help="Use one-sided inference (default two-sided).")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR alpha.")
    ap.add_argument("--cluster-k", type=int, default=10, help="Min cluster size with voxelwise FDR; 0 disables extent.")
    ap.add_argument("--unc-p", type=float, default=0.001, help="Uncorrected p for glass-brain.")
    ap.add_argument("--unc-z", type=float, default=3.1, help="Uncorrected Z for descriptive cluster table.")
    ap.add_argument("--unc-k", type=int, default=20, help="Min cluster size for descriptive uncorrected table.")
    ap.add_argument("--secondlevel-fwhm", type=float, default=8.0, help="Second-level smoothing FWHM (mm).")

    # permutation
    ap.add_argument("--perm-n", type=int, default=10000, help="If >0 and available, run permutation FWER.")
    ap.add_argument("--perm-mode", choices=["tfce","maxt"], default="tfce",
                    help="Permutation method: TFCE (primary) or maxT.")

    # extra PNGs
    ap.add_argument("--save-parametric-pmaps-pngs", action="store_true",
                    help="Also save PNGs for parametric p-map and Bonferroni -log10 p.")
    ap.add_argument("--save-permutation-pngs", action="store_true",
                    help="Also save PNGs for permutation outputs (neglog10 and binary p<0.05).")

    # provenance
    ap.add_argument("--design-order-json", default=None,
                    help="Optional: path to design_order_map.json for provenance logging only.")

    return ap.parse_args()

# -------------------- Utils --------------------
CANON_LABELS = [
    'happy_females','happy_males','angry_females',
    'angry_males','neutral_females','scrambled'
]

def normalize_sub(s):
    s = str(s).strip().lower()
    return s if s.startswith("sub-") else f"sub-{s}"

def read_placebo_subjects(csv_path):
    df = pd.read_csv(csv_path)
    if not {"subject","group"}.issubset(df.columns):
        raise SystemExit(f"[ERROR] CSV must have columns exactly: subject,group (got {df.columns.tolist()})")
    mask = df["group"].astype(str).str.strip().str.lower().eq("placebo")
    subs = df.loc[mask, "subject"].astype(str).str.strip().str.lower().unique().tolist()
    subs = [normalize_sub(s) for s in subs]
    return sorted(subs)

def nilearn_effect_path(base_dir, sub, runs_tag, datatype):
    return os.path.join(
        base_dir,
        f"{sub}_runs-{runs_tag}_{datatype}_bio_minus_scrambled_trim5tr_effect.nii.gz"
    )

def glmsingle_effect_path(base_dir, sub, datatype, branch):
    return os.path.join(
        base_dir,
        f"{sub}_{datatype}_{branch}_firstlevelEff_bio_minus_scrambled.nii.gz"
    )

def collect_effects(base_dir, subjects, builder, tag_name):
    found, missing = [], []
    for sub in subjects:
        p = builder(base_dir, sub)
        if os.path.isfile(p):
            found.append(p)
        else:
            missing.append({"subject": sub, "expected": p})
    print(f"[INFO] {tag_name}: found {len(found)} effect maps; missing {len(missing)}.")
    return found, missing

def make_full_mask_like(ref_img_path, out_path):
    """Create a full-volume (all-ones) mask with the same shape/affine/header as ref_img_path."""
    ref = nb.load(ref_img_path)
    data = np.ones(ref.shape, dtype=np.uint8)
    hdr = ref.header.copy(); hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data, ref.affine, hdr).to_filename(out_path)
    return out_path

def _get_mask_bool_like(mask_img, like_img):
    """
    Return a boolean mask array aligned to like_img.
    Resamples mask if shapes don't match.
    """
    from nilearn.image import resample_to_img
    mask_nii = mask_img if hasattr(mask_img, "get_fdata") else nb.load(mask_img)
    if mask_nii.shape != like_img.shape:
        mask_nii = resample_to_img(mask_nii, like_img, interpolation="nearest")
    return (mask_nii.get_fdata() > 0), mask_nii

# -------------------- Second-level core --------------------
def run_second_level(contrast_imgs, mask_img, two_sided, alpha, cluster_k, unc_p, unc_z, unc_k,
                     fwhm, outdir, tag, perm_n, perm_mode, save_param_pngs=False, save_perm_pngs=False):
    ensure_dir(outdir)
    # Design (one-sample t-test)
    dm = pd.DataFrame({"intercept": np.ones(len(contrast_imgs))})
    slm = SecondLevelModel(
        mask_img=mask_img,
        smoothing_fwhm=(None if (fwhm is None or fwhm <= 0) else float(fwhm)),
        minimize_memory=True
    ).fit(contrast_imgs, design_matrix=dm)

    # Pull the mask actually used by the model (handles auto-mask or user-supplied)
    auto_mask_img = slm.masker_.mask_img_
    mask_used_nii = auto_mask_img if hasattr(auto_mask_img, "get_fdata") else nb.load(auto_mask_img)

    # Helpers robust to nilearn version diffs
    def _safe_comp(output_type):
        try:
            return slm.compute_contrast("intercept", output_type=output_type, two_sided=two_sided)
        except TypeError:
            return slm.compute_contrast("intercept", output_type=output_type)

    # --- Z map (masked before saving) ---
    zmap_raw = _safe_comp("z_score")
    zdat = np.nan_to_num(zmap_raw.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Align mask to zmap grid if needed, then apply
    mask_bool, mask_aligned = _get_mask_bool_like(mask_used_nii, zmap_raw)
    zdat[~mask_bool] = 0.0
    zmap = nb.Nifti1Image(zdat, zmap_raw.affine, zmap_raw.header)
    zmap_path = os.path.join(outdir, f"group_zmap_{tag}.nii.gz"); zmap.to_filename(zmap_path)

    # Save the mask we actually used (QA in Mango)
    nb.Nifti1Image(mask_bool.astype(np.uint8), zmap.affine).to_filename(
        os.path.join(outdir, "mask_used.nii.gz")
    )

    # --- FDR-thresholded Z (mask applied) ---
    def _safe_thr(zimg):
        try:
            return threshold_stats_img(zimg, alpha=alpha, height_control="fdr",
                                       two_sided=two_sided,
                                       cluster_threshold=(int(cluster_k) if cluster_k and cluster_k>0 else 0))
        except TypeError:
            return threshold_stats_img(zimg, alpha=alpha, height_control="fdr",
                                       cluster_threshold=(int(cluster_k) if cluster_k and cluster_k>0 else 0))

    thr_img_raw, thr = _safe_thr(zmap)  # threshold zmap already masked
    thr_data = thr_img_raw.get_fdata().astype(np.float32)
    thr_data[~mask_bool] = 0.0
    thr_img = nb.Nifti1Image(thr_data, zmap.affine, thr_img_raw.header)

    tag_k = f"_k{cluster_k}" if cluster_k and cluster_k > 0 else ""
    thr_path = os.path.join(outdir, f"group_zmap_FDR0.05{tag_k}_{tag}.nii.gz")
    thr_img.to_filename(thr_path)

    # Cluster tables from the masked zmap
    try:
        if np.isfinite(thr):
            tbl = get_clusters_table(zmap, stat_threshold=float(thr), cluster_threshold=(cluster_k or 0))
            tbl.to_csv(os.path.join(outdir, f"group_clusters_FDR0.05{tag_k}_{tag}.csv"), index=False)
        else:
            open(os.path.join(outdir, f"group_clusters_FDR0.05{tag_k}_{tag}.csv"), "w").close()
    except Exception as e:
        print(f"[WARN] cluster table (FDR) failed: {e}", file=sys.stderr)

    try:
        if unc_z and unc_z > 0:
            tbl_unc = get_clusters_table(zmap, stat_threshold=float(unc_z), cluster_threshold=int(unc_k))
            tbl_unc.to_csv(os.path.join(outdir, f"group_clusters_Zgt{unc_z}_k{unc_k}_unc_{tag}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] uncorrected clusters failed: {e}", file=sys.stderr)

    # Glass brain (show masked zmap)
    try:
        disp = plotting.plot_glass_brain(zmap, threshold=float(ZPNG_THR), display_mode="z", plot_abs=False,
                                         title=f"group {tag} (Z≥{ZPNG_THR:g})")
        save_display(disp, os.path.join(outdir, f"group_glass_Zge{ZPNG_THR:g}_{tag}.png"))
    except Exception as e:
        print(f"[WARN] glass-brain plotting failed: {e}", file=sys.stderr)

    # Parametric p-maps (+ Bonferroni -log10 p); keep outside-mask as 1.0
    try:
        p_map = _safe_comp("p_value")
        pm = p_map.get_fdata()
        pm = np.where(mask_bool, np.clip(pm,0,1), 1.0).astype(np.float32)
        pmap_path = os.path.join(outdir, f"group_pmap_parametric_{tag}.nii.gz")
        nb.Nifti1Image(pm, p_map.affine, p_map.header).to_filename(pmap_path)

        from nilearn.image import math_img
        n_vox = int(np.sum(mask_bool))
        eps = 1e-300
        neglogp_bonf = math_img(f"-np.log10(np.maximum(np.minimum(img*{n_vox}, 1.0), {eps}))",
                                img=nb.Nifti1Image(pm, p_map.affine, p_map.header))
        neglogp_bonf.to_filename(os.path.join(outdir, f"group_neglog10p_bonf_parametric_{tag}.nii.gz"))

        if save_param_pngs:
            from nilearn.image import math_img as _math_img
            neglogp_unc = _math_img("-np.log10(np.maximum(img, 1e-300))",
                                    img=nb.Nifti1Image(pm, p_map.affine, p_map.header))
            plot_mosaic_safe(
                neglogp_unc,
                title=f"Parametric -log10(p) (unc) — {tag}",
                out_png=os.path.join(outdir, f"group_neglog10p_parametric_{tag}.png"),
                threshold=float(-np.log10(max(1e-6, alpha)))
            )
            plot_mosaic_safe(
                neglogp_bonf,
                title=f"Parametric -log10(p_bonf) — {tag}",
                out_png=os.path.join(outdir, f"group_neglog10p_bonf_parametric_{tag}.png"),
                threshold=float(-np.log10(0.05))
            )
    except Exception as e:
        print(f"[WARN] saving parametric p-maps failed: {e}", file=sys.stderr)

    # Permutation FWER (if enabled); mask already enforced via npi `mask=...`
    if perm_n > 0 and _HAS_PERM:
        mode = perm_mode.lower()
        tfce_flag = (mode == "tfce")
        mode_str = "tfce" if tfce_flag else "maxt"
        try:
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
            perm_mask = mask_img if mask_img is not None else mask_used_nii
            kwargs = dict(
                second_level_input=contrast_imgs, design_matrix=dm,
                second_level_contrast="intercept", model_intercept=True,
                n_perm=perm_n, two_sided_test=two_sided, n_jobs=n_jobs,
                mask=perm_mask,
                smoothing_fwhm=(None if (fwhm is None or fwhm <= 0) else float(fwhm))
            )
            if tfce_flag:
                try:
                    neglogp_perm = non_parametric_inference(tfce=True, **kwargs)
                except TypeError:
                    print("[WARN] TFCE not supported by your nilearn; falling back to maxT.", file=sys.stderr)
                    mode_str = "maxt"
                    neglogp_perm = non_parametric_inference(**kwargs)
            else:
                neglogp_perm = non_parametric_inference(**kwargs)

            neglogp_path = os.path.join(outdir, f"group_perm_{mode_str}_neglog10_vfwe_{perm_n}n_{tag}.nii.gz")
            neglogp_perm.to_filename(neglogp_path)

            from nilearn.image import math_img
            p_perm = math_img("10**(-img)", img=neglogp_perm)
            pm = p_perm.get_fdata()
            pm = np.where(mask_bool, np.clip(pm,0,1), 1.0).astype(np.float32)
            pmap_perm_path = os.path.join(outdir, f"group_perm_{mode_str}_vfwe_pmap_{perm_n}n_{tag}.nii.gz")
            nb.Nifti1Image(pm, p_perm.affine, p_perm.header).to_filename(pmap_perm_path)
            sig = (pm < 0.05) & mask_bool
            sig_path = os.path.join(outdir, f"group_perm_{mode_str}_vfwe_p_lt0p05_{perm_n}n_{tag}.nii.gz")
            nb.Nifti1Image(sig.astype(np.uint8), p_perm.affine).to_filename(sig_path)

            if save_perm_pngs:
                plot_mosaic_safe(
                    neglogp_perm,
                    title=f"Permutation FWER -log10(p) [{mode_str}, n={perm_n}] — {tag}",
                    out_png=os.path.join(outdir, f"group_perm_{mode_str}_neglog10_vfwe_{perm_n}n_{tag}.png"),
                    threshold=float(-np.log10(0.05))
                )
                sig_img = nb.Nifti1Image(sig.astype(np.uint8), p_perm.affine)
                plot_mosaic_safe(
                    sig_img,
                    title=f"Permutation FWER p<0.05 [{mode_str}, n={perm_n}] — {tag}",
                    out_png=os.path.join(outdir, f"group_perm_{mode_str}_vfwe_p_lt0p05_{perm_n}n_{tag}.png"),
                    threshold=0.5, colorbar=False
                )

        except Exception as e:
            print(f"[WARN] permutation inference failed ({tag}): {e}", file=sys.stderr)

    # -------------------- Summary PNGs for Z and FDR — masked --------------------
    try:
        plot_mosaic_safe(
            zmap,
            title=f"Unthresholded Z ({tag}) — Z≥{ZPNG_THR:g}",
            out_png=os.path.join(outdir, f"group_zmap_Zge{ZPNG_THR:g}_{tag}.png"),
            threshold=ZPNG_THR
        )
        plot_mosaic_safe(
            thr_img,
            title=f"FDR {alpha}" + (f" + k>={cluster_k}" if cluster_k else "") + f" ({tag})",
            out_png=os.path.join(outdir, f"group_zmap_FDR0.05{tag_k}_{tag}.png"),
            threshold=0.0
        )
    except Exception as e:
        print(f"[WARN] group plotting failed ({tag}): {e}", file=sys.stderr)

# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Subjects
    subs = set()
    if args.subjects:
        subs.update(normalize_sub(s) for s in args.subjects)
    if args.subject:
        subs.update(normalize_sub(s) for s in args.subject)
    if not subs and args.placebo_csv:
        subs.update(read_placebo_subjects(args.placebo_csv))
    subjects = sorted(subs)
    if not subjects:
        print("[ERROR] No subjects provided/found. Use --subjects or --subject (or --placebo-csv).", file=sys.stderr)
        sys.exit(2)

    # Provenance log
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"N subjects: {len(subjects)}\n")
        f.write("Subjects: " + " ".join(subjects) + "\n")
        f.write(f"One-sided: {args.one_sided}\n")
        f.write(f"Alpha (FDR): {args.alpha}; cluster-k: {args.cluster_k}\n")
        f.write(f"Unc p: {args.unc_p}; Unc Z: {args.unc_z}; Unc k: {args.unc_k}\n")
        f.write(f"Second-level FWHM: {args.secondlevel_fwhm}\n")
        f.write(f"Permutation: n={args.perm_n}, mode={args.perm_mode}\n")
        f.write(f"Nilearn dirs: {args.nilearn_dir}\n")
        f.write(f"GLMsingle dirs: {args.glms_dir}\n")
        f.write(f"Runs-tag: {args.runs_tag}; Datatype: {args.datatype}; Branch: {args.branch}\n")
        f.write(f"Mask: {args.mask_img}\n")
        if args.design_order_json:
            try:
                order_map = json.load(open(args.design_order_json, "r"))
                f.write(f"Design-order JSON loaded. (keys: {list(order_map.keys())[:5]}...)\n")
            except Exception as e:
                f.write(f"[WARN] Could not read design-order JSON: {e}\n")

    two_sided = (not args.one_sided)

    # ---- Process each Nilearn folder ----
    for ndir in (args.nilearn_dir or []):
        tag_base = f"nilearn__{os.path.basename(os.path.normpath(ndir))}"
        out_base = ensure_dir(os.path.join(args.outdir, tag_base))
        builder = lambda bdir, sub: nilearn_effect_path(bdir, sub, args.runs_tag, args.datatype)
        found, missing = collect_effects(ndir, subjects, lambda d, s: builder(ndir, s), tag_base)
        pd.DataFrame(missing).to_csv(os.path.join(out_base, "missing_subjects.csv"), index=False)
        pd.DataFrame({"path": found}).to_csv(os.path.join(out_base, "kept_subjects.csv"), index=False)

        if len(found) < 5:
            print(f"[WARN] {tag_base}: too few subjects ({len(found)}), skipping second-level.", file=sys.stderr)
            continue

        # Decide mask for this set
        mask_for_this = args.mask_img
        if isinstance(mask_for_this, str) and mask_for_this.strip().upper() == "FULL":
            fullmask_path = os.path.join(out_base, "fullmask_like_firstcontrast.nii.gz")
            try:
                make_full_mask_like(found[0], fullmask_path)
                mask_for_this = fullmask_path
                print(f"[INFO] Built full-volume mask matching {found[0]} -> {fullmask_path}")
            except Exception as e:
                print(f"[WARN] Could not build full-volume mask; falling back to auto mask: {e}", file=sys.stderr)
                mask_for_this = None

        # unsmoothed
        run_second_level(found, mask_for_this, two_sided, args.alpha, args.cluster_k,
                         args.unc_p, args.unc_z, args.unc_k,
                         fwhm=None, outdir=out_base, tag=f"{tag_base}_unsmoothed",
                         perm_n=args.perm_n, perm_mode=args.perm_mode,
                         save_param_pngs=args.save_parametric_pmaps_pngs,
                         save_perm_pngs=args.save_permutation_pngs)
        # smoothed
        run_second_level(found, mask_for_this, two_sided, args.alpha, args.cluster_k,
                         args.unc_p, args.unc_z, args.unc_k,
                         fwhm=args.secondlevel_fwhm, outdir=out_base, tag=f"{tag_base}_fwhm{args.secondlevel_fwhm:g}",
                         perm_n=args.perm_n, perm_mode=args.perm_mode,
                         save_param_pngs=args.save_parametric_pmaps_pngs,
                         save_perm_pngs=args.save_permutation_pngs)

    # ---- Process each GLMsingle folder ----
    for gdir in (args.glms_dir or []):
        tag_base = f"glmsingle__{os.path.basename(os.path.normpath(gdir))}"
        out_base = ensure_dir(os.path.join(args.outdir, tag_base))
        builder = lambda bdir, sub: glmsingle_effect_path(bdir, sub, args.datatype, args.branch)
        found, missing = collect_effects(gdir, subjects, lambda d, s: builder(gdir, s), tag_base)
        pd.DataFrame(missing).to_csv(os.path.join(out_base, "missing_subjects.csv"), index=False)
        pd.DataFrame({"path": found}).to_csv(os.path.join(out_base, "kept_subjects.csv"), index=False)

        if len(found) < 5:
            print(f"[WARN] {tag_base}: too few subjects ({len(found)}), skipping second-level.", file=sys.stderr)
            continue

        # Decide mask for this set
        mask_for_this = args.mask_img
        if isinstance(mask_for_this, str) and mask_for_this.strip().upper() == "FULL":
            fullmask_path = os.path.join(out_base, "fullmask_like_firstcontrast.nii.gz")
            try:
                make_full_mask_like(found[0], fullmask_path)
                mask_for_this = fullmask_path
                print(f"[INFO] Built full-volume mask matching {found[0]} -> {fullmask_path}")
            except Exception as e:
                print(f"[WARN] Could not build full-volume mask; falling back to auto mask: {e}", file=sys.stderr)
                mask_for_this = None

        # unsmoothed
        run_second_level(found, mask_for_this, two_sided, args.alpha, args.cluster_k,
                         args.unc_p, args.unc_z, args.unc_k,
                         fwhm=None, outdir=out_base, tag=f"{tag_base}_unsmoothed",
                         perm_n=args.perm_n, perm_mode=args.perm_mode,
                         save_param_pngs=args.save_parametric_pmaps_pngs,
                         save_perm_pngs=args.save_permutation_pngs)
        # smoothed
        run_second_level(found, mask_for_this, two_sided, args.alpha, args.cluster_k,
                         args.unc_p, args.unc_z, args.unc_k,
                         fwhm=args.secondlevel_fwhm, outdir=out_base, tag=f"{tag_base}_fwhm{args.secondlevel_fwhm:g}",
                         perm_n=args.perm_n, perm_mode=args.perm_mode,
                         save_param_pngs=args.save_parametric_pmaps_pngs,
                         save_perm_pngs=args.save_permutation_pngs)

if __name__ == "__main__":
    main()
