#!/usr/bin/env python3
"""
build_group_mask_from_rois.py

Build a group mask from per-subject binary ROI masks, and (optionally) create an
orthographic PNG overlay (like your make_mask_overlay.py).

Inputs:
  - per-subject ROI NIfTIs (binary-ish), e.g.
    GLMsingle/split_half/roi/sub-XXX_*_desc-cortexROI_runIntersect.nii.gz

Outputs:
  - <out_prefix>_present_percent.nii.gz
  - <out_prefix>_groupmask_covXX.nii.gz   (XX = save-frac*100)

Optional visualization:
  - --ortho-out <png> overlays the group mask on a reference image.

Notes:
  - For building the mask, all ROIs are resampled (nearest) to the first ROI grid if needed.
  - For visualization, you can resample the mask to the REF grid with --resample-mask-to-ref.
"""

import os, glob, math, argparse, tempfile
import numpy as np
import nibabel as nb

# plotting (only used if --ortho-out is set)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from nilearn.image import resample_to_img
    from nilearn.plotting import plot_img
    HAVE_NILEARN = True
except Exception:
    HAVE_NILEARN = False

# -------------------- helpers --------------------
def pick_first(glob_pat: str) -> str:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No files matched:\n  {glob_pat}")
    return paths[0]

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def try_resample_like(img, ref_img, interp="nearest"):
    """Resample img to ref_img grid if needed."""
    if img.shape == ref_img.shape and np.allclose(img.affine, ref_img.affine):
        return img
    if not HAVE_NILEARN:
        raise RuntimeError("Grid mismatch but nilearn is not available to resample_to_img.")
    return resample_to_img(img, ref_img, interpolation=interp, force_resample=True, copy_header=True)

def is_binary_mask_data(data: np.ndarray) -> bool:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return True
    u = np.unique(finite)
    # tolerate tiny numeric noise around 0/1
    if np.all(np.isclose(u, 0.0)):
        return True
    if np.all(np.isclose(u, 1.0)):
        return True
    if u.size <= 2 and np.all(np.isclose(np.sort(u), [0.0, 1.0])):
        return True
    # more permissive: everything close to 0 or 1
    return np.all((np.isclose(finite, 0.0) | np.isclose(finite, 1.0)))

def save_clean_ref_no_nan(ref_img: nb.Nifti1Image) -> str:
    """Write a temp NIfTI with NaNs replaced by 0 (like fslmaths -nan)."""
    data = ref_img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    tmp = tempfile.NamedTemporaryFile(prefix="ref_nanfix_", suffix=".nii.gz", dir="/tmp", delete=False).name
    nb.save(nb.Nifti1Image(data, ref_img.affine, ref_img.header), tmp)
    return tmp

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # mask-building
    ap.add_argument("--roi-dir", required=True, help="Directory with per-subject ROI NIfTIs.")
    ap.add_argument("--pattern", default="sub-*_desc-cortexROI_runIntersect.nii.gz",
                    help="Glob pattern within --roi-dir.")
    ap.add_argument("--save-frac", type=float, default=0.30, help="Keep voxels present in >= this fraction of subjects.")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9],
                    help="Fractions to report in console (voxel counts / volume).")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (no extension).")

    # visualization (optional)
    ap.add_argument("--ortho-out", default="", help="If set, write orthographic overlay PNG here.")
    ap.add_argument("--ref", default="", help="Background reference NIfTI for visualization.")
    ap.add_argument("--ref-glob",
                    default="/lustre/scratch/data/.../searchlight_RSA/unsmoothed_cov20/sub-*/sub-*_beta-model6_bio_vs_scrambled_rankortho.nii.gz",
                    help="If --ref not given, use first match of this glob.")
    ap.add_argument("--title", default="", help="Optional figure title.")
    ap.add_argument("--cut-coords", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", nargs=2, type=float, default=(12.0, 6.0))
    ap.add_argument("--overlay-alpha", type=float, default=0.7)
    ap.add_argument("--cmap", default="viridis")  # only used for continuous overlays
    ap.add_argument("--mask-vmin", type=float, default=None)
    ap.add_argument("--mask-vmax", type=float, default=None)
    ap.add_argument("--resample-mask-to-ref", action="store_true",
                    help="Resample the group mask to REF grid for visualization (nearest).")
    ap.add_argument("--keep-tmp", action="store_true")

    args = ap.parse_args()

    # ---------- find ROIs ----------
    roi_paths = sorted(glob.glob(os.path.join(args.roi_dir, args.pattern)))
    if not roi_paths:
        raise SystemExit(f"No ROI files found in {args.roi_dir} matching {args.pattern}")

    # ---------- reference grid for building (first ROI) ----------
    ref_img = nb.load(roi_paths[0])
    acc = np.zeros(ref_img.shape, dtype=np.int32)

    used = 0
    for p in roi_paths:
        img = nb.load(p)
        img = try_resample_like(img, ref_img, interp="nearest")
        m = img.get_fdata(dtype=np.float32) > 0.5
        acc += m.astype(np.int32)
        used += 1

    n = used
    print(f"[info] ROIs used: {n}")

    out_dir = ensure_dir(os.path.dirname(args.out_prefix) or ".")
    _ = out_dir  # keep linter quiet

    # ---------- percent-present map ----------
    pct = (acc.astype(np.float32) / float(n)) * 100.0
    out_pct = f"{args.out_prefix}_present_percent.nii.gz"
    nb.save(nb.Nifti1Image(pct, ref_img.affine, ref_img.header), out_pct)
    print(f"[ok] wrote: {out_pct}")

    # ---------- report volumes ----------
    voxel_vol_mm3 = abs(np.linalg.det(ref_img.affine[:3, :3]))
    for frac in sorted(set(args.fracs)):
        if not (0.0 < frac <= 1.0):
            print(f"[warn] skipping invalid frac {frac} (must be in (0,1]).")
            continue
        need = math.ceil(frac * n)
        mask = (acc >= need)
        vox = int(mask.sum())
        vol_ml = vox * voxel_vol_mm3 / 1000.0
        print(f"thr={int(round(frac*100))}% (>= {need}/{n}) voxels={vox} volume={vol_ml:.1f} mL")

    # ---------- final group mask ----------
    need = math.ceil(args.save_frac * n)
    group = (acc >= need).astype(np.uint8)
    out_mask = f"{args.out_prefix}_groupmask_cov{int(round(args.save_frac*100))}.nii.gz"
    nb.save(nb.Nifti1Image(group, ref_img.affine, ref_img.header), out_mask)
    print(f"[ok] wrote: {out_mask}")

    # ---------- optional visualization ----------
    if args.ortho_out:
        if not HAVE_NILEARN:
            raise SystemExit("ERROR: --ortho-out requires nilearn (plot_img + resample_to_img).")

        ref_path = args.ref if args.ref else pick_first(args.ref_glob)
        if not os.path.isfile(ref_path):
            raise SystemExit(f"ERROR: Reference NIfTI not found: {ref_path}")

        ref_vis_img = nb.load(ref_path)
        tmp_ref_clean = save_clean_ref_no_nan(ref_vis_img)

        mask_vis_img = nb.load(out_mask)
        if args.resample_mask_to_ref:
            mask_vis_img = try_resample_like(mask_vis_img, ref_vis_img, interp="nearest")

        data = mask_vis_img.get_fdata(dtype=np.float32)
        binary = is_binary_mask_data(data)

        print(f"[info] Writing orthographic overlay PNG → {args.ortho_out}")
        fig = plt.figure(figsize=tuple(args.figsize), dpi=args.dpi, constrained_layout=True)
        display = plot_img(tmp_ref_clean, display_mode="ortho",
                           cut_coords=tuple(args.cut_coords),
                           colorbar=False, figure=fig)

        if binary:
            display.add_overlay(mask_vis_img, cmap="autumn", alpha=float(args.overlay_alpha), colorbar=False)
        else:
            finite_vals = data[np.isfinite(data)]
            vmin = args.mask_vmin if args.mask_vmin is not None else (np.nanmin(finite_vals) if finite_vals.size else 0.0)
            vmax = args.mask_vmax if args.mask_vmax is not None else (np.nanmax(finite_vals) if finite_vals.size else 1.0)
            display.add_overlay(mask_vis_img, cmap=args.cmap, vmin=vmin, vmax=vmax, alpha=float(args.overlay_alpha), colorbar=True)

        if args.title:
            fig.suptitle(args.title, fontsize=18, fontweight="bold", x=0.02, y=0.99, ha="left")

        ensure_dir(os.path.dirname(args.ortho_out) or ".")
        fig.savefig(args.ortho_out, bbox_inches="tight")
        plt.close(fig)

        if os.path.isfile(args.ortho_out):
            print(f"[ok] saved: {args.ortho_out}")
        else:
            print(f"[warn] failed to save: {args.ortho_out}")

        if not args.keep_tmp:
            try:
                if os.path.exists(tmp_ref_clean):
                    os.remove(tmp_ref_clean)
            except OSError:
                pass

if __name__ == "__main__":
    main()
