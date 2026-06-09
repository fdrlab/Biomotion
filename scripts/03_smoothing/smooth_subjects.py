#!/usr/bin/env python3
"""
Fast 4D Gaussian smoother for fMRIPrep outputs (MNI, res-1.7).

- Loads the full 4D BOLD into float32 once.
- Smooths spatially across X/Y/Z in one call (sigma_t = 0 → no time smoothing).
- Applies brain mask AFTER smoothing (zeros non-brain voxels).
- Writes float32 NIfTI.

Typical memory: ~2x the 4D BOLD size in float32 (input + output).
For ~193×229×193×313 → ~10.7 GB per copy → ~21–24 GB plus overhead.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from nilearn.image import resample_to_img

def fwhm_to_sigma_vox(fwhm_mm: float, zooms_xyz: tuple[float, float, float]):
    """Convert FWHM (mm) to per-axis Gaussian sigma in voxel units."""
    sigma_mm = fwhm_mm / 2.3548200450309493  # sqrt(8 ln 2)
    zx, zy, zz = map(float, zooms_xyz)
    return (sigma_mm / zx, sigma_mm / zy, sigma_mm / zz)

def main():
    p = argparse.ArgumentParser(description="Gaussian smooth a subject's preprocessed BOLD (4D, spatial only).")
    p.add_argument("subject_id", help="e.g., sub-131")
    p.add_argument("--base-dir", default="/lustre/scratch/data/.../preprocessed")
    p.add_argument("--runs", nargs="+", default=["01", "02"], help="Run IDs to process (default: 01 02)")
    p.add_argument("--fwhm", type=float, default=3.0, help="Gaussian FWHM (mm), default 3")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = p.parse_args()

    sub = args.subject_id
    func_dir = os.path.join(args.base_dir, sub, "func")
    out_dir  = os.path.join(args.base_dir, sub, "smoothed")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nFast 4D smoothing for {sub} | runs={args.runs} | FWHM={args.fwhm} mm")

    bold_tmpl = "{s}_task-PLDs_run-{r}_space-MNI152NLin2009cAsym_res-1.7_desc-preproc_bold.nii.gz"
    mask_tmpl = "{s}_task-PLDs_run-{r}_space-MNI152NLin2009cAsym_res-1.7_desc-brain_mask.nii.gz"
    out_tmpl  = "{s}_task-PLDs_run-{r}_space-MNI152NLin2009cAsym_res-1.7_desc-smoothed_bold.nii.gz"

    for run in args.runs:
        bold_path = os.path.join(func_dir, bold_tmpl.format(s=sub, r=run))
        mask_path = os.path.join(func_dir, mask_tmpl.format(s=sub, r=run))
        out_path  = os.path.join(out_dir,  out_tmpl.format(s=sub, r=run))

        print(f"\n⏳ run-{run}")

        if not os.path.exists(bold_path):
            print(f"Missing BOLD: {bold_path} (skip)")
            continue
        if not os.path.exists(mask_path):
            print(f"Missing mask: {mask_path} (skip)")
            continue
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Exists, skipping (use --overwrite to redo): {out_path}")
            continue

        # Lazy load image objects (array proxy, not data yet)
        bold_img = nib.load(bold_path)
        mask_img = nib.load(mask_path)

        zooms = bold_img.header.get_zooms()[:3]
        sigma_xyz = fwhm_to_sigma_vox(args.fwhm, zooms)
        sigma = tuple(map(np.float32, sigma_xyz)) + (np.float32(0.0),)  # (sx, sy, sz, st=0)

        X, Y, Z, T = bold_img.shape
        print(f"   • Shape: {X}×{Y}×{Z}×{T}  |  zooms={zooms} mm  |  sigma_vox={sigma_xyz}")

        # Load full 4D into float32 (1x copy in RAM)
        data = bold_img.get_fdata(dtype=np.float32, caching="unchanged")

        # Prepare output array (same shape, float32)
        out = np.empty_like(data, dtype=np.float32)

        # Spatial-only 4D smoothing (no time smoothing via sigma_t=0)
        # Use mode="constant" with cval=0 so outside-brain padding doesn't smear NaNs.
        print("   • Applying 4D gaussian_filter (spatial only)…")
        gaussian_filter(data, sigma=sigma, output=out, mode="constant", cval=0.0)

        # Resample mask to bold grid; apply AFTER smoothing
        res_mask = resample_to_img(mask_img, bold_img, interpolation="nearest",
                                   force_resample=True, copy_header=True)
        mask_bool = res_mask.get_fdata().astype(bool)
        if mask_bool.shape != out.shape[:3]:
            raise RuntimeError(f"Mask/BOLD mismatch: {mask_bool.shape} vs {out.shape[:3]}")

        out[~mask_bool, :] = 0.0

        # Save float32 NIfTI
        out_img = nib.Nifti1Image(out, affine=bold_img.affine, header=bold_img.header)
        out_img.set_data_dtype(np.float32)
        print(f"Writing: {out_path}")
        nib.save(out_img, out_path)
        print(f"Done: {out_path}")

    print("\n All requested runs complete.")

if __name__ == "__main__":
    main()
