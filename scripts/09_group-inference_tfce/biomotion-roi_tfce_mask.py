#!/usr/bin/env python3
"""
biomotion-roi_tfce_mask.py

Create a final analysis mask for randomise by intersecting:
  final = TFCE_safe_mask ∩ biomotion_ROI_resampled_to_TFCE_grid

Also optionally writes a QC PNG overlay (plot_roi) on MNI152 template,
resampled to the TFCE grid.

Usage:
  python make_biomotion_tfce_mask.py \
    --tfce-mask /path/to/tfce_mask.nii.gz \
    --roi-mask  /path/to/biomotion_roi.nii.gz \
    --out-mask  /path/to/final_mask.nii.gz \
    --cut-coords 51 -67 1 \
    --dpi 200 \
    --debug
"""

import os
import argparse
import numpy as np
import nibabel as nb

# --- optional QC + resampling deps ---
try:
    import matplotlib
    matplotlib.use("Agg")
    from nilearn.image import resample_to_img
    from nilearn.plotting import plot_roi
    from nilearn.datasets import load_mni152_template
except Exception as e:
    matplotlib = None
    resample_to_img = None
    plot_roi = None
    load_mni152_template = None
    _QC_IMPORT_ERR = e


def vox_mm3(img: nb.Nifti1Image) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def as_bool(dat: np.ndarray, thr: float) -> np.ndarray:
    dat = np.asarray(dat, dtype=np.float32)
    return np.isfinite(dat) & (dat > float(thr))


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_qc_png(mask_img: nb.Nifti1Image,
                 ref_grid_img: nb.Nifti1Image,
                 out_png: str,
                 title: str,
                 cut_coords,
                 dpi: int,
                 debug: bool):
    """
    QC overlay similar to build_tfce_masks_and_qc_thresholds.py:
    - background: MNI152 T1
    - background is resampled to ref grid
    - overlay: binary mask (mask_img) in ref grid
    """
    if plot_roi is None or load_mni152_template is None or resample_to_img is None:
        raise RuntimeError(f"QC dependencies missing (matplotlib/nilearn). Import error: {_QC_IMPORT_ERR}")

    bg = load_mni152_template()
    bg_rs = resample_to_img(bg, ref_grid_img, interpolation="continuous")

    ensure_dir(os.path.dirname(out_png) or ".")

    if debug:
        d = mask_img.get_fdata(dtype=np.float32)
        print(f"[qc] writing {out_png} | vox={int(np.sum(d > 0.5))}")

    disp = plot_roi(
        mask_img,
        bg_img=bg_rs,
        display_mode="ortho",
        cut_coords=tuple(cut_coords),
        title=title,
        alpha=0.75,
        cmap="autumn",
        colorbar=False,
        black_bg=False,
        dim=0.35,
    )
    disp.savefig(out_png, dpi=int(dpi))
    disp.close()


def _default_qc_png_path(out_mask: str) -> str:
    # keep it simple + predictable
    d = os.path.dirname(out_mask) or "."
    bn = os.path.basename(out_mask)
    if bn.endswith(".nii.gz"):
        stem = bn[:-7]
    else:
        stem = os.path.splitext(bn)[0]
    return os.path.join(d, f"qc_{stem}.png")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tfce-mask", required=True, help="Your TFCE-safe mask (reference grid).")
    ap.add_argument("--roi-mask", required=True, help="Biomotion ROI mask (any grid).")
    ap.add_argument("--out-mask", required=True, help="Output final intersected mask.")

    ap.add_argument("--roi-thr", type=float, default=0.5,
                    help="Threshold for ROI binarization after resampling (default 0.5).")
    ap.add_argument("--tfce-thr", type=float, default=0.5,
                    help="Threshold for TFCE mask binarization (default 0.5).")

    # QC options
    ap.add_argument("--qc-png", default="",
                    help="Write QC overlay PNG to this path. If empty, auto-write next to out-mask.")
    ap.add_argument("--no-qc", action="store_true",
                    help="Disable QC PNG generation even if nilearn/matplotlib are available.")
    ap.add_argument("--cut-coords", nargs=3, type=float, default=(51.0, -67.0, 1.0),
                    help="Ortho cut coords (x y z) in mm for QC overlay.")
    ap.add_argument("--dpi", type=int, default=200, help="QC PNG DPI (default 200).")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tfce_img = nb.load(args.tfce_mask)
    tfce_dat = tfce_img.get_fdata(dtype=np.float32)
    tfce_bool = as_bool(tfce_dat, args.tfce_thr)

    roi_img = nb.load(args.roi_mask)

    # Resample ROI -> TFCE grid (nearest)
    if roi_img.shape != tfce_img.shape or (not np.allclose(roi_img.affine, tfce_img.affine)):
        if resample_to_img is None:
            raise RuntimeError(
                "ROI mask grid != TFCE grid and nilearn is not available for resampling. "
                "Install nilearn or provide an ROI mask already on the TFCE mask grid."
            )
        if args.debug:
            print("[info] Resampling ROI -> TFCE grid (nearest)")
            print("  ROI :", args.roi_mask)
            print("  TFCE:", args.tfce_mask)
        roi_rs = resample_to_img(roi_img, tfce_img, interpolation="nearest", force_resample=True, copy_header=True)
    else:
        roi_rs = roi_img

    roi_dat = roi_rs.get_fdata(dtype=np.float32)
    roi_bool = as_bool(roi_dat, args.roi_thr)

    final_bool = tfce_bool & roi_bool
    final_u8 = final_bool.astype(np.uint8)

    ensure_dir(os.path.dirname(args.out_mask) or ".")
    out_img = nb.Nifti1Image(final_u8, tfce_img.affine, tfce_img.header)
    out_img.set_data_dtype(np.uint8)
    nb.save(out_img, args.out_mask)

    v_ml = vox_mm3(tfce_img) / 1000.0  # mL per voxel
    n_tfce = int(tfce_bool.sum())
    n_roi  = int(roi_bool.sum())
    n_fin  = int(final_bool.sum())

    print("[ok] wrote:", args.out_mask)
    print(f"TFCE mask vox: {n_tfce}  ({n_tfce*v_ml:.3f} mL)")
    print(f"ROI  mask vox: {n_roi}  ({n_roi*v_ml:.3f} mL)  [after resample+thr]")
    print(f"FINAL vox    : {n_fin}  ({n_fin*v_ml:.3f} mL)  [intersection]")

    if n_fin == 0:
        print("[warn] FINAL mask is empty. Usual causes: wrong space/template, misalignment, or too strict threshold.")

    # QC PNG
    if args.no_qc:
        return

    qc_png = args.qc_png.strip() if args.qc_png.strip() else _default_qc_png_path(args.out_mask)

    if n_fin > 0:
        try:
            title = "Final TFCE mask = TFCE-safe ∩ biomotion ROI"
            write_qc_png(
                mask_img=nb.load(args.out_mask),
                ref_grid_img=tfce_img,
                out_png=qc_png,
                title=title,
                cut_coords=tuple(args.cut_coords),
                dpi=int(args.dpi),
                debug=args.debug,
            )
            print("[ok] wrote QC PNG:", qc_png)
        except Exception as e:
            print(f"[warn] QC PNG skipped/failed: {e}")
    else:
        print("[warn] QC PNG skipped: final mask empty")


if __name__ == "__main__":
    main()
