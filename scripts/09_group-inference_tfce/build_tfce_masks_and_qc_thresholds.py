#!/usr/bin/env python3
"""
build_tfce_masks_and_qc_thresholds.py

Build a TFCE-safe group mask family from RSA outputs using EFFECT maps (beta maps).

present(voxel, subject) :=
    cortex_universe
    & isfinite(effect_map)          # e.g., RSA beta map (recommended)
    & isfinite(coverage)
    & (coverage >= min_neigh)

Outputs:
  - count map (#subjects present at each voxel)
  - percent-present map (0..100)
  - binary masks for requested fractions (e.g., 1.0, .95, .90, .80, .70)
  - QC PNG overlays for each mask, using MNI152 T1 as background

Robustness:
  - subject exclusion list
  - file picking supports strict/newest/first
  - prefer/reject substrings to disambiguate multiple matches
  - auto-detect subject dirs even if nested below --rsa-root
"""

import os, glob, math, argparse
import numpy as np
import nibabel as nib

# QC deps
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


def normalize_sub(s: str) -> str:
    s = str(s).strip()
    if s.startswith("sub-"):
        return s
    if s.isdigit():
        return f"sub-{int(s):03d}"
    return f"sub-{s}"


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage) -> bool:
    return (a.shape == b.shape) and np.allclose(a.affine, b.affine)


def newest_by_mtime(paths):
    paths = [p for p in paths if p and os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def pick_one(glob_pat: str,
             mode: str,
             prefer_substr=None,
             reject_substr=None) -> str:
    hits = sorted(glob.glob(glob_pat, recursive=True))
    if reject_substr:
        hits = [h for h in hits if not any(r in h for r in reject_substr)]
    if prefer_substr:
        preferred = [h for h in hits if any(p in h for p in prefer_substr)]
        if preferred:
            hits = preferred

    if not hits:
        raise FileNotFoundError(f"No match: {glob_pat}")

    if mode == "strict":
        if len(hits) != 1:
            raise RuntimeError(
                f"Expected exactly 1 match for:\n  {glob_pat}\nGot {len(hits)} matches:\n  "
                + "\n  ".join(hits[:20]) + ("\n  ..." if len(hits) > 20 else "")
            )
        return hits[0]
    if mode == "first":
        return hits[0]
    return max(hits, key=os.path.getmtime)


def load_bool_mask(path: str):
    img = nib.load(path)
    dat = img.get_fdata(dtype=np.float32)
    m = np.isfinite(dat) & (dat > 0.5)
    return img, m


def load_cortex_on_ref(cortex_path: str, ref_img: nib.Nifti1Image, debug=False) -> np.ndarray:
    cortex_img, cortex = load_bool_mask(cortex_path)
    if same_grid(cortex_img, ref_img):
        return cortex

    if resample_to_img is None:
        raise RuntimeError(
            "Cortex mask grid != RSA grid and nilearn.resample_to_img unavailable.\n"
            f"Import error: {_QC_IMPORT_ERR}"
        )

    if debug:
        print(f"[info] Resampling cortex mask -> RSA grid (nearest): {cortex_path}")

    rs = resample_to_img(cortex_img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    dat = rs.get_fdata(dtype=np.float32)
    cortex_rs = np.isfinite(dat) & (dat > 0.5)
    if cortex_rs.shape != ref_img.shape:
        raise RuntimeError(f"Resampled cortex shape mismatch: {cortex_rs.shape} vs {ref_img.shape}")
    return cortex_rs


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def write_qc_png(mask_img: nib.Nifti1Image,
                 ref_grid_img: nib.Nifti1Image,
                 out_png: str,
                 title: str,
                 cut_coords,
                 dpi: int,
                 debug: bool):
    if plot_roi is None or load_mni152_template is None or resample_to_img is None:
        raise RuntimeError(f"QC dependencies missing (matplotlib/nilearn). Import error: {_QC_IMPORT_ERR}")

    bg = load_mni152_template()
    bg_rs = resample_to_img(bg, ref_grid_img, interpolation="continuous")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    if debug:
        data = mask_img.get_fdata(dtype=np.float32)
        print(f"[qc] {os.path.basename(out_png)} mask vox={int(np.sum(data > 0.5))}")

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
    disp.savefig(out_png, dpi=dpi)
    disp.close()


def find_subject_dirs(rsa_root: str):
    # First try direct subdirs
    direct = sorted(
        os.path.join(rsa_root, d) for d in os.listdir(rsa_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(rsa_root, d))
    )
    if direct:
        return direct

    # Fallback: recursive search
    rec = sorted(set(
        p for p in glob.glob(os.path.join(rsa_root, "**", "sub-*"), recursive=True)
        if os.path.isdir(p) and os.path.basename(p).startswith("sub-")
    ))
    return rec


def resolve_subject_dir(rsa_root: str, sub: str):
    sub = normalize_sub(sub)
    cand = os.path.join(rsa_root, sub)
    if os.path.isdir(cand):
        return cand

    hits = [p for p in glob.glob(os.path.join(rsa_root, "**", sub), recursive=True) if os.path.isdir(p)]
    if not hits:
        return None
    # pick newest subject dir (handles accidental duplicates)
    return newest_by_mtime(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsa-root", required=True)

    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Optional list of subjects. If omitted, auto-detect sub-*/ dirs (directly or recursively).")
    ap.add_argument("--exclude-subjects", nargs="*", default=[])

    # EFFECT MAP (beta) + coverage
    ap.add_argument("--effect-pattern", required=True,
                    help="Glob RELATIVE to subject dir (use ** for recursion). Should match RSA beta map(s).")
    ap.add_argument("--coverage-pattern", required=True,
                    help="Glob RELATIVE to subject dir (use ** for recursion).")

    ap.add_argument("--pick-mode", choices=["newest", "strict", "first"], default="newest")
    ap.add_argument("--prefer-substr", nargs="*", default=[],
                    help="If multiple matches, prefer paths containing any of these substrings.")
    ap.add_argument("--reject-substr", nargs="*", default=[],
                    help="Drop matches containing any of these substrings.")

    ap.add_argument("--min-neigh", type=int, default=60)
    ap.add_argument("--cortex-mask", required=True)

    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.95, 0.90, 0.80, 0.70])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="tfce")

    ap.add_argument("--cut-coords", nargs=3, type=float, default=(35.0, -20.0, 25.0))
    ap.add_argument("--dpi", type=int, default=200)

    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rsa_root = args.rsa_root
    if not os.path.isdir(rsa_root):
        raise SystemExit(f"--rsa-root not a dir: {rsa_root}")

    prefer = args.prefer_substr if args.prefer_substr else None
    reject = args.reject_substr if args.reject_substr else None

    # Build list of subject dirs
    excl = set(normalize_sub(s) for s in args.exclude_subjects)

    if args.subjects:
        subs = [normalize_sub(s) for s in args.subjects]
        subdirs = []
        for s in subs:
            if s in excl:
                continue
            sd = resolve_subject_dir(rsa_root, s)
            if sd is None:
                if args.skip_missing:
                    if args.debug:
                        print(f"[warn] {s}: could not resolve subject dir under {rsa_root}")
                    continue
                raise FileNotFoundError(f"Could not find directory for {s} under {rsa_root}")
            subdirs.append(sd)
    else:
        subdirs = find_subject_dirs(rsa_root)
        # filter excluded
        subdirs = [sd for sd in subdirs if os.path.basename(sd) not in excl]

    if not subdirs:
        raise SystemExit("No subject directories found after exclusion/selection.")

    if args.debug:
        print(f"[info] exclude: {sorted(excl)}")
        print(f"[info] subject dirs found: {len(subdirs)}")

    # reference grid: first usable subject
    ref_img = None
    ref_sub = None
    ref_sd = None

    for sd in subdirs:
        s = os.path.basename(sd)
        try:
            eff = pick_one(os.path.join(sd, args.effect_pattern), args.pick_mode, prefer, reject)
            cov = pick_one(os.path.join(sd, args.coverage_pattern), args.pick_mode, prefer, reject)
            ref_img = nib.load(eff)
            _ = nib.load(cov)
            if len(ref_img.shape) != 3:
                raise RuntimeError("reference effect image is not 3D")
            ref_sub = s
            ref_sd = sd
            break
        except Exception as e:
            if args.debug:
                print(f"[warn] {s}: not usable as reference ({e})")
            if not args.skip_missing:
                raise

    if ref_img is None:
        raise SystemExit("No usable subject found to define reference grid.")

    cortex = load_cortex_on_ref(args.cortex_mask, ref_img, debug=args.debug)

    count = np.zeros(ref_img.shape, dtype=np.int16)
    used = []
    manifest = []

    for sd in subdirs:
        s = os.path.basename(sd)

        try:
            eff = pick_one(os.path.join(sd, args.effect_pattern), args.pick_mode, prefer, reject)
            cov = pick_one(os.path.join(sd, args.coverage_pattern), args.pick_mode, prefer, reject)
        except Exception as e:
            if args.debug:
                print(f"[warn] {s}: missing effect/cov ({e})")
            if args.skip_missing:
                continue
            raise

        eimg = nib.load(eff)
        cimg = nib.load(cov)

        if not same_grid(eimg, ref_img):
            raise RuntimeError(f"{s}: effect grid mismatch vs ref: {eff}")
        if not same_grid(cimg, ref_img):
            raise RuntimeError(f"{s}: cov grid mismatch vs ref: {cov}")

        edat = eimg.get_fdata(dtype=np.float32)
        cdat = cimg.get_fdata(dtype=np.float32)

        present = cortex & np.isfinite(edat) & np.isfinite(cdat) & (cdat >= float(args.min_neigh))
        pv = int(present.sum())

        count += present.astype(np.int16)
        used.append(s)
        manifest.append((s, eff, cov, pv))

        if args.debug:
            print(f"[{s}] present vox={pv}")

    N = len(used)
    if N == 0:
        raise SystemExit("No subjects contributed.")

    os.makedirs(args.outdir, exist_ok=True)

    # Save count + percent maps
    out_count = os.path.join(args.outdir, f"{args.prefix}_count_N{N}_min{args.min_neigh}.nii.gz")
    nib.save(nib.Nifti1Image(count.astype(np.int16), ref_img.affine, ref_img.header), out_count)

    pct = (count.astype(np.float32) / float(N)) * 100.0
    out_pct = os.path.join(args.outdir, f"{args.prefix}_presentPercent_N{N}_min{args.min_neigh}.nii.gz")
    nib.save(nib.Nifti1Image(pct.astype(np.float32), ref_img.affine, ref_img.header), out_pct)

    # Manifest
    out_tsv = os.path.join(args.outdir, f"{args.prefix}_manifest_N{N}_min{args.min_neigh}.tsv")
    with open(out_tsv, "w") as f:
        f.write("subject\teffect_path\tcoverage_path\tpresent_vox\n")
        for s, eff, cov, pv in manifest:
            f.write(f"{s}\t{eff}\t{cov}\t{pv}\n")

    print(f"[ok] reference subject: {ref_sub}")
    print(f"[ok] reference subject dir: {ref_sd}")
    print(f"[ok] included subjects: {N}")
    print(f"[ok] wrote count map: {out_count}")
    print(f"[ok] wrote percent map: {out_pct}")
    print(f"[ok] wrote manifest: {out_tsv}")

    vmm3 = voxel_volume_mm3(ref_img)

    # Masks + QC
    fracs = []
    for fval in args.fractions:
        if not (0.0 < fval <= 1.0):
            print(f"[warn] skipping invalid fraction: {fval}")
            continue
        fracs.append(fval)

    fracs = sorted(set(fracs), reverse=True)

    print("\nthreshold\tneed/N\tvox\tmL\tmask_path\tqc_png")
    for fval in fracs:
        need = int(math.ceil(fval * N))
        mask = (count >= need).astype(np.uint8)
        vox = int(mask.sum())
        ml = vox * vmm3 / 1000.0

        p = int(round(fval * 100))
        mask_path = os.path.join(args.outdir, f"{args.prefix}_mask_p{p:03d}_{need}of{N}_min{args.min_neigh}.nii.gz")
        nib.save(nib.Nifti1Image(mask, ref_img.affine, ref_img.header), mask_path)

        qc_png = os.path.join(args.outdir, f"qc_{args.prefix}_mask_p{p:03d}_{need}of{N}_min{args.min_neigh}.png")
        title = f"TFCE group mask (>= {need}/{N} subjects; {p}%) | min_neigh={args.min_neigh}"

        if vox == 0:
            print(f"{p}%\t{need}/{N}\t0\t0.0\t{mask_path}\t(SKIP: empty)")
        else:
            write_qc_png(
                nib.load(mask_path),
                ref_img,
                qc_png,
                title=title,
                cut_coords=tuple(args.cut_coords),
                dpi=args.dpi,
                debug=args.debug,
            )
            print(f"{p}%\t{need}/{N}\t{vox}\t{ml:.3f}\t{mask_path}\t{qc_png}")


if __name__ == "__main__":
    main()
