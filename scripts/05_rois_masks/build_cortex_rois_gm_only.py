#!/usr/bin/env python3
import os, sys, glob, argparse, subprocess
import numpy as np
import nibabel as nb

BASE = ""
PREP = f"{BASE}/preprocessed"
GLM  = f"{BASE}/GLMsingle"
ROI_DIR = os.path.join(GLM, "split_half", "roi") # split_half/roi contains subject-specific, run-intersected cortex ROIs (e.g., sub-102_space-MNI152NLin2009cAsym_desc-cortexROI_runIntersect)
TMP_DIR = os.path.join(GLM, "split_half", "tmp")


def die(msg: str):
    raise RuntimeError(msg)


def norm_sub(s: str) -> str:
    s = str(s).strip()
    if s.startswith("sub-"):
        return s
    return f"sub-{s.zfill(3)}" if s.isdigit() else f"sub-{s}"


def first_glob(pat: str) -> str | None:
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def find_brainmask(sub: str, run: str) -> str:
    func = os.path.join(PREP, sub, "func")
    pat = os.path.join(func, f"{sub}_task-PLDs_run-{run}_space-MNI152NLin2009cAsym*_desc-brain_mask.nii.gz")
    p = first_glob(pat)
    if p:
        return p
    pat2 = os.path.join(func, f"{sub}_*run-{run}_space-MNI152NLin2009cAsym*_desc-brain_mask.nii.gz")
    p2 = first_glob(pat2)
    if p2:
        return p2
    die(f"[{sub}] Missing run-{run} MNI brain mask in {func}")


def find_ribbon_t1w(sub: str) -> str:
    anat = os.path.join(PREP, sub, "anat")
    cand = first_glob(os.path.join(anat, f"{sub}_desc-ribbon_mask.nii.gz"))
    if cand:
        return cand
    cand2 = first_glob(os.path.join(anat, f"{sub}_space-T1w*_desc-ribbon_mask.nii.gz"))
    if cand2:
        return cand2
    die(f"[{sub}] Missing T1w ribbon_mask in {anat}")


def find_t1_to_mni_h5(sub: str) -> str:
    anat = os.path.join(PREP, sub, "anat")
    p = os.path.join(anat, f"{sub}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")
    if os.path.isfile(p):
        return p
    die(f"[{sub}] Missing T1w→MNI .h5 transform: {p}")


def find_gm_probseg(sub: str) -> tuple[str, bool]:
    """Return (path, is_mni_space). Prefer MNI GM_probseg; fallback to T1w GM_probseg if present."""
    anat = os.path.join(PREP, sub, "anat")

    # Prefer MNI
    pat_mni = os.path.join(anat, f"{sub}_space-MNI152NLin2009cAsym*_label-GM_probseg.nii.gz")
    p = first_glob(pat_mni)
    if p:
        return p, True

    # Fallback: no-space GM probseg (likely T1w)
    p2 = first_glob(os.path.join(anat, f"{sub}_label-GM_probseg.nii.gz"))
    if p2:
        return p2, False

    die(f"[{sub}] Missing GM_probseg (MNI or T1w) in {anat}")


def run_cmd(cmd: list[str]) -> str:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd) + "\n\nSTDOUT:\n" + r.stdout + "\n\nSTDERR:\n" + r.stderr
        )
    return r.stdout


def apptainer_prefix(sif: str) -> list[str]:
    return ["apptainer", "exec", "--bind", "/lustre:/lustre", sif]


def ants_apply(
    ants_sif: str,
    inp: str,
    ref: str,
    out: str,
    transforms: list[str] | None,
    interp: str,
):
    """
    Apply ANTs transforms inside the fmriprep SIF.
      - transforms: e.g. [h5_path] or []/None for identity resampling
      - interp: "NearestNeighbor" or "Linear"
    """
    cmd = apptainer_prefix(ants_sif) + ["antsApplyTransforms", "-d", "3", "-i", inp, "-r", ref]
    if transforms:
        for t in transforms:
            cmd += ["-t", t]
    cmd += ["-n", interp, "-o", out]
    run_cmd(cmd)


def build_one(
    sub: str,
    ants_sif: str,
    gm_thr: float,
    overwrite: bool = False,
    allow_gm_only_fallback: bool = False,
) -> tuple[str, str, dict]:
    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    bm1 = find_brainmask(sub, "01")
    bm2 = find_brainmask(sub, "02")
    h5  = find_t1_to_mni_h5(sub)

    out_roi = os.path.join(ROI_DIR, f"{sub}_space-MNI152NLin2009cAsym_desc-cortexROI_runIntersect.nii.gz")

    # run coverage (functional space)
    m1 = nb.load(bm1).get_fdata() > 0.5
    m2 = nb.load(bm2).get_fdata() > 0.5
    cov = m1 & m2
    if not cov.any():
        die(f"[{sub}] coverage mask empty after run01∩run02")

    ref_img = nb.load(bm1)  # output grid (affine/header)

    # If ROI exists and no overwrite, just log basic info and return
    if os.path.isfile(out_roi) and not overwrite:
        roi_data = nb.load(out_roi).get_fdata()
        roi_mask = roi_data > 0.5
        n_roi = int(roi_mask.sum())
        n_cov = int(cov.sum())
        frac = (n_roi / n_cov) if n_cov > 0 else float("nan")
        roi_kind = "existing (not rebuilt)"
        stats = dict(
            gm_thr=float(gm_thr),
            roi_path=out_roi,
            roi_kind=roi_kind,
            n_roi=n_roi,
            n_cov=n_cov,
            frac=frac,
            n_gm=np.nan,
            n_ribbon=np.nan,
        )
        print(f"[{sub}] ROI exists, skip rebuild: {out_roi}")
        print(f"[{sub}] kind={roi_kind} | vox ROI={n_roi} | cov={n_cov} | frac={frac:.3f}")
        return out_roi, roi_kind, stats

    # ---------- ribbon (T1w -> bm1 grid) ----------
    ribbon_ok = True
    ribbon_mask = None
    n_ribbon = np.nan
    try:
        rib = find_ribbon_t1w(sub)
        rib_warped = os.path.join(TMP_DIR, f"{sub}_ribbon_warped_to_run01mask.nii.gz")
        ants_apply(
            ants_sif=ants_sif,
            inp=rib,
            ref=bm1,
            out=rib_warped,
            transforms=[h5],
            interp="NearestNeighbor",
        )
        w = nb.load(rib_warped).get_fdata()
        ribbon_mask = (w > 0.5)
        if not ribbon_mask.any():
            raise RuntimeError("warped ribbon produced empty mask")
        n_ribbon = int(ribbon_mask.sum())
    except Exception as e:
        ribbon_ok = False
        ribbon_mask = None
        n_ribbon = np.nan
        print(f"[{sub}] ribbon unavailable/failed: {e}")

    # ---------- GM probseg (to bm1 grid) ----------
    gm_path, gm_is_mni = find_gm_probseg(sub)
    gm_on_ref = os.path.join(TMP_DIR, f"{sub}_GM_probseg_on_run01mask.nii.gz")

    try:
        if gm_is_mni:
            # Identity resample to bm1 grid (no transforms)
            ants_apply(
                ants_sif=ants_sif,
                inp=gm_path,
                ref=bm1,
                out=gm_on_ref,
                transforms=[],
                interp="Linear",
            )
        else:
            # Warp T1w -> bm1 grid
            ants_apply(
                ants_sif=ants_sif,
                inp=gm_path,
                ref=bm1,
                out=gm_on_ref,
                transforms=[h5],
                interp="Linear",
            )
    except Exception as e:
        die(f"[{sub}] GM_probseg warp/resample failed: {e}")

    gm_img = nb.load(gm_on_ref)
    gm = gm_img.get_fdata(dtype=np.float32)
    gm = np.nan_to_num(gm, nan=0.0, posinf=0.0, neginf=0.0)
    gm_mask = (gm >= float(gm_thr))
    n_gm = int(gm_mask.sum())
    if not gm_mask.any():
        die(f"[{sub}] GM_probseg>=thr produced empty mask (thr={gm_thr})")

    # ---------- final ROI ----------
    if ribbon_ok and ribbon_mask is not None:
        roi = ribbon_mask & gm_mask & cov
        roi_kind = "ribbon∩GM∩coverage"
    else:
        if not allow_gm_only_fallback:
            die(f"[{sub}] No ribbon -> cannot build cortical GM ROI (use --allow-gm-only-fallback to build GM∩coverage).")
        roi = gm_mask & cov
        roi_kind = "GM∩coverage (NO RIBBON fallback)"

    if not roi.any():
        die(f"[{sub}] ROI empty after {roi_kind} (try lowering --gm-thr or inspect inputs).")

    # Save as uint8 with header from bm1
    out = nb.Nifti1Image(roi.astype(np.uint8), ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    out.to_filename(out_roi)

    n_roi = int(roi.sum())
    n_cov = int(cov.sum())
    frac = (n_roi / n_cov) if n_cov > 0 else float("nan")

    print(f"[{sub}] wrote ROI: {out_roi}")
    rib_msg = f"{int(n_ribbon)}" if np.isfinite(n_ribbon) else "NA"
    print(f"[{sub}] kind={roi_kind} | vox ROI={n_roi} | cov={n_cov} | frac={frac:.3f} | GMthr vox={n_gm} | ribbon={rib_msg}")

    stats = dict(
        gm_thr=float(gm_thr),
        roi_path=out_roi,
        roi_kind=roi_kind,
        n_roi=n_roi,
        n_cov=n_cov,
        frac=frac,
        n_gm=n_gm,
        n_ribbon=n_ribbon,
    )
    return out_roi, roi_kind, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True, help="e.g. 177 sub-177 105 ...")
    ap.add_argument("--ants-sif", required=True, help="Path to fmriprep SIF containing antsApplyTransforms")
    ap.add_argument("--gm-thr", type=float, default=0.30, help="Threshold for GM_probseg (default: 0.30)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--allow-gm-only-fallback",
        action="store_true",
        help="If ribbon missing/failed, fall back to GM_probseg>=thr ∩ coverage (NOT cortex-specific).",
    )
    args = ap.parse_args()

    log_tsv = os.path.join(GLM, "split_half", "roi_build_log.tsv")
    os.makedirs(os.path.dirname(log_tsv), exist_ok=True)

    rows = []   # success/fail records
    fails = []

    for s in args.subjects:
        sub = norm_sub(s)
        try:
            out_path, roi_kind, stats = build_one(
                sub=sub,
                ants_sif=args.ants_sif,
                gm_thr=args.gm_thr,
                overwrite=args.overwrite,
                allow_gm_only_fallback=args.allow_gm_only_fallback,
            )
            rows.append(dict(
                subject=sub,
                roi_kind=roi_kind,
                gm_thr=float(args.gm_thr),
                roi_path=out_path,
                n_roi=stats.get("n_roi", np.nan),
                n_cov=stats.get("n_cov", np.nan),
                frac=stats.get("frac", np.nan),
                n_gm=stats.get("n_gm", np.nan),
                n_ribbon=stats.get("n_ribbon", np.nan),
                error="",
            ))
        except Exception as e:
            msg = str(e)
            fails.append((sub, msg))
            rows.append(dict(
                subject=sub,
                roi_kind="FAIL",
                gm_thr=float(args.gm_thr),
                roi_path="",
                n_roi=np.nan,
                n_cov=np.nan,
                frac=np.nan,
                n_gm=np.nan,
                n_ribbon=np.nan,
                error=msg.replace("\t", " ").replace("\n", " "),
            ))

    # ---- write log TSV ----
    with open(log_tsv, "w") as f:
        f.write("subject\troi_kind\tgm_thr\troi_path\tn_roi\tn_cov\tfrac\tn_gm\tn_ribbon\terror\n")
        for r in rows:
            f.write(
                f"{r['subject']}\t{r['roi_kind']}\t{r['gm_thr']}\t{r['roi_path']}\t"
                f"{r['n_roi']}\t{r['n_cov']}\t{r['frac']}\t{r['n_gm']}\t{r['n_ribbon']}\t{r['error']}\n"
            )
    print(f"[ok] wrote log: {log_tsv}")

    # ---- list GM-only fallback subjects ----
    fallback_subs = [r["subject"] for r in rows if isinstance(r.get("roi_kind",""), str) and r["roi_kind"].startswith("GM∩coverage")]
    if fallback_subs:
        print("\nGM-only fallback subjects:")
        for s in fallback_subs:
            print(f"  - {s}")
    else:
        print("\nNo GM-only fallbacks.")

    # ---- failures ----
    if fails:
        print("\nFailures:")
        for sub, msg in fails:
            print(f"  - {sub}: {msg}")
        sys.exit(2)


if __name__ == "__main__":
    main()
