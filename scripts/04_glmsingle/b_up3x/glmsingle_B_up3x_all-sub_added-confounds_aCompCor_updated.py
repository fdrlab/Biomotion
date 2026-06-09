#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nb
from scipy.ndimage import label
from slicetime.main import run_slicetime
from glmsingle.glmsingle import GLM_single
from nilearn.image import mean_img, index_img
from nilearn import plotting
import matplotlib
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# === Paths to data directories ===
base_preproc_dir = ''
design_dir = ''
base_output_dir = '/output_B_aCompCor'

# === Argument parser ===
parser = argparse.ArgumentParser(description='Run GLMsingle for a single subject.')
parser.add_argument('--subject', required=False, default=None,
                    help='Subject ID (e.g. 103 or sub-103). If omitted, uses SLURM_ARRAY_TASK_ID to select a subject.')
parser.add_argument('--datatype', required=True, choices=['smoothed', 'unsmoothed'],
                    help='Type of data to process (smoothed or unsmoothed)')
parser.add_argument('--headless', action='store_true', help='Disable interactive plotting but save figures.')
args = parser.parse_args()

if args.headless:
    matplotlib.use('Agg')

# === GLM parameters ===
runs = ['01', '02']
orig_tr = 1.9
upsampling_factor = 3
new_tr = orig_tr / upsampling_factor
stimdur = 3.35
n_trim = 5  # drop first 5 TRs to match design
ab_tag = 'B_up3x'  # <— A/B tag to isolate outputs

# --- Nuisance (motion + spikes + optional aCompCor) configuration ---
FD_THRESH = 0.5         # mm (spike threshold)
ZDVARS_THRESH = 1.5     # std_dvars spike threshold
INCLUDE_NONSTEADY = True  # include non_steady_state_outlier* one-hots if present

# Optional aCompCor
INCLUDE_ACOMPCOR = True
N_ACOMPCOR = 6             # keep small to avoid over-regressing

# === GLMsingle options dictionary ===
opt = dict(
    wantlibrary=1,
    wantglmdenoise=0,
    wantfracridge=1,
    wantfileoutputs=[0, 0, 0, 0],
    wantmemoryoutputs=[0, 0, 0, 1],
    n_jobs=8,
    chunklen=10000
)

def _adjust_to_len(X, T):
    """Trim/pad (or make empty) to have exactly T rows."""
    if X is None or X.size == 0:
        return np.zeros((T, 0), dtype=np.float32)
    if X.shape[0] == T:
        return X.astype(np.float32, copy=False)
    if X.shape[0] > T:
        return X[:T, :].astype(np.float32, copy=False)
    pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=X.dtype)
    return np.vstack([X, pad]).astype(np.float32, copy=False)

def align_extras_to_design(extras_list, design_list, run_labels=None):
    """
    Given a list of extra-regressors (one per run, possibly misordered)
    and the list of design matrices, return a new list of extras
    whose i-th entry has the same #rows as design_list[i].
    """
    extras = [None if (X is None or X.size == 0) else X for X in extras_list]
    used   = [False] * len(extras)

    aligned = []
    for i, dm in enumerate(design_list):
        T = int(dm.shape[0])

        sel = None
        for j, X in enumerate(extras):
            if not used[j] and X is not None and int(X.shape[0]) == T:
                sel = j
                break

        if sel is None:
            best = None
            for j, X in enumerate(extras):
                if used[j]:
                    continue
                L = 0 if (X is None or X.size == 0) else int(X.shape[0])
                diff = abs(L - T)
                if best is None or diff < best[0]:
                    best = (diff, j)
            sel = best[1] if best is not None else None

        Xi = extras[sel] if sel is not None else None
        if sel is not None:
            used[sel] = True
        aligned.append(_adjust_to_len(Xi, T))

    if run_labels is None:
        run_labels = [f"{i:02d}" for i in range(len(design_list))]
    print("Extras↔Design alignment check:")
    for i, (dm, Xi) in enumerate(zip(design_list, aligned)):
        print(f"  run-{run_labels[i]}: design T={dm.shape[0]}  extras T={Xi.shape[0]}  p={Xi.shape[1]}")
    return aligned

# === Utility: contiguous-onset diagnostic ===
def contiguous_onset_counts(dm):
    per_cond, lengths = [], []
    total = 0
    for c in range(dm.shape[1]):
        col = (dm[:, c] != 0).astype(int)
        labeled, nseg = label(col)
        per_cond.append(int(nseg))
        total += int(nseg)
        seg_lens = [int(np.sum(labeled == s)) for s in range(1, nseg + 1)]
        lengths.append(seg_lens)
    return total, per_cond, lengths

# === Robust upsampling: one upsampled index per original onset ===
def upsample_sparse_design_matrix(design_matrix_orig, orig_tr, new_tr, total_upsampled_timepoints):
    n_orig_timepoints, n_regressors = design_matrix_orig.shape
    original_time_axis = np.arange(n_orig_timepoints) * orig_tr
    new_time_axis = np.arange(total_upsampled_timepoints) * new_tr
    up_dm = np.zeros((total_upsampled_timepoints, n_regressors), dtype=np.float32)
    for c in range(n_regressors):
        orig_onset_inds = np.where(design_matrix_orig[:, c] != 0)[0]
        for oi in orig_onset_inds:
            t = original_time_axis[oi]
            up_idx = int(np.argmin(np.abs(new_time_axis - t)))
            up_dm[up_idx, c] = 1.0
    print(f"Design matrix upsampling (singleton mapping) complete. New shape: {up_dm.shape}")
    return up_dm

# === Fallback: collapse contiguous upsampled blocks into singletons ===
def collapse_upsampled_to_single_onset_per_block(upsampled_dm):
    T_up, nconds = upsampled_dm.shape
    out = np.zeros_like(upsampled_dm, dtype=np.int8)
    for c in range(nconds):
        col = (upsampled_dm[:, c] > 0).astype(int)
        labeled, nseg = label(col)
        for seg_id in range(1, nseg + 1):
            seg_inds = np.where(labeled == seg_id)[0]
            if seg_inds.size == 0:
                continue
            first_up = int(seg_inds.min())
            out[first_up, c] = 1
    return out

# === Extra regressors (motion + spikes + optional aCompCor) ===
def build_extra_regs_for_run(sub, run_id, target_upsampled_T):
    if not HAVE_PANDAS:
        print(f"pandas not available; skipping extra regressors for run-{run_id}.")
        return None

    tsv = os.path.join(
        base_preproc_dir, sub, "func",
        f"{sub}_task-PLDs_run-{run_id}_desc-confounds_timeseries.tsv"
    )
    if not os.path.isfile(tsv):
        print(f"No confounds tsv for run-{run_id}; skipping extra regressors.")
        return None

    try:
        df = pd.read_csv(tsv, sep="\t")
    except Exception as e:
        print(f"Could not read confounds tsv ({e}); skipping extra regressors.")
        return None

    # 24-parameter motion
    base6 = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    deriv6 = [c + '_derivative1' for c in base6]
    sq12 = [c + '_power2' for c in base6 + deriv6]
    cols_24 = [c for c in (base6 + deriv6 + sq12) if c in df.columns]
    X_mot = df[cols_24].to_numpy(dtype=np.float32) if cols_24 else np.zeros((len(df), 0), np.float32)

    # Spikes (FD / std_dvars)
    spikes = []
    fd = pd.to_numeric(df['framewise_displacement'], errors='coerce').values if 'framewise_displacement' in df.columns else np.full(len(df), np.nan, np.float32)
    zd = (pd.to_numeric(df['std_dvars'], errors='coerce').values
          if 'std_dvars' in df.columns else
          pd.to_numeric(df['dvars'], errors='coerce').values
          if 'dvars' in df.columns else
          np.full(len(df), np.nan, np.float32))
    bad = (fd > FD_THRESH) | (zd > ZDVARS_THRESH)
    bad_idx = np.where(np.asarray(bad))[0]
    for idx in bad_idx:
        v = np.zeros(len(df), dtype=np.float32)
        v[idx] = 1.0
        spikes.append(v)
    X_spk = np.stack(spikes, axis=1) if spikes else np.zeros((len(df), 0), np.float32)

    # Optional non-steady-state outliers
    if INCLUDE_NONSTEADY:
        nss_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]
        for c in nss_cols:
            spikes.append(pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32).values)
        if spikes:
            X_spk = np.stack(spikes, axis=1)

    # Optional aCompCor
    X_ac = np.zeros((len(df), 0), np.float32)
    if INCLUDE_ACOMPCOR:
        cc_cols = [c for c in df.columns if c.startswith('a_comp_cor')]
        if cc_cols:
            cc_cols = cc_cols[:max(0, int(N_ACOMPCOR))]
            if cc_cols:
                X_ac = df[cc_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        print(f"aCompCor columns used (run-{run_id}): {X_ac.shape[1]}")

    # Trim first n_trim rows to match BOLD/design
    if len(df) <= n_trim:
        print(f"Not enough TRs after trimming for run-{run_id}; skipping extra regressors.")
        return None
    X = np.concatenate([X_mot[n_trim:], X_spk[n_trim:], X_ac[n_trim:]], axis=1)

    # Drop constant/all-zero/NaN, then z-score
    if X.size:
        keep = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if not np.isfinite(col).any(): continue
            if np.allclose(col, 0.0):      continue
            if np.nanstd(col) == 0:        continue
            keep.append(j)
        if keep:
            X = X[:, keep]
            X[~np.isfinite(X)] = 0.0
            mu = np.mean(X, axis=0, keepdims=True)
            sd = np.std(X, axis=0, keepdims=True); sd[sd == 0] = 1.0
            X = (X - mu) / sd
        else:
            X = np.zeros((X.shape[0], 0), dtype=np.float32)

    # Upsample by replication to B_up3x grid
    X_up = np.repeat(X, upsampling_factor, axis=0) if X.size else X

    # Ensure length matches target upsampled BOLD for this run
    Tu = X_up.shape[0]
    if Tu != target_upsampled_T:
        if Tu > target_upsampled_T:
            X_up = X_up[:target_upsampled_T, :]
        else:
            pad = np.zeros((target_upsampled_T - Tu, X_up.shape[1]), dtype=X_up.dtype)
            X_up = np.vstack([X_up, pad])
        print(f"Adjusted extra regressors length {Tu} -> {target_upsampled_T} for run-{run_id}.")

    print(f"Extra regressors (run-{run_id}): shape {X_up.shape} "
          f"[{len(cols_24)} motion, {X_spk.shape[1] if X_spk.size else 0} spikes, "
          f"{X_ac.shape[1] if INCLUDE_ACOMPCOR else 0} aCompCor]")
    return X_up.astype(np.float32, copy=False)

# === Subject selection ===
all_subject_ids = sorted(
    d for d in os.listdir(base_preproc_dir)
    if d.startswith('sub-') and os.path.isdir(os.path.join(base_preproc_dir, d))
)

if args.subject is not None:
    subject_clean = args.subject[4:] if args.subject.startswith('sub-') else args.subject
    subject_id = f"sub-{subject_clean}"
else:
    task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', None)
    if task_id_str is None:
        raise SystemExit("Provide --subject or submit as an array job with SLURM_ARRAY_TASK_ID set.")
    idx = int(task_id_str) - 1
    if idx < 0 or idx >= len(all_subject_ids):
        raise SystemExit(f"SLURM_ARRAY_TASK_ID {task_id_str} out of range (1..{len(all_subject_ids)}).")
    subject_id = all_subject_ids[idx]
    print(f"Array selection: SLURM_ARRAY_TASK_ID={task_id_str} -> subject={subject_id}")

data_type = args.datatype

# === Output dirs ===
data_output_dir_subject = os.path.join(base_output_dir, 'data', subject_id, data_type)
vis_output_dir_subject = os.path.join(base_output_dir, 'visualizations', subject_id, data_type)
os.makedirs(data_output_dir_subject, exist_ok=True)
os.makedirs(vis_output_dir_subject, exist_ok=True)

# === Paths/templates ===
if data_type == 'smoothed':
    bold_subdir = 'smoothed'; bold_desc = 'smoothed'
else:
    bold_subdir = 'func';     bold_desc = 'preproc'
mask_desc = 'brain_mask'; mask_subdir = 'func'

first_run_bold_path = os.path.join(
    base_preproc_dir, subject_id, bold_subdir,
    f"{subject_id}_task-PLDs_run-{runs[0]}_space-MNI152NLin2009cAsym_res-1.7_desc-{bold_desc}_bold.nii.gz"
)
if not os.path.isfile(first_run_bold_path):
    raise FileNotFoundError(f"Missing BOLD file for first run (to get reference shape): {first_run_bold_path}")

print(f"Loading initial BOLD for shape/affine reference from {first_run_bold_path}...")
initial_bold_img = nb.load(first_run_bold_path)
reference_affine = initial_bold_img.affine
original_bold_shape = initial_bold_img.shape
voxel_size = initial_bold_img.header.get_zooms()[:3]
print(f"Voxel size (mm): {voxel_size}")
print(f"Reference affine set from {first_run_bold_path}")
print(f"Original BOLD shape set to {original_bold_shape}")

print(f"\n=== Processing {subject_id} with {data_type} data — {ab_tag} ===")

# === Check cached results ===
results_save_path = os.path.join(
    data_output_dir_subject,
    f"{subject_id}_{data_type}_{ab_tag}_glmsingle_results.npz"
)

if os.path.exists(results_save_path):
    print(f"\nCached GLMsingle results found at: {results_save_path}")
    print("Skipping data preparation and GLMsingle analysis, loading results directly.")
    with np.load(results_save_path, allow_pickle=True) as z:
        results = z['results'].item() if 'results' in z else z[z.files[0]].item()
else:
    print(f"\n  No cached GLMsingle results found at: {results_save_path}")
    print("  Running full GLMsingle analysis pipeline.")

    bold_paths_for_glmsingle = []
    design_paths_for_glmsingle = []
    extra_regs_by_run = []
    
    for run in runs:
        print(f"\n Processing run-{run} for {subject_id} ({data_type})...")
        up_bold_path = os.path.join(
            data_output_dir_subject, f"{subject_id}_run-{run}_{ab_tag}_masked_bold_final.nii.gz"
        )
        up_dm_path = os.path.join(
            data_output_dir_subject, f"{subject_id}_task-PLDs_run-{run}_{ab_tag}_design_matrix.npy"
        )

        if os.path.exists(up_bold_path) and os.path.exists(up_dm_path):
            print(f" A/B files for run-{run} already exist. Skipping regeneration.")
            target_T = nb.load(up_bold_path).shape[3]
            X_up = build_extra_regs_for_run(subject_id, run, target_T)
            extra_regs_by_run.append(None if (X_up is None or X_up.shape[1] == 0) else X_up)
        else:
            bold_path = os.path.join(
                base_preproc_dir, subject_id, bold_subdir,
                f"{subject_id}_task-PLDs_run-{run}_space-MNI152NLin2009cAsym_res-1.7_desc-{bold_desc}_bold.nii.gz"
            )
            mask_path = os.path.join(
                base_preproc_dir, subject_id, mask_subdir,
                f"{subject_id}_task-PLDs_run-{run}_space-MNI152NLin2009cAsym_res-1.7_desc-{mask_desc}.nii.gz"
            )
            dm_orig_path = os.path.join(
                design_dir, f"{subject_id}_task-PLDs_run-{run}_desc-origevents_desc-conditionlevel_design_matrix.npy"
            )

            if not os.path.isfile(bold_path):
                print(f"Missing BOLD file for run-{run}: {bold_path}. Skipping for this run.")
                continue
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Missing brain mask file: {mask_path}")
            if not os.path.isfile(dm_orig_path):
                raise FileNotFoundError(f"Missing original (non-upsampled) design matrix: {dm_orig_path}")
            
            print(f"      Loading BOLD: {bold_path}")
            bold_img = nb.load(bold_path)
            bold_array = bold_img.get_fdata()
            assert bold_array.ndim == 4, f"ERROR: BOLD is not 4D, got {bold_array.shape}"
            
            print(f"      Loading mask: {mask_path}")
            mask_img = nb.load(mask_path)
            mask_array = mask_img.get_fdata()
            if mask_array.shape != bold_array.shape[:3]:
                raise ValueError(f"Mask {mask_array.shape} != BOLD spatial {bold_array.shape[:3]}")

            print("      Applying brain mask...")
            bold_array_masked = bold_array.copy()
            bold_array_masked[~mask_array.astype(bool)] = 0

            T_pre = bold_array_masked.shape[3]
            flat = bold_array_masked.reshape(-1, T_pre)[mask_array.flatten().astype(bool), :]
            if flat.size == 0:
                raise ValueError("ERROR: Masked BOLD empty after mask!")
            if np.sum(np.std(flat, axis=1) < 1e-6) > 0.05 * flat.shape[0]:
                raise ValueError("ERROR: Too many near-zero-variance voxels after masking.")
            print(f"   Masked BOLD variance check passed.")

            if T_pre <= n_trim:
                raise ValueError(f"ERROR: Only {T_pre} vols; cannot drop {n_trim} TRs.")
            T_trim = T_pre - n_trim
            print(f"      Trimming first {n_trim} TRs: {T_pre} -> {T_trim}")
            bold_trim = bold_array_masked[..., n_trim:]

            tmp_trim = os.path.join(data_output_dir_subject, f"{subject_id}_run-{run}_{ab_tag}_masked_bold_trimmed.nii.gz")
            nb.Nifti1Image(bold_trim, affine=bold_img.affine).to_filename(tmp_trim)

            print(f"      Loading original design (TR grid): {dm_orig_path}")
            dm_orig = np.load(dm_orig_path)
            if dm_orig.ndim != 2:
                raise ValueError(f"Design matrix is not 2D, got {dm_orig.shape}")
            if dm_orig.shape[0] != T_trim:
                raise ValueError(f"TR mismatch BEFORE upsampling: design rows={dm_orig.shape[0]} vs trimmed BOLD={T_trim}")

            print(f"      Upsampling BOLD {orig_tr:.2f}s -> {new_tr:.3f}s (×{upsampling_factor}) via slicetime...")
            z = bold_trim.shape[2]
            run_slicetime(
                inpath=tmp_trim,
                outpath=up_bold_path,
                slicetimes=np.zeros(z),
                tr_old=orig_tr,
                tr_new=new_tr,
                time_dim=2,
                offset=0
            )
            os.remove(tmp_trim)
            
            info = nb.load(up_bold_path)
            target_T = info.shape[3]
            print(f"      Upsampling design to {target_T} timepoints...")
            dm_up = upsample_sparse_design_matrix(dm_orig, orig_tr, new_tr, target_T)

            # Diagnostics on contiguity
            orig_ones = (dm_orig != 0).sum(axis=0).astype(int).tolist()
            total_orig = int(np.sum(orig_ones))
            total_seg, per_cond_seg, seg_lens = contiguous_onset_counts(dm_up)
            max_seg_len = max([max(l) if l else 0 for l in seg_lens])
            print("      Diagnostic: original onsets per condition:", orig_ones)
            print("      Diagnostic: upsampled contiguous segments per condition:", per_cond_seg)
            print(f"      Diagnostic: total original onsets={total_orig}, total upsampled segments={total_seg}")
            print(f"      Diagnostic: max contiguous-run length in upsampled DM = {max_seg_len}")

            if total_seg != total_orig or max_seg_len > 1:
                print("Collapsing upsampled blocks into singletons...")
                collapsed = collapse_upsampled_to_single_onset_per_block(dm_up)
                total_seg2, per_cond2, seg_lens2 = contiguous_onset_counts(collapsed)
                max_seg_len2 = max([max(l) if l else 0 for l in seg_lens2])
                print("      After collapse -> total segments:", total_seg2, "per condition:", per_cond2, "max seg len:", max_seg_len2)
                if total_seg2 == total_orig and max_seg_len2 == 1:
                    dm_up = collapsed.astype(np.float32)
                else:
                    raise ValueError("Collapsed design still mismatched relative to original onsets.")

            if dm_up.shape[0] != target_T:
                print(f"      ADJUSTING DM length {dm_up.shape[0]} -> BOLD {target_T}...")
                if dm_up.shape[0] > target_T:
                    dm_up = dm_up[:target_T, :]
                else:
                    pad = np.zeros((target_T - dm_up.shape[0], dm_up.shape[1]), dtype=dm_up.dtype)
                    dm_up = np.vstack((dm_up, pad))
                print(f"      ADJUSTED: DM new shape: {dm_up.shape}")

            np.save(up_dm_path, dm_up)
            print(f"Saved verified design: {up_dm_path}")

            X_up = build_extra_regs_for_run(subject_id, run, target_T)
            extra_regs_by_run.append(None if (X_up is None or X_up.shape[1] == 0) else X_up)

        bold_paths_for_glmsingle.append(up_bold_path)
        design_paths_for_glmsingle.append(up_dm_path)

    # === Final confirmation and run GLMsingle ===
    print("\n--- Final Design Matrix Confirmation ---")
    for i, dm_path in enumerate(design_paths_for_glmsingle):
        run_id = runs[i]
        T_bold = nb.load(bold_paths_for_glmsingle[i]).shape[3]
        dm_data = np.load(dm_path)
        assert dm_data.shape[0] == T_bold, f"Run {run_id}: DM ({dm_data.shape[0]}) vs BOLD ({T_bold}) mismatch."
        print(f"Run {run_id} DM shape {dm_data.shape} matches BOLD length {T_bold}.")
    print("--- End of Confirmation ---")

    print("\nLoading data into memory for GLMsingle...")
    loaded_bold_data = [nb.load(p).get_fdata() for p in bold_paths_for_glmsingle]
    loaded_design_data = [np.load(p) for p in design_paths_for_glmsingle]
    
    print("Performing pre-GLMsingle data consistency checks...")
    for i, (dm, bold) in enumerate(zip(loaded_design_data, loaded_bold_data)):
        run_id = runs[i]
        assert dm.shape[0] == bold.shape[3], f"ERROR: Run {run_id} - DM ({dm.shape[0]}) vs BOLD ({bold.shape[3]}) mismatch."
        if np.any(np.isnan(bold)) or np.any(np.isinf(bold)):
            raise ValueError(f"ERROR: Run {run_id} - BOLD contains NaN/Inf after upsampling!")
        if np.any(np.isnan(dm)) or np.any(np.isinf(dm)):
            raise ValueError(f"ERROR: Run {run_id} - Design contains NaN/Inf!")
    print("All DM/BOLD lengths match and are NaN/Inf-free.")

    # Build extras preliminary list in the exact run loop order
    prelim_extras = []
    for i, bold_path in enumerate(bold_paths_for_glmsingle):
        T_bold = nb.load(bold_path).shape[3]
        X = extra_regs_by_run[i]
        if X is None:
            prelim_extras.append(np.zeros((T_bold, 0), dtype=np.float32))
        else:
            prelim_extras.append(_adjust_to_len(X.astype(np.float32, copy=False), T_bold))

    # Align extras to the (current) design lengths
    aligned_extras = align_extras_to_design(prelim_extras, loaded_design_data, run_labels=runs)

    # === Make all runs the same length by TRIMMING (no padding) ===
    T_list   = [int(dm.shape[0]) for dm in loaded_design_data]
    T_target = min(T_list)  # shortest run
    if len(set(T_list)) > 1:
        print(f"Trimming all runs to the shortest length T={T_target}. Before: {T_list}")
        for i in range(len(loaded_design_data)):
            Ti = int(loaded_design_data[i].shape[0])
            if Ti > T_target:
                cut = Ti - T_target

                # (optional) safety: warn if we would remove any non-zero DM rows
                if np.any(loaded_design_data[i][-cut:, :] != 0):
                    print(f"run {i}: trimming will remove non-zero rows in the design tail ({cut} rows). "
                          "This is usually just buffer; double-check if unexpected.")

                # trim DESIGN
                loaded_design_data[i] = loaded_design_data[i][:-cut, :]

                # trim EXTRAS
                Xe = aligned_extras[i]
                aligned_extras[i] = Xe[:-cut, :]

                # trim BOLD (time is last axis)
                loaded_bold_data[i] = loaded_bold_data[i][..., :-cut]

        T_list_after = [int(dm.shape[0]) for dm in loaded_design_data]
        print(f"All runs now have identical T: {T_list_after}")
    else:
        print("Runs already equal length; no trimming needed.")

    # Attach extras *after* standardization (and skip any pre-normalize-by-length step)
    opt["extra_regressors"] = aligned_extras
    
    # Final safety checks
    assert len(loaded_design_data) == len(loaded_bold_data) == len(aligned_extras), \
       "Design/BOLD/extras lists must have the same number of runs."

    for i in range(len(loaded_design_data)):
        Ti = loaded_design_data[i].shape[0]
        Tb = loaded_bold_data[i].shape[3]
        Xe = opt["extra_regressors"][i]
        assert Ti == Tb, f"Run {i}: design T={Ti} != BOLD T={Tb}"
        assert Xe.shape[0] == Ti, f"Run {i}: extras T={Xe.shape[0]} != design T={Ti}"
    print("Same-length + extras sanity checks passed.")

    # Run GLMsingle
    print("\nRunning GLMsingle...")
    glmsingle_obj = GLM_single(opt)
    results = glmsingle_obj.fit(loaded_design_data, loaded_bold_data, stimdur=stimdur, tr=new_tr)
    np.savez_compressed(results_save_path, results=results)
    print(f"Results saved to {results_save_path}")


# === VISUALIZATIONS of GLMsingle results (merged) ===
print("\n=== Starting Visualization of GLMsingle results ===")
try:
    mask_file = os.path.join(
        base_preproc_dir, subject_id, mask_subdir,
        f"{subject_id}_task-PLDs_run-01_space-MNI152NLin2009cAsym_res-1.7_desc-brain_mask.nii.gz"
    )
    ref_img   = nb.load(first_run_bold_path)
    mask_img  = nb.load(mask_file)
    mask_array = mask_img.get_fdata().astype(bool)
    affine    = ref_img.affine
    xyz       = ref_img.shape[:3]

    out_vis = vis_output_dir_subject
    out_dat = data_output_dir_subject
    os.makedirs(out_vis, exist_ok=True)

    def get_typed_field(typed_dict, key):
        if key in typed_dict:
            return typed_dict[key]
        for k in typed_dict.keys():
            if k.lower() == key.lower():
                return typed_dict[k]
        raise KeyError(f"Field '{key}' not found in results['typed'].")

    # --- helpers for robust unmasking and NaN-safe plotting ---
    def to_volume(data, mask_bool, xyz):
        if data.ndim == 4:
            data = np.nanmean(data, axis=3)
        if data.ndim == 3:
            if data.shape == xyz:
                return data.astype(np.float32, copy=False)
            raise ValueError(f"3D data has wrong shape {data.shape}, expected {xyz}.")
        if data.ndim == 1:
            nmask = int(mask_bool.sum())
            nvox  = int(np.prod(xyz))
            if data.size == nmask:
                vol = np.full(xyz, np.nan, dtype=np.float32)
                vol[mask_bool] = data.astype(np.float32, copy=False)
                return vol
            if data.size == nvox:
                return data.reshape(xyz).astype(np.float32, copy=False)
            raise ValueError(f"1D data length {data.size} != mask {nmask} or grid {nvox}.")
        raise ValueError(f"Unsupported ndim={data.ndim} for visualization.")

    def sanitize_for_plot(vol3d):
        pane = np.array(vol3d, dtype=np.float32, copy=True)
        pane[~np.isfinite(pane)] = 0.0
        return pane

    typed = results['typed']

    # Mean BOLD per run (A/B-specific files)
    print("\n--- Generating Mean BOLD Slice Visualizations for Each Run ---")
    for run in runs:
        bold_file = os.path.join(data_output_dir_subject, f"{subject_id}_run-{run}_{ab_tag}_masked_bold_final.nii.gz")
        if os.path.exists(bold_file):
            mean_bold = mean_img(nb.load(bold_file), copy_header=True)
            output_file = os.path.join(out_vis, f"{subject_id}_{data_type}_{ab_tag}_run-{run}_mean_bold.png")
            plotting.plot_epi(mean_bold, title=f"{subject_id} Run-{run} Mean BOLD ({ab_tag})", output_file=output_file)
            print(f"Saved mean BOLD plot for run-{run} to {output_file}")
        else:
            print(f"File not found for run-{run}: {bold_file}")
    print("--- Mean BOLD Visualizations Complete ---")

    # Handle betas
    betas = get_typed_field(typed, 'betasmd')
    print(f"\n--- Betas Summary ---")
    print(f"'betasmd' shape: {betas.shape}")

    if betas.ndim == 2:
        n_vox, n_cond = betas.shape
        betas_4d = np.full(xyz + (n_cond,), np.nan, dtype=np.float32)
        m = mask_array
        for c in range(n_cond):
            vol = np.full(xyz, np.nan, dtype=np.float32)
            vol[m] = betas[:, c]
            betas_4d[..., c] = vol
        print(f"Unmasked betas to 4D: {betas_4d.shape} (conditions = {n_cond})")
    elif betas.ndim == 4:
        betas_4d = betas.astype(np.float32, copy=False)
        n_cond   = betas_4d.shape[3]
        print(f"Detected 4D betas; conditions = {n_cond}, ~voxels = {np.prod(xyz)}")
    else:
        raise ValueError(f"Unexpected 'betasmd' ndim={betas.ndim}; expected 2 or 4.")

    betas_img  = nb.Nifti1Image(betas_4d.astype(np.float32), affine=affine)
    betas_path = os.path.join(out_dat, f"{subject_id}_{data_type}_{ab_tag}_betas.nii.gz")
    betas_img.to_filename(betas_path)
    print(f"Betas saved to {betas_path}")

    # Glass brain
    try:
        mean_beta_img = mean_img(betas_img, copy_header=True)
        mean_beta_arr = np.nan_to_num(mean_beta_img.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
        mean_beta_img = nb.Nifti1Image(mean_beta_arr.astype(np.float32), affine)
        gb_png = os.path.join(out_vis, f"{subject_id}_{data_type}_{ab_tag}_glass_brain.png")
        plotting.plot_glass_brain(mean_beta_img, title=f"{subject_id} Mean GLM Betas ({data_type}, {ab_tag})", output_file=gb_png)
        print(f"Glass brain saved: {gb_png}")
    except Exception as e:
        print(f"Glass brain plot failed: {e}")

    # Per-condition maps
    try:
        idxs_to_plot = [0] + ([1] if n_cond > 1 else []) + ([n_cond - 1] if n_cond > 2 else [])
        for ci in idxs_to_plot:
            con_img = index_img(betas_img, ci)
            con_arr = np.nan_to_num(con_img.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
            con_img = nb.Nifti1Image(con_arr.astype(np.float32), affine)
            out_png = os.path.join(out_vis, f"{subject_id}_{data_type}_{ab_tag}_beta_{ci:03d}.png")
            plotting.plot_stat_map(con_img, title=f"{subject_id} Beta {ci} ({data_type}, {ab_tag})", output_file=out_png, threshold=2.0)
            print(f"Saved beta map for condition {ci}: {out_png}")
    except Exception as e:
        print(f"Per-condition beta maps failed: {e}")

    # Mid-slice average betas heatmap
    try:
        mid_slice = xyz[2] // 2
        avg_betas_3d = np.nanmean(betas_4d, axis=3)
        pane = np.nan_to_num(avg_betas_3d[:, :, mid_slice], nan=0.0, posinf=0.0, neginf=0.0)
        plt.figure(figsize=(8, 6))
        plt.imshow(pane, cmap='RdBu_r', vmin=-5, vmax=5)
        plt.colorbar()
        plt.title(f'{subject_id} Average GLM Betas (Mid slice, {data_type}, {ab_tag})')
        plt.axis('off')
        mid_png = os.path.join(out_vis, f"{subject_id}_{data_type}_{ab_tag}_mid_slice_betas.png")
        plt.savefig(mid_png, bbox_inches='tight')
        plt.close()
        print(f"Mid-slice average betas saved: {mid_png}")
    except Exception as e:
        print(f"Mid-slice avg betas plot failed: {e}")

    # Summary panel with numeric readouts
    fields = ['betasmd', 'R2', 'HRFindex', 'FRACvalue']
    cmaps  = ['RdBu_r', 'hot', 'jet', 'jet']
    clims  = [[-5, 5], [0, 85], None, [0, 1]]  # R² in percent (0–100)

    try:
        plt.figure(figsize=(12, 8))
        for i, field in enumerate(fields):
            plt.subplot(2, 2, i + 1)
            data = get_typed_field(typed, field)

            if field == 'betasmd':
                if data.ndim == 4:
                    data_to_plot = np.nanmean(data, axis=3)
                elif data.ndim == 2:
                    mean_over_cond = np.nanmean(data, axis=1)
                    vol = np.full(xyz, np.nan, dtype=np.float32)
                    vol[mask_array] = mean_over_cond
                    data_to_plot = vol
                else:
                    data_to_plot = np.zeros(xyz, dtype=np.float32)
                pane = sanitize_for_plot(data_to_plot)
                title = 'Avg GLM Betas (typed)'

            elif field.lower() == 'r2':
                if data.ndim == 1:
                    vol = np.full(xyz, np.nan, dtype=np.float32); vol[mask_array] = data; data_to_plot = vol
                elif data.ndim == 4:
                    data_to_plot = np.nanmean(data, axis=3)
                else:
                    data_to_plot = np.zeros(xyz, dtype=np.float32)
                pane = sanitize_for_plot(data_to_plot)
                slice2 = pane[:, :, xyz[2] // 2].copy()
                slice2[slice2 < 2.0] = 0.0  # display-only threshold
                title = 'R²'
                r2_vals = data[np.isfinite(data)]
                if r2_vals.size:
                    print("\n🔎  R² Quality Summary (percent):")
                    print(f"  Max R²:  {np.max(r2_vals):.2f}")
                    print(f"  Mean R²: {np.nanmean(r2_vals):.2f}")
                    for t in [0, 2, 4, 6, 8, 10, 12, 14]:
                        print(f"  R² > {t:2d}: {np.sum(r2_vals > t)}")

            elif field.lower() in ('hrfindex',):
                if data.ndim == 1:
                    vol = np.full(xyz, np.nan, dtype=np.float32); vol[mask_array] = data; data_to_plot = vol
                else:
                    data_to_plot = np.nanmean(data, axis=3) if data.ndim == 4 else data
                pane = sanitize_for_plot(data_to_plot)
                title = 'HRFindex'

            elif field.lower() in ('fracvalue',):
                if data.ndim == 1:
                    vol = np.full(xyz, np.nan, dtype=np.float32); vol[mask_array] = data; data_to_plot = vol
                else:
                    data_to_plot = np.nanmean(data, axis=3) if data.ndim == 4 else data
                pane = sanitize_for_plot(data_to_plot)
                title = 'FRACvalue'

            else:
                data_to_plot = np.zeros(xyz, dtype=np.float32)
                pane = data_to_plot
                title = field

            clim = clims[i]
            mid = pane[:, :, xyz[2] // 2]
            if clim is None:
                plt.imshow(mid, cmap=cmaps[i])
            else:
                plt.imshow(mid, cmap=cmaps[i], vmin=clim[0], vmax=clim[1])
            plt.colorbar()
            plt.title(f'{subject_id} {title} ({data_type}, {ab_tag})')
            plt.axis('off')

        plt.tight_layout()
        summary_png = os.path.join(out_vis, f"{subject_id}_{data_type}_{ab_tag}_summary_fields.png")
        plt.savefig(summary_png, bbox_inches='tight')
        plt.close()
        print(f"Summary fields plot saved: {summary_png}")
    except Exception as e:
        print(f"Summary fields visualization failed: {e}")

    print("=== Visualization complete. ===")

except Exception as e:
    print(f"Visualization block failed (continuing): {e}")
