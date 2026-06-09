#!/bin/bash
#SBATCH --job-name=searchlight
#SBATCH --output=searchlight_output_%A_%a.txt
#SBATCH --error=searchlight_error_%A_%a.txt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_short
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
module purge

# === Python env ===
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

# Threading caps
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE="/lustre/scratch/data/..."

# Script (make sure this exists at this path)
SCRIPT="$BASE/searchlight_RSA/biomotion-roi_tfce_mask.py"

# Your TFCE-safe reference mask (grid to match your RSA outputs)
TFCE_MASK="$BASE/searchlight_RSA/unsmoothed_cov20_max_dist5_k60/group_masks/tfce_thresholds_excl114_139/bio_rankortho_excl114_139_mask_p100_58of58_min1.nii.gz"

# <<< SET THIS >>> biomotion ROI from other study
BIOMOTION_ROI="$BASE/group_masks_new/BM>BMscram_0p05FWEclus_0p001u_k40.nii"

# Output
OUTDIR="$BASE/searchlight_RSA/unsmoothed_cov20_max_dist5_k60/group_masks/roi_intersections"
FINAL_MASK="$OUTDIR/tfce_p100_excl114_139_min1__INTERSECT__biomotionROI.nii.gz"

# Optional sanity imports (nilearn+matplotlib needed for QC PNG + resampling)
python - <<'PY'
import sys
mods = ["numpy","nibabel","nilearn","matplotlib"]
missing=[]
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m,str(e)))
if missing:
    print("Missing/broken modules (QC/resampling may fail if nilearn/matplotlib missing):")
    for m,e in missing:
        print(" ", m, "->", e)
    # don't hard-fail; the script itself will hard-fail only if resampling is needed
print("OK: python environment reachable")
PY

mkdir -p "$OUTDIR"

python -u "$SCRIPT" \
  --tfce-mask "$TFCE_MASK" \
  --roi-mask  "$BIOMOTION_ROI" \
  --out-mask  "$FINAL_MASK" \
  --roi-thr 0.5 \
  --tfce-thr 0.5 \
  --cut-coords 51 -67 1 \
  --dpi 200 \
  --debug

echo "[done] Final mask: $FINAL_MASK"
echo "[done] QC PNG next to mask (auto): qc_$(basename "${FINAL_MASK%.nii.gz}").png"

