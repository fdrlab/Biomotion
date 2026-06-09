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
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1

BASE="/lustre/scratch/data/..."

# Point this to something that contains your RSA outputs somewhere underneath.
# If you want to be precise, set it to the exact .../runs folder.
RSA_ROOT="$BASE/searchlight_RSA/unsmoothed_cov20_max_dist5_k60"

SCRIPT="$BASE/searchlight_RSA/build_tfce_masks_and_qc_thresholds.py"

OUTDIR="$BASE/searchlight_RSA/unsmoothed_cov20_max_dist5_k60/group_masks/tfce_thresholds_excl114_139"
PREFIX="bio_rankortho_excl114_139"

PY="/home/.../.conda/envs/glmsingle_env/bin/python"

# sanity imports
$PY - <<'PY'
import sys
mods = ["numpy","nibabel","nilearn","matplotlib"]
missing=[]
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m,str(e)))
if missing:
    print("Missing/broken modules:")
    for m,e in missing:
        print(" ", m, "->", e)
    sys.exit(2)
print("OK: imports working")
PY

$PY -u "$SCRIPT" \
  --rsa-root "$RSA_ROOT" \
  --effect-pattern "**/*_beta-model6_castaldi_bmvsc_STD.nii.gz" \
  --coverage-pattern "**/*_glm_fit_ok.nii.gz" \
  --pick-mode newest \
  --min-neigh 60 \
  --cortex-mask "$BASE/group_masks_new/cortexROI_runIntersect_groupmask_cov20.nii.gz" \
  --exclude-subjects sub-114 sub-139 \
  --fractions 1.0 \
  --outdir "$OUTDIR" \
  --prefix "$PREFIX" \
  --cut-coords 35 -20 25 \
  --dpi 600 \
  --debug