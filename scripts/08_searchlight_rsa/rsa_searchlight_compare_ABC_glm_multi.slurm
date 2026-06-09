#!/bin/bash
#SBATCH --job-name=searchlight
#SBATCH --output=searchlight_output_%A_%a.txt
#SBATCH --error=searchlight_error_%A_%a.txt
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --partition=intelsr_short
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=1-60

set -euo pipefail
mkdir -p logs

source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate glmsingle_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CONDA_VERBOSITY=0

BASE=/lustre/scratch/data/...
SCRIPT=$BASE/searchlight_RSA/rsa_searchlight_compare_ABC_glm_multi.py

DTYPE="unsmoothed"
APP="B_up3x"

# Subjects: from B-branch output
B_DATA_DIR=$BASE/GLMsingle/output_upsampled_slicetime_changed/data
mapfile -t SUBJECTS < <(find "$B_DATA_DIR" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort)
NSUB=${#SUBJECTS[@]}

tid=$((SLURM_ARRAY_TASK_ID - 1))
if (( tid < 0 || tid >= NSUB )); then
  echo "Task id out of range: tid=$tid NSUB=$NSUB"
  exit 0
fi

SUB="${SUBJECTS[$tid]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${SUB} | ${DTYPE} | ${APP}"

# GROUP cortex universe mask (used for centers AND neighbor pool unless --neighbor-mask is set)
UNIVERSE_MASK=$BASE/group_masks_new/cortexROI_runIntersect_groupmask_cov20.nii.gz
if [[ ! -f "$UNIVERSE_MASK" ]]; then
  echo "Missing UNIVERSE_MASK: $UNIVERSE_MASK"
  exit 1
fi

# adaptive-K settings
K_NEIGH=60

HARD_MAXDIST="--k-hard-maxdist --k-maxdist-vox 5.0"

OUT_TAG="knn_${SUB}_${DTYPE}_${APP}"

python -u "$SCRIPT" \
  --subjects "$SUB" \
  --datatype "$DTYPE" \
  --approaches B_up3x \
  --center-mask "$UNIVERSE_MASK" \
  --neigh-mode adaptiveK \
  --k-neigh "$K_NEIGH" \
  $HARD_MAXDIST \
  --model-rdms \
    $BASE/searchlight_RSA/RDM_models/model6_castaldi_bmvsc_STD.npy \
    $BASE/searchlight_RSA/RDM_models/model6_castaldi_fvp_STD.npy \
    $BASE/searchlight_RSA/RDM_models/model6_castaldi_gtvfp_STD.npy \
  --metric crossnobis \
  --out-tag "$OUT_TAG" \
  --save-searchlight-rdmvec \
  --glm-rank \
  --save-summary-rdms \
  --debug 1